//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/compaction/compaction_picker_rl.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <mutex>
#include <string>
#include <vector>

#include "db/compaction/rl_compaction_telemetry.h"
#include "logging/logging.h"

namespace ROCKSDB_NAMESPACE {

namespace {

double Clamp01(double value) { return std::max(0.0, std::min(1.0, value)); }

double NormalizeBytes(uint64_t value, uint64_t norm) {
  return norm == 0 ? 0.0 : Clamp01(static_cast<double>(value) / norm);
}

}  // namespace

RLCompactionPicker::RLRewardBreakdown RLCompactionPicker::ComputeReward(
    int l0_files, uint64_t pcb,
    const RLCompactionTelemetrySnapshot& telemetry) const {
  RLRewardBreakdown reward;
  const double delta_f0_pos =
      std::max(0.0, static_cast<double>(l0_files - rl_prev_l0_files_));
  const double delta_pcb_pos =
      std::max(0.0, static_cast<double>(static_cast<int64_t>(pcb) -
                                        static_cast<int64_t>(rl_prev_pcb_)));
  const bool l0_pressure_relieved = l0_files < rl_prev_l0_files_;
  const bool pcb_pressure_relieved = pcb < rl_prev_pcb_;

  reward.delta_f0_norm = Clamp01(delta_f0_pos / kL0HardCap);
  reward.delta_pcb_norm = Clamp01(delta_pcb_pos / kPcbHardCap);
  reward.stall_norm =
      telemetry.stall_count > 0 || telemetry.stop_count > 0 ? 1.0 : 0.0;
  reward.compact_cost = telemetry.l0_compactions_completed > 0 ? 1.0 : 0.0;
  reward.bytes_compacted_norm = NormalizeBytes(
      telemetry.compaction_bytes_read + telemetry.compaction_bytes_written,
      kCompactionBytesNorm);
  reward.pressure_relieved =
      l0_pressure_relieved || pcb_pressure_relieved ? 1.0 : 0.0;

  reward.reward = -0.30 * reward.delta_f0_norm - 0.30 * reward.delta_pcb_norm -
                  0.30 * reward.stall_norm - 0.10 * reward.compact_cost -
                  0.05 * reward.bytes_compacted_norm +
                  0.50 * reward.pressure_relieved;
  return reward;
}

double RLCompactionPicker::L0CompactionScore(
    const VersionStorageInfo* vstorage) const {
  for (int i = 0; i <= vstorage->MaxInputLevel(); ++i) {
    if (vstorage->CompactionScoreLevel(i) == 0) {
      return vstorage->CompactionScore(i);
    }
  }
  return 0.0;
}

bool RLCompactionPicker::QueryRL(const VersionStorageInfo* vstorage,
                                 int l0_files, uint64_t pcb) const {
  auto now = std::chrono::steady_clock::now();
  const double l0_score = L0CompactionScore(vstorage);
  const bool l0_score_emergency = l0_score >= 1.0 && l0_files > 0;
  if (!l0_score_emergency && now - rl_last_query_time_ < kMinQueryInterval) {
    return rl_last_decision_;
  }

  rl_last_compaction_picked_ = false;
  RLCompactionTelemetrySnapshot telemetry =
      RLCompactionTelemetry::Get().Consume();
  RLRewardBreakdown reward = ComputeReward(l0_files, pcb, telemetry);

  const double f0_norm = Clamp01(static_cast<double>(l0_files) / kL0HardCap);
  const double df0 = static_cast<double>(l0_files - rl_prev_l0_files_);
  const double df0_norm = std::max(-1.0, std::min(1.0, df0 / kL0HardCap));
  const double s0_norm = Clamp01(l0_score);
  const double pcb_norm = NormalizeBytes(pcb, kPcbHardCap);
  const double stall = reward.stall_norm;
  const double bw = NormalizeBytes(telemetry.flushed_bytes, kWriteBytesNorm);
  const bool stall_emergency = telemetry.stop_count > 0 && l0_files > 0;

  RLState state{f0_norm, df0_norm, s0_norm,       pcb_norm,
                stall,   bw,       reward.reward, false};

  ROCKS_LOG_INFO(
      ioptions_.logger,
      "RL compaction state: f0=%.4f df0=%.4f score0=%.4f pcb=%.4f "
      "stall=%.4f bw=%.4f reward=%.4f "
      "reward_components={delta_f0=%.4f,delta_pcb=%.4f,stall=%.4f,"
      "compact=%.4f,bytes=%.4f,relieved=%.4f} "
      "telemetry={flushed=%" PRIu64 ",compact_read=%" PRIu64
      ",compact_written=%" PRIu64 ",l0_completed=%" PRIu64
      ",l0_scheduled=%" PRIu64 ",stall_count=%" PRIu64 ",stop_count=%" PRIu64
      "}",
      state.f0, state.df0, state.s0, state.pcb, state.stall, state.bw,
      state.reward, reward.delta_f0_norm, reward.delta_pcb_norm,
      reward.stall_norm, reward.compact_cost, reward.bytes_compacted_norm,
      reward.pressure_relieved, telemetry.flushed_bytes,
      telemetry.compaction_bytes_read, telemetry.compaction_bytes_written,
      telemetry.l0_compactions_completed, telemetry.l0_compactions_scheduled,
      telemetry.stall_count, telemetry.stop_count);

  RLQueryResult result = RLCompactionClient::Get().QueryAction(state);

  rl_prev_l0_files_ = l0_files;
  rl_prev_pcb_ = pcb;
  rl_last_query_time_ = now;

  if (!result.ok) {
    if (stall_emergency || l0_score_emergency) {
      rl_force_l0_compaction_pending_ = true;
      rl_last_decision_ = true;
      rl_cooldown_steps_ = 0;
      return true;
    }
    rl_force_l0_compaction_pending_ = false;
    rl_last_decision_ = LevelCompactionPicker::NeedsCompaction(vstorage);
    if (!rl_fallback_logged_) {
      ROCKS_LOG_WARN(
          ioptions_.logger,
          "RL compaction server unavailable or timed out; falling back to "
          "normal leveled compaction thresholds.");
      rl_fallback_logged_ = true;
    }
    return rl_last_decision_;
  }
  rl_fallback_logged_ = false;

  if (stall_emergency || l0_score_emergency) {
    ROCKS_LOG_WARN(ioptions_.logger,
                   "RL compaction action overridden by safety guard: "
                   "l0_score=%.4f stall_emergency=%d l0_files=%d",
                   l0_score, static_cast<int>(stall_emergency), l0_files);
    rl_force_l0_compaction_pending_ = true;
    rl_last_decision_ = true;
    rl_cooldown_steps_ = 0;
    return true;
  }

  switch (result.action) {
    case RLAction::kCompactNow:
      rl_force_l0_compaction_pending_ = true;
      rl_last_decision_ = true;
      return true;
    case RLAction::kDelay:
      rl_cooldown_steps_ = 1;
      rl_force_l0_compaction_pending_ = false;
      rl_last_decision_ = false;
      return false;
    case RLAction::kDoNothing:
    default:
      rl_force_l0_compaction_pending_ = false;
      rl_last_decision_ = false;
      return false;
  }
}

bool RLCompactionPicker::NeedsCompaction(
    const VersionStorageInfo* vstorage) const {
  if (!vstorage->ExpiredTtlFiles().empty() ||
      !vstorage->FilesMarkedForPeriodicCompaction().empty() ||
      !vstorage->BottommostFilesMarkedForCompaction().empty() ||
      !vstorage->FilesMarkedForCompaction().empty() ||
      !vstorage->FilesMarkedForForcedBlobGC().empty()) {
    std::unique_lock<std::mutex> lock(rl_mu_);
    rl_force_l0_compaction_pending_ = false;
    return true;
  }

  for (int i = 0; i <= vstorage->MaxInputLevel(); i++) {
    if (vstorage->CompactionScoreLevel(i) != 0 &&
        vstorage->CompactionScore(i) >= 1) {
      std::unique_lock<std::mutex> lock(rl_mu_);
      rl_force_l0_compaction_pending_ = false;
      return true;
    }
  }

  int l0_files = vstorage->NumLevelFiles(0);
  uint64_t pcb = vstorage->estimated_compaction_needed_bytes();

  if (l0_files >= kL0HardCap || pcb >= kPcbHardCap) {
    std::unique_lock<std::mutex> lock(rl_mu_);
    rl_prev_l0_files_ = l0_files;
    rl_prev_pcb_ = pcb;
    rl_last_decision_ = true;
    rl_force_l0_compaction_pending_ = l0_files > 0;
    rl_cooldown_steps_ = 0;
    return true;
  }

  std::unique_lock<std::mutex> lock(rl_mu_);

  if (rl_cooldown_steps_ > 0) {
    if (L0CompactionScore(vstorage) < 1.0 || l0_files == 0) {
      --rl_cooldown_steps_;
      return false;
    }
    rl_cooldown_steps_ = 0;
  }

  return QueryRL(vstorage, l0_files, pcb);
}

Compaction* RLCompactionPicker::PickCompaction(
    const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
    const MutableDBOptions& mutable_db_options,
    const std::vector<SequenceNumber>& existing_snapshots,
    const SnapshotChecker* snapshot_checker, VersionStorageInfo* vstorage,
    LogBuffer* log_buffer, const std::string& full_history_ts_low,
    bool require_max_output_level) {
  {
    std::unique_lock<std::mutex> lock(rl_mu_);
    rl_l0_trigger_ = mutable_cf_options.level0_file_num_compaction_trigger;
  }

  bool force_l0_compaction = false;
  {
    std::unique_lock<std::mutex> lock(rl_mu_);
    force_l0_compaction = rl_force_l0_compaction_pending_;
    rl_force_l0_compaction_pending_ = false;
  }

  if (force_l0_compaction) {
    Compaction* forced_l0 = PickCompactionFromLevel(
        cf_name, mutable_cf_options, mutable_db_options, vstorage, log_buffer,
        full_history_ts_low, /*forced_start_level=*/0,
        L0CompactionScore(vstorage), CompactionReason::kLevelL0FilesNum);
    if (forced_l0 != nullptr) {
      RLCompactionTelemetry::Get().RecordL0CompactionScheduled();
      std::unique_lock<std::mutex> lock(rl_mu_);
      rl_last_compaction_picked_ = true;
      return forced_l0;
    }
    std::unique_lock<std::mutex> lock(rl_mu_);
    rl_last_decision_ = false;
  }

  Compaction* c = LevelCompactionPicker::PickCompaction(
      cf_name, mutable_cf_options, mutable_db_options, existing_snapshots,
      snapshot_checker, vstorage, log_buffer, full_history_ts_low,
      require_max_output_level);
  if (c != nullptr && c->start_level() == 0) {
    RLCompactionTelemetry::Get().RecordL0CompactionScheduled();
    std::unique_lock<std::mutex> lock(rl_mu_);
    rl_last_compaction_picked_ = true;
  }
  return c;
}

}  // namespace ROCKSDB_NAMESPACE
