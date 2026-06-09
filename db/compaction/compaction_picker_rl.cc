//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/compaction/compaction_picker_rl.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include "logging/logging.h"

namespace ROCKSDB_NAMESPACE {

double RLCompactionPicker::ComputeReward(int l0_files, uint64_t pcb,
                                         bool did_compact) const {
  double delta_f0_pos =
      std::max(0.0, static_cast<double>(l0_files - rl_prev_l0_files_));
  double delta_pcb_pos =
      std::max(0.0, static_cast<double>(static_cast<int64_t>(pcb) -
                                        static_cast<int64_t>(rl_prev_pcb_)));

  double delta_f0_norm = std::min(1.0, delta_f0_pos / 20.0);
  double delta_pcb_norm = std::min(1.0, delta_pcb_pos / 1e10);
  double stall = 0.0;
  double compact_cost = did_compact ? 1.0 : 0.0;
  double bytes_compacted_norm = 0.0;
  double pressure_relieved =
      (l0_files < rl_prev_l0_files_ && l0_files < rl_l0_trigger_) ? 1.0 : 0.0;

  return -0.30 * delta_f0_norm - 0.30 * delta_pcb_norm - 0.30 * stall -
         0.10 * compact_cost - 0.05 * bytes_compacted_norm +
         0.50 * pressure_relieved;
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
  if (now - rl_last_query_time_ < kMinQueryInterval) return rl_last_decision_;

  bool did_compact = rl_last_compaction_picked_;
  rl_last_compaction_picked_ = false;
  double reward = ComputeReward(l0_files, pcb, did_compact);

  double f0_norm = std::min(1.0, l0_files / 20.0);
  double df0 = static_cast<double>(l0_files - rl_prev_l0_files_);
  double df0_norm = std::max(-1.0, std::min(1.0, df0 / 20.0));
  double s0_norm = std::max(0.0, std::min(1.0, L0CompactionScore(vstorage)));
  double pcb_norm = std::min(1.0, static_cast<double>(pcb) / 1e10);
  double stall = 0.0;
  double bw = std::max(0.0, df0_norm);

  RLState state{f0_norm, df0_norm, s0_norm, pcb_norm, stall, bw, reward, false};
  RLQueryResult result = RLCompactionClient::Get().QueryAction(state);

  rl_prev_l0_files_ = l0_files;
  rl_prev_pcb_ = pcb;
  rl_last_query_time_ = now;

  if (!result.ok) {
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
    --rl_cooldown_steps_;
    return false;
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
    std::unique_lock<std::mutex> lock(rl_mu_);
    rl_last_compaction_picked_ = true;
  }
  return c;
}

}  // namespace ROCKSDB_NAMESPACE
