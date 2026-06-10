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

  const bool stall_emergency = telemetry.stop_count > 0 && l0_files > 0;

  RLState state{l0_files,
                vstorage->NumLevelBytes(0),
                l0_score,
                vstorage->l0_delay_trigger_count(),
                rl_l0_trigger_,
                rl_l0_slowdown_trigger_,
                rl_l0_stop_trigger_,
                pcb,
                telemetry.flushed_bytes,
                telemetry.compaction_bytes_read,
                telemetry.compaction_bytes_written,
                telemetry.compactions_completed,
                telemetry.l0_compactions_completed,
                telemetry.l0_compactions_scheduled,
                telemetry.stall_count,
                telemetry.stop_count,
                l0_score >= 1.0,
                false};

  ROCKS_LOG_INFO(
      ioptions_.logger,
      "RL compaction state: l0_files=%d l0_size=%" PRIu64
      " l0_score=%.4f l0_delay=%d trigger=%d slowdown=%d stop=%d "
      "pending_compaction_bytes=%" PRIu64 " telemetry={flushed=%" PRIu64
      ",compact_read=%" PRIu64 ",compact_written=%" PRIu64
      ",compactions=%" PRIu64 ",l0_completed=%" PRIu64 ",l0_scheduled=%" PRIu64
      ",stall_count=%" PRIu64 ",stop_count=%" PRIu64 "}",
      state.l0_files, state.l0_size_bytes, state.l0_score,
      state.l0_delay_trigger_count, state.l0_compaction_trigger,
      state.l0_slowdown_trigger, state.l0_stop_trigger,
      state.pending_compaction_bytes, telemetry.flushed_bytes,
      telemetry.compaction_bytes_read, telemetry.compaction_bytes_written,
      telemetry.compactions_completed, telemetry.l0_compactions_completed,
      telemetry.l0_compactions_scheduled, telemetry.stall_count,
      telemetry.stop_count);

  RLQueryResult result = RLCompactionClient::Get().QueryAction(state);

  rl_last_query_time_ = now;

  if (!result.ok) {
    if (stall_emergency || l0_score_emergency) {
      rl_force_l0_compaction_pending_ = true;
      rl_last_decision_ = true;
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
    return true;
  }

  switch (result.action) {
    case RLAction::kCompactNow:
      rl_force_l0_compaction_pending_ = true;
      rl_last_decision_ = true;
      return true;
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
  std::unique_lock<std::mutex> lock(rl_mu_);
  const int l0_safety_cap = std::max(1, rl_l0_stop_trigger_);

  if (l0_files >= l0_safety_cap || pcb >= kPcbHardCap) {
    rl_last_decision_ = true;
    rl_force_l0_compaction_pending_ = l0_files > 0;
    return true;
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
    rl_l0_slowdown_trigger_ = mutable_cf_options.level0_slowdown_writes_trigger;
    rl_l0_stop_trigger_ = mutable_cf_options.level0_stop_writes_trigger;
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
