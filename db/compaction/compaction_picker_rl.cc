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

namespace ROCKSDB_NAMESPACE {

double RLCompactionPicker::ComputeReward(int l0_files, uint64_t pcb,
                                         bool did_compact) const {
  double delta_f0_pos =
      std::max(0.0, static_cast<double>(l0_files - rl_prev_l0_files_));
  double delta_pcb_pos = std::max(
      0.0, static_cast<double>(static_cast<int64_t>(pcb) -
                               static_cast<int64_t>(rl_prev_pcb_)));

  double delta_f0_norm  = std::min(1.0, delta_f0_pos  / 20.0);
  double delta_pcb_norm = std::min(1.0, delta_pcb_pos / 1e10);
  double stall               = 0.0;
  double compact_cost        = did_compact ? 1.0 : 0.0;
  double bytes_compacted_norm = 0.0;
  double pressure_relieved =
      (l0_files < rl_prev_l0_files_ && l0_files < rl_l0_trigger_) ? 1.0 : 0.0;

  return -0.30 * delta_f0_norm
       - 0.30 * delta_pcb_norm
       - 0.30 * stall
       - 0.10 * compact_cost
       - 0.05 * bytes_compacted_norm
       + 0.50 * pressure_relieved;
}

bool RLCompactionPicker::QueryRL(int l0_files, uint64_t pcb) const {
  auto now = std::chrono::steady_clock::now();
  if (now - rl_last_query_time_ < kMinQueryInterval) return rl_last_decision_;

  bool   did_compact = rl_last_decision_;
  double reward      = ComputeReward(l0_files, pcb, did_compact);

  double f0_norm  = std::min(1.0, l0_files / 20.0);
  double df0      = static_cast<double>(l0_files - rl_prev_l0_files_);
  double df0_norm = std::max(-1.0, std::min(1.0, df0 / 20.0));
  double s0_norm  = std::min(
      1.0, l0_files / (static_cast<double>(std::max(1, rl_l0_trigger_))));
  double pcb_norm = std::min(1.0, static_cast<double>(pcb) / 1e10);
  double stall    = 0.0;
  double bw       = std::max(0.0, df0_norm);

  RLState state{f0_norm, df0_norm, s0_norm, pcb_norm, stall, bw, reward, false};
  RLAction action = RLCompactionClient::Get().QueryAction(state);

  rl_prev_l0_files_   = l0_files;
  rl_prev_pcb_        = pcb;
  rl_last_query_time_ = now;

  switch (action) {
    case RLAction::kCompactNow:
      rl_last_decision_ = true;  return true;
    case RLAction::kDelay:
      rl_cooldown_steps_ = 1;
      rl_last_decision_  = false; return false;
    case RLAction::kDoNothing:
    default:
      rl_last_decision_ = false; return false;
  }
}

bool RLCompactionPicker::NeedsCompaction(
    const VersionStorageInfo* vstorage) const {
  if (!vstorage->ExpiredTtlFiles().empty())                    return true;
  if (!vstorage->FilesMarkedForPeriodicCompaction().empty())   return true;
  if (!vstorage->BottommostFilesMarkedForCompaction().empty()) return true;
  if (!vstorage->FilesMarkedForCompaction().empty())           return true;
  if (!vstorage->FilesMarkedForForcedBlobGC().empty())         return true;

  for (int i = 0; i <= vstorage->MaxInputLevel(); i++) {
    if (vstorage->CompactionScoreLevel(i) != 0 &&
        vstorage->CompactionScore(i) >= 1) {
      return true;
    }
  }

  int      l0_files = vstorage->NumLevelFiles(0);
  uint64_t pcb      = vstorage->estimated_compaction_needed_bytes();

  if (l0_files >= kL0HardCap || pcb >= kPcbHardCap) {
    std::unique_lock<std::mutex> lock(rl_mu_);
    rl_prev_l0_files_  = l0_files;
    rl_prev_pcb_       = pcb;
    rl_last_decision_  = true;
    rl_cooldown_steps_ = 0;
    return true;
  }

  std::unique_lock<std::mutex> lock(rl_mu_);

  if (rl_cooldown_steps_ > 0) {
    --rl_cooldown_steps_;
    return false;
  }

  // QueryAction() handles lazy connection internally and falls back to
  // kCompactNow when the server is unreachable, so always call QueryRL().
  return QueryRL(l0_files, pcb);
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
  return LevelCompactionPicker::PickCompaction(
      cf_name, mutable_cf_options, mutable_db_options, existing_snapshots,
      snapshot_checker, vstorage, log_buffer, full_history_ts_low,
      require_max_output_level);
}

}  // namespace ROCKSDB_NAMESPACE
