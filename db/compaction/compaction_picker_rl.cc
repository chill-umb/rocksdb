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

#include "logging/logging.h"

namespace ROCKSDB_NAMESPACE {

double RLCompactionPicker::LevelScore(const VersionStorageInfo* vstorage,
                                      int level) const {
  for (int i = 0; i <= vstorage->MaxInputLevel(); ++i) {
    if (vstorage->CompactionScoreLevel(i) == level) {
      return vstorage->CompactionScore(i);
    }
  }
  return 0.0;
}

uint64_t RLCompactionPicker::NextLevelOverlapBytes(
    const VersionStorageInfo* vstorage, int level) const {
  const int next = level + 1;
  if (next >= vstorage->num_levels()) return 0;
  const std::vector<FileMetaData*>& files = vstorage->LevelFiles(level);
  if (files.empty() || vstorage->NumLevelFiles(next) == 0) return 0;

  const InternalKey* smallest = &files[0]->smallest;
  const InternalKey* largest = &files[0]->largest;
  for (size_t i = 1; i < files.size(); ++i) {
    if (icmp_->Compare(files[i]->smallest, *smallest) < 0) {
      smallest = &files[i]->smallest;
    }
    if (icmp_->Compare(files[i]->largest, *largest) > 0) {
      largest = &files[i]->largest;
    }
  }

  std::vector<FileMetaData*> overlaps;
  vstorage->GetOverlappingInputs(next, smallest, largest, &overlaps);
  uint64_t bytes = 0;
  for (const FileMetaData* f : overlaps) {
    bytes += f->fd.GetFileSize();
  }
  return bytes;
}

bool RLCompactionPicker::AnyForcePending() const {
  for (int i = 0; i < kMaxRLLevels; ++i) {
    if (rl_force_pending_[i]) return true;
  }
  return false;
}

void RLCompactionPicker::ClearForcePending() const {
  rl_force_pending_.fill(false);
  rl_force_score_.fill(0.0);
}

bool RLCompactionPicker::QueryRL(const VersionStorageInfo* vstorage,
                                 uint64_t pcb) const {
  auto now = std::chrono::steady_clock::now();
  const int l0_files = vstorage->NumLevelFiles(0);
  const double l0_score = LevelScore(vstorage, 0);
  const bool l0_score_emergency = l0_score >= 1.0 && l0_files > 0;
  if (!l0_score_emergency && now - rl_last_query_time_ < kMinQueryInterval) {
    return rl_last_decision_;
  }

  RLCompactionTelemetrySnapshot telemetry =
      RLCompactionTelemetry::Get().Consume();

  const bool stall_emergency = telemetry.stop_count > 0 && l0_files > 0;

  // Candidate levels: L0 always; deeper levels only when they hold files.
  // MaxInputLevel bounds the levels a compaction may start from.
  const int max_input_level =
      std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
  RLStateV2 state;
  state.pending_compaction_bytes = pcb;
  state.flushed_bytes = telemetry.flushed_bytes;
  state.compaction_bytes_read = telemetry.compaction_bytes_read;
  state.compaction_bytes_written = telemetry.compaction_bytes_written;
  state.compactions_completed = telemetry.compactions_completed;
  state.stall_count = telemetry.stall_count;
  state.stop_count = telemetry.stop_count;
  state.l0_compaction_trigger = rl_l0_trigger_;
  state.l0_slowdown_trigger = rl_l0_slowdown_trigger_;
  state.l0_stop_trigger = rl_l0_stop_trigger_;
  state.l0_delay_trigger_count = vstorage->l0_delay_trigger_count();
  state.done = false;

  for (int level = 0; level <= max_input_level; ++level) {
    if (level > 0 && vstorage->NumLevelFiles(level) == 0) continue;

    RLLevelState ls;
    ls.level = level;
    ls.files = vstorage->NumLevelFiles(level);
    ls.bytes = vstorage->NumLevelBytes(level);
    ls.score = LevelScore(vstorage, level);
    ls.target_bytes = level > 0 ? vstorage->MaxBytesForLevel(level) : 0;
    ls.is_last = (level + 1 >= vstorage->num_levels());
    if (!ls.is_last) {
      const int next = level + 1;
      ls.next_level_files = vstorage->NumLevelFiles(next);
      ls.next_level_bytes = vstorage->NumLevelBytes(next);
      ls.next_level_score =
          next <= vstorage->MaxInputLevel() ? LevelScore(vstorage, next) : 0.0;
      ls.next_level_target_bytes = vstorage->MaxBytesForLevel(next);
      ls.overlap_bytes = NextLevelOverlapBytes(vstorage, level);
    }
    const int t = std::min(level, kRLTelemetryMaxLevels - 1);
    ls.bytes_in = telemetry.bytes_into_level[t];
    ls.bytes_read_out = telemetry.compaction_read_from_level[t];
    ls.bytes_written_out = telemetry.compaction_written_from_level[t];
    ls.compactions_from = telemetry.compactions_from_level[t];
    ls.compactions_scheduled = telemetry.compactions_scheduled_from_level[t];
    ls.default_needed = ls.score >= 1.0;
    state.levels.push_back(ls);
  }

  ROCKS_LOG_INFO(
      ioptions_.logger,
      "RL compaction state: levels=%zu l0_files=%d l0_score=%.4f "
      "pending_compaction_bytes=%" PRIu64 " telemetry={flushed=%" PRIu64
      ",compact_read=%" PRIu64 ",compact_written=%" PRIu64
      ",compactions=%" PRIu64 ",stall_count=%" PRIu64 ",stop_count=%" PRIu64
      "}",
      state.levels.size(), l0_files, l0_score, pcb, telemetry.flushed_bytes,
      telemetry.compaction_bytes_read, telemetry.compaction_bytes_written,
      telemetry.compactions_completed, telemetry.stall_count,
      telemetry.stop_count);

  RLMultiQueryResult result = RLCompactionClient::Get().QueryActions(state);

  rl_last_query_time_ = now;

  if (!result.ok) {
    ClearForcePending();
    ++rl_fallback_count_;
    if (stall_emergency || l0_score_emergency) {
      rl_force_pending_[0] = true;
      rl_force_score_[0] = l0_score;
      rl_last_decision_ = true;
      return true;
    }
    rl_last_decision_ = LevelCompactionPicker::NeedsCompaction(vstorage);
    // Warn on the first fallback and every 200th after: a permanently broken
    // RL path (server down, unparseable/mis-sized response) must stay visible
    // in the LOG rather than silently degrading to leveled behavior.
    if (!rl_fallback_logged_ || rl_fallback_count_ % 200 == 0) {
      ROCKS_LOG_WARN(
          ioptions_.logger,
          "RL compaction server unavailable, timed out, or returned an "
          "unusable response; falling back to normal leveled thresholds "
          "(fallback_count=%" PRIu64 ").",
          rl_fallback_count_);
      rl_fallback_logged_ = true;
    }
    return rl_last_decision_;
  }
  rl_fallback_logged_ = false;
  rl_fallback_count_ = 0;

  ClearForcePending();
  for (size_t i = 0; i < state.levels.size(); ++i) {
    if (result.actions[i] == RLAction::kCompactNow) {
      const int level = state.levels[i].level;
      rl_force_pending_[level] = true;
      rl_force_score_[level] = state.levels[i].score;
    }
  }

  if (stall_emergency || l0_score_emergency) {
    ROCKS_LOG_WARN(ioptions_.logger,
                   "RL compaction action overridden by safety guard: "
                   "l0_score=%.4f stall_emergency=%d l0_files=%d",
                   l0_score, static_cast<int>(stall_emergency), l0_files);
    rl_force_pending_[0] = true;
    rl_force_score_[0] = l0_score;
    rl_last_decision_ = true;
    return true;
  }

  rl_last_decision_ = AnyForcePending();
  return rl_last_decision_;
}

bool RLCompactionPicker::NeedsCompaction(
    const VersionStorageInfo* vstorage) const {
  if (!vstorage->ExpiredTtlFiles().empty() ||
      !vstorage->FilesMarkedForPeriodicCompaction().empty() ||
      !vstorage->BottommostFilesMarkedForCompaction().empty() ||
      !vstorage->FilesMarkedForCompaction().empty() ||
      !vstorage->FilesMarkedForForcedBlobGC().empty()) {
    std::unique_lock<std::mutex> lock(rl_mu_);
    ClearForcePending();
    return true;
  }

  for (int i = 0; i <= vstorage->MaxInputLevel(); i++) {
    if (vstorage->CompactionScoreLevel(i) != 0 &&
        vstorage->CompactionScore(i) >= 1) {
      std::unique_lock<std::mutex> lock(rl_mu_);
      ClearForcePending();
      return true;
    }
  }

  int l0_files = vstorage->NumLevelFiles(0);
  uint64_t pcb = vstorage->estimated_compaction_needed_bytes();
  std::unique_lock<std::mutex> lock(rl_mu_);
  const int l0_safety_cap = std::max(1, rl_l0_stop_trigger_);

  if (l0_files >= l0_safety_cap || pcb >= kPcbHardCap) {
    rl_last_decision_ = true;
    if (l0_files > 0) {
      rl_force_pending_[0] = true;
      rl_force_score_[0] = LevelScore(vstorage, 0);
    }
    return true;
  }

  return QueryRL(vstorage, pcb);
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

  // Serve RL force-pending levels first, highest score first (stall-risk
  // proxy; L0 outranks deeper levels when pressured). A level whose pick
  // fails (e.g. all of its files are already being compacted) is skipped;
  // the remaining pending levels stay queued for subsequent picks.
  while (true) {
    int chosen = -1;
    double chosen_score = -1.0;
    {
      std::unique_lock<std::mutex> lock(rl_mu_);
      for (int level = 0; level < kMaxRLLevels; ++level) {
        if (rl_force_pending_[level] && rl_force_score_[level] > chosen_score) {
          chosen = level;
          chosen_score = rl_force_score_[level];
        }
      }
      if (chosen >= 0) {
        rl_force_pending_[chosen] = false;
      }
    }
    if (chosen < 0) break;

    Compaction* forced = PickCompactionFromLevel(
        cf_name, mutable_cf_options, mutable_db_options, vstorage, log_buffer,
        full_history_ts_low, /*forced_start_level=*/chosen, chosen_score,
        chosen == 0 ? CompactionReason::kLevelL0FilesNum
                    : CompactionReason::kLevelMaxLevelSize);
    if (forced != nullptr) {
      RLCompactionTelemetry::Get().RecordCompactionScheduled(chosen);
      return forced;
    }
  }

  Compaction* c = LevelCompactionPicker::PickCompaction(
      cf_name, mutable_cf_options, mutable_db_options, existing_snapshots,
      snapshot_checker, vstorage, log_buffer, full_history_ts_low,
      require_max_output_level);
  if (c != nullptr) {
    RLCompactionTelemetry::Get().RecordCompactionScheduled(c->start_level());
  }
  return c;
}

}  // namespace ROCKSDB_NAMESPACE
