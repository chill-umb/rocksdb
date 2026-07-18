#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>

#include "db/compaction/compaction_picker_level.h"
#include "db/compaction/rl_compaction_client.h"
#include "db/compaction/rl_compaction_telemetry.h"

namespace ROCKSDB_NAMESPACE {

// RLCompactionPicker — a leveled compaction picker whose per-level compaction
// triggers are governed by online DQN agents (one per level) running in a
// separate Python process.
//
// Every decision cadence the picker assembles one state entry per candidate
// level (own level + next-level observables, including the bytes in level+1
// overlapping this level's key range) and sends them in a single batched
// request; the server returns one action per level. Levels whose agent
// answers kCompactNow are marked force-pending; PickCompaction resolves
// contention by compacting the pending level with the highest RocksDB score
// first (stall-risk proxy), leaving the rest pending for subsequent picks.
//
// Non-score triggers (TTL, periodic compaction, marked files, blob GC) are
// preserved unchanged via the parent picker.
//
// Emergency safeguards override the RL agents and force an L0 compaction
// when:
//   • L0 file count reaches the live level0_stop_writes_trigger, OR
//   • estimated pending compaction bytes reaches kPcbHardCap, OR
//   • a write-stop was observed in the telemetry window.
//
// If the RL server is unreachable the picker transparently falls back to the
// standard level-based thresholds so the database stays operational.
class RLCompactionPicker : public LevelCompactionPicker {
 public:
  RLCompactionPicker(const ImmutableOptions& ioptions,
                     const InternalKeyComparator* icmp)
      : LevelCompactionPicker(ioptions, icmp) {}

  // Overrides LevelCompactionPicker::NeedsCompaction.
  // Non-score triggers delegate to the parent. Score-based decisions for
  // levels 0..MaxInputLevel are routed to the per-level RL agents (with
  // safeguard overrides).
  bool NeedsCompaction(const VersionStorageInfo* vstorage) const override;

  // Overrides PickCompaction to serve RL force-pending levels (highest score
  // first) before delegating to the parent.
  Compaction* PickCompaction(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options,
      const std::vector<SequenceNumber>& existing_snapshots,
      const SnapshotChecker* snapshot_checker, VersionStorageInfo* vstorage,
      LogBuffer* log_buffer, const std::string& full_history_ts_low,
      bool require_max_output_level = false) override;

 private:
  static constexpr int kMaxRLLevels = kRLTelemetryMaxLevels;

  // -----------------------------------------------------------------------
  // Per-instance RL tracking state.
  // All fields are mutable because NeedsCompaction is const.
  // -----------------------------------------------------------------------
  mutable std::mutex rl_mu_;

  mutable bool rl_last_decision_{true};
  mutable bool rl_fallback_logged_{false};
  // Counts consecutive fallbacks (server unavailable / bad response). Re-warned
  // periodically so a permanently broken RL path can never fail silently.
  mutable uint64_t rl_fallback_count_{0};
  // Per-level force flags set by kCompactNow answers, consumed by
  // PickCompaction. rl_force_score_ caches the level's score at decision
  // time for arbiter ordering and forced-pick bookkeeping.
  mutable std::array<bool, kMaxRLLevels> rl_force_pending_{};
  mutable std::array<double, kMaxRLLevels> rl_force_score_{};
  mutable int rl_l0_trigger_{4};  // refreshed by PickCompaction
  mutable int rl_l0_slowdown_trigger_{20};
  mutable int rl_l0_stop_trigger_{36};
  mutable std::chrono::steady_clock::time_point rl_last_query_time_{};

  // Rate-limit: minimum wall-clock gap between successive RL server queries.
  static constexpr std::chrono::milliseconds kMinQueryInterval{50};

  // -----------------------------------------------------------------------
  // Emergency safeguard threshold for pending compaction bytes.
  // L0 file-count safety uses the live level0_stop_writes_trigger.
  // -----------------------------------------------------------------------
  static constexpr uint64_t kPcbHardCap = 10ULL * 1024 * 1024 * 1024;  // 10 GB

  // Batch-query the RL server for all candidate levels; updates
  // rl_force_pending_/rl_force_score_ and returns whether any compaction is
  // wanted. Caller must hold rl_mu_.
  bool QueryRL(const VersionStorageInfo* vstorage, uint64_t pcb) const;

  // RocksDB's compaction score for `level` (0.0 when absent).
  double LevelScore(const VersionStorageInfo* vstorage, int level) const;

  // Bytes in level+1 whose key range overlaps level's span. 0 for an empty
  // level or the last level.
  uint64_t NextLevelOverlapBytes(const VersionStorageInfo* vstorage,
                                 int level) const;

  bool AnyForcePending() const;  // caller must hold rl_mu_
  void ClearForcePending() const;  // caller must hold rl_mu_
};

}  // namespace ROCKSDB_NAMESPACE
