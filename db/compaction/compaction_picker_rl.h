#pragma once

#include <chrono>
#include <mutex>

#include "db/compaction/compaction_picker_level.h"
#include "db/compaction/rl_compaction_client.h"

namespace ROCKSDB_NAMESPACE {

// RLCompactionPicker — a leveled compaction picker whose L0 trigger is
// governed by an online DQN agent running in a separate Python process.
//
// For all levels above L0 and for all non-score-based triggers (TTL,
// periodic compaction, etc.) the original LevelCompactionPicker logic is
// preserved unchanged.  Only the L0 NeedsCompaction decision is intercepted
// and routed to the RL server via a Unix domain socket.
//
// Emergency safeguards override the RL agent and force a compaction when:
//   • L0 file count reaches kL0HardCap, OR
//   • estimated pending compaction bytes reaches kPcbHardCap, OR
//   • (future) a stall condition is detected.
//
// If the RL server is unreachable the picker transparently falls back to the
// standard level-based threshold so the database stays operational.
class RLCompactionPicker : public LevelCompactionPicker {
 public:
  RLCompactionPicker(const ImmutableOptions& ioptions,
                     const InternalKeyComparator* icmp)
      : LevelCompactionPicker(ioptions, icmp) {}

  // Overrides LevelCompactionPicker::NeedsCompaction.
  // Non-L0 decisions and non-score triggers delegate to the parent.
  // L0 decisions are routed to the RL agent (with safeguard overrides).
  bool NeedsCompaction(const VersionStorageInfo* vstorage) const override;

  // Overrides PickCompaction only to keep the cached L0 trigger in sync
  // with the live MutableCFOptions before delegating to the parent.
  Compaction* PickCompaction(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options,
      const std::vector<SequenceNumber>& existing_snapshots,
      const SnapshotChecker* snapshot_checker, VersionStorageInfo* vstorage,
      LogBuffer* log_buffer, const std::string& full_history_ts_low,
      bool require_max_output_level = false) override;

 private:
  // -----------------------------------------------------------------------
  // Per-instance RL tracking state.
  // All fields are mutable because NeedsCompaction is const.
  // -----------------------------------------------------------------------
  mutable std::mutex rl_mu_;

  mutable int rl_prev_l0_files_{0};
  mutable uint64_t rl_prev_pcb_{0};
  mutable int rl_cooldown_steps_{0};
  mutable bool rl_last_decision_{true};
  mutable bool rl_last_compaction_picked_{false};
  mutable bool rl_force_l0_compaction_pending_{false};
  mutable bool rl_fallback_logged_{false};
  mutable int rl_l0_trigger_{4};  // refreshed by PickCompaction
  mutable std::chrono::steady_clock::time_point rl_last_query_time_{};

  // Rate-limit: minimum wall-clock gap between successive RL server queries.
  static constexpr std::chrono::milliseconds kMinQueryInterval{50};

  // -----------------------------------------------------------------------
  // Emergency safeguard thresholds — must match the Python config.
  // -----------------------------------------------------------------------
  static constexpr int kL0HardCap = 20;
  static constexpr uint64_t kPcbHardCap = 10ULL * 1024 * 1024 * 1024;  // 10 GB

  // Compute the scalar reward for the step that just completed.
  double ComputeReward(int l0_files, uint64_t pcb, bool did_compact) const;

  // Normalise state, call RLCompactionClient, update tracking, return decision.
  bool QueryRL(const VersionStorageInfo* vstorage, int l0_files,
               uint64_t pcb) const;

  double L0CompactionScore(const VersionStorageInfo* vstorage) const;
};

}  // namespace ROCKSDB_NAMESPACE
