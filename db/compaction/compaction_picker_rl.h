#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "db/compaction/compaction_picker_level.h"
#include "db/compaction/rl_compaction_client.h"
#include "db/compaction/rl_compaction_telemetry.h"

namespace ROCKSDB_NAMESPACE {

// Candidate-aware leveled picker. The socket worker can authorize at most one
// exact SST per actuation. Every authorization is a versioned, single-use
// lease; stale or blocked candidates fail closed and are reported to the next
// observation rather than being replaced by a different file or level.
class RLCompactionPicker : public LevelCompactionPicker {
 public:
  RLCompactionPicker(const ImmutableOptions& ioptions,
                     const InternalKeyComparator* icmp);
  ~RLCompactionPicker() override;

  bool NeedsCompaction(const VersionStorageInfo* vstorage) const override;

  Compaction* PickCompaction(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options,
      const std::vector<SequenceNumber>& existing_snapshots,
      const SnapshotChecker* snapshot_checker, VersionStorageInfo* vstorage,
      LogBuffer* log_buffer, const std::string& full_history_ts_low,
      bool require_max_output_level = false) override;

 private:
  friend class RLCompactionPickerTestPeer;

  static constexpr int kMaxRLLevels = kRLTelemetryMaxLevels;
  static constexpr size_t kMaxCandidatesPerLevel = 8;
  static constexpr uint64_t kPcbHardCap = 10ULL * 1024 * 1024 * 1024;

  enum class ActionReason : int {
    kPolicy = 0,
    kBudget = 1,
    kMaintenance = 2,
    kEmergency = 3,
    kFallback = 4,
    kDrain = 5,
  };

  struct ActionLease {
    bool active = false;
    uint64_t decision_id = 0;
    uint64_t snapshot_epoch = 0;
    uint64_t candidate_file_number = 0;
    int level = -1;
    double score = 0.0;
    ActionReason reason = ActionReason::kPolicy;
  };

  int decision_interval_ms_;
  int observe_interval_ms_;
  int max_defer_steps_;
  int max_defer_steps_l0_;
  bool allow_defer_;
  int protocol_version_;

  int MaxDeferSteps(int level) const {
    return level == 0 ? max_defer_steps_l0_ : max_defer_steps_;
  }

  // Snapshot handoff. Structural preview is built while DBImpl holds its
  // mutex; network I/O and learning remain on worker_.
  mutable std::mutex snap_mu_;
  mutable std::condition_variable snap_cv_;
  mutable RLStateV2 pending_snapshot_;
  mutable bool snapshot_valid_{false};
  mutable std::vector<int> last_sent_levels_;
  mutable std::atomic<uint64_t> last_publish_micros_{0};
  std::thread worker_;
  std::atomic<bool> worker_stop_{false};

  // The options required by RocksDB's real candidate builder become available
  // on the first PickCompaction call. They are subsequently refreshed on every
  // call (all accesses occur under DBImpl's mutex).
  mutable MutableCFOptions cached_cf_options_;
  mutable MutableDBOptions cached_db_options_;
  mutable std::string cached_cf_name_;
  mutable std::string cached_history_ts_low_;
  mutable bool picker_options_valid_{false};

  // Per-level leases and outcome attribution cross the worker/DB threads.
  mutable std::mutex lease_mu_;
  mutable ActionLease leases_[kMaxRLLevels];
  mutable std::atomic<int> defer_count_[kMaxRLLevels] = {};
  mutable std::atomic<int> last_effective_action_[kMaxRLLevels] = {};
  mutable std::atomic<bool> last_action_overridden_[kMaxRLLevels] = {};
  mutable std::atomic<bool> compaction_picked_[kMaxRLLevels] = {};
  mutable std::atomic<uint64_t> last_decision_id_[kMaxRLLevels] = {};
  mutable std::atomic<uint64_t> last_snapshot_epoch_[kMaxRLLevels] = {};
  mutable std::atomic<uint64_t> last_candidate_file_[kMaxRLLevels] = {};
  mutable std::atomic<int> last_scheduling_result_[kMaxRLLevels] = {};
  mutable std::atomic<int> last_override_reason_[kMaxRLLevels] = {};
  mutable std::atomic<bool> last_transition_valid_[kMaxRLLevels] = {};

  mutable std::atomic<bool> rl_available_{false};
  // Only maintenance and drain use the ordinary parent picker. The reason is
  // consumed by PickCompaction and stamped on the resulting Compaction.
  mutable std::atomic<int> parent_bypass_reason_{-1};
  std::atomic<int> rl_l0_trigger_{4};
  std::atomic<int> rl_l0_slowdown_trigger_{20};
  std::atomic<int> rl_l0_stop_trigger_{36};

  mutable std::atomic<uint64_t> rl_fallback_count_{0};
  mutable std::atomic<uint64_t> rl_query_count_{0};
  mutable std::atomic<uint64_t> rl_actuation_count_{0};
  mutable std::atomic<uint64_t> rl_skipped_ticks_{0};
  mutable std::atomic<uint64_t> rl_bypass_count_{0};
  mutable std::atomic<bool> rl_fallback_logged_{false};
  mutable std::atomic<uint64_t> rl_nc_calls_{0};
  mutable std::atomic<uint64_t> rl_nc_nanos_{0};
  mutable std::atomic<uint64_t> rl_publish_calls_{0};
  mutable std::atomic<uint64_t> rl_publish_nanos_{0};

  static constexpr int kNumReadTickers = 9;
  uint64_t prev_read_tickers_[kNumReadTickers] = {};
  static constexpr uint64_t kDiagnosticsEveryQueries = 200;

  void WorkerLoop();
  void RunDecisionCycle(RLStateV2& state, bool actuate);
  void PopulateReadStats(RLStateV2& state);
  void LogDiagnostics(bool final) const;
  void SendDoneMessage();

  void BuildSnapshot(const VersionStorageInfo* vstorage, uint64_t pcb,
                     RLStateV2* state) const;
  void PublishSnapshot(const VersionStorageInfo* vstorage, uint64_t pcb) const;
  uint64_t SnapshotEpoch(const VersionStorageInfo* vstorage) const;
  double LevelScore(const VersionStorageInfo* vstorage, int level) const;
  uint64_t NextLevelOverlapBytes(const VersionStorageInfo* vstorage,
                                 int level) const;

  bool HasActiveLease() const;
  void ExpireLeasesAtActuation() const;
  void InstallLease(const ActionLease& lease) const;
  bool TakeLease(ActionLease* lease) const;
  int HighestDueLevel(const VersionStorageInfo* vstorage) const;
  uint64_t FirstValidCandidate(const RLLevelState& level) const;
  void InstallLocalLease(const VersionStorageInfo* vstorage, int level,
                         ActionReason reason) const;
};

}  // namespace ROCKSDB_NAMESPACE
