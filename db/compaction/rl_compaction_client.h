#pragma once

#include <sys/un.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

// Actions the RL agent can return for a per-level compaction decision.
enum class RLAction : int {
  kDoNothing = 0,
  kCompactNow = 1,
};

struct RLCandidateState {
  uint64_t snapshot_epoch = 0;
  uint64_t source_file_number = 0;
  int source_level = 0;
  int output_level = 0;
  uint64_t source_bytes = 0;
  uint64_t expanded_source_bytes = 0;
  std::vector<uint64_t> expanded_source_files;
  std::vector<uint64_t> overlap_files;
  uint64_t overlap_bytes = 0;
  uint64_t estimated_read_bytes = 0;
  uint64_t estimated_write_bytes = 0;
  double overlap_ratio = 0.0;
  uint64_t num_entries = 0;
  uint64_t num_deletions = 0;
  uint64_t compensated_size = 0;
  double projected_source_fullness = 0.0;
  double projected_output_fullness = 0.0;
  bool empties_source_level = false;
  int priority_rank = 0;
  bool conflict = false;
};

// Raw observable state for one LSM level. RocksDB only reports observables;
// the Python agent owns normalization, reward computation, and learning.
struct RLLevelState {
  int level = 0;
  int files = 0;
  uint64_t bytes = 0;
  double score = 0.0;           // RocksDB compaction score for this level
  uint64_t target_bytes = 0;    // MaxBytesForLevel (0 for L0: use triggers)
  int next_level_files = 0;     // 0 when is_last
  uint64_t next_level_bytes = 0;
  double next_level_score = 0.0;
  uint64_t next_level_target_bytes = 0;
  uint64_t overlap_bytes = 0;   // bytes in level+1 overlapping this level
  // Telemetry deltas since the previous RL query, scoped to this level.
  uint64_t bytes_in = 0;            // arrived via flush/compaction output
  uint64_t bytes_read_out = 0;      // compaction reads with this base level
  uint64_t bytes_written_out = 0;   // compaction writes with this base level
  uint64_t compactions_from = 0;    // completed compactions from this level
  uint64_t compactions_scheduled = 0;
  // Compactions from this level that the RL agent itself forced, as opposed to
  // ones the parent (leveled) picker chose. Without this split a level's agent
  // is credited with compactions it never asked for.
  uint64_t compactions_forced = 0;
  bool default_needed = false;  // would RocksDB's own trigger fire (score>=1)
  bool is_last = false;         // no next level below this one
  std::vector<RLCandidateState> candidates;

  // --- Outcome of the PREVIOUS decision for this level -------------------
  // The agent's chosen action and the action actually imposed on RocksDB can
  // differ (safety guard, deferral bound, or a forced pick that found nothing
  // to compact). Training must key the transition on what was executed, not on
  // what was chosen, or every override becomes a mislabelled sample.
  int prev_action_executed = 0;       // 0=do_nothing, 1=compact_now (effective)
  bool prev_action_overridden = false;  // safety guard / deferral bound stepped in
  bool prev_compaction_picked = false;  // a compaction really started from here
  uint64_t prev_decision_id = 0;
  uint64_t prev_snapshot_epoch = 0;
  uint64_t prev_candidate_file_number = 0;
  // 0=none, 1=scheduled, 2=validation failed, 3=expired at next actuation.
  int prev_scheduling_result = 0;
  // 0=not observed, 1=completed, 2=job failed.
  int prev_completion_result = 0;
  uint64_t prev_completed_decision_id = 0;
  uint64_t prev_completed_candidate_file_number = 0;
  // 0=policy, 1=budget, 2=maintenance, 3=emergency, 4=fallback, 5=drain.
  int prev_override_reason = 0;
  bool prev_transition_valid = true;
  // Consecutive decisions this level has been deferred while due (score>=1).
  int defer_count = 0;
};

// Full request: global state + one entry per candidate level.
struct RLStateV2 {
  int protocol_version = 3;
  uint64_t snapshot_epoch = 0;
  uint64_t pending_compaction_bytes = 0;
  uint64_t flushed_bytes = 0;
  uint64_t compaction_bytes_read = 0;
  uint64_t compaction_bytes_written = 0;
  uint64_t compactions_completed = 0;
  uint64_t stall_count = 0;
  uint64_t stop_count = 0;
  uint64_t fallback_count = 0;
  int l0_compaction_trigger = 4;
  int l0_slowdown_trigger = 20;
  int l0_stop_trigger = 36;
  int l0_delay_trigger_count = 0;
  // Wall-clock span the telemetry deltas above cover. Without it every
  // delta-valued field is a total over an unknown window, and two identical
  // feature vectors can describe completely different physical situations.
  uint64_t interval_micros = 0;
  // --- Read-path telemetry (deltas over interval_micros) -----------------
  // Sourced from the Statistics object RocksDB already maintains, so these
  // cost nothing on the read hot path. Compaction exists to bound read
  // amplification; without these the reward cannot see its own objective.
  uint64_t keys_read = 0;               // NUMBER_KEYS_READ
  uint64_t seeks = 0;                   // NUMBER_DB_SEEK
  uint64_t get_hit_l0 = 0;              // GET_HIT_L0
  uint64_t get_hit_l1 = 0;              // GET_HIT_L1
  uint64_t get_hit_l2_and_up = 0;       // GET_HIT_L2_AND_UP
  uint64_t bloom_useful = 0;            // BLOOM_FILTER_USEFUL
  uint64_t non_last_level_read_count = 0;
  uint64_t last_level_read_count = 0;
  // Logical foreground work and rolling latency summaries.
  uint64_t user_logical_write_bytes = 0;
  uint64_t point_sst_probes = 0;
  uint64_t scan_returned_entries = 0;
  uint64_t scan_internal_skipped = 0;
  uint64_t scan_sorted_run_seeks = 0;
  uint64_t physical_sst_bytes = 0;
  uint64_t live_logical_bytes = 0;
  uint64_t stall_duration_micros = 0;
  uint64_t get_latency_count = 0;
  double get_latency_avg_ns = 0.0;
  uint64_t get_latency_p95_ns = 0;
  uint64_t scan_latency_count = 0;
  double scan_latency_avg_ns = 0.0;
  uint64_t scan_latency_p95_ns = 0;
  uint64_t write_latency_count = 0;
  double write_latency_avg_ns = 0.0;
  uint64_t write_latency_p95_ns = 0;
  bool done = false;
  std::vector<RLLevelState> levels;
};

// Result of a multi-level query: one action per requested level, in request
// order. `ok=false` means the caller should use its local fallback policy.
struct RLMultiQueryResult {
  bool ok = false;
  uint64_t decision_id = 0;
  uint64_t snapshot_epoch = 0;
  std::vector<RLAction> actions;
  std::vector<uint64_t> candidate_file_numbers;
};

// Singleton client that communicates with the Python RL server over a
// Unix domain socket.  The C++ compaction picker calls QueryActions()
// on each decision evaluation; the server returns per-level actions and
// trains its per-level agents online.
//
// Thread-safe: a single mutex serialises socket I/O.
class RLCompactionClient {
 public:
  // Returns the process-wide singleton, lazily initialised.
  static RLCompactionClient& Get();

  // Sends `state` to the Python server and returns one action per level in
  // `state.levels` (same order). `ok=false` on connection failure, timeout,
  // or a malformed/mis-sized response.
  RLMultiQueryResult QueryActions(const RLStateV2& state);

  bool IsConnected() const { return fd_ >= 0; }

 private:
  RLCompactionClient();
  ~RLCompactionClient();

  // Non-copyable.
  RLCompactionClient(const RLCompactionClient&) = delete;
  RLCompactionClient& operator=(const RLCompactionClient&) = delete;

  bool Connect();
  void Disconnect();
  bool SetSocketTimeouts();
  bool ConnectWithTimeout(::sockaddr_un* addr);

  // Send a newline-terminated string; returns false on error.
  bool SendLine(const std::string& line);

  // Read until '\n'; returns empty string on error.
  std::string RecvLine();

  std::string socket_path_;
  int socket_timeout_ms_{100};
  int fd_{-1};
  mutable std::mutex mu_;
};

}  // namespace ROCKSDB_NAMESPACE
