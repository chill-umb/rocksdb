#pragma once

#include <sys/un.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include "db/compaction/rl_latency_histogram.h"
#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

// Actions the RL agent can return for a per-level compaction decision.
enum class RLAction : int {
  kDoNothing = 0,
  kCompactNow = 1,
};

enum class RLRewardInvalidReason : uint64_t {
  kSocketOrQueryFallback = 1ULL << 0,
  kWatchdogNativeFallback = 1ULL << 1,
  kMalformedProtocol = 1ULL << 2,
  kRejectedManifest = 1ULL << 3,
  kUnknownControlOwnership = 1ULL << 4,
};

// Raw observable state for one LSM level. RocksDB only reports observables;
// the Python agent owns normalization, reward computation, and learning.
struct RLLevelState {
  int level = 0;
  int files = 0;
  uint64_t bytes = 0;
  double score = 0.0;         // RocksDB compaction score for this level
  uint64_t target_bytes = 0;  // MaxBytesForLevel (0 for L0: use triggers)
  int next_level_files = 0;   // 0 when is_last
  uint64_t next_level_bytes = 0;
  double next_level_score = 0.0;
  uint64_t next_level_target_bytes = 0;
  uint64_t overlap_bytes = 0;  // bytes in level+1 overlapping this level
  // Telemetry deltas since the previous RL query, scoped to this level.
  uint64_t bytes_in = 0;           // arrived via flush/compaction output
  uint64_t bytes_read_out = 0;     // compaction reads with this base level
  uint64_t bytes_written_out = 0;  // compaction writes with this base level
  uint64_t compactions_from = 0;   // completed compactions from this level
  uint64_t compactions_scheduled = 0;
  // Compactions from this level that the RL agent itself forced, as opposed to
  // ones the parent (leveled) picker chose. Without this split a level's agent
  // is credited with compactions it never asked for.
  uint64_t compactions_forced = 0;
  bool default_needed = false;  // would RocksDB's own trigger fire (score>=1)
  bool is_last = false;         // no next level below this one
  // --- Outcome of the PREVIOUS decision for this level -------------------
  // The agent's chosen action and the action actually imposed on RocksDB can
  // differ (safety guard, deferral bound, or a forced pick that found nothing
  // to compact). Training must key the transition on what was executed, not on
  // what was chosen, or every override becomes a mislabelled sample.
  int prev_action_executed = 0;  // 0=do_nothing, 1=compact_now (effective)
  bool prev_action_overridden =
      false;  // safety guard / deferral bound stepped in
  bool prev_compaction_picked = false;  // a compaction really started from here
  uint64_t prev_decision_id = 0;
  uint64_t prev_snapshot_epoch = 0;
  // 0=none, 1=scheduled, 2=native picker found no valid work,
  // 3=expired at next actuation.
  int prev_scheduling_result = 0;
  // 0=not observed, 1=completed, 2=job failed.
  int prev_completion_result = 0;
  uint64_t prev_completed_decision_id = 0;
  uint64_t prev_completed_decision_generation = 0;
  uint64_t prev_completed_eligibility_generation = 0;
  int prev_completed_override_reason = 0;
  // 0=policy, 1=budget, 2=maintenance, 3=emergency, 4=fallback, 5=drain,
  // 6=latency/space SLO, 7=invalid/mismatched manifest,
  // 8=structural refresh deadline missed.
  int prev_override_reason = 0;
  // Consecutive decisions this level has been deferred while due (score>=1).
  int defer_count = 0;
  // Shadow safety clocks from the accepted active score-event stream. Phase
  // 1a exports these without changing admission behavior.
  uint64_t due_age_micros = 0;
  double pressure_score_micros = 0.0;
  uint64_t score_event_generation = 0;
  bool gate_open = false;
  int gate_mode = 0;
  uint64_t jobs_attempted = 0;
  uint64_t jobs_blocked = 0;
  uint64_t jobs_scheduled = 0;
  uint64_t jobs_completed = 0;
  // Time from the current eligibility interval opening to its first successful
  // native schedule. Zero while the gate is closed or nothing has been
  // admitted yet; it separates "the gate opened" from "the plant responded".
  uint64_t decision_to_first_schedule_micros = 0;
  // Native trivial moves attributed to this source level. A level drained by
  // moves costs almost no write amplification, so a policy comparison that
  // cannot see them misreads cheap progress as expensive progress.
  uint64_t trivial_move_jobs = 0;
  uint64_t trivial_move_bytes = 0;
  int consecutive_blocked = 0;
  bool in_backoff = false;
};

// Full request: global state + one entry per observable input level.
struct RLStateV2 {
  // Protocol v2 is intentionally trigger-only. The agent chooses compact or
  // defer for a level; RocksDB's native leveled picker chooses the SSTs.
  int protocol_version = 2;
  int credit_assignment_version = 2;
  // Nonzero ID of this request/proposal. Python must echo it exactly; C++
  // installs no action from an unacknowledged response.
  uint64_t decision_id = 0;
  // Atomically accumulated since the preceding observation. Bits are defined
  // by RLRewardInvalidReason and are set only when reward/action attribution is
  // genuinely unavailable, never for a known policy override.
  uint64_t prev_reward_invalid_reason_mask = 0;
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
  uint64_t observation_micros = 0;
  uint64_t structural_snapshot_age_micros = 0;
  uint64_t structural_dirty_age_micros = 0;
  uint64_t structural_source_generation = 0;
  uint64_t structural_built_generation = 0;
  uint64_t score_event_generation = 0;
  // --- Read-path telemetry (deltas over interval_micros) -----------------
  // Sourced from the Statistics object RocksDB already maintains, so these
  // cost nothing on the read hot path. Compaction exists to bound read
  // amplification; without these the reward cannot see its own objective.
  uint64_t keys_read = 0;          // NUMBER_KEYS_READ
  uint64_t seeks = 0;              // NUMBER_DB_SEEK
  uint64_t get_hit_l0 = 0;         // GET_HIT_L0
  uint64_t get_hit_l1 = 0;         // GET_HIT_L1
  uint64_t get_hit_l2_and_up = 0;  // GET_HIT_L2_AND_UP
  uint64_t bloom_useful = 0;       // BLOOM_FILTER_USEFUL
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
  // Bottom output level is tree state, not an action-bearing source level.
  int output_only_level_files = 0;
  uint64_t output_only_level_bytes = 0;
  uint64_t output_only_level_target_bytes = 0;
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
  // Internal safety inputs. The socket serializer deliberately continues to
  // send only count/average/p95 to Python; the bucket arrays are used by the
  // in-process live guard and optional compact calibration log.
  RLLatencyHistogram get_latency_buckets{};
  RLLatencyHistogram scan_latency_buckets{};
  RLLatencyHistogram write_latency_buckets{};
  bool done = false;
  std::vector<RLLevelState> levels;
};

// Result of a multi-level query: one action per requested level, in request
// order. `ok=false` means the caller should use its local fallback policy.
struct RLMultiQueryResult {
  enum class Failure : int {
    kNone = 0,
    kTransport = 1,
    kMalformedResponse = 2,
    kDecisionIdMismatch = 3,
    kDuplicateDecisionId = 4,
  };
  bool ok = false;
  uint64_t decision_id = 0;
  Failure failure = Failure::kNone;
  std::vector<RLAction> actions;
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

  // Process-wide monotonic request identity. Multiple column-family pickers
  // share the socket and must not allocate colliding per-picker IDs.
  uint64_t AllocateDecisionId();

  bool IsConnected() const { return fd_ >= 0; }

 private:
  friend class RLCompactionClientTestPeer;
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
  static RLMultiQueryResult DecodeResponse(
      const RLStateV2& state, const std::string& response,
      uint64_t last_acknowledged_decision_id);
  static std::string FormatRequest(const RLStateV2& state);

  std::string socket_path_;
  int socket_timeout_ms_{100};
  int fd_{-1};
  std::atomic<uint64_t> next_decision_id_{1};
  uint64_t last_acknowledged_decision_id_{0};
  mutable std::mutex mu_;
};

}  // namespace ROCKSDB_NAMESPACE
