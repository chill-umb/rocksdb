#pragma once

#include <atomic>
#include <cstdint>

#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/types.h"

namespace ROCKSDB_NAMESPACE {

// Upper bound on the number of LSM levels tracked per level. Levels beyond
// this are clamped into the last bucket (never expected in practice: RocksDB
// defaults to 7 levels, the experiments use 10).
constexpr int kRLTelemetryMaxLevels = 16;

// Drain mode. While set, RLCompactionPicker delegates entirely to the parent
// leveled picker.
//
// This exists because WaitForCompact() only waits for compactions that are
// already *queued*, and the RL policy decides what gets queued: a policy that
// defers can simply refuse to drain, leaving outstanding debt that the
// compaction totals never charge it for. Settling both arms through the same
// leveled invariant is what makes their compaction I/O comparable.
void SetRLDrainMode(bool enabled);
bool RLDrainMode();

struct RLCompactionTelemetrySnapshot {
  // Global counters (kept for the legacy single-level protocol and as
  // aggregate features).
  uint64_t flushed_bytes = 0;
  uint64_t compaction_bytes_read = 0;
  uint64_t compaction_bytes_written = 0;
  uint64_t compactions_completed = 0;
  uint64_t l0_compactions_completed = 0;
  uint64_t l0_compactions_scheduled = 0;
  uint64_t stall_count = 0;
  uint64_t stop_count = 0;
  uint64_t stall_duration_micros = 0;
  uint64_t user_logical_write_bytes = 0;
  uint64_t scan_returned_entries = 0;
  uint64_t scan_internal_skipped = 0;
  uint64_t scan_sorted_run_seeks = 0;
  uint64_t foreground_count[3] = {};
  uint64_t foreground_latency_sum_ns[3] = {};
  uint64_t foreground_latency_p95_ns[3] = {};

  // Wall-clock span these deltas cover, measured between successive Consume()
  // calls. Every counter above is a delta, so without this they are totals
  // over an unknown window.
  uint64_t interval_micros = 0;

  // Per-level counters, indexed by level (clamped to kRLTelemetryMaxLevels-1).
  // bytes_into_level: bytes arriving in a level (flush output for L0,
  //   compaction output for deeper levels) — the per-level arrival-rate signal.
  // *_from_level: compaction work whose input base level was this level.
  uint64_t bytes_into_level[kRLTelemetryMaxLevels] = {};
  uint64_t compaction_read_from_level[kRLTelemetryMaxLevels] = {};
  uint64_t compaction_written_from_level[kRLTelemetryMaxLevels] = {};
  uint64_t compactions_from_level[kRLTelemetryMaxLevels] = {};
  uint64_t compactions_scheduled_from_level[kRLTelemetryMaxLevels] = {};
  // Subset of compactions_scheduled_from_level that the RL agent forced. The
  // remainder came from the parent leveled picker and must not be attributed
  // to the agent as an effect of its own action.
  uint64_t compactions_forced_from_level[kRLTelemetryMaxLevels] = {};
  // Trivial moves are compactions that relink a file into the next level
  // without rewriting it. They drain a level at almost no write-amplification
  // cost, so a decision that produced ten moves is not comparable to one that
  // produced ten rewrites, and the aggregate byte counters cannot separate
  // them.
  uint64_t trivial_moves_from_level[kRLTelemetryMaxLevels] = {};
  uint64_t trivial_move_bytes_from_level[kRLTelemetryMaxLevels] = {};
  uint64_t last_completed_decision_id[kRLTelemetryMaxLevels] = {};
  uint64_t last_completed_decision_generation[kRLTelemetryMaxLevels] = {};
  uint64_t last_completed_eligibility_generation[kRLTelemetryMaxLevels] = {};
  int last_completed_override_reason[kRLTelemetryMaxLevels] = {};
  int last_completion_result[kRLTelemetryMaxLevels] = {};
};

class RLCompactionTelemetry {
 public:
  enum class ForegroundOperation : int { kGet = 0, kScan = 1, kWrite = 2 };
  static RLCompactionTelemetry& Get();

  void RecordFlushBytes(uint64_t bytes);
  // `rl_forced` marks a compaction the RL agent asked for, as opposed to one
  // the parent leveled picker selected on its own.
  void RecordCompactionScheduled(int level, bool rl_forced = false);
  // Legacy name kept for existing call sites.
  void RecordL0CompactionScheduled() { RecordCompactionScheduled(0); }
  void RecordCompactionCompleted(int base_input_level, int output_level,
                                 uint64_t bytes_read, uint64_t bytes_written,
                                 uint64_t decision_id = 0,
                                 uint64_t decision_generation = 0,
                                 uint64_t eligibility_generation = 0,
                                 int override_reason = 0,
                                 bool successful = true,
                                 bool trivial_move = false);
  void RecordWriteStall(WriteStallCondition condition);
  void RecordForegroundOperation(ForegroundOperation operation,
                                 uint64_t latency_ns,
                                 uint64_t logical_write_bytes = 0,
                                 uint64_t scan_returned_entries = 0,
                                 uint64_t scan_internal_skipped = 0,
                                 uint64_t scan_sorted_run_seeks = 0);

  RLCompactionTelemetrySnapshot Snapshot() const;
  RLCompactionTelemetrySnapshot Consume();

 private:
  RLCompactionTelemetry() = default;

  static int ClampLevel(int level) {
    if (level < 0) return 0;
    if (level >= kRLTelemetryMaxLevels) return kRLTelemetryMaxLevels - 1;
    return level;
  }

  std::atomic<uint64_t> flushed_bytes_{0};
  std::atomic<uint64_t> compaction_bytes_read_{0};
  std::atomic<uint64_t> compaction_bytes_written_{0};
  std::atomic<uint64_t> compactions_completed_{0};
  std::atomic<uint64_t> l0_compactions_completed_{0};
  std::atomic<uint64_t> l0_compactions_scheduled_{0};
  std::atomic<uint64_t> stall_count_{0};
  std::atomic<uint64_t> stop_count_{0};
  std::atomic<uint64_t> stall_duration_micros_{0};
  std::atomic<uint64_t> stall_started_micros_{0};
  std::atomic<uint64_t> user_logical_write_bytes_{0};
  std::atomic<uint64_t> scan_returned_entries_{0};
  std::atomic<uint64_t> scan_internal_skipped_{0};
  std::atomic<uint64_t> scan_sorted_run_seeks_{0};
  std::atomic<uint64_t> foreground_count_[3] = {};
  std::atomic<uint64_t> foreground_latency_sum_ns_[3] = {};
  // Log2 nanosecond buckets provide a bounded, lock-free rolling p95.
  static constexpr int kLatencyBuckets = 64;
  std::atomic<uint64_t> foreground_latency_buckets_[3][kLatencyBuckets] = {};

  std::atomic<uint64_t> bytes_into_level_[kRLTelemetryMaxLevels] = {};
  std::atomic<uint64_t> compaction_read_from_level_[kRLTelemetryMaxLevels] = {};
  std::atomic<uint64_t> compaction_written_from_level_[kRLTelemetryMaxLevels] =
      {};
  std::atomic<uint64_t> compactions_from_level_[kRLTelemetryMaxLevels] = {};
  std::atomic<uint64_t>
      compactions_scheduled_from_level_[kRLTelemetryMaxLevels] = {};
  std::atomic<uint64_t> compactions_forced_from_level_[kRLTelemetryMaxLevels] =
      {};
  std::atomic<uint64_t> trivial_moves_from_level_[kRLTelemetryMaxLevels] = {};
  std::atomic<uint64_t>
      trivial_move_bytes_from_level_[kRLTelemetryMaxLevels] = {};
  std::atomic<uint64_t> last_completed_decision_id_[kRLTelemetryMaxLevels] = {};
  std::atomic<uint64_t>
      last_completed_decision_generation_[kRLTelemetryMaxLevels] = {};
  std::atomic<uint64_t>
      last_completed_eligibility_generation_[kRLTelemetryMaxLevels] = {};
  std::atomic<int> last_completed_override_reason_[kRLTelemetryMaxLevels] = {};
  std::atomic<int> last_completion_result_[kRLTelemetryMaxLevels] = {};

  // steady_clock microseconds at the last Consume(); 0 until the first one.
  std::atomic<uint64_t> last_consume_micros_{0};
};

}  // namespace ROCKSDB_NAMESPACE
