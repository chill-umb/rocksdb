#include "db/compaction/rl_compaction_telemetry.h"

#include <chrono>
#include <limits>

#include "db/compaction/compaction_pressure_observer.h"

namespace ROCKSDB_NAMESPACE {

namespace {
uint64_t NowMicros() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

uint64_t LatencyBucketUpperBound(int bucket) {
  return bucket >= 63 ? std::numeric_limits<uint64_t>::max()
                      : (1ULL << (bucket + 1)) - 1;
}
}  // namespace

namespace {
std::atomic<bool> g_rl_drain_mode{false};
}  // namespace

void SetRLDrainMode(bool enabled) {
  if (enabled) {
    // Close every open due episode at the workload/drain boundary. Drain hands
    // the tree back to the parent leveled picker, so an episode spanning this
    // instant would mix policy-governed pressure with settle-up pressure in one
    // calibration record.
    //
    // Deliberately unconditional on compaction style: db_bench calls this from
    // WaitForCompaction() for every arm, and both regular and RL column
    // families construct a pressure observer. Gating it would give the two arms
    // differently segmented episode logs, which is the same class of defect as
    // comparing arms through instruments only one of them has.
    CompactionPressureObserver::FlushAllOpenEpisodes("workload_end");
  }
  g_rl_drain_mode.store(enabled, std::memory_order_release);
}

bool RLDrainMode() { return g_rl_drain_mode.load(std::memory_order_acquire); }

RLCompactionTelemetry& RLCompactionTelemetry::Get() {
  static RLCompactionTelemetry telemetry;
  return telemetry;
}

void RLCompactionTelemetry::RecordFlushBytes(uint64_t bytes) {
  flushed_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  bytes_into_level_[0].fetch_add(bytes, std::memory_order_relaxed);
}

void RLCompactionTelemetry::RecordCompactionScheduled(int level,
                                                      bool rl_forced) {
  const int idx = ClampLevel(level);
  compactions_scheduled_from_level_[idx].fetch_add(1,
                                                   std::memory_order_relaxed);
  if (rl_forced) {
    compactions_forced_from_level_[idx].fetch_add(1, std::memory_order_relaxed);
  }
  if (idx == 0) {
    l0_compactions_scheduled_.fetch_add(1, std::memory_order_relaxed);
  }
}

void RLCompactionTelemetry::RecordCompactionCompleted(
    int base_input_level, int output_level, uint64_t bytes_read,
    uint64_t bytes_written, uint64_t decision_id,
    uint64_t decision_generation, uint64_t eligibility_generation,
    int override_reason, bool successful, bool trivial_move) {
  const int in_idx = ClampLevel(base_input_level);
  // Decision zero is meaningful: it identifies an explicit maintenance,
  // fallback, emergency, or drain bypass. Preserve that completion as well so
  // the next observation cannot retain a stale policy completion record.
  last_completed_decision_id_[in_idx].store(decision_id,
                                            std::memory_order_relaxed);
  last_completed_decision_generation_[in_idx].store(
      decision_generation, std::memory_order_relaxed);
  last_completed_eligibility_generation_[in_idx].store(
      eligibility_generation, std::memory_order_relaxed);
  last_completed_override_reason_[in_idx].store(
      override_reason, std::memory_order_relaxed);
  last_completion_result_[in_idx].store(successful ? 1 : 2,
                                        std::memory_order_relaxed);
  if (!successful) return;
  compactions_completed_.fetch_add(1, std::memory_order_relaxed);
  compaction_bytes_read_.fetch_add(bytes_read, std::memory_order_relaxed);
  compaction_bytes_written_.fetch_add(bytes_written, std::memory_order_relaxed);

  const int out_idx = ClampLevel(output_level);
  if (trivial_move) {
    trivial_moves_from_level_[in_idx].fetch_add(1, std::memory_order_relaxed);
    // A move rewrites nothing, so its progress is the input size.
    trivial_move_bytes_from_level_[in_idx].fetch_add(
        bytes_read, std::memory_order_relaxed);
  }
  compactions_from_level_[in_idx].fetch_add(1, std::memory_order_relaxed);
  compaction_read_from_level_[in_idx].fetch_add(bytes_read,
                                                std::memory_order_relaxed);
  compaction_written_from_level_[in_idx].fetch_add(bytes_written,
                                                   std::memory_order_relaxed);
  bytes_into_level_[out_idx].fetch_add(bytes_written,
                                       std::memory_order_relaxed);

  if (in_idx == 0) {
    l0_compactions_completed_.fetch_add(1, std::memory_order_relaxed);
  }
}

void RLCompactionTelemetry::RecordWriteStall(WriteStallCondition condition) {
  if (condition != WriteStallCondition::kNormal) {
    stall_count_.fetch_add(1, std::memory_order_relaxed);
  }
  if (condition == WriteStallCondition::kStopped) {
    stop_count_.fetch_add(1, std::memory_order_relaxed);
  }
  const uint64_t now = NowMicros();
  if (condition != WriteStallCondition::kNormal) {
    uint64_t expected = 0;
    stall_started_micros_.compare_exchange_strong(expected, now,
                                                  std::memory_order_relaxed);
  } else {
    const uint64_t started =
        stall_started_micros_.exchange(0, std::memory_order_relaxed);
    if (started != 0 && now >= started) {
      stall_duration_micros_.fetch_add(now - started,
                                       std::memory_order_relaxed);
    }
  }
}

void RLCompactionTelemetry::RecordForegroundOperation(
    ForegroundOperation operation, uint64_t latency_ns,
    uint64_t logical_write_bytes, uint64_t scan_returned_entries,
    uint64_t scan_internal_skipped, uint64_t scan_sorted_run_seeks) {
  const int index = static_cast<int>(operation);
  if (index < 0 || index >= 3) return;
  foreground_count_[index].fetch_add(1, std::memory_order_relaxed);
  foreground_latency_sum_ns_[index].fetch_add(latency_ns,
                                              std::memory_order_relaxed);
  int bucket = 0;
  uint64_t value = latency_ns;
  while (value > 1 && bucket + 1 < kLatencyBuckets) {
    value >>= 1;
    ++bucket;
  }
  foreground_latency_buckets_[index][bucket].fetch_add(
      1, std::memory_order_relaxed);
  user_logical_write_bytes_.fetch_add(logical_write_bytes,
                                      std::memory_order_relaxed);
  scan_returned_entries_.fetch_add(scan_returned_entries,
                                   std::memory_order_relaxed);
  scan_internal_skipped_.fetch_add(scan_internal_skipped,
                                   std::memory_order_relaxed);
  scan_sorted_run_seeks_.fetch_add(scan_sorted_run_seeks,
                                   std::memory_order_relaxed);
}

RLCompactionTelemetrySnapshot RLCompactionTelemetry::Snapshot() const {
  RLCompactionTelemetrySnapshot snapshot;
  snapshot.flushed_bytes = flushed_bytes_.load(std::memory_order_acquire);
  snapshot.compaction_bytes_read =
      compaction_bytes_read_.load(std::memory_order_acquire);
  snapshot.compaction_bytes_written =
      compaction_bytes_written_.load(std::memory_order_acquire);
  snapshot.compactions_completed =
      compactions_completed_.load(std::memory_order_acquire);
  snapshot.l0_compactions_completed =
      l0_compactions_completed_.load(std::memory_order_acquire);
  snapshot.l0_compactions_scheduled =
      l0_compactions_scheduled_.load(std::memory_order_acquire);
  snapshot.stall_count = stall_count_.load(std::memory_order_acquire);
  snapshot.stop_count = stop_count_.load(std::memory_order_acquire);
  snapshot.stall_duration_micros =
      stall_duration_micros_.load(std::memory_order_acquire);
  const uint64_t active_stall_started =
      stall_started_micros_.load(std::memory_order_acquire);
  if (active_stall_started != 0) {
    const uint64_t now = NowMicros();
    if (now >= active_stall_started) {
      snapshot.stall_duration_micros += now - active_stall_started;
    }
  }
  snapshot.user_logical_write_bytes =
      user_logical_write_bytes_.load(std::memory_order_acquire);
  snapshot.scan_returned_entries =
      scan_returned_entries_.load(std::memory_order_acquire);
  snapshot.scan_internal_skipped =
      scan_internal_skipped_.load(std::memory_order_acquire);
  snapshot.scan_sorted_run_seeks =
      scan_sorted_run_seeks_.load(std::memory_order_acquire);
  for (int op = 0; op < 3; ++op) {
    snapshot.foreground_count[op] =
        foreground_count_[op].load(std::memory_order_acquire);
    snapshot.foreground_latency_sum_ns[op] =
        foreground_latency_sum_ns_[op].load(std::memory_order_acquire);
    const uint64_t rank = (snapshot.foreground_count[op] * 95 + 99) / 100;
    uint64_t seen = 0;
    for (int bucket = 0; bucket < kLatencyBuckets; ++bucket) {
      seen += foreground_latency_buckets_[op][bucket].load(
          std::memory_order_acquire);
      if (rank != 0 && seen >= rank) {
        snapshot.foreground_latency_p95_ns[op] =
            LatencyBucketUpperBound(bucket);
        break;
      }
    }
  }
  for (int i = 0; i < kRLTelemetryMaxLevels; ++i) {
    snapshot.bytes_into_level[i] =
        bytes_into_level_[i].load(std::memory_order_acquire);
    snapshot.compaction_read_from_level[i] =
        compaction_read_from_level_[i].load(std::memory_order_acquire);
    snapshot.compaction_written_from_level[i] =
        compaction_written_from_level_[i].load(std::memory_order_acquire);
    snapshot.compactions_from_level[i] =
        compactions_from_level_[i].load(std::memory_order_acquire);
    snapshot.compactions_scheduled_from_level[i] =
        compactions_scheduled_from_level_[i].load(std::memory_order_acquire);
    snapshot.compactions_forced_from_level[i] =
        compactions_forced_from_level_[i].load(std::memory_order_acquire);
    snapshot.last_completed_decision_id[i] =
        last_completed_decision_id_[i].load(std::memory_order_acquire);
    snapshot.last_completed_decision_generation[i] =
        last_completed_decision_generation_[i].load(
            std::memory_order_acquire);
    snapshot.last_completed_eligibility_generation[i] =
        last_completed_eligibility_generation_[i].load(
            std::memory_order_acquire);
    snapshot.last_completed_override_reason[i] =
        last_completed_override_reason_[i].load(std::memory_order_acquire);
    snapshot.last_completion_result[i] =
        last_completion_result_[i].load(std::memory_order_acquire);
  }
  return snapshot;
}

RLCompactionTelemetrySnapshot RLCompactionTelemetry::Consume() {
  RLCompactionTelemetrySnapshot snapshot;
  snapshot.flushed_bytes =
      flushed_bytes_.exchange(0, std::memory_order_acq_rel);
  snapshot.compaction_bytes_read =
      compaction_bytes_read_.exchange(0, std::memory_order_acq_rel);
  snapshot.compaction_bytes_written =
      compaction_bytes_written_.exchange(0, std::memory_order_acq_rel);
  snapshot.compactions_completed =
      compactions_completed_.exchange(0, std::memory_order_acq_rel);
  snapshot.l0_compactions_completed =
      l0_compactions_completed_.exchange(0, std::memory_order_acq_rel);
  snapshot.l0_compactions_scheduled =
      l0_compactions_scheduled_.exchange(0, std::memory_order_acq_rel);
  snapshot.stall_count = stall_count_.exchange(0, std::memory_order_acq_rel);
  snapshot.stop_count = stop_count_.exchange(0, std::memory_order_acq_rel);
  snapshot.stall_duration_micros =
      stall_duration_micros_.exchange(0, std::memory_order_acq_rel);
  // Split an ongoing stall at the telemetry-window boundary. This attributes
  // every stalled microsecond exactly once while keeping the stall open for
  // the next interval.
  const uint64_t stall_window_end = NowMicros();
  uint64_t active_stall_started =
      stall_started_micros_.load(std::memory_order_acquire);
  while (active_stall_started != 0 &&
         !stall_started_micros_.compare_exchange_weak(
             active_stall_started, stall_window_end, std::memory_order_acq_rel,
             std::memory_order_acquire)) {
  }
  if (active_stall_started != 0 && stall_window_end >= active_stall_started) {
    snapshot.stall_duration_micros += stall_window_end - active_stall_started;
  }
  snapshot.user_logical_write_bytes =
      user_logical_write_bytes_.exchange(0, std::memory_order_acq_rel);
  snapshot.scan_returned_entries =
      scan_returned_entries_.exchange(0, std::memory_order_acq_rel);
  snapshot.scan_internal_skipped =
      scan_internal_skipped_.exchange(0, std::memory_order_acq_rel);
  snapshot.scan_sorted_run_seeks =
      scan_sorted_run_seeks_.exchange(0, std::memory_order_acq_rel);
  for (int op = 0; op < 3; ++op) {
    snapshot.foreground_count[op] =
        foreground_count_[op].exchange(0, std::memory_order_acq_rel);
    snapshot.foreground_latency_sum_ns[op] =
        foreground_latency_sum_ns_[op].exchange(0, std::memory_order_acq_rel);
    const uint64_t rank = (snapshot.foreground_count[op] * 95 + 99) / 100;
    uint64_t seen = 0;
    for (int bucket = 0; bucket < kLatencyBuckets; ++bucket) {
      seen += foreground_latency_buckets_[op][bucket].exchange(
          0, std::memory_order_acq_rel);
      if (snapshot.foreground_latency_p95_ns[op] == 0 && rank != 0 &&
          seen >= rank) {
        snapshot.foreground_latency_p95_ns[op] =
            LatencyBucketUpperBound(bucket);
      }
    }
  }
  for (int i = 0; i < kRLTelemetryMaxLevels; ++i) {
    snapshot.bytes_into_level[i] =
        bytes_into_level_[i].exchange(0, std::memory_order_acq_rel);
    snapshot.compaction_read_from_level[i] =
        compaction_read_from_level_[i].exchange(0, std::memory_order_acq_rel);
    snapshot.compaction_written_from_level[i] =
        compaction_written_from_level_[i].exchange(0,
                                                   std::memory_order_acq_rel);
    snapshot.compactions_from_level[i] =
        compactions_from_level_[i].exchange(0, std::memory_order_acq_rel);
    snapshot.compactions_scheduled_from_level[i] =
        compactions_scheduled_from_level_[i].exchange(
            0, std::memory_order_acq_rel);
    snapshot.compactions_forced_from_level[i] =
        compactions_forced_from_level_[i].exchange(0,
                                                   std::memory_order_acq_rel);
    snapshot.trivial_moves_from_level[i] =
        trivial_moves_from_level_[i].exchange(0, std::memory_order_acq_rel);
    snapshot.trivial_move_bytes_from_level[i] =
        trivial_move_bytes_from_level_[i].exchange(0,
                                                   std::memory_order_acq_rel);
    snapshot.last_completed_decision_id[i] =
        last_completed_decision_id_[i].exchange(0, std::memory_order_acq_rel);
    snapshot.last_completed_decision_generation[i] =
        last_completed_decision_generation_[i].exchange(
            0, std::memory_order_acq_rel);
    snapshot.last_completed_eligibility_generation[i] =
        last_completed_eligibility_generation_[i].exchange(
            0, std::memory_order_acq_rel);
    snapshot.last_completed_override_reason[i] =
        last_completed_override_reason_[i].exchange(
            0, std::memory_order_acq_rel);
    snapshot.last_completion_result[i] =
        last_completion_result_[i].exchange(0, std::memory_order_acq_rel);
  }

  // Stamp the window these deltas cover. The first Consume() has no previous
  // mark, so it reports 0 and the consumer treats the sample as rate-less.
  const uint64_t now = stall_window_end;
  const uint64_t previous =
      last_consume_micros_.exchange(now, std::memory_order_acq_rel);
  snapshot.interval_micros = previous == 0 ? 0 : now - previous;
  return snapshot;
}

}  // namespace ROCKSDB_NAMESPACE
