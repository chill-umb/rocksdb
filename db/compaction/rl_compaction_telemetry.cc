#include "db/compaction/rl_compaction_telemetry.h"

namespace ROCKSDB_NAMESPACE {

RLCompactionTelemetry& RLCompactionTelemetry::Get() {
  static RLCompactionTelemetry telemetry;
  return telemetry;
}

void RLCompactionTelemetry::RecordFlushBytes(uint64_t bytes) {
  flushed_bytes_.fetch_add(bytes, std::memory_order_relaxed);
}

void RLCompactionTelemetry::RecordL0CompactionScheduled() {
  l0_compactions_scheduled_.fetch_add(1, std::memory_order_relaxed);
}

void RLCompactionTelemetry::RecordCompactionCompleted(int base_input_level,
                                                      uint64_t bytes_read,
                                                      uint64_t bytes_written) {
  compactions_completed_.fetch_add(1, std::memory_order_relaxed);
  compaction_bytes_read_.fetch_add(bytes_read, std::memory_order_relaxed);
  compaction_bytes_written_.fetch_add(bytes_written, std::memory_order_relaxed);
  if (base_input_level == 0) {
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
  return snapshot;
}

}  // namespace ROCKSDB_NAMESPACE
