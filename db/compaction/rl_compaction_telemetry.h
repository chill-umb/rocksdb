#pragma once

#include <atomic>
#include <cstdint>

#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/types.h"

namespace ROCKSDB_NAMESPACE {

struct RLCompactionTelemetrySnapshot {
  uint64_t flushed_bytes = 0;
  uint64_t compaction_bytes_read = 0;
  uint64_t compaction_bytes_written = 0;
  uint64_t compactions_completed = 0;
  uint64_t l0_compactions_completed = 0;
  uint64_t l0_compactions_scheduled = 0;
  uint64_t stall_count = 0;
  uint64_t stop_count = 0;
};

class RLCompactionTelemetry {
 public:
  static RLCompactionTelemetry& Get();

  void RecordFlushBytes(uint64_t bytes);
  void RecordL0CompactionScheduled();
  void RecordCompactionCompleted(int base_input_level, uint64_t bytes_read,
                                 uint64_t bytes_written);
  void RecordWriteStall(WriteStallCondition condition);

  RLCompactionTelemetrySnapshot Snapshot() const;
  RLCompactionTelemetrySnapshot Consume();

 private:
  RLCompactionTelemetry() = default;

  std::atomic<uint64_t> flushed_bytes_{0};
  std::atomic<uint64_t> compaction_bytes_read_{0};
  std::atomic<uint64_t> compaction_bytes_written_{0};
  std::atomic<uint64_t> compactions_completed_{0};
  std::atomic<uint64_t> l0_compactions_completed_{0};
  std::atomic<uint64_t> l0_compactions_scheduled_{0};
  std::atomic<uint64_t> stall_count_{0};
  std::atomic<uint64_t> stop_count_{0};
};

}  // namespace ROCKSDB_NAMESPACE
