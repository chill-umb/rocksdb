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

  // Per-level counters, indexed by level (clamped to kRLTelemetryMaxLevels-1).
  // bytes_into_level: bytes arriving in a level (flush output for L0,
  //   compaction output for deeper levels) — the per-level arrival-rate signal.
  // *_from_level: compaction work whose input base level was this level.
  uint64_t bytes_into_level[kRLTelemetryMaxLevels] = {};
  uint64_t compaction_read_from_level[kRLTelemetryMaxLevels] = {};
  uint64_t compaction_written_from_level[kRLTelemetryMaxLevels] = {};
  uint64_t compactions_from_level[kRLTelemetryMaxLevels] = {};
  uint64_t compactions_scheduled_from_level[kRLTelemetryMaxLevels] = {};
};

class RLCompactionTelemetry {
 public:
  static RLCompactionTelemetry& Get();

  void RecordFlushBytes(uint64_t bytes);
  void RecordCompactionScheduled(int level);
  // Legacy name kept for existing call sites.
  void RecordL0CompactionScheduled() { RecordCompactionScheduled(0); }
  void RecordCompactionCompleted(int base_input_level, int output_level,
                                 uint64_t bytes_read, uint64_t bytes_written);
  void RecordWriteStall(WriteStallCondition condition);

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

  std::atomic<uint64_t> bytes_into_level_[kRLTelemetryMaxLevels] = {};
  std::atomic<uint64_t> compaction_read_from_level_[kRLTelemetryMaxLevels] = {};
  std::atomic<uint64_t> compaction_written_from_level_[kRLTelemetryMaxLevels] =
      {};
  std::atomic<uint64_t> compactions_from_level_[kRLTelemetryMaxLevels] = {};
  std::atomic<uint64_t>
      compactions_scheduled_from_level_[kRLTelemetryMaxLevels] = {};
};

}  // namespace ROCKSDB_NAMESPACE
