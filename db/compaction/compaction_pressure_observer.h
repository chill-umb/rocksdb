// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

class VersionStorageInfo;

// Immutable event-time state for one level. Pressure is measured in
// score-microseconds using a zero-order hold between accepted score events.
struct CompactionPressureLevelState {
  double score_at_event = 0.0;
  uint64_t event_micros = 0;
  uint64_t due_since_micros = 0;
  // Integrated excess score since due_since_micros. This resets to zero when
  // the level becomes healthy; manifest limits are calibrated per episode.
  double pressure_at_event = 0.0;
};

struct CompactionPressureSnapshot {
  uint64_t generation = 0;
  std::vector<CompactionPressureLevelState> levels;
};

// Refcounted publication surface retained by a worker instead of a live
// ColumnFamilyData or VersionStorageInfo pointer.
class CompactionPressureView {
 public:
  std::shared_ptr<const CompactionPressureSnapshot> Load() const;

 private:
  friend class CompactionPressureObserver;
  void Publish(std::shared_ptr<const CompactionPressureSnapshot> snapshot);

  std::shared_ptr<const CompactionPressureSnapshot> snapshot_;
};

// Per-column-family writer-side observer. Observe() is called while the DB
// mutex protects the active VersionStorageInfo. Readers only load immutable
// publications.
class CompactionPressureObserver {
 public:
  explicit CompactionPressureObserver(int num_levels, uint32_t cf_id = 0);

  void Observe(const VersionStorageInfo* vstorage, uint64_t now_micros = 0);
  std::shared_ptr<CompactionPressureView> view() const { return view_; }

  static uint64_t NowMicros();
  static CompactionPressureSnapshot ExtendTo(
      const CompactionPressureSnapshot& snapshot, uint64_t now_micros);
  // Longest observed score-event publication, process-wide. This is the
  // T_score_event_publication term of the admission-latency budget: the delay
  // between an accepted active-score change and its availability to a worker.
  static uint64_t MaxObserveMicros();

 private:
  struct MutableLevelState {
    double score = 0.0;
    uint64_t event_micros = 0;
    uint64_t due_since_micros = 0;
    double pressure = 0.0;
    double episode_start_pressure = 0.0;
    double episode_max_score = 0.0;
    double episode_max_pending_debt_ratio = 0.0;
  };

  void ExportCompletedEpisode(int level, const MutableLevelState& state,
                              uint64_t end_micros) const;

  uint64_t generation_ = 0;
  bool initialized_ = false;
  uint32_t cf_id_ = 0;
  std::string episode_log_path_;
  std::vector<MutableLevelState> levels_;
  std::shared_ptr<CompactionPressureView> view_;
};

}  // namespace ROCKSDB_NAMESPACE
