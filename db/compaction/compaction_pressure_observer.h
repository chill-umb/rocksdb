// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
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
  ~CompactionPressureObserver();

  void Observe(const VersionStorageInfo* vstorage, uint64_t now_micros = 0);
  std::shared_ptr<CompactionPressureView> view() const { return view_; }

  // Export every level currently inside a due episode, tagged `truncated`, and
  // start a fresh episode for any level that is still due.
  //
  // Without this, an episode is exported only on a due->healthy transition, so
  // a level that ends a run or a phase while still due never writes its record
  // at all. That is a systematic loss of the largest observations, and
  // 06_select_baseline_slo.py estimates its upper tolerance bounds from exactly
  // that tail: the safety limits come out tighter than the baseline warrants,
  // which suppresses the deferral behavior the experiment is about.
  //
  // `phase` distinguishes the workload/drain boundary from process teardown, so
  // one truncated record per level per phase is expected rather than suspect.
  void FlushOpenEpisodes(const char* phase);

  // Process-wide flush for callers that reach the boundary without a column
  // family handle and without the DB mutex — the drain transition runs on the
  // benchmark thread. Both regular and RL column families construct an
  // observer, so this must stay unconditional on compaction style or the two
  // arms end up with differently segmented episode logs.
  static void FlushAllOpenEpisodes(const char* phase);

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
                              uint64_t end_micros, bool truncated,
                              const char* phase) const;

  uint64_t generation_ = 0;
  bool initialized_ = false;
  uint32_t cf_id_ = 0;
  std::string episode_log_path_;
  // Observe() runs under the DB mutex, so it never races another Observe() on
  // the same column family. FlushOpenEpisodes() can arrive from the drain
  // thread without that mutex, which is the only reason this exists; the lock
  // is uncontended on the normal path.
  mutable std::mutex state_mu_;
  std::vector<MutableLevelState> levels_;
  std::shared_ptr<CompactionPressureView> view_;
};

}  // namespace ROCKSDB_NAMESPACE
