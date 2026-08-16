// Copyright (c) Facebook, Inc. and its affiliates.

#include "db/compaction/compaction_pressure_observer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <utility>

#include "db/version_set.h"

namespace ROCKSDB_NAMESPACE {
namespace {

std::mutex& EpisodeLogMutex() {
  static std::mutex mutex;
  return mutex;
}

std::atomic<uint64_t>& MaxObserveMicrosSlot() {
  static std::atomic<uint64_t> slot{0};
  return slot;
}

}  // namespace

uint64_t CompactionPressureObserver::MaxObserveMicros() {
  return MaxObserveMicrosSlot().load(std::memory_order_relaxed);
}

std::shared_ptr<const CompactionPressureSnapshot>
CompactionPressureView::Load() const {
  return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
}

void CompactionPressureView::Publish(
    std::shared_ptr<const CompactionPressureSnapshot> snapshot) {
  std::atomic_store_explicit(&snapshot_, std::move(snapshot),
                             std::memory_order_release);
}

CompactionPressureObserver::CompactionPressureObserver(int num_levels,
                                                       uint32_t cf_id)
    : cf_id_(cf_id),
      levels_(std::max(0, num_levels)),
      view_(std::make_shared<CompactionPressureView>()) {
  const char* episode_log = std::getenv("RL_PRESSURE_EPISODE_LOG");
  if (episode_log != nullptr) episode_log_path_ = episode_log;
  auto initial = std::make_shared<CompactionPressureSnapshot>();
  initial->levels.resize(levels_.size());
  view_->Publish(std::move(initial));
}

void CompactionPressureObserver::ExportCompletedEpisode(
    int level, const MutableLevelState& state, uint64_t end_micros) const {
  if (episode_log_path_.empty() || state.due_since_micros == 0 ||
      end_micros < state.due_since_micros) {
    return;
  }
  // Regular leveled and RL runs use this identical event-time exporter. One
  // append is protected process-wide so multiple column families cannot
  // interleave JSON records.
  std::lock_guard<std::mutex> lock(EpisodeLogMutex());
  std::ofstream output(episode_log_path_, std::ios::out | std::ios::app);
  if (!output) return;
  output << "{\"schema_version\":1,\"cf_id\":" << cf_id_
         << ",\"level\":" << level
         << ",\"start_micros\":" << state.due_since_micros
         << ",\"end_micros\":" << end_micros
         << ",\"duration_micros\":"
         << (end_micros - state.due_since_micros)
         << ",\"max_score\":" << state.episode_max_score
         << ",\"integrated_excess_score_micros\":"
         << (state.pressure - state.episode_start_pressure)
         << ",\"max_pending_debt_ratio\":"
         << state.episode_max_pending_debt_ratio << "}\n";
}

uint64_t CompactionPressureObserver::NowMicros() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void CompactionPressureObserver::Observe(const VersionStorageInfo* vstorage,
                                         uint64_t now_micros) {
  if (vstorage == nullptr) return;
  if (now_micros == 0) now_micros = NowMicros();

  std::vector<double> new_scores(levels_.size(), 0.0);
  const int max_rank = std::min(vstorage->MaxInputLevel(),
                                static_cast<int>(levels_.size()) - 1);
  for (int rank = 0; rank <= max_rank; ++rank) {
    const int level = vstorage->CompactionScoreLevel(rank);
    if (level >= 0 && level < static_cast<int>(new_scores.size())) {
      new_scores[level] = vstorage->CompactionScore(rank);
    }
  }

  double pending_debt_ratio = 0.0;
  if (!episode_log_path_.empty()) {
    const uint64_t live_bytes = vstorage->EstimateLiveDataSize();
    pending_debt_ratio =
        static_cast<double>(vstorage->estimated_compaction_needed_bytes()) /
        static_cast<double>(std::max<uint64_t>(live_bytes, 1));
  }

  auto publication = std::make_shared<CompactionPressureSnapshot>();
  publication->generation = ++generation_;
  publication->levels.resize(levels_.size());
  for (size_t level = 0; level < levels_.size(); ++level) {
    MutableLevelState& current = levels_[level];
    if (initialized_ && now_micros >= current.event_micros) {
      current.pressure +=
          std::max(0.0, current.score - 1.0) *
          static_cast<double>(now_micros - current.event_micros);
    }

    const bool was_due = initialized_ && current.score >= 1.0;
    const bool is_due = new_scores[level] >= 1.0;
    if (!was_due && is_due) {
      current.due_since_micros = now_micros;
      current.episode_start_pressure = current.pressure;
      current.episode_max_score = new_scores[level];
      current.episode_max_pending_debt_ratio = pending_debt_ratio;
    } else if (was_due && !is_due) {
      ExportCompletedEpisode(static_cast<int>(level), current, now_micros);
      current.due_since_micros = 0;
      current.episode_start_pressure = current.pressure;
      current.episode_max_score = 0.0;
      current.episode_max_pending_debt_ratio = 0.0;
    } else if (is_due) {
      current.episode_max_score =
          std::max(current.episode_max_score, new_scores[level]);
      current.episode_max_pending_debt_ratio =
          std::max(current.episode_max_pending_debt_ratio,
                   pending_debt_ratio);
    }
    current.score = new_scores[level];
    current.event_micros = now_micros;

    CompactionPressureLevelState& out = publication->levels[level];
    out.score_at_event = current.score;
    out.event_micros = current.event_micros;
    out.due_since_micros = current.due_since_micros;
    out.pressure_at_event = current.due_since_micros == 0
                                ? 0.0
                                : current.pressure -
                                      current.episode_start_pressure;
  }
  initialized_ = true;
  view_->Publish(std::move(publication));

  // Publication runs inside the accepted score recomputation, so this is the
  // whole delay between the score changing and a worker being able to read it.
  const uint64_t published_micros = NowMicros();
  if (published_micros > now_micros) {
    std::atomic<uint64_t>& slot = MaxObserveMicrosSlot();
    const uint64_t elapsed = published_micros - now_micros;
    uint64_t current = slot.load(std::memory_order_relaxed);
    while (current < elapsed &&
           !slot.compare_exchange_weak(current, elapsed,
                                       std::memory_order_relaxed)) {
    }
  }
}

CompactionPressureSnapshot CompactionPressureObserver::ExtendTo(
    const CompactionPressureSnapshot& snapshot, uint64_t now_micros) {
  CompactionPressureSnapshot result = snapshot;
  for (CompactionPressureLevelState& level : result.levels) {
    if (level.due_since_micros != 0 &&
        now_micros >= level.event_micros) {
      level.pressure_at_event +=
          std::max(0.0, level.score_at_event - 1.0) *
          static_cast<double>(now_micros - level.event_micros);
      level.event_micros = now_micros;
    }
  }
  return result;
}

}  // namespace ROCKSDB_NAMESPACE
