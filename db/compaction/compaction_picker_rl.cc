// Copyright (c) Facebook, Inc. and its affiliates.

#include "db/compaction/compaction_picker_rl.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <limits>
#include <utility>

#include "logging/log_buffer.h"
#include "logging/logging.h"
#include "rocksdb/statistics.h"

namespace ROCKSDB_NAMESPACE {
namespace {

int EnvInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  const int parsed = std::atoi(value);
  return parsed > 0 ? parsed : fallback;
}

bool EnvBool(const char* name, bool fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  return !(value[0] == '0' && value[1] == '\0');
}

uint64_t NowMicros() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

uint64_t MixEpoch(uint64_t hash, uint64_t value) {
  hash ^= value;
  return hash * 1099511628211ULL;
}

}  // namespace

RLCompactionPicker::RLCompactionPicker(const ImmutableOptions& ioptions,
                                       const InternalKeyComparator* icmp)
    : LevelCompactionPicker(ioptions, icmp),
      decision_interval_ms_(EnvInt("RL_DECISION_INTERVAL_MS", 50)),
      observe_interval_ms_(
          std::min(EnvInt("RL_OBSERVE_INTERVAL_MS", decision_interval_ms_),
                   decision_interval_ms_)),
      max_defer_steps_(EnvInt("RL_MAX_DEFER_STEPS", 50)),
      max_defer_steps_l0_(EnvInt("RL_MAX_DEFER_STEPS_L0", 1)),
      allow_defer_(EnvBool("RL_ALLOW_DEFER", true)) {
  for (int level = 0; level < kMaxRLLevels; ++level) {
    last_transition_valid_[level].store(true, std::memory_order_relaxed);
  }
  worker_ = std::thread([this] { WorkerLoop(); });
}

RLCompactionPicker::~RLCompactionPicker() {
  worker_stop_.store(true, std::memory_order_release);
  snap_cv_.notify_all();
  if (worker_.joinable()) worker_.join();
  SendDoneMessage();
}

double RLCompactionPicker::LevelScore(const VersionStorageInfo* vstorage,
                                      int level) const {
  for (int rank = 0; rank <= vstorage->MaxInputLevel(); ++rank) {
    if (vstorage->CompactionScoreLevel(rank) == level) {
      return vstorage->CompactionScore(rank);
    }
  }
  return 0.0;
}

uint64_t RLCompactionPicker::SnapshotEpoch(
    const VersionStorageInfo* vstorage) const {
  // A structural fingerprint is a stronger lease epoch than a local counter:
  // any flush, deletion, compaction, or conflict-bit change invalidates the
  // preview even if no new observation has yet been published.
  uint64_t epoch = 1469598103934665603ULL;
  epoch = MixEpoch(epoch, static_cast<uint64_t>(vstorage->num_levels()));
  epoch = MixEpoch(epoch, static_cast<uint64_t>(vstorage->base_level()));
  for (int level = 0; level < vstorage->num_levels(); ++level) {
    epoch = MixEpoch(epoch, static_cast<uint64_t>(level));
    for (const FileMetaData* file : vstorage->LevelFiles(level)) {
      epoch = MixEpoch(epoch, file->fd.GetNumber());
      epoch = MixEpoch(epoch, file->fd.GetFileSize());
      epoch = MixEpoch(epoch, file->being_compacted ? 1 : 0);
    }
  }
  return epoch == 0 ? 1 : epoch;
}

uint64_t RLCompactionPicker::NextLevelOverlapBytes(
    const VersionStorageInfo* vstorage, int level) const {
  const int next = level + 1;
  if (next >= vstorage->num_levels()) return 0;
  const auto& files = vstorage->LevelFiles(level);
  if (files.empty() || vstorage->NumLevelFiles(next) == 0) return 0;
  const InternalKey* smallest = &files.front()->smallest;
  const InternalKey* largest = &files.front()->largest;
  for (const FileMetaData* file : files) {
    if (icmp_->Compare(file->smallest, *smallest) < 0)
      smallest = &file->smallest;
    if (icmp_->Compare(file->largest, *largest) > 0) largest = &file->largest;
  }
  std::vector<FileMetaData*> overlaps;
  vstorage->GetOverlappingInputs(next, smallest, largest, &overlaps);
  uint64_t bytes = 0;
  for (const FileMetaData* file : overlaps) bytes += file->fd.GetFileSize();
  return bytes;
}

void RLCompactionPicker::BuildSnapshot(const VersionStorageInfo* vstorage,
                                       uint64_t pcb, RLStateV2* state) const {
  state->protocol_version = 2;
  state->snapshot_epoch = SnapshotEpoch(vstorage);
  state->pending_compaction_bytes = pcb;
  state->l0_compaction_trigger = rl_l0_trigger_.load(std::memory_order_relaxed);
  state->l0_slowdown_trigger =
      rl_l0_slowdown_trigger_.load(std::memory_order_relaxed);
  state->l0_stop_trigger = rl_l0_stop_trigger_.load(std::memory_order_relaxed);
  state->l0_delay_trigger_count = vstorage->l0_delay_trigger_count();
  state->fallback_count = rl_fallback_count_.load(std::memory_order_relaxed);
  state->done = false;
  for (int level = 0; level < vstorage->num_levels(); ++level) {
    state->physical_sst_bytes += vstorage->NumLevelBytes(level);
  }
  state->live_logical_bytes = vstorage->EstimateLiveDataSize();

  const int max_input_level =
      std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
  for (int level = 0; level <= max_input_level; ++level) {
    if (level > 0 && vstorage->NumLevelFiles(level) == 0) {
      continue;
    }
    RLLevelState ls;
    ls.level = level;
    ls.files = vstorage->NumLevelFiles(level);
    ls.bytes = vstorage->NumLevelBytes(level);
    ls.score = LevelScore(vstorage, level);
    ls.target_bytes = level > 0 ? vstorage->MaxBytesForLevel(level) : 0;
    ls.is_last = level + 1 >= vstorage->num_levels();
    if (!ls.is_last) {
      const int next = level + 1;
      ls.next_level_files = vstorage->NumLevelFiles(next);
      ls.next_level_bytes = vstorage->NumLevelBytes(next);
      ls.next_level_score =
          next <= vstorage->MaxInputLevel() ? LevelScore(vstorage, next) : 0.0;
      ls.next_level_target_bytes = vstorage->MaxBytesForLevel(next);
      ls.overlap_bytes = NextLevelOverlapBytes(vstorage, level);
    }
    ls.default_needed = ls.score >= 1.0;
    ls.defer_count = defer_count_[level].load(std::memory_order_relaxed);
    ls.prev_action_executed =
        last_effective_action_[level].load(std::memory_order_relaxed);
    ls.prev_action_overridden =
        last_action_overridden_[level].load(std::memory_order_relaxed);
    ls.prev_compaction_picked =
        compaction_picked_[level].load(std::memory_order_relaxed);
    ls.prev_decision_id =
        last_decision_id_[level].load(std::memory_order_relaxed);
    ls.prev_snapshot_epoch =
        last_snapshot_epoch_[level].load(std::memory_order_relaxed);
    ls.prev_scheduling_result =
        last_scheduling_result_[level].load(std::memory_order_relaxed);
    ls.prev_override_reason =
        last_override_reason_[level].load(std::memory_order_relaxed);
    ls.prev_transition_valid =
        last_transition_valid_[level].load(std::memory_order_relaxed);

    state->levels.push_back(std::move(ls));
  }
}

void RLCompactionPicker::PublishSnapshot(const VersionStorageInfo* vstorage,
                                         uint64_t pcb) const {
  const uint64_t now = NowMicros();
  const uint64_t previous =
      last_publish_micros_.load(std::memory_order_relaxed);
  const uint64_t min_gap =
      static_cast<uint64_t>(observe_interval_ms_) * 1000 / 4;
  if (previous != 0 && now - previous < min_gap) return;
  last_publish_micros_.store(now, std::memory_order_relaxed);
  const uint64_t start = NowMicros();
  RLStateV2 state;
  BuildSnapshot(vstorage, pcb, &state);
  rl_publish_calls_.fetch_add(1, std::memory_order_relaxed);
  rl_publish_nanos_.fetch_add((NowMicros() - start) * 1000,
                              std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    pending_snapshot_ = std::move(state);
    snapshot_valid_ = true;
  }
  snap_cv_.notify_one();
}

void RLCompactionPicker::PopulateReadStats(RLStateV2& state) {
  Statistics* stats = ioptions_.stats;
  if (stats == nullptr) return;
  const uint32_t tickers[kNumReadTickers] = {NUMBER_KEYS_READ,
                                             NUMBER_DB_SEEK,
                                             GET_HIT_L0,
                                             GET_HIT_L1,
                                             GET_HIT_L2_AND_UP,
                                             BLOOM_FILTER_USEFUL,
                                             NON_LAST_LEVEL_READ_COUNT,
                                             LAST_LEVEL_READ_COUNT,
                                             POINT_SST_PROBE};
  uint64_t delta[kNumReadTickers] = {};
  for (int i = 0; i < kNumReadTickers; ++i) {
    const uint64_t current = stats->getTickerCount(tickers[i]);
    delta[i] =
        current >= prev_read_tickers_[i] ? current - prev_read_tickers_[i] : 0;
    prev_read_tickers_[i] = current;
  }
  state.keys_read = delta[0];
  state.seeks = delta[1];
  state.get_hit_l0 = delta[2];
  state.get_hit_l1 = delta[3];
  state.get_hit_l2_and_up = delta[4];
  state.bloom_useful = delta[5];
  state.non_last_level_read_count = delta[6];
  state.last_level_read_count = delta[7];
  state.point_sst_probes = delta[8];
}

void RLCompactionPicker::ExpireLeasesAtActuation() const {
  std::lock_guard<std::mutex> lock(lease_mu_);
  for (int level = 0; level < kMaxRLLevels; ++level) {
    if (!leases_[level].active) continue;
    leases_[level].active = false;
    last_scheduling_result_[level].store(3, std::memory_order_relaxed);
    last_transition_valid_[level].store(false, std::memory_order_relaxed);
  }
}

void RLCompactionPicker::InstallLease(const ActionLease& lease) const {
  if (lease.level < 0 || lease.level >= kMaxRLLevels) return;
  std::lock_guard<std::mutex> lock(lease_mu_);
  leases_[lease.level] = lease;
  leases_[lease.level].active = true;
  last_decision_id_[lease.level].store(lease.decision_id,
                                       std::memory_order_relaxed);
  last_snapshot_epoch_[lease.level].store(lease.snapshot_epoch,
                                          std::memory_order_relaxed);
  last_scheduling_result_[lease.level].store(0, std::memory_order_relaxed);
  last_override_reason_[lease.level].store(static_cast<int>(lease.reason),
                                           std::memory_order_relaxed);
  last_transition_valid_[lease.level].store(
      lease.reason != ActionReason::kFallback, std::memory_order_relaxed);
}

bool RLCompactionPicker::HasActiveLease() const {
  std::lock_guard<std::mutex> lock(lease_mu_);
  for (const ActionLease& lease : leases_) {
    if (lease.active) return true;
  }
  return false;
}

bool RLCompactionPicker::TakeLease(ActionLease* lease) const {
  std::lock_guard<std::mutex> lock(lease_mu_);
  int chosen = -1;
  double score = -std::numeric_limits<double>::infinity();
  for (int level = 0; level < kMaxRLLevels; ++level) {
    if (leases_[level].active && leases_[level].score > score) {
      chosen = level;
      score = leases_[level].score;
    }
  }
  if (chosen < 0) return false;
  *lease = leases_[chosen];
  leases_[chosen].active = false;  // consume before validation/scheduling
  return true;
}

int RLCompactionPicker::HighestDueLevel(
    const VersionStorageInfo* vstorage) const {
  int chosen = -1;
  double score = -1.0;
  const int max_level = std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
  for (int level = 0; level <= max_level; ++level) {
    const double candidate_score = LevelScore(vstorage, level);
    if (candidate_score >= 1.0 && candidate_score > score &&
        vstorage->NumLevelFiles(level) > 0) {
      chosen = level;
      score = candidate_score;
    }
  }
  return chosen;
}

void RLCompactionPicker::InstallLocalLease(const VersionStorageInfo* vstorage,
                                           int level,
                                           ActionReason reason) const {
  if (level < 0 || level >= kMaxRLLevels) return;
  {
    std::lock_guard<std::mutex> lock(lease_mu_);
    if (leases_[level].active && leases_[level].reason == reason) return;
    // A synchronous safety/fallback verdict supersedes a decision made from
    // an older snapshot. Keep exactly one authority globally and make the
    // displaced transition explicitly invalid rather than leaving two levels
    // able to schedule from one scheduler wakeup.
    for (int index = 0; index < kMaxRLLevels; ++index) {
      if (!leases_[index].active) continue;
      leases_[index].active = false;
      last_scheduling_result_[index].store(3, std::memory_order_relaxed);
      last_transition_valid_[index].store(false, std::memory_order_relaxed);
    }
  }
  ActionLease lease;
  lease.active = true;
  lease.snapshot_epoch = SnapshotEpoch(vstorage);
  lease.level = level;
  lease.score = LevelScore(vstorage, level);
  lease.reason = reason;
  InstallLease(lease);
}

void RLCompactionPicker::RunDecisionCycle(RLStateV2& state, bool actuate) {
  const RLCompactionTelemetrySnapshot telemetry =
      RLCompactionTelemetry::Get().Consume();
  state.flushed_bytes = telemetry.flushed_bytes;
  state.compaction_bytes_read = telemetry.compaction_bytes_read;
  state.compaction_bytes_written = telemetry.compaction_bytes_written;
  state.compactions_completed = telemetry.compactions_completed;
  state.stall_count = telemetry.stall_count;
  state.stop_count = telemetry.stop_count;
  state.stall_duration_micros = telemetry.stall_duration_micros;
  state.user_logical_write_bytes = telemetry.user_logical_write_bytes;
  state.scan_returned_entries = telemetry.scan_returned_entries;
  state.scan_internal_skipped = telemetry.scan_internal_skipped;
  state.scan_sorted_run_seeks = telemetry.scan_sorted_run_seeks;
  state.interval_micros = telemetry.interval_micros;
  state.get_latency_count = telemetry.foreground_count[0];
  state.get_latency_avg_ns =
      telemetry.foreground_count[0] == 0
          ? 0.0
          : static_cast<double>(telemetry.foreground_latency_sum_ns[0]) /
                telemetry.foreground_count[0];
  state.get_latency_p95_ns = telemetry.foreground_latency_p95_ns[0];
  state.scan_latency_count = telemetry.foreground_count[1];
  state.scan_latency_avg_ns =
      telemetry.foreground_count[1] == 0
          ? 0.0
          : static_cast<double>(telemetry.foreground_latency_sum_ns[1]) /
                telemetry.foreground_count[1];
  state.scan_latency_p95_ns = telemetry.foreground_latency_p95_ns[1];
  state.write_latency_count = telemetry.foreground_count[2];
  state.write_latency_avg_ns =
      telemetry.foreground_count[2] == 0
          ? 0.0
          : static_cast<double>(telemetry.foreground_latency_sum_ns[2]) /
                telemetry.foreground_count[2];
  state.write_latency_p95_ns = telemetry.foreground_latency_p95_ns[2];
  state.fallback_count = rl_fallback_count_.load(std::memory_order_relaxed);
  PopulateReadStats(state);

  for (RLLevelState& level : state.levels) {
    const int index = std::min(level.level, kRLTelemetryMaxLevels - 1);
    level.bytes_in = telemetry.bytes_into_level[index];
    level.bytes_read_out = telemetry.compaction_read_from_level[index];
    level.bytes_written_out = telemetry.compaction_written_from_level[index];
    level.compactions_from = telemetry.compactions_from_level[index];
    level.compactions_scheduled =
        telemetry.compactions_scheduled_from_level[index];
    level.compactions_forced = telemetry.compactions_forced_from_level[index];
    level.prev_completed_decision_id =
        telemetry.last_completed_decision_id[index];
    level.prev_completion_result = telemetry.last_completion_result[index];
  }

  const RLMultiQueryResult result =
      RLCompactionClient::Get().QueryActions(state);
  const uint64_t query_id =
      rl_query_count_.fetch_add(1, std::memory_order_relaxed) + 1;
  if (!result.ok) {
    rl_available_.store(false, std::memory_order_release);
    const uint64_t failures =
        rl_fallback_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (actuate) {
      ExpireLeasesAtActuation();
      const RLLevelState* due = nullptr;
      for (const RLLevelState& level : state.levels) {
        if (level.default_needed &&
            (due == nullptr || level.score > due->score)) {
          due = &level;
        }
      }
      if (due != nullptr) {
        ActionLease fallback;
        fallback.active = true;
        fallback.snapshot_epoch = state.snapshot_epoch;
        fallback.level = due->level;
        fallback.score = due->score;
        fallback.reason = ActionReason::kFallback;
        InstallLease(fallback);
      }
    }
    if (!rl_fallback_logged_.exchange(true, std::memory_order_relaxed) ||
        failures % 200 == 0) {
      ROCKS_LOG_WARN(ioptions_.logger,
                     "RL response unavailable; installed a level-scoped "
                     "fallback lease (cumulative_fallbacks=%" PRIu64 ")",
                     failures);
    }
    return;
  }

  rl_available_.store(true, std::memory_order_release);
  rl_fallback_logged_.store(false, std::memory_order_relaxed);
  if (!actuate) return;
  rl_actuation_count_.fetch_add(1, std::memory_order_relaxed);
  ExpireLeasesAtActuation();

  ActionLease chosen;
  int chosen_priority = -1;
  for (size_t i = 0; i < state.levels.size(); ++i) {
    const RLLevelState& level = state.levels[i];
    const int source_level = level.level;
    bool compact = result.actions[i] == RLAction::kCompactNow;
    bool overridden = false;
    ActionReason reason = ActionReason::kPolicy;

    if (level.default_needed && !compact) {
      const int deferred =
          defer_count_[source_level].fetch_add(1, std::memory_order_relaxed) +
          1;
      if (!allow_defer_ || deferred >= MaxDeferSteps(source_level)) {
        compact = true;
        overridden = true;
        reason = ActionReason::kBudget;
      }
    } else if (!level.default_needed || compact) {
      defer_count_[source_level].store(0, std::memory_order_relaxed);
    }
    if (telemetry.stop_count > 0 && source_level == 0 && level.files > 0) {
      compact = true;
      overridden = true;
      reason = ActionReason::kEmergency;
    }

    last_effective_action_[source_level].store(compact ? 1 : 0,
                                               std::memory_order_relaxed);
    last_action_overridden_[source_level].store(overridden,
                                                std::memory_order_relaxed);
    compaction_picked_[source_level].store(false, std::memory_order_relaxed);
    if (!compact) continue;

    int priority = reason == ActionReason::kEmergency
                       ? 3
                       : (reason == ActionReason::kBudget ? 2 : 1);
    if (priority > chosen_priority ||
        (priority == chosen_priority && level.score > chosen.score)) {
      if (chosen.level >= 0) {
        last_effective_action_[chosen.level].store(0,
                                                   std::memory_order_relaxed);
        last_action_overridden_[chosen.level].store(true,
                                                    std::memory_order_relaxed);
      }
      chosen.active = true;
      // Protocol v2 deliberately returns actions only. Generate attribution
      // locally so file identity never has to cross the policy boundary.
      chosen.decision_id = query_id;
      chosen.snapshot_epoch = state.snapshot_epoch;
      chosen.level = source_level;
      chosen.score = level.score;
      chosen.reason = reason;
      chosen_priority = priority;
    } else {
      // A protocol response never grants more than one scheduling authority.
      last_effective_action_[source_level].store(0, std::memory_order_relaxed);
      last_action_overridden_[source_level].store(true,
                                                  std::memory_order_relaxed);
    }
  }
  if (chosen.level >= 0) InstallLease(chosen);
}

void RLCompactionPicker::WorkerLoop() {
  auto next_tick = std::chrono::steady_clock::now();
  auto next_actuation = next_tick;
  const auto observe_interval = std::chrono::milliseconds(observe_interval_ms_);
  const auto actuation_interval =
      std::chrono::milliseconds(decision_interval_ms_);
  while (!worker_stop_.load(std::memory_order_acquire)) {
    next_tick += observe_interval;
    const auto now = std::chrono::steady_clock::now();
    if (next_tick < now) next_tick = now + observe_interval;
    RLStateV2 state;
    {
      std::unique_lock<std::mutex> lock(snap_mu_);
      snap_cv_.wait_until(lock, next_tick, [this] {
        return worker_stop_.load(std::memory_order_acquire);
      });
      if (worker_stop_.load(std::memory_order_acquire)) break;
      if (!snapshot_valid_) {
        rl_skipped_ticks_.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      state = std::move(pending_snapshot_);
      pending_snapshot_ = RLStateV2();
      snapshot_valid_ = false;
    }
    if (state.levels.empty()) continue;
    std::vector<int> levels;
    for (const RLLevelState& level : state.levels)
      levels.push_back(level.level);
    const bool actuate = next_tick >= next_actuation;
    if (actuate) next_actuation = next_tick + actuation_interval;
    RunDecisionCycle(state, actuate);
    {
      std::lock_guard<std::mutex> lock(snap_mu_);
      last_sent_levels_ = std::move(levels);
    }
    LogDiagnostics(false);
  }
  LogDiagnostics(true);
}

void RLCompactionPicker::SendDoneMessage() {
  std::vector<int> levels;
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    levels = last_sent_levels_;
  }
  if (levels.empty()) return;
  RLStateV2 state;
  state.protocol_version = 2;
  state.done = true;
  for (int source_level : levels) {
    RLLevelState level;
    level.level = source_level;
    level.prev_action_executed =
        last_effective_action_[source_level].load(std::memory_order_relaxed);
    level.prev_compaction_picked =
        compaction_picked_[source_level].load(std::memory_order_relaxed);
    level.prev_decision_id =
        last_decision_id_[source_level].load(std::memory_order_relaxed);
    level.prev_transition_valid =
        last_transition_valid_[source_level].load(std::memory_order_relaxed);
    state.levels.push_back(std::move(level));
  }
  RLCompactionClient::Get().QueryActions(state);
}

void RLCompactionPicker::LogDiagnostics(bool final) const {
  const uint64_t queries = rl_query_count_.load(std::memory_order_relaxed);
  if (!final && (queries == 0 || queries % kDiagnosticsEveryQueries != 0)) {
    return;
  }
  ROCKS_LOG_INFO(ioptions_.logger,
                 "RL trigger diagnostics: protocol=2 queries=%" PRIu64
                 " actuations=%" PRIu64 " bypasses=%" PRIu64
                 " skipped_ticks=%" PRIu64 " cumulative_fallbacks=%" PRIu64
                 " available=%d nc_calls=%" PRIu64 " nc_total_ms=%" PRIu64
                 " publish_calls=%" PRIu64 " publish_total_ms=%" PRIu64,
                 queries, rl_actuation_count_.load(std::memory_order_relaxed),
                 rl_bypass_count_.load(std::memory_order_relaxed),
                 rl_skipped_ticks_.load(std::memory_order_relaxed),
                 rl_fallback_count_.load(std::memory_order_relaxed),
                 rl_available_.load(std::memory_order_relaxed) ? 1 : 0,
                 rl_nc_calls_.load(std::memory_order_relaxed),
                 rl_nc_nanos_.load(std::memory_order_relaxed) / 1000000,
                 rl_publish_calls_.load(std::memory_order_relaxed),
                 rl_publish_nanos_.load(std::memory_order_relaxed) / 1000000);
}

bool RLCompactionPicker::NeedsCompaction(
    const VersionStorageInfo* vstorage) const {
  const uint64_t start = NowMicros();
  struct Timer {
    const RLCompactionPicker* picker;
    uint64_t start;
    ~Timer() {
      picker->rl_nc_calls_.fetch_add(1, std::memory_order_relaxed);
      picker->rl_nc_nanos_.fetch_add((NowMicros() - start) * 1000,
                                     std::memory_order_relaxed);
    }
  } timer{this, start};

  if (RLDrainMode()) {
    ExpireLeasesAtActuation();
    parent_bypass_reason_.store(static_cast<int>(ActionReason::kDrain),
                                std::memory_order_relaxed);
    return LevelCompactionPicker::NeedsCompaction(vstorage);
  }
  if (!vstorage->ExpiredTtlFiles().empty() ||
      !vstorage->FilesMarkedForPeriodicCompaction().empty() ||
      !vstorage->BottommostFilesMarkedForCompaction().empty() ||
      !vstorage->FilesMarkedForCompaction().empty() ||
      !vstorage->FilesMarkedForForcedBlobGC().empty()) {
    ExpireLeasesAtActuation();
    parent_bypass_reason_.store(static_cast<int>(ActionReason::kMaintenance),
                                std::memory_order_relaxed);
    rl_bypass_count_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  parent_bypass_reason_.store(-1, std::memory_order_relaxed);

  const uint64_t pcb = vstorage->estimated_compaction_needed_bytes();
  PublishSnapshot(vstorage, pcb);
  const int l0_files = vstorage->NumLevelFiles(0);
  if (l0_files >=
          std::max(1, rl_l0_stop_trigger_.load(std::memory_order_relaxed)) ||
      pcb >= kPcbHardCap) {
    const int level = l0_files > 0 ? 0 : HighestDueLevel(vstorage);
    InstallLocalLease(vstorage, level, ActionReason::kEmergency);
    rl_bypass_count_.fetch_add(1, std::memory_order_relaxed);
    return HasActiveLease();
  }

  const int max_level = std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
  for (int level = 0; level <= max_level; ++level) {
    if (LevelScore(vstorage, level) >= 1.0 &&
        defer_count_[level].load(std::memory_order_relaxed) >=
            MaxDeferSteps(level)) {
      InstallLocalLease(vstorage, level, ActionReason::kBudget);
      return true;
    }
  }

  if (!rl_available_.load(std::memory_order_acquire)) {
    const int due_level = HighestDueLevel(vstorage);
    if (due_level >= 0) {
      InstallLocalLease(vstorage, due_level, ActionReason::kFallback);
      rl_bypass_count_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }
  return HasActiveLease();
}

Compaction* RLCompactionPicker::PickCompaction(
    const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
    const MutableDBOptions& mutable_db_options,
    const std::vector<SequenceNumber>& existing_snapshots,
    const SnapshotChecker* snapshot_checker, VersionStorageInfo* vstorage,
    LogBuffer* log_buffer, const std::string& full_history_ts_low,
    bool require_max_output_level) {
  rl_l0_trigger_.store(mutable_cf_options.level0_file_num_compaction_trigger,
                       std::memory_order_relaxed);
  rl_l0_slowdown_trigger_.store(
      mutable_cf_options.level0_slowdown_writes_trigger,
      std::memory_order_relaxed);
  rl_l0_stop_trigger_.store(mutable_cf_options.level0_stop_writes_trigger,
                            std::memory_order_relaxed);

  ActionLease lease;
  if (TakeLease(&lease)) {
    Compaction* compaction = nullptr;
    const uint64_t current_epoch = SnapshotEpoch(vstorage);
    const CompactionReason reason = lease.level == 0
                                        ? CompactionReason::kLevelL0FilesNum
                                        : CompactionReason::kLevelMaxLevelSize;
    // Force only the source level. PickCompactionFromLevel immediately enters
    // RocksDB's normal FilesByCompactionPri/clean-cut/overlap path; the RL
    // response never supplies an SST identity.
    compaction = PickCompactionFromLevel(
        cf_name, mutable_cf_options, mutable_db_options, vstorage, log_buffer,
        full_history_ts_low, lease.level, lease.score, reason);
    if (compaction == nullptr) {
      last_scheduling_result_[lease.level].store(2, std::memory_order_relaxed);
      last_transition_valid_[lease.level].store(false,
                                                std::memory_order_relaxed);
      ROCKS_LOG_WARN(ioptions_.logger,
                     "RL level trigger scheduling failed: decision=%" PRIu64
                     " epoch=%" PRIu64 " current_epoch=%" PRIu64
                     " level=%d reason=%d",
                     lease.decision_id, lease.snapshot_epoch, current_epoch,
                     lease.level, static_cast<int>(lease.reason));
      return nullptr;
    }
    compaction->SetRLDecisionAttribution(lease.decision_id,
                                         lease.snapshot_epoch,
                                         static_cast<int>(lease.reason));
    last_scheduling_result_[lease.level].store(1, std::memory_order_relaxed);
    compaction_picked_[lease.level].store(true, std::memory_order_relaxed);
    defer_count_[lease.level].store(0, std::memory_order_relaxed);
    RLCompactionTelemetry::Get().RecordCompactionScheduled(
        lease.level, lease.reason == ActionReason::kPolicy);
    ROCKS_LOG_INFO(ioptions_.logger,
                   "RL lease scheduled: decision=%" PRIu64 " epoch=%" PRIu64
                   " level=%d reason=%d",
                   lease.decision_id, lease.snapshot_epoch, lease.level,
                   static_cast<int>(lease.reason));
    return compaction;
  }

  const int parent_reason =
      parent_bypass_reason_.exchange(-1, std::memory_order_relaxed);
  if (parent_reason < 0) return nullptr;
  Compaction* compaction = LevelCompactionPicker::PickCompaction(
      cf_name, mutable_cf_options, mutable_db_options, existing_snapshots,
      snapshot_checker, vstorage, log_buffer, full_history_ts_low,
      require_max_output_level);
  if (compaction != nullptr) {
    compaction->SetRLDecisionAttribution(0, SnapshotEpoch(vstorage),
                                         parent_reason);
    RLCompactionTelemetry::Get().RecordCompactionScheduled(
        compaction->start_level(), false);
  }
  return compaction;
}

}  // namespace ROCKSDB_NAMESPACE
