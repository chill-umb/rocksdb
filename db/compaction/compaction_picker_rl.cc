// Copyright (c) Facebook, Inc. and its affiliates.

#include "db/compaction/compaction_picker_rl.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>

#include "db/compaction/compaction_pressure_observer.h"
#include "db/compaction/rl_control_coordinator.h"
#include "db/compaction/rl_safety_manifest.h"
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

double EnvDouble(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  return end != value ? parsed : fallback;
}

std::string EnvString(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
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

// Scores are doubles recomputed from the same integers on both sides, so any
// real divergence is far larger than floating-point noise.
constexpr double kPressureScoreTolerance = 1e-6;

void UpdateMaximum(std::atomic<uint64_t>* maximum, uint64_t value) {
  uint64_t current = maximum->load(std::memory_order_relaxed);
  while (current < value &&
         !maximum->compare_exchange_weak(current, value,
                                         std::memory_order_relaxed)) {
  }
}

}  // namespace

RLCompactionPicker::RLCompactionPicker(const ImmutableOptions& ioptions,
                                       const InternalKeyComparator* icmp)
    : LevelCompactionPicker(ioptions, icmp),
      decision_interval_ms_(
          std::max(1, EnvInt("RL_DECISION_INTERVAL_MS", 50))),
      // A protocol-v2 request both closes the preceding reward interval and
      // returns the next action. There is no observe-only message, so issuing
      // faster observations would either discard selected actions or discard
      // physical-cost deltas. Keep one worker observation per actuation.
      observe_interval_ms_(decision_interval_ms_),
      blocked_attempt_limit_(EnvInt("RL_BLOCKED_ATTEMPT_LIMIT", 4)),
      blocked_backoff_ms_(EnvInt("RL_BLOCKED_BACKOFF_MS", 100)),
      // One knob with the Python mask. The learner may only choose compact
      // below the native threshold where the bridge will actually admit it;
      // two different thresholds silently discard a band of chosen actions.
      optional_min_score_(EnvDouble("RL_OPTIONAL_MIN_SCORE", 0.10)),
      oracle_mode_(EnvBool("RL_TRIGGER_ORACLE", false)),
      safety_enabled_(EnvBool("RL_SAFETY_ENFORCEMENT", true)),
      l0_allow_defer_(EnvBool("RL_L0_ALLOW_DEFER", false)),
      crossing_posture_compact_(
          EnvString("RL_CROSSING_POSTURE") != "defer"),
      epsilon_bound_micros_(
          static_cast<uint64_t>(std::max(EnvInt("RL_EPSILON_BOUND_MS", 0), 0)) *
          1000),
      safety_due_age_micros_(
          static_cast<uint64_t>(EnvInt("RL_SAFETY_DUE_AGE_MS", 1000)) * 1000),
      safety_pressure_score_micros_(
          EnvDouble("RL_SAFETY_PRESSURE_SCORE_SECONDS", 0.25) * 1000000.0),
      safety_score_cap_(EnvDouble("RL_SAFETY_SCORE_CAP", 1.25)),
      safety_debt_ratio_cap_(EnvDouble("RL_SAFETY_DEBT_RATIO_CAP", 0.50)),
      structural_dirty_deadline_micros_(static_cast<uint64_t>(std::max(
          EnvInt("RL_STRUCTURAL_DIRTY_DEADLINE_MS", 250), 1)) * 1000) {
  const int requested_observe_interval =
      std::max(1, EnvInt("RL_OBSERVE_INTERVAL_MS", decision_interval_ms_));
  if (requested_observe_interval != decision_interval_ms_) {
    ROCKS_LOG_WARN(
        ioptions_.logger,
        "Protocol v2 has no observe-only exchange; using decision interval "
        "%d ms instead of RL_OBSERVE_INTERVAL_MS=%d",
        decision_interval_ms_, requested_observe_interval);
  }
  trigger_trace_path_ = EnvString("RL_TRIGGER_TRACE_PATH");
  if (!trigger_trace_path_.empty()) {
    trigger_trace_.open(trigger_trace_path_, std::ios::out | std::ios::app);
  }
  std::string manifest_error;
  slo_safety_ = RLSafetyController::Load(
      EnvString("RL_BASELINE_SLO_PATH"),
      EnvString("RL_EXPERIMENT_FINGERPRINT"),
      EnvBool("RL_REQUIRE_BASELINE_SLO", false), &manifest_error);
  if (slo_safety_->invalid()) {
    slo_manifest_invalid_.store(true, std::memory_order_relaxed);
    ROCKS_LOG_WARN(ioptions_.logger,
                   "RL baseline SLO manifest rejected; learned trigger will "
                   "use conservative all-due native eligibility: %s",
                   manifest_error.c_str());
  } else if (!slo_safety_->calibrated()) {
    ROCKS_LOG_WARN(ioptions_.logger,
                   "RL safety is using explicitly uncalibrated bootstrap "
                   "limits; do not report this as baseline-calibrated");
  }
  for (int level = 0; level < kMaxRLLevels; ++level) {
    last_transition_valid_[level].store(true, std::memory_order_relaxed);
  }
}

RLCompactionPicker::~RLCompactionPicker() {
  StopWorker();
}

void RLCompactionPicker::AttachControl(
    uint32_t cf_id, uint64_t registration_generation,
    std::shared_ptr<RLControlHandle> control_handle,
    std::shared_ptr<CompactionPressureView> pressure_view,
    const MutableCFOptions& mutable_cf_options,
    const VersionStorageInfo* vstorage, uint64_t pcb) {
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    if (attached_.load(std::memory_order_relaxed)) return;
    cf_id_ = cf_id;
    registration_generation_ = registration_generation;
    control_handle_ = std::move(control_handle);
    pressure_view_ = std::move(pressure_view);
    attached_.store(true, std::memory_order_release);
    worker_stop_.store(false, std::memory_order_release);
  }
  // The initial immutable views are installed before the network worker can
  // emit its first observation.
  UpdateTriggerOptions(mutable_cf_options);
  OnStructuralChange(vstorage, pcb);
  // Oracle mode is the native-trigger parity arm and never waits for a
  // server response. Make native eligibility authoritative from the first
  // scheduler pass instead of allowing an initial fallback interval.
  if (oracle_mode_) {
    std::shared_ptr<const RLStructuralSnapshot> structural;
    {
      std::lock_guard<std::mutex> lock(snap_mu_);
      structural = structural_snapshot_;
    }
    if (structural != nullptr) {
      RLStateV2 state;
      BuildObservation(*structural, &state);
      std::vector<RLAction> actions;
      actions.reserve(state.levels.size());
      for (const RLLevelState& level : state.levels) {
        actions.push_back(level.score >= 1.0 ? RLAction::kCompactNow
                                             : RLAction::kDoNothing);
      }
      const uint64_t query_id =
          rl_query_count_.fetch_add(1, std::memory_order_relaxed) + 1;
      rl_actuation_count_.fetch_add(1, std::memory_order_relaxed);
      std::vector<SchedulingToken> wake_tokens;
      InstallPolicyFrame(state, actions, query_id, &wake_tokens);
      for (const SchedulingToken& token : wake_tokens) {
        RequestScheduling(token, NowMicros());
      }
    }
    rl_available_.store(true, std::memory_order_release);
  }
  worker_started_.store(true, std::memory_order_release);
  worker_ = std::thread([this] { WorkerLoop(); });
}

void RLCompactionPicker::UpdateTriggerOptions(
    const MutableCFOptions& mutable_cf_options) {
  rl_l0_trigger_.store(mutable_cf_options.level0_file_num_compaction_trigger,
                       std::memory_order_relaxed);
  rl_l0_slowdown_trigger_.store(
      mutable_cf_options.level0_slowdown_writes_trigger,
      std::memory_order_relaxed);
  rl_l0_stop_trigger_.store(mutable_cf_options.level0_stop_writes_trigger,
                            std::memory_order_relaxed);
}

void RLCompactionPicker::DetachControl(uint64_t registration_generation) {
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    if (!attached_.load(std::memory_order_relaxed) ||
        registration_generation_ != registration_generation) {
      return;
    }
    attached_.store(false, std::memory_order_release);
    control_handle_.reset();
    pressure_view_.reset();
  }
  worker_stop_.store(true, std::memory_order_release);
  snap_cv_.notify_all();
}

void RLCompactionPicker::StopWorker() {
  worker_stop_.store(true, std::memory_order_release);
  snap_cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
    if (!oracle_mode_) SendDoneMessage();
  }
  worker_started_.store(false, std::memory_order_release);
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
    if (level > 0) {
      epoch = MixEpoch(epoch, vstorage->MaxBytesForLevel(level));
    }
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

std::shared_ptr<const RLCompactionPicker::RLStructuralSnapshot>
RLCompactionPicker::BuildStructuralSnapshot(
    const VersionStorageInfo* vstorage, uint64_t pcb,
    uint64_t built_generation) const {
  const uint64_t start = NowMicros();
  auto snapshot = std::make_shared<RLStructuralSnapshot>();
  snapshot->snapshot_epoch = SnapshotEpoch(vstorage);
  snapshot->build_micros = start;
  snapshot->built_generation = built_generation;
  snapshot->pending_compaction_bytes = pcb;
  snapshot->l0_delay_trigger_count = vstorage->l0_delay_trigger_count();
  for (int level = 0; level < vstorage->num_levels(); ++level) {
    snapshot->physical_sst_bytes += vstorage->NumLevelBytes(level);
  }
  snapshot->live_logical_bytes = vstorage->EstimateLiveDataSize();

  const int max_input_level =
      std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
  for (int level = 0; level <= max_input_level; ++level) {
    RLStructuralLevel ls;
    ls.level = level;
    ls.files = vstorage->NumLevelFiles(level);
    ls.bytes = vstorage->NumLevelBytes(level);
    ls.target_bytes = level > 0 ? vstorage->MaxBytesForLevel(level) : 0;
    ls.is_last = level + 1 >= vstorage->num_levels();
    if (!ls.is_last) {
      const int next = level + 1;
      ls.next_level_files = vstorage->NumLevelFiles(next);
      ls.next_level_bytes = vstorage->NumLevelBytes(next);
      ls.next_level_target_bytes = vstorage->MaxBytesForLevel(next);
      ls.overlap_bytes = NextLevelOverlapBytes(vstorage, level);
    }
    snapshot->levels.push_back(std::move(ls));
  }
  // The bottom level is observable tree state but not an action-bearing
  // source. Export it globally so whole-tree reward conservation sees output
  // growth without creating an unactuatable DQN head.
  const int output_only_level = vstorage->num_levels() - 1;
  if (output_only_level > max_input_level && output_only_level >= 0) {
    snapshot->output_only_level_files =
        vstorage->NumLevelFiles(output_only_level);
    snapshot->output_only_level_bytes =
        vstorage->NumLevelBytes(output_only_level);
    snapshot->output_only_level_target_bytes =
        vstorage->MaxBytesForLevel(output_only_level);
  }
  const uint64_t build_nanos = (NowMicros() - start) * 1000;
  rl_publish_calls_.fetch_add(1, std::memory_order_relaxed);
  rl_publish_nanos_.fetch_add(build_nanos, std::memory_order_relaxed);
  UpdateMaximum(&rl_publish_max_nanos_, build_nanos);
  return snapshot;
}

void RLCompactionPicker::BuildObservation(
    const RLStructuralSnapshot& structural, RLStateV2* state) const {
  const uint64_t now = NowMicros();
  state->protocol_version = 2;
  state->snapshot_epoch = structural.snapshot_epoch;
  state->pending_compaction_bytes = structural.pending_compaction_bytes;
  state->physical_sst_bytes = structural.physical_sst_bytes;
  state->live_logical_bytes = structural.live_logical_bytes;
  state->output_only_level_files = structural.output_only_level_files;
  state->output_only_level_bytes = structural.output_only_level_bytes;
  state->output_only_level_target_bytes =
      structural.output_only_level_target_bytes;
  state->l0_delay_trigger_count = structural.l0_delay_trigger_count;
  state->l0_compaction_trigger = rl_l0_trigger_.load(std::memory_order_relaxed);
  state->l0_slowdown_trigger =
      rl_l0_slowdown_trigger_.load(std::memory_order_relaxed);
  state->l0_stop_trigger = rl_l0_stop_trigger_.load(std::memory_order_relaxed);
  state->fallback_count = rl_fallback_count_.load(std::memory_order_relaxed);
  state->observation_micros = now;
  state->structural_snapshot_age_micros =
      now >= structural.build_micros ? now - structural.build_micros : 0;
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    state->structural_source_generation = structural_source_generation_;
    state->structural_built_generation = structural_built_generation_;
    state->structural_dirty_age_micros =
        structural_dirty_since_micros_ != 0 &&
                now >= structural_dirty_since_micros_
            ? now - structural_dirty_since_micros_
            : 0;
  }

  std::shared_ptr<CompactionPressureView> pressure_view;
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    pressure_view = pressure_view_;
  }
  std::shared_ptr<const CompactionPressureSnapshot> pressure;
  if (pressure_view != nullptr) pressure = pressure_view->Load();
  CompactionPressureSnapshot extended;
  if (pressure != nullptr) {
    extended = CompactionPressureObserver::ExtendTo(*pressure, now);
    state->score_event_generation = pressure->generation;
  }

  // A response is a frame, so its observation must also see one complete
  // permit frame. Locking separately for every level could otherwise combine
  // levels from two decision generations.
  LevelPermit permit_snapshot[kMaxRLLevels];
  {
    std::lock_guard<std::mutex> lock(permit_mu_);
    for (int level = 0; level < kMaxRLLevels; ++level) {
      permit_snapshot[level] = permits_[level];
    }
  }

  for (const RLStructuralLevel& source : structural.levels) {
    const int level = source.level;
    RLLevelState ls;
    ls.level = level;
    ls.files = source.files;
    ls.bytes = source.bytes;
    ls.target_bytes = source.target_bytes;
    ls.next_level_files = source.next_level_files;
    ls.next_level_bytes = source.next_level_bytes;
    ls.next_level_target_bytes = source.next_level_target_bytes;
    ls.overlap_bytes = source.overlap_bytes;
    ls.is_last = source.is_last;
    if (source.level >= 0 &&
        source.level < static_cast<int>(extended.levels.size())) {
      const CompactionPressureLevelState& clock =
          extended.levels[source.level];
      ls.score = clock.score_at_event;
      ls.pressure_score_micros = clock.pressure_at_event;
      ls.due_age_micros =
          clock.due_since_micros != 0 && now >= clock.due_since_micros
              ? now - clock.due_since_micros
              : 0;
      ls.score_event_generation = state->score_event_generation;
    }
    if (source.level + 1 < static_cast<int>(extended.levels.size())) {
      ls.next_level_score = extended.levels[source.level + 1].score_at_event;
    }
    ls.default_needed = ls.score >= 1.0;
    // Kept in protocol v2 for compatibility; held gates use wall-clock due
    // and pressure state rather than a decision-count deferral budget.
    ls.defer_count = 0;
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
    const LevelPermit& permit = permit_snapshot[level];
    ls.gate_open = permit.mode != PermitMode::kClosed;
    ls.gate_mode = static_cast<int>(permit.mode);
    ls.consecutive_blocked = permit.consecutive_blocked;
    ls.in_backoff = permit.backoff_until_micros > now;
    ls.jobs_attempted =
        pick_attempts_[level].load(std::memory_order_relaxed);
    ls.jobs_blocked = pick_blocked_[level].load(std::memory_order_relaxed);
    ls.jobs_scheduled = pick_scheduled_[level].load(std::memory_order_relaxed);
    ls.jobs_completed = jobs_completed_[level].load(std::memory_order_relaxed);
    ls.decision_to_first_schedule_micros =
        first_schedule_latency_micros_[level].load(std::memory_order_relaxed);
    ls.trivial_move_jobs =
        trivial_move_jobs_[level].load(std::memory_order_relaxed);
    ls.trivial_move_bytes =
        trivial_move_bytes_[level].load(std::memory_order_relaxed);

    state->levels.push_back(std::move(ls));
  }
}

void RLCompactionPicker::InstallStructuralSnapshot(
    std::shared_ptr<const RLStructuralSnapshot> snapshot) const {
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    if (!attached_.load(std::memory_order_relaxed) || snapshot == nullptr) {
      return;
    }
    // A build is accepted only for the newest source generation. DBImpl's
    // mutex normally serializes this, but the check documents and protects
    // the publication invariant.
    if (snapshot->built_generation != structural_source_generation_) return;
    structural_snapshot_ = std::move(snapshot);
    structural_built_generation_ = structural_source_generation_;
    last_build_micros_ = structural_snapshot_->build_micros;
    structural_dirty_since_micros_ = 0;
  }
  snap_cv_.notify_all();
}

void RLCompactionPicker::OnStructuralChange(
    const VersionStorageInfo* vstorage, uint64_t pcb) const {
  if (vstorage == nullptr ||
      !attached_.load(std::memory_order_acquire)) {
    return;
  }
  const uint64_t now = NowMicros();
  const uint64_t epoch = SnapshotEpoch(vstorage);
  const uint64_t min_gap =
      std::max<uint64_t>(1, static_cast<uint64_t>(observe_interval_ms_) *
                                1000 / 4);
  bool build_now = false;
  uint64_t generation = 0;
  uint64_t not_before = 0;
  uint32_t cf_id = 0;
  uint64_t registration_generation = 0;
  std::shared_ptr<RLControlHandle> handle;
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    if (!attached_.load(std::memory_order_relaxed)) return;
    if (structural_snapshot_ != nullptr && epoch == last_structural_epoch_) {
      return;
    }
    last_structural_epoch_ = epoch;
    generation = ++structural_source_generation_;
    if (structural_dirty_since_micros_ == 0) {
      structural_dirty_since_micros_ = now;
    }
    build_now = last_build_micros_ == 0 || now >= last_build_micros_ + min_gap;
    if (!build_now) {
      not_before = last_build_micros_ + min_gap;
      cf_id = cf_id_;
      registration_generation = registration_generation_;
      handle = control_handle_;
    }
  }
  if (build_now) {
    InstallStructuralSnapshot(
        BuildStructuralSnapshot(vstorage, pcb, generation));
  } else if (handle != nullptr) {
    handle->RequestSnapshotRefresh(cf_id, registration_generation, generation,
                                   not_before);
  }
}

void RLCompactionPicker::RefreshDeferredSnapshot(
    const VersionStorageInfo* vstorage, uint64_t pcb,
    uint64_t requested_source_generation,
    uint64_t registration_generation) const {
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    if (!attached_.load(std::memory_order_relaxed) ||
        registration_generation_ != registration_generation ||
        structural_built_generation_ >= requested_source_generation ||
        structural_built_generation_ == structural_source_generation_) {
      return;
    }
    generation = structural_source_generation_;
  }
  InstallStructuralSnapshot(
      BuildStructuralSnapshot(vstorage, pcb, generation));
}

bool RLCompactionPicker::IsPermitEligible(const LevelPermit& permit,
                                          double current_score,
                                          uint64_t now_micros) const {
  if (permit.mode == PermitMode::kClosed) return false;
  if (permit.backoff_until_micros > now_micros) return false;
  if (current_score >= 1.0) {
    return permit.action == PolicyAction::kCompact ||
           permit.mode == PermitMode::kForcedOpen;
  }
  return permit.action == PolicyAction::kCompact &&
         permit.optional_token_available;
}

bool RLCompactionPicker::ApplyCrossingPosture(int level, LevelPermit& permit,
                                              double current_score,
                                              uint64_t now_micros) const {
  // Callers hold permit_mu_.
  //
  // D3a. The permit deferred a level that was below its trigger when the frame
  // was installed, so it expressed no judgement about a due level at all. The
  // level has since crossed. Binding the stale answer is what makes trigger
  // latency track the decision interval: RocksDB has already woken its
  // scheduler on this crossing, and this picker would otherwise answer that
  // nothing is eligible until the next frame arrives.
  if (!crossing_posture_compact_) return false;
  if (current_score < 1.0) return false;
  if (!permit.issued_below_threshold) return false;
  if (permit.action == PolicyAction::kCompact ||
      permit.mode == PermitMode::kForcedOpen) {
    return false;
  }
  if (permit.mode == PermitMode::kDueOpen &&
      permit.reason == ActionReason::kPosture) {
    return false;  // already promoted for this crossing
  }
  permit.action = PolicyAction::kCompact;
  permit.mode = PermitMode::kDueOpen;
  permit.reason = ActionReason::kPosture;
  permit.optional_token_available = false;
  permit.eligibility_generation = ++next_eligibility_generation_;
  permit.consecutive_blocked = 0;
  permit.backoff_until_micros = 0;
  permit.retry_generation = 0;
  permit.eligibility_opened_micros = now_micros;
  permit.first_schedule_micros = 0;
  // Unlike ForceOpenLevel, the transition stays VALID for replay. This is a
  // fixed, declared property of the environment — the same treatment
  // RL_L0_ALLOW_DEFER=0 already receives — not a reactive safety intervention,
  // and the learner keys its sample on the executed action.
  permit.transition_valid = true;
  last_effective_action_[level].store(1, std::memory_order_relaxed);
  last_action_overridden_[level].store(true, std::memory_order_relaxed);
  last_override_reason_[level].store(static_cast<int>(ActionReason::kPosture),
                                     std::memory_order_relaxed);
  posture_admissions_[level].fetch_add(1, std::memory_order_relaxed);
  return true;
}

void RLCompactionPicker::RecordDueAdmission(int level,
                                            uint64_t due_since_micros,
                                            uint64_t now_micros) const {
  if (level < 0 || level >= kMaxRLLevels || due_since_micros == 0 ||
      now_micros < due_since_micros) {
    return;
  }
  // One sample per due episode. The episode's own start timestamp is its
  // identity, so a gate that stays open across many PickCompaction calls
  // contributes exactly once.
  uint64_t recorded =
      recorded_due_since_[level].load(std::memory_order_relaxed);
  if (recorded == due_since_micros) return;
  if (!recorded_due_since_[level].compare_exchange_strong(
          recorded, due_since_micros, std::memory_order_relaxed)) {
    return;
  }
  const uint64_t latency = now_micros - due_since_micros;
  int bucket = 0;
  uint64_t value = latency;
  while (value > 0 && bucket < kLatencyBuckets - 1) {
    value >>= 1;
    ++bucket;
  }
  due_admission_buckets_[bucket].fetch_add(1, std::memory_order_relaxed);
  due_admission_count_.fetch_add(1, std::memory_order_relaxed);
  UpdateMaximum(&due_admission_max_micros_, latency);
}

void RLCompactionPicker::ObserveDueEpisodeTransitions() const {
  if (pressure_view_ == nullptr) return;
  std::shared_ptr<const CompactionPressureSnapshot> snapshot =
      pressure_view_->Load();
  if (snapshot == nullptr) return;
  const int levels =
      std::min(static_cast<int>(snapshot->levels.size()), kMaxRLLevels);
  for (int level = 0; level < levels; ++level) {
    const uint64_t due_since = snapshot->levels[level].due_since_micros;
    const uint64_t previous =
        last_seen_due_since_[level].exchange(due_since,
                                            std::memory_order_relaxed);
    if (previous == 0 || previous == due_since) continue;
    // The previous due episode ended. If it never reached the admission
    // recorder, the controller held that level closed for its whole life.
    if (recorded_due_since_[level].load(std::memory_order_relaxed) !=
        previous) {
      due_never_admitted_.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

uint64_t RLCompactionPicker::DueAdmissionPercentileMicros(
    double fraction) const {
  const uint64_t total = due_admission_count_.load(std::memory_order_relaxed);
  if (total == 0) return 0;
  const uint64_t target =
      static_cast<uint64_t>(fraction * static_cast<double>(total));
  uint64_t cumulative = 0;
  for (int bucket = 0; bucket < kLatencyBuckets; ++bucket) {
    cumulative += due_admission_buckets_[bucket].load(std::memory_order_relaxed);
    if (cumulative > target) {
      // Upper bound of the bucket: never understates the latency, which is the
      // safe direction for a threshold that is supposed to catch a regression.
      return bucket == 0 ? 0 : ((1ULL << bucket) - 1);
    }
  }
  return due_admission_max_micros_.load(std::memory_order_relaxed);
}

void RLCompactionPicker::RequestScheduling(
    const SchedulingToken& token, uint64_t not_before_micros) const {
  std::shared_ptr<RLControlHandle> handle;
  uint32_t cf_id = 0;
  uint64_t registration = 0;
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    if (!attached_.load(std::memory_order_relaxed)) return;
    handle = control_handle_;
    cf_id = cf_id_;
    registration = registration_generation_;
  }
  if (handle == nullptr || token.eligibility_generation == 0) return;
  handle->RequestScheduling(cf_id, registration,
                            token.eligibility_generation,
                            token.retry_generation, not_before_micros);
  scheduling_wakes_.fetch_add(1, std::memory_order_relaxed);
}

void RLCompactionPicker::InstallPolicyFrame(
    const RLStateV2& state, const std::vector<RLAction>& actions,
    uint64_t decision_id, std::vector<SchedulingToken>* wake_tokens) {
  if (actions.size() != state.levels.size()) return;
  const uint64_t now = NowMicros();
  std::lock_guard<std::mutex> lock(permit_mu_);
  if (last_valid_response_micros_ != 0 &&
      now >= last_valid_response_micros_) {
    response_spacings_micros_.push_back(now - last_valid_response_micros_);
    if (response_spacings_micros_.size() > 31) {
      response_spacings_micros_.pop_front();
    }
  }
  last_valid_response_micros_ = now;
  const uint64_t frame_generation = ++decision_generation_;
  bool present[kMaxRLLevels] = {};
  for (size_t index = 0; index < state.levels.size(); ++index) {
    const RLLevelState& observed = state.levels[index];
    const int level = observed.level;
    if (level < 0 || level >= kMaxRLLevels) continue;
    present[level] = true;
    LevelPermit previous = permits_[level];
    LevelPermit next = previous;
    next.decision_id = decision_id;
    next.decision_generation = frame_generation;
    next.structural_generation = state.structural_built_generation;
    next.installed_micros = now;
    next.reason = ActionReason::kPolicy;
    next.transition_valid = true;
    next.action = actions[index] == RLAction::kCompactNow
                      ? PolicyAction::kCompact
                      : PolicyAction::kDefer;
    // Record the band this action was chosen against, so a `defer` selected
    // over a level that was not yet due can be told apart from a `defer` that
    // deliberately held a due level back. Only the latter is a deferral
    // decision; see ApplyCrossingPosture.
    next.issued_below_threshold = observed.score < 1.0;
    const bool safety_held =
        previous.mode == PermitMode::kForcedOpen && observed.score >= 1.0 &&
        (previous.reason == ActionReason::kBudget ||
         previous.reason == ActionReason::kEmergency ||
         previous.reason == ActionReason::kSLO ||
         previous.reason == ActionReason::kManifest ||
         previous.reason == ActionReason::kStaleStructure);
    if (safety_held) {
      // A policy frame updates attribution but cannot transiently close a
      // safety interval. EvaluateWorkerSafety clears the interval only after
      // its own exit condition is satisfied.
      next.mode = previous.mode;
      next.reason = previous.reason;
      next.action = PolicyAction::kCompact;
      next.optional_token_available = false;
      next.transition_valid = false;
    } else if (next.action == PolicyAction::kDefer) {
      next.mode = PermitMode::kClosed;
      next.optional_token_available = false;
    } else if (observed.score >= 1.0) {
      next.mode = PermitMode::kDueOpen;
      next.optional_token_available = false;
    } else if (!oracle_mode_ && observed.score >= optional_min_score_) {
      next.mode = PermitMode::kOptionalOpen;
      next.optional_token_available = true;
    } else {
      next.mode = PermitMode::kClosed;
      next.optional_token_available = false;
      next.action = PolicyAction::kDefer;
    }
    if (!l0_allow_defer_ && level == 0 && observed.score >= 1.0 &&
        next.mode == PermitMode::kClosed) {
      // Review decision 6: L0 stays non-deferring during bridge validation, so
      // it matches native leveled behavior on the axis that most directly
      // drives write stalls while deeper-level control is evaluated.
      //
      // The transition stays valid for replay. This is a fixed, known part of
      // the environment rather than an unpredictable safety intervention, and
      // the learner keys its sample on the executed action, so L0's sample is
      // correctly labelled `compact` rather than mislabelled or discarded.
      next.action = PolicyAction::kCompact;
      next.mode = PermitMode::kDueOpen;
      next.optional_token_available = false;
      next.reason = ActionReason::kPosture;
    }

    const bool was_open =
        previous.mode != PermitMode::kClosed &&
        (previous.mode != PermitMode::kOptionalOpen ||
         previous.optional_token_available);
    const bool is_open =
        next.mode != PermitMode::kClosed &&
        (next.mode != PermitMode::kOptionalOpen ||
         next.optional_token_available);
    const bool newly_granted_optional =
        next.mode == PermitMode::kOptionalOpen &&
        next.optional_token_available;
    if (was_open != is_open || newly_granted_optional) {
      next.eligibility_generation = ++next_eligibility_generation_;
      next.consecutive_blocked = 0;
      next.backoff_until_micros = 0;
      next.retry_generation = 0;
      next.eligibility_opened_micros = is_open ? now : 0;
      next.first_schedule_micros = 0;
      if (is_open) {
        wake_tokens->push_back(
            {next.eligibility_generation, next.retry_generation});
      }
    } else if (is_open) {
      // A repeated due-compact frame updates attribution without resetting a
      // blocked window or producing a duplicate eligibility edge. Optional
      // compact responses intentionally grant a fresh one-shot generation.
      next.eligibility_generation = previous.eligibility_generation;
      next.consecutive_blocked = previous.consecutive_blocked;
      next.backoff_until_micros = previous.backoff_until_micros;
      next.retry_generation = previous.retry_generation;
    }
    permits_[level] = next;
    const bool selected_compact =
        actions[index] == RLAction::kCompactNow;
    const bool effective_compact =
        next.action == PolicyAction::kCompact &&
        next.mode != PermitMode::kOptionalOpen;
    last_effective_action_[level].store(effective_compact ? 1 : 0,
                                        std::memory_order_relaxed);
    last_action_overridden_[level].store(
        next.reason != ActionReason::kPolicy ||
            effective_compact != selected_compact,
        std::memory_order_relaxed);
    compaction_picked_[level].store(false, std::memory_order_relaxed);
    last_scheduling_result_[level].store(0, std::memory_order_relaxed);
    last_override_reason_[level].store(static_cast<int>(next.reason),
                                       std::memory_order_relaxed);
    last_transition_valid_[level].store(next.transition_valid,
                                         std::memory_order_relaxed);
    last_decision_id_[level].store(decision_id, std::memory_order_relaxed);
    last_snapshot_epoch_[level].store(state.snapshot_epoch,
                                      std::memory_order_relaxed);
  }
  for (int level = 0; level < kMaxRLLevels; ++level) {
    if (present[level]) continue;
    LevelPermit& permit = permits_[level];
    if (permit.mode != PermitMode::kClosed) {
      permit = LevelPermit();
      permit.eligibility_generation = ++next_eligibility_generation_;
    }
  }
}

void RLCompactionPicker::InstallFallbackFrame(
    const RLStateV2& state, std::vector<SchedulingToken>* wake_tokens) {
  const uint64_t now = NowMicros();
  std::lock_guard<std::mutex> lock(permit_mu_);
  const uint64_t frame_generation = ++decision_generation_;
  for (const RLLevelState& observed : state.levels) {
    const int level = observed.level;
    if (level < 0 || level >= kMaxRLLevels) continue;
    LevelPermit previous = permits_[level];
    LevelPermit next;
    next.decision_generation = frame_generation;
    next.structural_generation = state.structural_built_generation;
    next.installed_micros = now;
    next.action = observed.score >= 1.0 ? PolicyAction::kCompact
                                        : PolicyAction::kDefer;
    next.mode = observed.score >= 1.0 ? PermitMode::kForcedOpen
                                      : PermitMode::kClosed;
    next.reason = ActionReason::kFallback;
    next.transition_valid = false;
    if (next.mode == PermitMode::kForcedOpen &&
        previous.mode == PermitMode::kForcedOpen &&
        previous.reason == ActionReason::kFallback) {
      next.eligibility_generation = previous.eligibility_generation;
      next.consecutive_blocked = previous.consecutive_blocked;
      next.backoff_until_micros = previous.backoff_until_micros;
      next.retry_generation = previous.retry_generation;
    } else {
      next.eligibility_generation = ++next_eligibility_generation_;
      next.eligibility_opened_micros =
          next.mode == PermitMode::kForcedOpen ? now : 0;
      next.first_schedule_micros = 0;
      if (next.mode == PermitMode::kForcedOpen) {
        wake_tokens->push_back(
            {next.eligibility_generation, next.retry_generation});
      }
    }
    permits_[level] = next;
    last_effective_action_[level].store(
        next.action == PolicyAction::kCompact ? 1 : 0,
        std::memory_order_relaxed);
    last_action_overridden_[level].store(true, std::memory_order_relaxed);
    compaction_picked_[level].store(false, std::memory_order_relaxed);
    last_scheduling_result_[level].store(0, std::memory_order_relaxed);
    last_override_reason_[level].store(static_cast<int>(ActionReason::kFallback),
                                       std::memory_order_relaxed);
    last_transition_valid_[level].store(false, std::memory_order_relaxed);
    last_decision_id_[level].store(0, std::memory_order_relaxed);
    last_snapshot_epoch_[level].store(state.snapshot_epoch,
                                      std::memory_order_relaxed);
  }
}

std::vector<bool> RLCompactionPicker::BuildDueAllowedMask(
    const VersionStorageInfo* vstorage, uint64_t now_micros) const {
  std::vector<bool> allowed(vstorage->num_levels(), false);
  std::shared_ptr<const CompactionPressureSnapshot> pressure =
      pressure_view_ == nullptr ? nullptr : pressure_view_->Load();
  {
    std::lock_guard<std::mutex> lock(permit_mu_);
    const int max_level = std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
    for (int level = 0; level <= max_level; ++level) {
      const double score = LevelScore(vstorage, level);
      ApplyCrossingPosture(level, permits_[level], score, now_micros);
      if (score >= 1.0 &&
          IsPermitEligible(permits_[level], score, now_micros)) {
        allowed[level] = true;
      }
    }
  }
  for (int level = 0; level < static_cast<int>(allowed.size()); ++level) {
    if (!allowed[level] || pressure == nullptr ||
        level >= static_cast<int>(pressure->levels.size())) {
      continue;
    }
    RecordDueAdmission(level, pressure->levels[level].due_since_micros,
                       now_micros);
  }
  return allowed;
}

bool RLCompactionPicker::ValidateSchedulingRequest(
    const VersionStorageInfo* vstorage, uint64_t eligibility_generation,
    uint64_t retry_generation) const {
  const uint64_t now = NowMicros();
  std::lock_guard<std::mutex> lock(permit_mu_);
  const int max_level = std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
  for (int level = 0; level <= max_level; ++level) {
    const LevelPermit& permit = permits_[level];
    if (permit.eligibility_generation != eligibility_generation) continue;
    if (retry_generation != 0 && permit.retry_generation != retry_generation) {
      continue;
    }
    if (IsPermitEligible(permit, LevelScore(vstorage, level), now)) return true;
  }
  return false;
}

void RLCompactionPicker::RecordBlockedAttempt(
    int level, uint64_t now_micros,
    uint64_t eligibility_generation, uint64_t decision_generation) const {
  SchedulingToken retry;
  uint64_t deadline = 0;
  {
    std::lock_guard<std::mutex> lock(permit_mu_);
    if (level < 0 || level >= kMaxRLLevels) return;
    // These are physical-attempt diagnostics and remain cumulative even when
    // the response frame changes while the native builder is running.
    pick_blocked_[level].fetch_add(1, std::memory_order_relaxed);
    blocked_attempts_.fetch_add(1, std::memory_order_relaxed);
    LevelPermit& permit = permits_[level];
    // The native attempt was admitted from an immutable frame copy. Do not
    // charge its outcome to a response that superseded that frame meanwhile.
    if (permit.mode == PermitMode::kClosed ||
        permit.eligibility_generation != eligibility_generation) {
      return;
    }
    if (permit.decision_generation == decision_generation) {
      last_scheduling_result_[level].store(2, std::memory_order_relaxed);
      compaction_picked_[level].store(false, std::memory_order_relaxed);
    }
    ++permit.consecutive_blocked;
    if (permit.consecutive_blocked < blocked_attempt_limit_) return;
    permit.consecutive_blocked = 0;
    permit.backoff_until_micros =
        now_micros + static_cast<uint64_t>(blocked_backoff_ms_) * 1000;
    permit.retry_generation = ++next_retry_generation_;
    retry = {permit.eligibility_generation, permit.retry_generation};
    deadline = permit.backoff_until_micros;
    blocked_windows_.fetch_add(1, std::memory_order_relaxed);
  }
  RequestScheduling(retry, deadline);
}

bool RLCompactionPicker::RecordSuccessfulSchedule(
    int level, bool optional, uint64_t eligibility_generation,
    uint64_t decision_generation) const {
  std::lock_guard<std::mutex> lock(permit_mu_);
  if (level < 0 || level >= kMaxRLLevels) return false;
  LevelPermit& permit = permits_[level];
  pick_scheduled_[level].fetch_add(1, std::memory_order_relaxed);
  if (permit.eligibility_generation != eligibility_generation) {
    return false;
  }
  permit.consecutive_blocked = 0;
  permit.backoff_until_micros = 0;
  if (permit.first_schedule_micros == 0) {
    const uint64_t now = NowMicros();
    permit.first_schedule_micros = now;
    if (permit.eligibility_opened_micros != 0 &&
        now >= permit.eligibility_opened_micros) {
      first_schedule_latency_micros_[level].store(
          now - permit.eligibility_opened_micros, std::memory_order_relaxed);
    }
  }
  if (optional) {
    permit.optional_token_available = false;
    permit.mode = PermitMode::kClosed;
    permit.eligibility_generation = ++next_eligibility_generation_;
    permit.eligibility_opened_micros = 0;
    permit.first_schedule_micros = 0;
  }
  if (permit.decision_generation == decision_generation) {
    last_effective_action_[level].store(1, std::memory_order_relaxed);
    last_action_overridden_[level].store(false, std::memory_order_relaxed);
    last_scheduling_result_[level].store(1, std::memory_order_relaxed);
    compaction_picked_[level].store(true, std::memory_order_relaxed);
  }
  return true;
}

void RLCompactionPicker::ForceOpenLevel(const VersionStorageInfo* vstorage,
                                        int level,
                                        ActionReason reason) const {
  if (level < 0 || level >= kMaxRLLevels ||
      LevelScore(vstorage, level) < 1.0) {
    return;
  }
  std::lock_guard<std::mutex> lock(permit_mu_);
  LevelPermit& permit = permits_[level];
  if (permit.mode != PermitMode::kForcedOpen || permit.reason != reason) {
    permit.eligibility_generation = ++next_eligibility_generation_;
    permit.consecutive_blocked = 0;
    permit.backoff_until_micros = 0;
    permit.retry_generation = 0;
    permit.eligibility_opened_micros = NowMicros();
    permit.first_schedule_micros = 0;
  }
  permit.action = PolicyAction::kCompact;
  permit.mode = PermitMode::kForcedOpen;
  permit.reason = reason;
  permit.optional_token_available = false;
  permit.transition_valid = false;
  permit.installed_micros = NowMicros();
  last_effective_action_[level].store(1, std::memory_order_relaxed);
  last_action_overridden_[level].store(true, std::memory_order_relaxed);
  last_override_reason_[level].store(static_cast<int>(reason),
                                     std::memory_order_relaxed);
  // The learner uses one cooperative whole-tree reward. A forced compaction
  // changes that reward for every head, so the complete response frame is
  // unattributable even when only one source level was forced.
  for (int index = 0; index < kMaxRLLevels; ++index) {
    last_transition_valid_[index].store(false, std::memory_order_relaxed);
  }
}

uint64_t RLCompactionPicker::ResponseWatchdogMicros() const {
  std::lock_guard<std::mutex> lock(permit_mu_);
  uint64_t median =
      static_cast<uint64_t>(decision_interval_ms_) * 1000;
  if (!response_spacings_micros_.empty()) {
    std::vector<uint64_t> samples(response_spacings_micros_.begin(),
                                  response_spacings_micros_.end());
    std::sort(samples.begin(), samples.end());
    median = samples[samples.size() / 2];
  }
  return std::max<uint64_t>(1000000, 8 * median);
}

bool RLCompactionPicker::DebtRatioBreach(
    uint64_t pending_compaction_bytes) const {
  uint64_t live_bytes = 0;
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    if (structural_snapshot_ != nullptr) {
      live_bytes = structural_snapshot_->live_logical_bytes;
    }
  }
  // With no live data there is no debt to normalize against, and an empty tree
  // cannot be in debt. Reporting a breach here would force gates open on every
  // freshly opened database.
  if (live_bytes == 0) return false;
  const double limit = slo_safety_ != nullptr && slo_safety_->calibrated()
                           ? slo_safety_->pending_debt_ratio_limit()
                           : safety_debt_ratio_cap_;
  if (limit <= 0.0) return false;
  return static_cast<double>(pending_compaction_bytes) /
             static_cast<double>(live_bytes) >=
         limit;
}

void RLCompactionPicker::CheckPressureDivergence(
    const VersionStorageInfo* vstorage) const {
  // The observer hook lives inside VersionStorageInfo::ComputeCompactionScore
  // and only an active version carries a non-null observer, so no production
  // call site can bypass it by construction. This is the runtime proof of that
  // claim: if any path ever changed an active score without publishing, the
  // held score would drift from the score admission is deciding against, and
  // the safety clocks would be integrating a score RocksDB no longer has.
  //
  // Sampled rather than exhaustive: this runs on RocksDB's scheduling path, and
  // a path that bypassed the observer would do so persistently, not once.
  static constexpr uint64_t kAuditStride = 16;
  if (pressure_audit_attempts_.fetch_add(1, std::memory_order_relaxed) %
          kAuditStride !=
      0) {
    return;
  }
  std::shared_ptr<CompactionPressureView> view;
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    view = pressure_view_;
  }
  if (view == nullptr) return;
  std::shared_ptr<const CompactionPressureSnapshot> pressure = view->Load();
  if (pressure == nullptr) return;
  const int max_level = std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
  double worst = 0.0;
  for (int level = 0; level <= max_level; ++level) {
    if (level >= static_cast<int>(pressure->levels.size())) break;
    worst = std::max(worst, std::fabs(pressure->levels[level].score_at_event -
                                      LevelScore(vstorage, level)));
  }
  pressure_checks_.fetch_add(1, std::memory_order_relaxed);
  if (worst <= kPressureScoreTolerance) return;
  pressure_divergences_.fetch_add(1, std::memory_order_relaxed);
  UpdateMaximum(&pressure_max_divergence_milli_,
                static_cast<uint64_t>(worst * 1000.0));
}

void RLCompactionPicker::NotifyWakeDispatched(
    uint64_t dispatch_micros, uint64_t queue_delay_micros) const {
  UpdateMaximum(&control_queue_max_micros_, queue_delay_micros);
  wake_dispatch_micros_.store(dispatch_micros, std::memory_order_release);
}

void RLCompactionPicker::RecordEpsilonSample(uint64_t admission_micros) const {
  UpdateMaximum(&admission_max_micros_, admission_micros);
  // epsilon = score-event publication + worker tick + control queue +
  // scheduler admission. Each term is measured; the exported bound is the sum
  // of the observed maxima, so `H_i + epsilon` is a testable claim rather than
  // an assumed constant.
  const uint64_t publication_micros =
      CompactionPressureObserver::MaxObserveMicros();
  const uint64_t tick_micros = tick_gap_max_micros_.load(
      std::memory_order_relaxed);
  const uint64_t queue_micros = control_queue_max_micros_.load(
      std::memory_order_relaxed);
  const uint64_t total = publication_micros + tick_micros + queue_micros +
                         admission_max_micros_.load(std::memory_order_relaxed);
  UpdateMaximum(&epsilon_max_micros_, total);
  if (epsilon_bound_micros_ != 0 && total > epsilon_bound_micros_) {
    epsilon_violations_.fetch_add(1, std::memory_order_relaxed);
  }
}

void RLCompactionPicker::EvaluateWorkerSafety(const RLStateV2& state) {
  std::vector<SchedulingToken> wake_tokens;
  bool tree_transition_overridden = false;
  const uint64_t now = NowMicros();
  const RLSLOBreachState slo = slo_safety_->Update(state);
  slo_read_breach_.store(slo.read, std::memory_order_relaxed);
  slo_write_breach_.store(slo.write, std::memory_order_relaxed);
  slo_space_breach_.store(slo.space, std::memory_order_relaxed);
  slo_manifest_invalid_.store(slo.manifest_invalid,
                              std::memory_order_relaxed);
  if (safety_enabled_ &&
      (slo.read || slo.write || slo.space || slo.manifest_invalid)) {
    slo_masked_windows_.fetch_add(1, std::memory_order_relaxed);
  }
  const bool dirty_deadline_miss = StructuralDirtyDeadlineMiss(now);

  if (!oracle_mode_ && rl_available_.load(std::memory_order_acquire)) {
    uint64_t last_response = 0;
    {
      std::lock_guard<std::mutex> lock(permit_mu_);
      last_response = last_valid_response_micros_;
    }
    if (last_response != 0 && now > last_response &&
        now - last_response > ResponseWatchdogMicros()) {
      rl_available_.store(false, std::memory_order_release);
      watchdog_expiries_.fetch_add(1, std::memory_order_relaxed);
      InstallFallbackFrame(state, &wake_tokens);
    }
  }

  if (safety_enabled_) {
    const bool global_debt_breach =
        state.live_logical_bytes > 0 &&
        DebtRatioBreach(state.pending_compaction_bytes);
    const bool slo_force_due =
        slo.manifest_invalid || slo.read || slo.space ||
        slo.simultaneous_read_write || dirty_deadline_miss;
    const bool prohibit_optional =
        slo.manifest_invalid || slo.write || dirty_deadline_miss;
    const bool l0_slowdown = !state.levels.empty() &&
        state.levels.front().level == 0 &&
        state.levels.front().files >=
            rl_l0_slowdown_trigger_.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(permit_mu_);
    for (const RLLevelState& observed : state.levels) {
      const int level = observed.level;
      if (level < 0 || level >= kMaxRLLevels) continue;
      LevelPermit& permit = permits_[level];
      const bool due = observed.score >= 1.0;
      uint64_t due_age_limit = safety_due_age_micros_;
      double pressure_limit = safety_pressure_score_micros_;
      double score_limit = safety_score_cap_;
      if (const RLSafetyLevelLimit* limit =
              slo_safety_->LevelLimit(level)) {
        due_age_limit = limit->due_age_limit_micros;
        pressure_limit = limit->pressure_limit_score_micros;
        score_limit = limit->score_limit;
      }
      const bool force =
          due &&
          (observed.due_age_micros >= due_age_limit ||
           observed.pressure_score_micros >= pressure_limit ||
           observed.score >= score_limit || global_debt_breach ||
           slo_force_due || l0_slowdown);
      if (prohibit_optional && permit.mode == PermitMode::kOptionalOpen) {
        permit = LevelPermit();
        permit.eligibility_generation = ++next_eligibility_generation_;
        last_transition_valid_[level].store(false,
                                             std::memory_order_relaxed);
        tree_transition_overridden = true;
      }
      if (force) {
        ActionReason force_reason = ActionReason::kBudget;
        if (slo.manifest_invalid) {
          force_reason = ActionReason::kManifest;
        } else if (dirty_deadline_miss) {
          force_reason = ActionReason::kStaleStructure;
        } else if (slo_force_due) {
          force_reason = ActionReason::kSLO;
        } else if (level == 0 &&
                   observed.files >= rl_l0_stop_trigger_.load(
                                         std::memory_order_relaxed)) {
          force_reason = ActionReason::kEmergency;
        }
        const bool new_force_interval =
            permit.mode != PermitMode::kForcedOpen ||
            permit.reason != force_reason;
        if (new_force_interval) {
          permit.eligibility_generation = ++next_eligibility_generation_;
          permit.consecutive_blocked = 0;
          permit.backoff_until_micros = 0;
          permit.retry_generation = 0;
          permit.eligibility_opened_micros = now;
          permit.first_schedule_micros = 0;
          wake_tokens.push_back(
              {permit.eligibility_generation, permit.retry_generation});
        }
        permit.action = PolicyAction::kCompact;
        permit.mode = PermitMode::kForcedOpen;
        permit.reason = force_reason;
        permit.optional_token_available = false;
        permit.transition_valid = false;
        permit.installed_micros = now;
        last_action_overridden_[level].store(true,
                                             std::memory_order_relaxed);
        last_override_reason_[level].store(static_cast<int>(permit.reason),
                                           std::memory_order_relaxed);
        last_transition_valid_[level].store(false,
                                             std::memory_order_relaxed);
        tree_transition_overridden = true;
      } else if (permit.mode == PermitMode::kForcedOpen &&
                  ((!due && (permit.reason == ActionReason::kBudget ||
                            permit.reason == ActionReason::kEmergency)) ||
                  (permit.reason == ActionReason::kSLO && !slo_force_due) ||
                  (permit.reason == ActionReason::kStaleStructure &&
                   !dirty_deadline_miss))) {
        permit = LevelPermit();
        permit.eligibility_generation = ++next_eligibility_generation_;
      }
    }
  }

  if (tree_transition_overridden) {
    for (int level = 0; level < kMaxRLLevels; ++level) {
      last_transition_valid_[level].store(false, std::memory_order_relaxed);
    }
  }

  // The control queue is called after permit_mu_ has been released.
  for (const SchedulingToken& token : wake_tokens) {
    RequestScheduling(token, now);
  }
  TraceControlState(state);
}

bool RLCompactionPicker::StructuralDirtyDeadlineMiss(
    uint64_t now_micros) const {
  bool missed = false;
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    missed = structural_source_generation_ != structural_built_generation_ &&
             structural_dirty_since_micros_ != 0 &&
             now_micros >= structural_dirty_since_micros_ &&
             now_micros - structural_dirty_since_micros_ >=
                 structural_dirty_deadline_micros_;
  }
  const bool was_active =
      dirty_deadline_active_.exchange(missed, std::memory_order_relaxed);
  if (missed && !was_active) {
    dirty_deadline_misses_.fetch_add(1, std::memory_order_relaxed);
  }
  return missed;
}

bool RLCompactionPicker::HasMaintenanceWork(
    const VersionStorageInfo* vstorage) const {
  return !vstorage->ExpiredTtlFiles().empty() ||
         !vstorage->FilesMarkedForPeriodicCompaction().empty() ||
         !vstorage->BottommostFilesMarkedForCompaction().empty() ||
         !vstorage->FilesMarkedForCompaction().empty() ||
         !vstorage->FilesMarkedForForcedBlobGC().empty();
}

void RLCompactionPicker::TraceControlState(const RLStateV2& state) const {
  if (!trigger_trace_) return;
  const uint64_t now = NowMicros();
  LevelPermit frame[kMaxRLLevels];
  {
    std::lock_guard<std::mutex> lock(permit_mu_);
    for (int level = 0; level < kMaxRLLevels; ++level) {
      frame[level] = permits_[level];
    }
  }
  trigger_trace_ << "{\"schema_version\":1,\"time_micros\":" << now
                 << ",\"snapshot_epoch\":" << state.snapshot_epoch
                 << ",\"structural_source_generation\":"
                 << state.structural_source_generation
                 << ",\"structural_built_generation\":"
                 << state.structural_built_generation
                 << ",\"pending_compaction_bytes\":"
                 << state.pending_compaction_bytes
                 << ",\"physical_sst_bytes\":" << state.physical_sst_bytes
                 << ",\"live_logical_bytes\":" << state.live_logical_bytes
                 << ",\"slo_read\":"
                 << (slo_read_breach_.load(std::memory_order_relaxed) ? "true"
                                                                      : "false")
                 << ",\"slo_write\":"
                 << (slo_write_breach_.load(std::memory_order_relaxed) ? "true"
                                                                       : "false")
                 << ",\"slo_space\":"
                 << (slo_space_breach_.load(std::memory_order_relaxed) ? "true"
                                                                       : "false")
                 << ",\"manifest_invalid\":"
                 << (slo_manifest_invalid_.load(std::memory_order_relaxed)
                         ? "true"
                         : "false")
                 << ",\"structural_dirty_deadline_miss\":"
                 << (dirty_deadline_active_.load(std::memory_order_relaxed)
                         ? "true"
                         : "false")
                 << ",\"levels\":[";
  bool first = true;
  for (const RLLevelState& observed : state.levels) {
    const int level = observed.level;
    if (level < 0 || level >= kMaxRLLevels) continue;
    if (!first) trigger_trace_ << ',';
    first = false;
    const LevelPermit& permit = frame[level];
    trigger_trace_ << "{\"level\":" << level << ",\"score\":"
                   << observed.score << ",\"due_age_micros\":"
                   << observed.due_age_micros
                   << ",\"pressure_score_micros\":"
                   << observed.pressure_score_micros
                   << ",\"selected_action\":"
                   << last_effective_action_[level].load(
                          std::memory_order_relaxed)
                   << ",\"gate_mode\":" << static_cast<int>(permit.mode)
                   << ",\"reason\":" << static_cast<int>(permit.reason)
                   << ",\"decision_id\":" << permit.decision_id
                   << ",\"decision_generation\":"
                   << permit.decision_generation
                   << ",\"eligibility_generation\":"
                   << permit.eligibility_generation
                   << ",\"transition_valid\":"
                   << (permit.transition_valid ? "true" : "false")
                   << ",\"in_backoff\":"
                   << (permit.backoff_until_micros > now ? "true" : "false")
                   << '}';
  }
  trigger_trace_ << "]}\n";
  trigger_trace_.flush();
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
    level.prev_completed_decision_generation =
        telemetry.last_completed_decision_generation[index];
    level.prev_completed_eligibility_generation =
        telemetry.last_completed_eligibility_generation[index];
    level.prev_completed_override_reason =
        telemetry.last_completed_override_reason[index];
    level.prev_completion_result = telemetry.last_completion_result[index];
    // Telemetry deltas are consumed once per cycle; accumulate the cumulative
    // plant outcomes the observation reports alongside the picker's own
    // cumulative attempt/blocked/scheduled counters.
    if (level.level >= 0 && level.level < kMaxRLLevels) {
      jobs_completed_[level.level].fetch_add(
          telemetry.compactions_from_level[index], std::memory_order_relaxed);
      trivial_move_jobs_[level.level].fetch_add(
          telemetry.trivial_moves_from_level[index],
          std::memory_order_relaxed);
      trivial_move_bytes_[level.level].fetch_add(
          telemetry.trivial_move_bytes_from_level[index],
          std::memory_order_relaxed);
    }
  }

  // Protocol v2 pins observation and actuation cadence together. Keep this
  // defensive guard so a future cadence change cannot query Python and then
  // silently discard the selected action, corrupting replay attribution.
  if (!actuate) return;

  RLMultiQueryResult result;
  if (oracle_mode_) {
    result.ok = true;
    for (const RLLevelState& level : state.levels) {
      result.actions.push_back(level.score >= 1.0 ? RLAction::kCompactNow
                                                  : RLAction::kDoNothing);
    }
  } else {
    result = RLCompactionClient::Get().QueryActions(state);
  }
  const uint64_t query_id =
      rl_query_count_.fetch_add(1, std::memory_order_relaxed) + 1;
  std::vector<SchedulingToken> wake_tokens;
  if (!result.ok) {
    rl_available_.store(false, std::memory_order_release);
    const uint64_t failures =
        rl_fallback_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (actuate) {
      InstallFallbackFrame(state, &wake_tokens);
      for (const SchedulingToken& token : wake_tokens) {
        RequestScheduling(token, NowMicros());
      }
    }
    if (!rl_fallback_logged_.exchange(true, std::memory_order_relaxed) ||
        failures % 200 == 0) {
      ROCKS_LOG_WARN(ioptions_.logger,
                     "RL response unavailable; enabled native due fallback "
                     "for all due levels (cumulative_fallbacks=%" PRIu64 ")",
                     failures);
    }
    return;
  }

  uint64_t current_source_generation = 0;
  {
    std::lock_guard<std::mutex> lock(snap_mu_);
    current_source_generation = structural_source_generation_;
  }
  if (state.structural_built_generation == 0 ||
      state.structural_built_generation != current_source_generation) {
    // Advice produced from a superseded structure cannot close safety gates or
    // grant optional work. Preserve the prior frame in full.
    for (const RLLevelState& level : state.levels) {
      if (level.level >= 0 && level.level < kMaxRLLevels) {
        last_transition_valid_[level.level].store(false,
                                                   std::memory_order_relaxed);
      }
    }
    return;
  }
  InstallPolicyFrame(state, result.actions, query_id, &wake_tokens);
  rl_actuation_count_.fetch_add(1, std::memory_order_relaxed);
  rl_fallback_logged_.store(false, std::memory_order_relaxed);
  // Publish availability only after the complete permit frame is visible.
  // A scheduler thread that sees `true` must never observe bootstrap-closed
  // gates in place of this valid response.
  rl_available_.store(true, std::memory_order_release);
  for (const SchedulingToken& token : wake_tokens) {
    RequestScheduling(token, NowMicros());
  }
}

void RLCompactionPicker::WorkerLoop() {
  auto next_tick = std::chrono::steady_clock::now();
  auto next_actuation = next_tick;
  uint64_t last_tick_micros = 0;
  const auto observe_interval = std::chrono::milliseconds(observe_interval_ms_);
  const auto actuation_interval =
      std::chrono::milliseconds(decision_interval_ms_);
  while (!worker_stop_.load(std::memory_order_acquire)) {
    next_tick += observe_interval;
    const auto now = std::chrono::steady_clock::now();
    if (next_tick < now) next_tick = now + observe_interval;
    std::shared_ptr<const RLStructuralSnapshot> structural;
    {
      std::unique_lock<std::mutex> lock(snap_mu_);
      snap_cv_.wait_until(lock, next_tick, [this] {
        return worker_stop_.load(std::memory_order_acquire);
      });
      if (worker_stop_.load(std::memory_order_acquire)) break;
      // Measured, not nominal: the worker tick is one epsilon term, and a tick
      // that slipped is exactly what would make a deadline miss its budget.
      const uint64_t tick_micros = NowMicros();
      if (last_tick_micros != 0 && tick_micros > last_tick_micros) {
        UpdateMaximum(&tick_gap_max_micros_, tick_micros - last_tick_micros);
      }
      last_tick_micros = tick_micros;
      structural = structural_snapshot_;
      if (structural == nullptr) {
        rl_skipped_ticks_.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
    }
    RLStateV2 state;
    BuildObservation(*structural, &state);
    if (state.levels.empty()) continue;
    std::vector<int> levels;
    for (const RLLevelState& level : state.levels)
      levels.push_back(level.level);
    const bool actuate = next_tick >= next_actuation;
    if (actuate) next_actuation = next_tick + actuation_interval;
    RunDecisionCycle(state, actuate);
    EvaluateWorkerSafety(state);
    // Close out due episodes that ended without ever being admitted, so the
    // admission-latency histogram has a visible denominator rather than
    // silently describing only the episodes that were served.
    ObserveDueEpisodeTransitions();
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
  uint64_t source_generation = 0;
  uint64_t built_generation = 0;
  uint64_t dirty_age_micros = 0;
  uint64_t snapshot_age_micros = 0;
  {
    const uint64_t now = NowMicros();
    std::lock_guard<std::mutex> lock(snap_mu_);
    source_generation = structural_source_generation_;
    built_generation = structural_built_generation_;
    if (structural_dirty_since_micros_ != 0 &&
        now >= structural_dirty_since_micros_) {
      dirty_age_micros = now - structural_dirty_since_micros_;
    }
    if (structural_snapshot_ != nullptr &&
        now >= structural_snapshot_->build_micros) {
      snapshot_age_micros = now - structural_snapshot_->build_micros;
    }
  }
  uint64_t posture_admissions_total = 0;
  for (int level = 0; level < kMaxRLLevels; ++level) {
    posture_admissions_total +=
        posture_admissions_[level].load(std::memory_order_relaxed);
  }
  ROCKS_LOG_INFO(ioptions_.logger,
                 "RL trigger diagnostics: protocol=2 queries=%" PRIu64
                 " actuations=%" PRIu64 " bypasses=%" PRIu64
                 " skipped_ticks=%" PRIu64 " cumulative_fallbacks=%" PRIu64
                 " available=%d nc_calls=%" PRIu64 " nc_total_ms=%" PRIu64
                 " publish_calls=%" PRIu64 " publish_total_ms=%" PRIu64
                 " publish_max_us=%" PRIu64
                 " source_generation=%" PRIu64 " built_generation=%" PRIu64
                 " dirty_age_us=%" PRIu64 " snapshot_age_us=%" PRIu64
                 " scheduling_wakes=%" PRIu64 " blocked_attempts=%" PRIu64
                 " blocked_windows=%" PRIu64 " watchdog_expiries=%" PRIu64
                 " slo_masked_windows=%" PRIu64
                 " dirty_deadline_misses=%" PRIu64
                 " dirty_deadline_active=%d slo_read=%d slo_write=%d"
                 " slo_space=%d manifest_invalid=%d"
                 " pressure_checks=%" PRIu64 " pressure_divergences=%" PRIu64
                 " pressure_max_divergence_milli=%" PRIu64
                 " epsilon_publish_us=%" PRIu64 " epsilon_tick_us=%" PRIu64
                 " epsilon_queue_us=%" PRIu64 " epsilon_admission_us=%" PRIu64
                 " epsilon_max_us=%" PRIu64 " epsilon_bound_us=%" PRIu64
                 " epsilon_violations=%" PRIu64
                 " posture_admissions=%" PRIu64
                 " due_to_admission_micros_p50=%" PRIu64
                 " due_to_admission_micros_p90=%" PRIu64
                 " due_to_admission_micros_max=%" PRIu64
                 " due_admission_samples=%" PRIu64
                 " due_never_admitted=%" PRIu64,
                 queries, rl_actuation_count_.load(std::memory_order_relaxed),
                 rl_bypass_count_.load(std::memory_order_relaxed),
                 rl_skipped_ticks_.load(std::memory_order_relaxed),
                 rl_fallback_count_.load(std::memory_order_relaxed),
                 rl_available_.load(std::memory_order_relaxed) ? 1 : 0,
                 rl_nc_calls_.load(std::memory_order_relaxed),
                 rl_nc_nanos_.load(std::memory_order_relaxed) / 1000000,
                 rl_publish_calls_.load(std::memory_order_relaxed),
                 rl_publish_nanos_.load(std::memory_order_relaxed) / 1000000,
                 rl_publish_max_nanos_.load(std::memory_order_relaxed) / 1000,
                 source_generation, built_generation, dirty_age_micros,
                 snapshot_age_micros,
                 scheduling_wakes_.load(std::memory_order_relaxed),
                 blocked_attempts_.load(std::memory_order_relaxed),
                 blocked_windows_.load(std::memory_order_relaxed),
                 watchdog_expiries_.load(std::memory_order_relaxed),
                 slo_masked_windows_.load(std::memory_order_relaxed),
                 dirty_deadline_misses_.load(std::memory_order_relaxed),
                 dirty_deadline_active_.load(std::memory_order_relaxed) ? 1 : 0,
                 slo_read_breach_.load(std::memory_order_relaxed) ? 1 : 0,
                 slo_write_breach_.load(std::memory_order_relaxed) ? 1 : 0,
                 slo_space_breach_.load(std::memory_order_relaxed) ? 1 : 0,
                 slo_manifest_invalid_.load(std::memory_order_relaxed) ? 1 : 0,
                 pressure_checks_.load(std::memory_order_relaxed),
                 pressure_divergences_.load(std::memory_order_relaxed),
                 pressure_max_divergence_milli_.load(std::memory_order_relaxed),
                 CompactionPressureObserver::MaxObserveMicros(),
                 tick_gap_max_micros_.load(std::memory_order_relaxed),
                 control_queue_max_micros_.load(std::memory_order_relaxed),
                 admission_max_micros_.load(std::memory_order_relaxed),
                 epsilon_max_micros_.load(std::memory_order_relaxed),
                 epsilon_bound_micros_,
                 epsilon_violations_.load(std::memory_order_relaxed),
                 posture_admissions_total,
                 DueAdmissionPercentileMicros(0.50),
                 DueAdmissionPercentileMicros(0.90),
                 due_admission_max_micros_.load(std::memory_order_relaxed),
                 due_admission_count_.load(std::memory_order_relaxed),
                 due_never_admitted_.load(std::memory_order_relaxed));
  if (final) {
    for (int level = 0; level < kMaxRLLevels; ++level) {
      const uint64_t attempted =
          pick_attempts_[level].load(std::memory_order_relaxed);
      const uint64_t blocked =
          pick_blocked_[level].load(std::memory_order_relaxed);
      const uint64_t scheduled =
          pick_scheduled_[level].load(std::memory_order_relaxed);
      if (attempted == 0 && blocked == 0 && scheduled == 0) continue;
      ROCKS_LOG_INFO(
          ioptions_.logger,
          "RL trigger level diagnostics: level=%d attempted=%" PRIu64
          " blocked=%" PRIu64 " scheduled=%" PRIu64 " completed=%" PRIu64
          " trivial_moves=%" PRIu64 " trivial_move_bytes=%" PRIu64
          " first_schedule_latency_us=%" PRIu64,
          level, attempted, blocked, scheduled,
          jobs_completed_[level].load(std::memory_order_relaxed),
          trivial_move_jobs_[level].load(std::memory_order_relaxed),
          trivial_move_bytes_[level].load(std::memory_order_relaxed),
          first_schedule_latency_micros_[level].load(
              std::memory_order_relaxed));
    }
  }
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
    for (int level = 0; level < kMaxRLLevels; ++level) {
      last_transition_valid_[level].store(false, std::memory_order_relaxed);
    }
    return LevelCompactionPicker::NeedsCompaction(vstorage);
  }
  if (HasMaintenanceWork(vstorage)) {
    rl_bypass_count_.fetch_add(1, std::memory_order_relaxed);
    for (int level = 0; level < kMaxRLLevels; ++level) {
      last_transition_valid_[level].store(false, std::memory_order_relaxed);
    }
    return true;
  }

  const uint64_t pcb = vstorage->estimated_compaction_needed_bytes();
  OnStructuralChange(vstorage, pcb);
  CheckPressureDivergence(vstorage);
  const int l0_files = vstorage->NumLevelFiles(0);
  if (l0_files >=
          std::max(1, rl_l0_stop_trigger_.load(std::memory_order_relaxed)) ||
      DebtRatioBreach(pcb)) {
    // An L0 stop or hard debt cap can be caused by a blocked downstream
    // level. Open every independently due source and let native score order,
    // conflict checks, and L0/base-level starvation protection choose which
    // legal job makes progress first.
    const int max_level =
        std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
    for (int level = 0; level <= max_level; ++level) {
      if (LevelScore(vstorage, level) >= 1.0) {
        ForceOpenLevel(vstorage, level, ActionReason::kEmergency);
      }
    }
    rl_bypass_count_.fetch_add(1, std::memory_order_relaxed);
  }

  if (!oracle_mode_ && rl_available_.load(std::memory_order_acquire)) {
    uint64_t last_response = 0;
    {
      std::lock_guard<std::mutex> lock(permit_mu_);
      last_response = last_valid_response_micros_;
    }
    const uint64_t now = NowMicros();
    if (last_response != 0 && now > last_response &&
        now - last_response > ResponseWatchdogMicros()) {
      rl_available_.store(false, std::memory_order_release);
      watchdog_expiries_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  if (!rl_available_.load(std::memory_order_acquire)) {
    for (int level = 0; level < kMaxRLLevels; ++level) {
      last_transition_valid_[level].store(false, std::memory_order_relaxed);
    }
    return LevelCompactionPicker::NeedsCompaction(vstorage);
  }

  const uint64_t now = NowMicros();
  if (StructuralDirtyDeadlineMiss(now)) {
    const int max_level =
        std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
    for (int level = 0; level <= max_level; ++level) {
      if (LevelScore(vstorage, level) >= 1.0) {
        ForceOpenLevel(vstorage, level, ActionReason::kStaleStructure);
      }
    }
  }
  const std::vector<bool> allowed = BuildDueAllowedMask(vstorage, now);
  for (bool level_allowed : allowed) {
    if (level_allowed) return true;
  }
  std::lock_guard<std::mutex> lock(permit_mu_);
  const int max_level = std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
  for (int level = 0; level <= max_level; ++level) {
    const double score = LevelScore(vstorage, level);
    if (score < 1.0 && IsPermitEligible(permits_[level], score, now)) {
      return true;
    }
  }
  return false;
}

Compaction* RLCompactionPicker::PickCompaction(
    const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
    const MutableDBOptions& mutable_db_options,
    const std::vector<SequenceNumber>& existing_snapshots,
    const SnapshotChecker* snapshot_checker, VersionStorageInfo* vstorage,
    LogBuffer* log_buffer, const std::string& full_history_ts_low,
    bool require_max_output_level) {
  UpdateTriggerOptions(mutable_cf_options);

  int parent_reason = -1;
  if (RLDrainMode()) {
    parent_reason = static_cast<int>(ActionReason::kDrain);
  } else if (HasMaintenanceWork(vstorage)) {
    parent_reason = static_cast<int>(ActionReason::kMaintenance);
  } else if (!rl_available_.load(std::memory_order_acquire)) {
    parent_reason = static_cast<int>(ActionReason::kFallback);
  }
  if (parent_reason >= 0) {
    Compaction* compaction = LevelCompactionPicker::PickCompaction(
        cf_name, mutable_cf_options, mutable_db_options, existing_snapshots,
        snapshot_checker, vstorage, log_buffer, full_history_ts_low,
        require_max_output_level);
    if (compaction != nullptr) {
      compaction->SetRLDecisionAttribution(0, SnapshotEpoch(vstorage),
                                           parent_reason);
      const int level = std::max(
          0, std::min(compaction->start_level(), kMaxRLLevels - 1));
      last_transition_valid_[level].store(false, std::memory_order_relaxed);
      RLCompactionTelemetry::Get().RecordCompactionScheduled(
          compaction->start_level(), false);
    }
    return compaction;
  }

  const uint64_t now = NowMicros();
  // Close the admission-latency measurement for a coordinator wake: this call
  // is the moment the permit reached RocksDB's picker.
  const uint64_t dispatched =
      wake_dispatch_micros_.exchange(0, std::memory_order_acq_rel);
  if (dispatched != 0 && now >= dispatched) {
    RecordEpsilonSample(now - dispatched);
  }
  std::vector<bool> due_allowed(vstorage->num_levels(), false);
  LevelPermit admission[kMaxRLLevels];
  std::shared_ptr<const CompactionPressureSnapshot> pressure =
      pressure_view_ == nullptr ? nullptr : pressure_view_->Load();
  {
    // This is the linearization point for admission. A later response may
    // close the held gate, but it cannot retroactively cancel native work that
    // was already admitted under the previous frame.
    std::lock_guard<std::mutex> lock(permit_mu_);
    const int max_level =
        std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
    for (int level = 0; level <= max_level; ++level) {
      const double score = LevelScore(vstorage, level);
      ApplyCrossingPosture(level, permits_[level], score, now);
      admission[level] = permits_[level];
      due_allowed[level] =
          score >= 1.0 && IsPermitEligible(admission[level], score, now);
    }
  }
  for (int level = 0; level < static_cast<int>(due_allowed.size()); ++level) {
    if (!due_allowed[level] || pressure == nullptr ||
        level >= static_cast<int>(pressure->levels.size())) {
      continue;
    }
    RecordDueAdmission(level, pressure->levels[level].due_since_micros, now);
  }

  bool have_due = false;
  for (bool allowed : due_allowed) have_due = have_due || allowed;
  if (have_due) {
    std::vector<int> attempted_levels;
    Compaction* compaction = PickCompactionFromAllowedLevels(
        cf_name, mutable_cf_options, mutable_db_options, vstorage, log_buffer,
        full_history_ts_low, due_allowed, &attempted_levels);
    for (int attempted_level : attempted_levels) {
      if (attempted_level >= 0 && attempted_level < kMaxRLLevels) {
        pick_attempts_[attempted_level].fetch_add(1,
                                                  std::memory_order_relaxed);
      }
    }
    if (compaction != nullptr) {
      const int level = compaction->start_level();
      // Every higher-priority source actually visited before the successful
      // source was blocked. Charge only the immutable frame that admitted it.
      for (int attempted_level : attempted_levels) {
        if (attempted_level != level && attempted_level >= 0 &&
            attempted_level < kMaxRLLevels) {
          RecordBlockedAttempt(
              attempted_level, now,
              admission[attempted_level].eligibility_generation,
              admission[attempted_level].decision_generation);
        }
      }
      const LevelPermit& permit = admission[level];
      compaction->SetRLDecisionAttribution(
          permit.decision_id, SnapshotEpoch(vstorage),
          static_cast<int>(permit.reason), permit.decision_generation,
          permit.eligibility_generation);
      RecordSuccessfulSchedule(level, /*optional=*/false,
                               permit.eligibility_generation,
                               permit.decision_generation);
      RLCompactionTelemetry::Get().RecordCompactionScheduled(
          level, permit.reason == ActionReason::kPolicy);
      return compaction;
    }
    for (int attempted_level : attempted_levels) {
      if (attempted_level >= 0 && attempted_level < kMaxRLLevels) {
        RecordBlockedAttempt(
            attempted_level, now,
            admission[attempted_level].eligibility_generation,
            admission[attempted_level].decision_generation);
      }
    }
  }

  int optional_level = -1;
  double optional_score = -1.0;
  LevelPermit optional_permit;
  {
    std::lock_guard<std::mutex> lock(permit_mu_);
    const int max_level =
        std::min(vstorage->MaxInputLevel(), kMaxRLLevels - 1);
    for (int level = 0; level <= max_level; ++level) {
      const double score = LevelScore(vstorage, level);
      if (score < 1.0 && score > optional_score &&
          IsPermitEligible(permits_[level], score, now)) {
        optional_level = level;
        optional_score = score;
        optional_permit = permits_[level];
      }
    }
  }
  if (optional_level < 0) return nullptr;
  pick_attempts_[optional_level].fetch_add(1, std::memory_order_relaxed);
  const CompactionReason reason =
      optional_level == 0 ? CompactionReason::kLevelL0FilesNum
                          : CompactionReason::kLevelMaxLevelSize;
  Compaction* optional = PickCompactionFromLevel(
      cf_name, mutable_cf_options, mutable_db_options, vstorage, log_buffer,
      full_history_ts_low, optional_level, optional_score, reason);
  if (optional == nullptr) {
    RecordBlockedAttempt(optional_level, now,
                         optional_permit.eligibility_generation,
                         optional_permit.decision_generation);
    return nullptr;
  }
  optional->SetRLDecisionAttribution(
      optional_permit.decision_id, SnapshotEpoch(vstorage),
      static_cast<int>(optional_permit.reason),
      optional_permit.decision_generation,
      optional_permit.eligibility_generation);
  RecordSuccessfulSchedule(optional_level, /*optional=*/true,
                           optional_permit.eligibility_generation,
                           optional_permit.decision_generation);
  RLCompactionTelemetry::Get().RecordCompactionScheduled(
      optional_level, optional_permit.reason == ActionReason::kPolicy);
  return optional;
}

}  // namespace ROCKSDB_NAMESPACE
