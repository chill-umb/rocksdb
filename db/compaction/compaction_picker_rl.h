#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "db/compaction/compaction_picker_level.h"
#include "db/compaction/rl_compaction_client.h"
#include "db/compaction/rl_compaction_telemetry.h"

namespace ROCKSDB_NAMESPACE {

class CompactionPressureView;
class RLControlHandle;
class RLSafetyController;
struct RLSLOBreachState;

// Trigger-only leveled picker. Each response atomically opens or closes a
// per-level trigger gate for one control interval. RocksDB's native leveled
// picker retains sole authority over which SSTs every compaction consumes.
class RLCompactionPicker : public LevelCompactionPicker {
 public:
  RLCompactionPicker(const ImmutableOptions& ioptions,
                     const InternalKeyComparator* icmp);
  ~RLCompactionPicker() override;

  // DBImpl attaches after the recovered/dynamically-created CF is visible and
  // has an active version. Construction alone never starts a worker.
  void AttachControl(
      uint32_t cf_id, uint64_t registration_generation,
      std::shared_ptr<RLControlHandle> control_handle,
      std::shared_ptr<CompactionPressureView> pressure_view,
      const MutableCFOptions& mutable_cf_options,
      const VersionStorageInfo* vstorage, uint64_t pcb);
  void DetachControl(uint64_t registration_generation);
  void StopWorker();

  // Called under DBImpl's mutex at structural change points. A rate-limited
  // change is coalesced by the DB-owned coordinator, never discarded.
  void OnStructuralChange(const VersionStorageInfo* vstorage,
                          uint64_t pcb) const;
  void RefreshDeferredSnapshot(const VersionStorageInfo* vstorage,
                               uint64_t pcb,
                               uint64_t requested_source_generation,
                               uint64_t registration_generation) const;
  bool ValidateSchedulingRequest(const VersionStorageInfo* vstorage,
                                 uint64_t eligibility_generation,
                                 uint64_t retry_generation) const;
  // The coordinator stamps the moment it hands a wake to RocksDB's scheduler,
  // so the picker can measure the remaining scheduler-admission delay when
  // PickCompaction is finally entered.
  void NotifyWakeDispatched(uint64_t dispatch_micros,
                            uint64_t queue_delay_micros) const;
  void UpdateTriggerOptions(const MutableCFOptions& mutable_cf_options);

  bool NeedsCompaction(const VersionStorageInfo* vstorage) const override;

  Compaction* PickCompaction(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options,
      const std::vector<SequenceNumber>& existing_snapshots,
      const SnapshotChecker* snapshot_checker, VersionStorageInfo* vstorage,
      LogBuffer* log_buffer, const std::string& full_history_ts_low,
      bool require_max_output_level = false) override;

 private:
  friend class RLCompactionPickerTestPeer;

  static constexpr int kMaxRLLevels = kRLTelemetryMaxLevels;

  enum class ActionReason : int {
    kPolicy = 0,
    kBudget = 1,
    kMaintenance = 2,
    kEmergency = 3,
    kFallback = 4,
    kDrain = 5,
    kSLO = 6,
    kManifest = 7,
    kStaleStructure = 8,
    // Configured control posture rather than a reaction to state: during
    // bridge validation L0 is held non-deferring so it matches native leveled
    // behavior while deeper-level control is evaluated.
    kPosture = 9,
  };

  enum class PolicyAction : int { kDefer = 0, kCompact = 1 };
  enum class PermitMode : int {
    kClosed = 0,
    kDueOpen = 1,
    kOptionalOpen = 2,
    kForcedOpen = 3,
  };

  struct LevelPermit {
    uint64_t decision_id = 0;
    uint64_t decision_generation = 0;
    uint64_t eligibility_generation = 0;
    uint64_t structural_generation = 0;
    uint64_t installed_micros = 0;
    PolicyAction action = PolicyAction::kDefer;
    PermitMode mode = PermitMode::kClosed;
    ActionReason reason = ActionReason::kPolicy;
    bool optional_token_available = false;
    bool transition_valid = true;
    // True when this permit's action was selected against a level observed
    // BELOW its trigger. Such a `defer` is not a decision to defer anything —
    // nothing was due — so binding it once the level crosses applies a permit
    // outside the state it was issued against. See crossing_posture_compact_.
    bool issued_below_threshold = false;
    int consecutive_blocked = 0;
    uint64_t backoff_until_micros = 0;
    uint64_t retry_generation = 0;
    // Start of the current eligibility interval, and the first successful
    // native schedule inside it. Together they separate "the gate opened" from
    // "the plant actually admitted work", which aggregate counters cannot.
    uint64_t eligibility_opened_micros = 0;
    uint64_t first_schedule_micros = 0;
  };

  struct SchedulingToken {
    uint64_t eligibility_generation = 0;
    uint64_t retry_generation = 0;
  };

  struct SafetyLevelEvaluation {
    bool observed = false;
    bool due = false;
    bool revoke_optional = false;
    bool force = false;
    ActionReason force_reason = ActionReason::kBudget;
  };

  struct SafetyFrameEvaluation {
    uint64_t now_micros = 0;
    bool guard_ready = false;
    bool dirty_deadline_miss = false;
    bool slo_force_due = false;
    bool prohibit_optional = false;
    bool would_invalidate_frame = false;
    uint64_t reason_mask = 0;
    SafetyLevelEvaluation levels[kMaxRLLevels];
  };

  struct RLStructuralLevel {
    int level = 0;
    int files = 0;
    uint64_t bytes = 0;
    uint64_t target_bytes = 0;
    int next_level_files = 0;
    uint64_t next_level_bytes = 0;
    uint64_t next_level_target_bytes = 0;
    uint64_t overlap_bytes = 0;
    bool is_last = false;
  };

  struct RLStructuralSnapshot {
    uint64_t snapshot_epoch = 0;
    uint64_t build_micros = 0;
    uint64_t built_generation = 0;
    uint64_t pending_compaction_bytes = 0;
    uint64_t physical_sst_bytes = 0;
    uint64_t live_logical_bytes = 0;
    int output_only_level_files = 0;
    uint64_t output_only_level_bytes = 0;
    uint64_t output_only_level_target_bytes = 0;
    int l0_delay_trigger_count = 0;
    std::vector<RLStructuralLevel> levels;
  };

  int decision_interval_ms_;
  int observe_interval_ms_;
  int blocked_attempt_limit_;
  int blocked_backoff_ms_;
  double optional_min_score_;
  bool oracle_mode_;
  bool safety_enabled_;
  // Review decision 6: no learned L0 deferral during bridge validation. L0 is
  // the level whose deferral most directly changes stall behavior, so holding
  // it at native semantics isolates deeper-level control while the bridge is
  // being validated. Set RL_L0_ALLOW_DEFER=1 to hand L0 to the policy.
  bool l0_allow_defer_;
  // D3a. When a level crosses its trigger between two decisions it carries the
  // previous frame's action, which is `defer` precisely because the level was
  // not yet due. RocksDB already wakes its scheduler on that crossing
  // (EnqueuePendingCompaction after ComputeCompactionScore), so the whole
  // observed trigger latency is this picker answering `false` to a
  // NeedsCompaction query that native leveled would have answered `true`.
  //
  // With the posture enabled, such a permit does not bind and the level is
  // admitted under kPosture — a fixed property of the environment, like
  // RL_L0_ALLOW_DEFER=0, rather than a reactive safety intervention, so the
  // transition stays valid for replay. Set RL_CROSSING_POSTURE=defer to
  // reproduce the binding behaviour for ablation.
  bool crossing_posture_compact_;
  // Measured admission-latency budget. `epsilon` in the repair plan is the sum
  // of score-event publication, worker tick, control-queue and scheduler
  // admission delay; a zero bound disables the violation counter but never
  // the measurement.
  uint64_t epsilon_bound_micros_;
  uint64_t safety_due_age_micros_;
  double safety_pressure_score_micros_;
  double safety_score_cap_;
  double safety_debt_ratio_cap_;
  uint64_t structural_dirty_deadline_micros_;
  std::unique_ptr<RLSafetyController> slo_safety_;
  std::string experiment_fingerprint_;
  std::string trigger_trace_path_;
  mutable std::ofstream trigger_trace_;
  std::string latency_window_log_path_;
  mutable std::ofstream latency_window_log_;
  std::string safety_shadow_log_path_;
  mutable std::ofstream safety_shadow_log_;

  // Immutable structural cache. It is built while DBImpl holds its mutex;
  // network I/O, overlay construction, and learning remain on worker_.
  mutable std::mutex snap_mu_;
  mutable std::condition_variable snap_cv_;
  mutable std::shared_ptr<const RLStructuralSnapshot> structural_snapshot_;
  mutable std::vector<int> last_sent_levels_;
  mutable uint64_t structural_source_generation_{0};
  mutable uint64_t structural_built_generation_{0};
  mutable uint64_t structural_dirty_since_micros_{0};
  mutable uint64_t last_structural_epoch_{0};
  mutable uint64_t last_build_micros_{0};
  mutable uint64_t registration_generation_{0};
  mutable uint32_t cf_id_{0};
  mutable std::shared_ptr<RLControlHandle> control_handle_;
  std::shared_ptr<CompactionPressureView> pressure_view_;
  std::thread worker_;
  std::atomic<bool> worker_stop_{false};
  std::atomic<bool> worker_started_{false};
  std::atomic<bool> attached_{false};

  // Whole response frames are installed atomically under permit_mu_. A due
  // open permit is held across multiple native jobs; an optional token is
  // consumed after one successful below-threshold schedule.
  mutable std::mutex permit_mu_;
  mutable LevelPermit permits_[kMaxRLLevels];
  mutable uint64_t decision_generation_{0};
  mutable uint64_t next_eligibility_generation_{0};
  mutable uint64_t next_retry_generation_{0};
  mutable uint64_t last_valid_response_micros_{0};
  mutable std::deque<uint64_t> response_spacings_micros_;
  mutable std::atomic<uint64_t> watchdog_expiries_{0};
  mutable std::atomic<uint64_t> blocked_attempts_{0};
  mutable std::atomic<uint64_t> blocked_windows_{0};
  mutable std::atomic<uint64_t> scheduling_wakes_{0};
  mutable std::atomic<uint64_t> slo_masked_windows_{0};
  mutable std::atomic<uint64_t> safety_would_override_windows_{0};
  mutable std::atomic<uint64_t> safety_applied_override_windows_{0};
  mutable std::atomic<uint64_t> dirty_deadline_misses_{0};
  mutable std::atomic<bool> dirty_deadline_active_{false};
  mutable std::atomic<bool> slo_read_breach_{false};
  mutable std::atomic<bool> slo_write_breach_{false};
  mutable std::atomic<bool> slo_space_breach_{false};
  mutable std::atomic<bool> slo_manifest_invalid_{false};
  mutable std::atomic<uint64_t> pick_attempts_[kMaxRLLevels] = {};
  mutable std::atomic<uint64_t> pick_blocked_[kMaxRLLevels] = {};
  mutable std::atomic<uint64_t> pick_scheduled_[kMaxRLLevels] = {};
  // Cumulative per-level plant outcomes. Telemetry deltas are consumed once
  // per decision cycle, so these accumulate them for the observation.
  mutable std::atomic<uint64_t> jobs_completed_[kMaxRLLevels] = {};
  mutable std::atomic<uint64_t> trivial_move_jobs_[kMaxRLLevels] = {};
  mutable std::atomic<uint64_t> trivial_move_bytes_[kMaxRLLevels] = {};
  mutable std::atomic<uint64_t> first_schedule_latency_micros_[kMaxRLLevels] =
      {};

  // Continuous audit that every accepted active-score recomputation reached
  // the shared pressure observer. A production path that changes the active
  // score without publishing shows up here as a divergence between the
  // observer's held score and the score the scheduler is admitting against.
  mutable std::atomic<uint64_t> pressure_audit_attempts_{0};
  mutable std::atomic<uint64_t> pressure_checks_{0};
  mutable std::atomic<uint64_t> pressure_divergences_{0};
  mutable std::atomic<uint64_t> pressure_max_divergence_milli_{0};

  // D3a attribution: admissions granted because a below-threshold permit did
  // not bind across the due crossing.
  mutable std::atomic<uint64_t> posture_admissions_[kMaxRLLevels] = {};

  // D7 item 5. Event-time due->admission latency, in microseconds.
  //
  // The trace-derived figure in 09_evaluate_oracle_parity.py is reconstructed
  // from a log written once per worker tick, so it is quantised to the
  // observation interval and cannot express an acceptance threshold below one
  // tick. Both endpoints are available here at microsecond resolution: the
  // pressure observer stamps due_since_micros on the crossing, and this picker
  // knows when it first admits the level. Log-scale buckets keep the recording
  // path lock-free on a hot admission check.
  static constexpr int kLatencyBuckets = 32;
  mutable std::atomic<uint64_t> due_admission_buckets_[kLatencyBuckets] = {};
  mutable std::atomic<uint64_t> due_admission_count_{0};
  mutable std::atomic<uint64_t> due_admission_max_micros_{0};
  mutable std::atomic<uint64_t> due_never_admitted_{0};
  mutable std::atomic<uint64_t> recorded_due_since_[kMaxRLLevels] = {};
  mutable std::atomic<uint64_t> last_seen_due_since_[kMaxRLLevels] = {};

  // epsilon components, in microseconds.
  mutable std::atomic<uint64_t> tick_gap_max_micros_{0};
  mutable std::atomic<uint64_t> control_queue_max_micros_{0};
  mutable std::atomic<uint64_t> admission_max_micros_{0};
  mutable std::atomic<uint64_t> wake_dispatch_micros_{0};
  mutable std::atomic<uint64_t> epsilon_max_micros_{0};
  mutable std::atomic<uint64_t> epsilon_violations_{0};

  // Per-level outcome attribution crosses the worker/DB threads.
  mutable std::atomic<int> last_effective_action_[kMaxRLLevels] = {};
  mutable std::atomic<bool> last_action_overridden_[kMaxRLLevels] = {};
  mutable std::atomic<bool> compaction_picked_[kMaxRLLevels] = {};
  mutable std::atomic<uint64_t> last_decision_id_[kMaxRLLevels] = {};
  mutable std::atomic<uint64_t> last_snapshot_epoch_[kMaxRLLevels] = {};
  mutable std::atomic<int> last_scheduling_result_[kMaxRLLevels] = {};
  mutable std::atomic<int> last_override_reason_[kMaxRLLevels] = {};
  mutable std::atomic<bool> last_transition_valid_[kMaxRLLevels] = {};

  mutable std::atomic<bool> rl_available_{false};
  std::atomic<int> rl_l0_trigger_{4};
  std::atomic<int> rl_l0_slowdown_trigger_{20};
  std::atomic<int> rl_l0_stop_trigger_{36};

  mutable std::atomic<uint64_t> rl_fallback_count_{0};
  mutable std::atomic<uint64_t> rl_query_count_{0};
  mutable std::atomic<uint64_t> rl_actuation_count_{0};
  mutable std::atomic<uint64_t> rl_skipped_ticks_{0};
  mutable std::atomic<uint64_t> rl_bypass_count_{0};
  mutable std::atomic<bool> rl_fallback_logged_{false};
  mutable std::atomic<uint64_t> rl_nc_calls_{0};
  mutable std::atomic<uint64_t> rl_nc_nanos_{0};
  mutable std::atomic<uint64_t> rl_publish_calls_{0};
  mutable std::atomic<uint64_t> rl_publish_nanos_{0};
  mutable std::atomic<uint64_t> rl_publish_max_nanos_{0};

  static constexpr int kNumReadTickers = 9;
  uint64_t prev_read_tickers_[kNumReadTickers] = {};
  static constexpr uint64_t kDiagnosticsEveryQueries = 200;

  void WorkerLoop();
  void RunDecisionCycle(RLStateV2& state, bool actuate);
  void PopulateReadStats(RLStateV2& state);
  void LogDiagnostics(bool final) const;
  void SendDoneMessage();

  void InstallPolicyFrame(const RLStateV2& state,
                          const std::vector<RLAction>& actions,
                          uint64_t decision_id,
                          std::vector<SchedulingToken>* wake_tokens);
  void InstallFallbackFrame(const RLStateV2& state,
                            std::vector<SchedulingToken>* wake_tokens);
  bool IsPermitEligible(const LevelPermit& permit, double current_score,
                        uint64_t now_micros) const;
  // Promotes a permit whose `defer` was selected below the trigger once the
  // level has crossed it. Callers must hold permit_mu_. Returns true when the
  // permit was promoted by this call.
  bool ApplyCrossingPosture(int level, LevelPermit& permit,
                            double current_score, uint64_t now_micros) const;
  void RecordDueAdmission(int level, uint64_t due_since_micros,
                          uint64_t now_micros) const;
  void ObserveDueEpisodeTransitions() const;
  uint64_t DueAdmissionPercentileMicros(double fraction) const;
  std::vector<bool> BuildDueAllowedMask(
      const VersionStorageInfo* vstorage, uint64_t now_micros) const;
  void RequestScheduling(const SchedulingToken& token,
                         uint64_t not_before_micros) const;
  void RecordBlockedAttempt(int level, uint64_t now_micros,
                            uint64_t eligibility_generation,
                            uint64_t decision_generation) const;
  bool RecordSuccessfulSchedule(int level, bool optional,
                                uint64_t eligibility_generation,
                                uint64_t decision_generation) const;
  void ForceOpenLevel(const VersionStorageInfo* vstorage, int level,
                      ActionReason reason) const;
  uint64_t ResponseWatchdogMicros() const;
  // Normalized pending debt against the manifest limit, or the configured
  // bootstrap cap. A byte-valued cap cannot serve 1M and 50M workloads with
  // the same number, so there is no absolute byte threshold anywhere.
  bool DebtRatioBreach(uint64_t pending_compaction_bytes) const;
  void CheckPressureDivergence(const VersionStorageInfo* vstorage) const;
  void RecordEpsilonSample(uint64_t admission_micros) const;
  SafetyFrameEvaluation ClassifyWorkerSafety(
      const RLStateV2& state, const RLSLOBreachState& slo,
      bool dirty_deadline_miss, uint64_t now_micros) const;
  bool ApplyWorkerSafety(const RLStateV2& state,
                         const SafetyFrameEvaluation& evaluation);
  void EvaluateWorkerSafety(const RLStateV2& state, bool actuate);
  void TraceControlState(const RLStateV2& state) const;
  void TraceLatencyWindow(
      const RLCompactionTelemetrySnapshot& telemetry) const;
  void TraceSafetyShadow(const RLStateV2& state,
                         const SafetyFrameEvaluation& evaluation,
                         bool actuate, bool intervention_applied) const;
  bool StructuralDirtyDeadlineMiss(uint64_t now_micros) const;
  bool HasMaintenanceWork(const VersionStorageInfo* vstorage) const;

  std::shared_ptr<const RLStructuralSnapshot> BuildStructuralSnapshot(
      const VersionStorageInfo* vstorage, uint64_t pcb,
      uint64_t built_generation) const;
  void BuildObservation(const RLStructuralSnapshot& structural,
                        RLStateV2* state) const;
  void InstallStructuralSnapshot(
      std::shared_ptr<const RLStructuralSnapshot> snapshot) const;
  uint64_t SnapshotEpoch(const VersionStorageInfo* vstorage) const;
  double LevelScore(const VersionStorageInfo* vstorage, int level) const;
  uint64_t NextLevelOverlapBytes(const VersionStorageInfo* vstorage,
                                 int level) const;

};

}  // namespace ROCKSDB_NAMESPACE
