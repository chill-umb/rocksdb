// Copyright (c) Facebook, Inc. and its affiliates.

#include "db/compaction/rl_control_coordinator.h"

#include <algorithm>
#include <chrono>

#include "db/column_family.h"
#include "db/compaction/compaction_picker_rl.h"
#include "db/db_impl/db_impl.h"
#include "db/version_set.h"

namespace ROCKSDB_NAMESPACE {
namespace {

void UpdateMaximum(std::atomic<uint64_t>* maximum, uint64_t value) {
  uint64_t current = maximum->load(std::memory_order_relaxed);
  while (current < value &&
         !maximum->compare_exchange_weak(current, value,
                                         std::memory_order_relaxed)) {
  }
}

}  // namespace

RLControlCoordinator::RLControlCoordinator(DBImpl* db) : db_(db) {}

RLControlCoordinator::~RLControlCoordinator() { Stop(); }

uint64_t RLControlCoordinator::NowMicros() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

uint64_t RLControlCoordinator::RegisterColumnFamily(uint32_t cf_id) {
  db_->mutex()->AssertHeld();
  // Ordinary leveled DBs never pay for a dormant control thread.
  if (!executor_.joinable()) {
    executor_ = std::thread([this] { ExecutorLoop(); });
  }
  const uint64_t generation = ++next_registration_generation_;
  registrations_[cf_id] = generation;
  return generation;
}

void RLControlCoordinator::UnregisterColumnFamily(
    uint32_t cf_id, uint64_t registration_generation) {
  db_->mutex()->AssertHeld();
  auto registration = registrations_.find(cf_id);
  if (registration != registrations_.end() &&
      registration->second == registration_generation) {
    registrations_.erase(registration);
  }
  {
    std::lock_guard<std::mutex> lock(queue_mu_);
    refresh_requests_.erase({cf_id, registration_generation});
    scheduling_requests_.erase({cf_id, registration_generation});
  }
  queue_cv_.notify_all();
}

void RLControlCoordinator::RequestScheduling(
    uint32_t cf_id, uint64_t registration_generation,
    uint64_t eligibility_generation, uint64_t retry_generation,
    uint64_t not_before_micros) {
  if (closing_.load(std::memory_order_acquire)) return;
  const RegistrationKey key{cf_id, registration_generation};
  {
    std::lock_guard<std::mutex> lock(queue_mu_);
    if (closing_.load(std::memory_order_relaxed)) return;
    SchedulingRequest request;
    request.cf_id = cf_id;
    request.registration_generation = registration_generation;
    request.eligibility_generation = eligibility_generation;
    request.retry_generation = retry_generation;
    request.not_before_micros = not_before_micros;
    request.queued_micros = NowMicros();
    auto existing = scheduling_requests_.find(key);
    if (existing == scheduling_requests_.end()) {
      scheduling_requests_.emplace(key, request);
    } else if (eligibility_generation >
                   existing->second.eligibility_generation ||
               (eligibility_generation ==
                    existing->second.eligibility_generation &&
                retry_generation > existing->second.retry_generation)) {
      existing->second = request;
      coalesced_requests_.fetch_add(1, std::memory_order_relaxed);
    } else if (eligibility_generation ==
                   existing->second.eligibility_generation &&
               retry_generation == existing->second.retry_generation &&
               not_before_micros < existing->second.not_before_micros) {
      existing->second.not_before_micros = not_before_micros;
      coalesced_requests_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  queue_cv_.notify_all();
}

bool RLControlCoordinator::IsRegistered(
    uint32_t cf_id, uint64_t registration_generation) const {
  db_->mutex()->AssertHeld();
  auto registration = registrations_.find(cf_id);
  return registration != registrations_.end() &&
         registration->second == registration_generation;
}

uint64_t RLControlCoordinator::RegistrationGeneration(uint32_t cf_id) const {
  db_->mutex()->AssertHeld();
  auto registration = registrations_.find(cf_id);
  return registration == registrations_.end() ? 0 : registration->second;
}

void RLControlCoordinator::RequestSnapshotRefresh(
    uint32_t cf_id, uint64_t registration_generation,
    uint64_t source_generation, uint64_t not_before_micros) {
  if (closing_.load(std::memory_order_acquire)) return;
  const RegistrationKey key{cf_id, registration_generation};
  {
    std::lock_guard<std::mutex> lock(queue_mu_);
    if (closing_.load(std::memory_order_relaxed)) return;
    auto existing = refresh_requests_.find(key);
    if (existing == refresh_requests_.end()) {
      RefreshRequest request;
      request.cf_id = cf_id;
      request.registration_generation = registration_generation;
      request.source_generation = source_generation;
      request.not_before_micros = not_before_micros;
      request.queued_micros = NowMicros();
      refresh_requests_.emplace(key, request);
    } else {
      existing->second.source_generation =
          std::max(existing->second.source_generation, source_generation);
      // The limiter deadline is based on the last successful build. A newer
      // coalesced change does not push that deadline into the future.
      existing->second.not_before_micros =
          std::min(existing->second.not_before_micros, not_before_micros);
      coalesced_requests_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  queue_cv_.notify_all();
}

void RLControlCoordinator::Stop() {
  if (!closing_.exchange(true, std::memory_order_acq_rel)) {
    queue_cv_.notify_all();
  }
  if (executor_.joinable()) executor_.join();
  std::lock_guard<std::mutex> lock(queue_mu_);
  refresh_requests_.clear();
  scheduling_requests_.clear();
}

void RLControlCoordinator::ExecutorLoop() {
  while (!closing_.load(std::memory_order_acquire)) {
    RefreshRequest request;
    bool have_request = false;
    {
      std::unique_lock<std::mutex> lock(queue_mu_);
      while (!closing_.load(std::memory_order_relaxed)) {
        if (refresh_requests_.empty() && scheduling_requests_.empty()) {
          queue_cv_.wait(lock);
          continue;
        }
        auto next_refresh = std::min_element(
            refresh_requests_.begin(), refresh_requests_.end(),
            [](const auto& left, const auto& right) {
              return left.second.not_before_micros <
                     right.second.not_before_micros;
            });
        auto next_scheduling = std::min_element(
            scheduling_requests_.begin(), scheduling_requests_.end(),
            [](const auto& left, const auto& right) {
              return left.second.not_before_micros <
                     right.second.not_before_micros;
            });
        const bool use_refresh =
            next_scheduling == scheduling_requests_.end() ||
            (next_refresh != refresh_requests_.end() &&
             next_refresh->second.not_before_micros <=
                 next_scheduling->second.not_before_micros);
        const uint64_t deadline =
            use_refresh ? next_refresh->second.not_before_micros
                        : next_scheduling->second.not_before_micros;
        const uint64_t now = NowMicros();
        if (deadline > now) {
          queue_cv_.wait_for(
              lock, std::chrono::microseconds(deadline - now));
          continue;
        }
        if (use_refresh) {
          request = next_refresh->second;
          refresh_requests_.erase(next_refresh);
          have_request = true;
        } else {
          SchedulingRequest scheduling = next_scheduling->second;
          scheduling_requests_.erase(next_scheduling);
          lock.unlock();
          if (!closing_.load(std::memory_order_acquire)) {
            ProcessScheduling(scheduling);
          }
          lock.lock();
        }
        break;
      }
    }
    // queue_mu_ is deliberately released before DBImpl's mutex is acquired.
    if (have_request && !closing_.load(std::memory_order_acquire)) {
      ProcessRefresh(request);
    }
  }
}

void RLControlCoordinator::ProcessScheduling(
    const SchedulingRequest& request) {
  db_->mutex()->Lock();
  const uint64_t dispatch_micros = NowMicros();
  const uint64_t eligible_micros =
      std::max(request.queued_micros, request.not_before_micros);
  const uint64_t delay = dispatch_micros >= eligible_micros
                             ? dispatch_micros - eligible_micros
                             : 0;
  scheduling_dispatches_.fetch_add(1, std::memory_order_relaxed);
  scheduling_total_delay_micros_.fetch_add(delay, std::memory_order_relaxed);
  UpdateMaximum(&scheduling_max_delay_micros_, delay);
  if (closing_.load(std::memory_order_acquire) ||
      !IsRegistered(request.cf_id, request.registration_generation)) {
    db_->mutex()->Unlock();
    dropped_registrations_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  ColumnFamilyData* cfd = db_->GetVersionSet()
                              ->GetColumnFamilySet()
                              ->GetColumnFamily(request.cf_id);
  if (cfd == nullptr || cfd->IsDropped() ||
      cfd->ioptions().compaction_style != kCompactionStyleRL) {
    db_->mutex()->Unlock();
    dropped_registrations_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  auto* picker = static_cast<RLCompactionPicker*>(cfd->compaction_picker());
  if (picker->ValidateSchedulingRequest(
          cfd->current()->storage_info(), request.eligibility_generation,
          request.retry_generation)) {
    // Stamp the hand-off so the picker can attribute the remaining
    // scheduler-admission delay when PickCompaction is finally entered.
    picker->NotifyWakeDispatched(NowMicros(), delay);
    db_->EnqueuePendingCompaction(cfd);
    db_->MaybeScheduleFlushOrCompaction();
  }
  db_->mutex()->Unlock();
}

void RLControlCoordinator::ProcessRefresh(const RefreshRequest& request) {
  db_->mutex()->Lock();
  const uint64_t dispatch_micros = NowMicros();
  const uint64_t eligible_micros =
      std::max(request.queued_micros, request.not_before_micros);
  const uint64_t delay = dispatch_micros >= eligible_micros
                             ? dispatch_micros - eligible_micros
                             : 0;
  refresh_dispatches_.fetch_add(1, std::memory_order_relaxed);
  refresh_total_delay_micros_.fetch_add(delay, std::memory_order_relaxed);
  UpdateMaximum(&refresh_max_delay_micros_, delay);
  if (closing_.load(std::memory_order_acquire) ||
      !IsRegistered(request.cf_id, request.registration_generation)) {
    db_->mutex()->Unlock();
    dropped_registrations_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  ColumnFamilyData* cfd = db_->GetVersionSet()
                              ->GetColumnFamilySet()
                              ->GetColumnFamily(request.cf_id);
  if (cfd == nullptr || cfd->IsDropped() ||
      cfd->ioptions().compaction_style != kCompactionStyleRL) {
    db_->mutex()->Unlock();
    dropped_registrations_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  auto* picker = static_cast<RLCompactionPicker*>(cfd->compaction_picker());
  picker->RefreshDeferredSnapshot(
      cfd->current()->storage_info(),
      cfd->current()->storage_info()->estimated_compaction_needed_bytes(),
      request.source_generation, request.registration_generation);
  db_->mutex()->Unlock();
}

}  // namespace ROCKSDB_NAMESPACE
