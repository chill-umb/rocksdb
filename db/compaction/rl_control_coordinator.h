// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

class DBImpl;

// Narrow producer interface retained by a picker worker. Enqueue never takes
// DBImpl's mutex and never dereferences DB or column-family state.
class RLControlHandle {
 public:
  virtual ~RLControlHandle() = default;
  virtual void RequestSnapshotRefresh(uint32_t cf_id,
                                      uint64_t registration_generation,
                                      uint64_t source_generation,
                                      uint64_t not_before_micros) = 0;
  virtual void RequestScheduling(uint32_t cf_id,
                                 uint64_t registration_generation,
                                 uint64_t eligibility_generation,
                                 uint64_t retry_generation,
                                 uint64_t not_before_micros) = 0;
};

// DB-owned executor for coalesced, per-CF deferred structural refreshes.
// Phase 1b extends this same queue with scheduler wake requests.
class RLControlCoordinator final : public RLControlHandle {
 public:
  explicit RLControlCoordinator(DBImpl* db);
  ~RLControlCoordinator() override;

  // All registration methods require DBImpl's mutex.
  uint64_t RegisterColumnFamily(uint32_t cf_id);
  void UnregisterColumnFamily(uint32_t cf_id,
                              uint64_t registration_generation);
  bool IsRegistered(uint32_t cf_id,
                    uint64_t registration_generation) const;
  uint64_t RegistrationGeneration(uint32_t cf_id) const;

  void RequestSnapshotRefresh(uint32_t cf_id,
                              uint64_t registration_generation,
                              uint64_t source_generation,
                              uint64_t not_before_micros) override;
  void RequestScheduling(uint32_t cf_id, uint64_t registration_generation,
                         uint64_t eligibility_generation,
                         uint64_t retry_generation,
                         uint64_t not_before_micros) override;

  // Idempotent. Must be called without DBImpl's mutex because it joins the
  // executor, which may already be waiting to acquire that mutex.
  void Stop();

  uint64_t coalesced_requests() const {
    return coalesced_requests_.load(std::memory_order_relaxed);
  }
  uint64_t dropped_registrations() const {
    return dropped_registrations_.load(std::memory_order_relaxed);
  }
  uint64_t refresh_dispatches() const {
    return refresh_dispatches_.load(std::memory_order_relaxed);
  }
  uint64_t refresh_total_delay_micros() const {
    return refresh_total_delay_micros_.load(std::memory_order_relaxed);
  }
  uint64_t refresh_max_delay_micros() const {
    return refresh_max_delay_micros_.load(std::memory_order_relaxed);
  }
  uint64_t scheduling_dispatches() const {
    return scheduling_dispatches_.load(std::memory_order_relaxed);
  }
  uint64_t scheduling_total_delay_micros() const {
    return scheduling_total_delay_micros_.load(std::memory_order_relaxed);
  }
  uint64_t scheduling_max_delay_micros() const {
    return scheduling_max_delay_micros_.load(std::memory_order_relaxed);
  }

 private:
  struct RefreshRequest {
    uint32_t cf_id = 0;
    uint64_t registration_generation = 0;
    uint64_t source_generation = 0;
    uint64_t not_before_micros = 0;
    uint64_t queued_micros = 0;
  };

  struct SchedulingRequest {
    uint32_t cf_id = 0;
    uint64_t registration_generation = 0;
    uint64_t eligibility_generation = 0;
    uint64_t retry_generation = 0;
    uint64_t not_before_micros = 0;
    uint64_t queued_micros = 0;
  };

  using RegistrationKey = std::pair<uint32_t, uint64_t>;

  void ExecutorLoop();
  void ProcessRefresh(const RefreshRequest& request);
  void ProcessScheduling(const SchedulingRequest& request);
  static uint64_t NowMicros();

  DBImpl* const db_;
  mutable std::mutex queue_mu_;
  std::condition_variable queue_cv_;
  std::map<RegistrationKey, RefreshRequest> refresh_requests_;
  std::map<RegistrationKey, SchedulingRequest> scheduling_requests_;

  // Accessed only while DBImpl's mutex is held.
  std::map<uint32_t, uint64_t> registrations_;
  uint64_t next_registration_generation_ = 0;

  std::thread executor_;
  std::atomic<bool> closing_{false};
  std::atomic<uint64_t> coalesced_requests_{0};
  std::atomic<uint64_t> dropped_registrations_{0};
  std::atomic<uint64_t> refresh_dispatches_{0};
  std::atomic<uint64_t> refresh_total_delay_micros_{0};
  std::atomic<uint64_t> refresh_max_delay_micros_{0};
  std::atomic<uint64_t> scheduling_dispatches_{0};
  std::atomic<uint64_t> scheduling_total_delay_micros_{0};
  std::atomic<uint64_t> scheduling_max_delay_micros_{0};
};

}  // namespace ROCKSDB_NAMESPACE
