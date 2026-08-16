// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

struct RLStateV2;

struct RLSafetyLevelLimit {
  int level = 0;
  bool calibrated = false;
  uint64_t due_age_limit_micros = 0;
  double pressure_limit_score_micros = 0.0;
  double score_limit = 0.0;
};

struct RLSLOBreachState {
  bool manifest_invalid = false;
  bool read = false;
  bool write = false;
  bool space = false;
  bool simultaneous_read_write = false;
  uint64_t get_samples = 0;
  uint64_t scan_samples = 0;
  uint64_t write_samples = 0;
};

// Validates the workload-specific baseline manifest and applies its rolling
// latency/space classifiers. This class changes only trigger eligibility; it
// never identifies an SST or invokes the file picker.
class RLSafetyController {
 public:
  // No path means an explicitly uncalibrated bootstrap controller. A supplied
  // but invalid path is conservative failure and sets manifest_invalid.
  static std::unique_ptr<RLSafetyController> Load(
      const std::string& path, const std::string& expected_fingerprint,
      bool manifest_required, std::string* error);

  RLSLOBreachState Update(const RLStateV2& state);
  const RLSafetyLevelLimit* LevelLimit(int level) const;
  bool calibrated() const { return calibrated_; }
  bool invalid() const { return invalid_; }
  double pending_debt_ratio_limit() const {
    return pending_debt_ratio_limit_;
  }

 private:
  friend class RLCompactionPickerTestPeer;

  struct LatencySample {
    uint64_t count = 0;
    double sum_ns = 0.0;
    uint64_t p95_ns = 0;
  };
  struct Hysteresis {
    bool breached = false;
    int above = 0;
    int below = 0;
  };

  RLSafetyController() = default;
  bool Parse(const std::string& json, const std::string& fingerprint,
             std::string* error);
  bool UpdateLatency(std::deque<LatencySample>* samples,
                     Hysteresis* hysteresis, uint64_t count, double avg_ns,
                     uint64_t p95_ns, double avg_limit_ns,
                     uint64_t p95_limit_ns, uint64_t* sample_count);
  void UpdateHysteresis(bool classifiable, bool above, Hysteresis* state);

  bool calibrated_ = false;
  bool invalid_ = false;
  uint64_t minimum_samples_ = 299;
  size_t rolling_window_count_ = 20;
  int enter_windows_ = 3;
  int exit_windows_ = 3;
  uint64_t physical_sst_bytes_limit_ = 0;
  double pending_debt_ratio_limit_ = 0.50;
  double get_avg_limit_ns_ = 0.0;
  uint64_t get_p95_limit_ns_ = 0;
  double scan_avg_limit_ns_ = 0.0;
  uint64_t scan_p95_limit_ns_ = 0;
  double write_avg_limit_ns_ = 0.0;
  uint64_t write_p95_limit_ns_ = 0;
  std::vector<RLSafetyLevelLimit> level_limits_;
  std::deque<LatencySample> get_samples_;
  std::deque<LatencySample> scan_samples_;
  std::deque<LatencySample> write_samples_;
  Hysteresis get_hysteresis_;
  Hysteresis scan_hysteresis_;
  Hysteresis write_hysteresis_;
  Hysteresis space_hysteresis_;
};

}  // namespace ROCKSDB_NAMESPACE
