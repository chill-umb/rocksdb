// Copyright (c) Facebook, Inc. and its affiliates.

#include "db/compaction/rl_safety_manifest.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <utility>

#include "db/compaction/rl_compaction_client.h"
#include "db/compaction/rl_compaction_telemetry.h"

namespace ROCKSDB_NAMESPACE {
namespace {

bool FindValueStart(const std::string& text, const std::string& key,
                    size_t* position) {
  const std::string needle = "\"" + key + "\"";
  size_t found = text.find(needle);
  if (found == std::string::npos) return false;
  found = text.find(':', found + needle.size());
  if (found == std::string::npos) return false;
  ++found;
  while (found < text.size() &&
         std::isspace(static_cast<unsigned char>(text[found]))) {
    ++found;
  }
  *position = found;
  return true;
}

bool IsValueTerminator(const std::string& text, const char* end) {
  const char* limit = text.c_str() + text.size();
  while (end < limit && std::isspace(static_cast<unsigned char>(*end))) {
    ++end;
  }
  return end == limit || *end == ',' || *end == '}' || *end == ']';
}

bool NumberField(const std::string& text, const std::string& key,
                 double* value) {
  size_t position = 0;
  if (!FindValueStart(text, key, &position)) return false;
  if (position >= text.size() ||
      (text[position] != '-' &&
       !std::isdigit(static_cast<unsigned char>(text[position])))) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(text.c_str() + position, &end);
  if (end == text.c_str() + position || errno == ERANGE ||
      !std::isfinite(parsed) || !IsValueTerminator(text, end)) {
    return false;
  }
  *value = parsed;
  return true;
}

bool UintField(const std::string& text, const std::string& key,
               uint64_t* value) {
  size_t position = 0;
  if (!FindValueStart(text, key, &position) || position >= text.size() ||
      !std::isdigit(static_cast<unsigned char>(text[position]))) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed =
      std::strtoull(text.c_str() + position, &end, 10);
  if (end == text.c_str() + position || errno == ERANGE ||
      !IsValueTerminator(text, end)) {
    return false;
  }
  *value = static_cast<uint64_t>(parsed);
  return true;
}

bool StringField(const std::string& text, const std::string& key,
                 std::string* value) {
  size_t position = 0;
  if (!FindValueStart(text, key, &position) || position >= text.size() ||
      text[position] != '"') {
    return false;
  }
  const size_t end = text.find('"', position + 1);
  if (end == std::string::npos ||
      !IsValueTerminator(text, text.c_str() + end + 1)) {
    return false;
  }
  *value = text.substr(position + 1, end - position - 1);
  return true;
}

bool BoolField(const std::string& text, const std::string& key, bool* value) {
  size_t position = 0;
  if (!FindValueStart(text, key, &position)) return false;
  if (text.compare(position, 4, "true") == 0) {
    if (!IsValueTerminator(text, text.c_str() + position + 4)) return false;
    *value = true;
    return true;
  }
  if (text.compare(position, 5, "false") == 0) {
    if (!IsValueTerminator(text, text.c_str() + position + 5)) return false;
    *value = false;
    return true;
  }
  return false;
}

std::vector<std::string> ArrayObjects(const std::string& text,
                                      const std::string& key) {
  std::vector<std::string> result;
  size_t position = 0;
  if (!FindValueStart(text, key, &position) || position >= text.size() ||
      text[position] != '[') {
    return result;
  }
  int depth = 0;
  size_t start = std::string::npos;
  for (++position; position < text.size(); ++position) {
    if (text[position] == '{') {
      if (depth++ == 0) start = position;
    } else if (text[position] == '}') {
      if (depth <= 0) break;
      if (--depth == 0 && start != std::string::npos) {
        result.push_back(text.substr(start, position - start + 1));
        start = std::string::npos;
      }
    } else if (text[position] == ']' && depth == 0) {
      break;
    }
  }
  return result;
}

}  // namespace

std::unique_ptr<RLSafetyController> RLSafetyController::Load(
    const std::string& path, const std::string& expected_fingerprint,
    bool manifest_required, std::string* error) {
  std::unique_ptr<RLSafetyController> controller(new RLSafetyController());
  if (path.empty()) {
    controller->invalid_ = manifest_required;
    if (manifest_required && error != nullptr) {
      *error = "RL_BASELINE_SLO_PATH is required but unset";
    }
    return controller;
  }
  std::ifstream input(path);
  if (!input) {
    controller->invalid_ = true;
    if (error != nullptr) *error = "cannot open baseline SLO manifest: " + path;
    return controller;
  }
  const std::string json((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  if (!controller->Parse(json, expected_fingerprint, error)) {
    controller->invalid_ = true;
    return controller;
  }
  if (manifest_required && !controller->calibrated_) {
    controller->invalid_ = true;
    if (error != nullptr) {
      *error = "baseline SLO live guard is not calibrated";
    }
  }
  return controller;
}

bool RLSafetyController::Parse(const std::string& json,
                               const std::string& fingerprint,
                               std::string* error) {
  uint64_t schema = 0;
  std::string definitions;
  std::string manifest_fingerprint;
  uint64_t rolling = 0;
  uint64_t enter = 0;
  uint64_t exit = 0;
  bool guard_calibrated = false;
  std::string p95_method;
  if (!UintField(json, "schema_version", &schema) || schema != 2 ||
      !StringField(json, "metric_definitions_version", &definitions) ||
      definitions != "trigger-v2-logical-v2" ||
      !StringField(json, "experiment_fingerprint", &manifest_fingerprint) ||
      fingerprint.empty() || manifest_fingerprint != fingerprint ||
      !BoolField(json, "guard_calibrated", &guard_calibrated) ||
      !UintField(json, "guard_minimum_samples", &minimum_samples_) ||
      !UintField(json, "guard_rolling_window_count", &rolling) ||
      !UintField(json, "guard_hysteresis_enter_windows", &enter) ||
      !UintField(json, "guard_hysteresis_exit_windows", &exit) ||
      !StringField(json, "guard_p95_method", &p95_method) ||
      p95_method != "merged_log2_histogram" ||
      !UintField(json, "allowed_physical_sst_bytes",
                 &physical_sst_bytes_limit_) ||
      !NumberField(json, "allowed_pending_debt_ratio",
                   &pending_debt_ratio_limit_) ||
      minimum_samples_ == 0 || rolling == 0 || enter == 0 || exit == 0 ||
      rolling > std::numeric_limits<size_t>::max() ||
      enter > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      exit > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      physical_sst_bytes_limit_ == 0 || pending_debt_ratio_limit_ <= 0.0) {
    if (error != nullptr) {
      *error = "baseline SLO schema, fingerprint, or required limits mismatch";
    }
    return false;
  }
  rolling_window_count_ = static_cast<size_t>(rolling);
  enter_windows_ = static_cast<int>(enter);
  exit_windows_ = static_cast<int>(exit);
  calibrated_ = guard_calibrated;
  if (calibrated_ &&
      (!NumberField(json, "guard_get_latency_avg_ns_limit",
                    &get_avg_limit_ns_) ||
       !UintField(json, "guard_get_latency_p95_ns_limit",
                  &get_p95_limit_ns_) ||
       !NumberField(json, "guard_scan_latency_avg_ns_limit",
                    &scan_avg_limit_ns_) ||
       !UintField(json, "guard_scan_latency_p95_ns_limit",
                  &scan_p95_limit_ns_) ||
       !NumberField(json, "guard_write_latency_avg_ns_limit",
                    &write_avg_limit_ns_) ||
       !UintField(json, "guard_write_latency_p95_ns_limit",
                  &write_p95_limit_ns_) ||
       get_avg_limit_ns_ <= 0.0 || get_p95_limit_ns_ == 0 ||
       scan_avg_limit_ns_ <= 0.0 || scan_p95_limit_ns_ == 0 ||
       write_avg_limit_ns_ <= 0.0 || write_p95_limit_ns_ == 0)) {
    if (error != nullptr) *error = "invalid calibrated live-guard limits";
    return false;
  }

  std::set<int> seen_levels;
  for (const std::string& object : ArrayObjects(json, "level_limits")) {
    uint64_t level = 0;
    bool calibrated = false;
    RLSafetyLevelLimit limit;
    if (!UintField(object, "level", &level) ||
        !BoolField(object, "calibrated", &calibrated) ||
        !UintField(object, "due_age_limit_micros",
                   &limit.due_age_limit_micros) ||
        !NumberField(object, "pressure_limit_score_micros",
                     &limit.pressure_limit_score_micros) ||
        !NumberField(object, "score_limit", &limit.score_limit) ||
        level >= static_cast<uint64_t>(kRLTelemetryMaxLevels) ||
        !seen_levels.insert(static_cast<int>(level)).second ||
        limit.due_age_limit_micros == 0 ||
        limit.pressure_limit_score_micros <= 0.0 || limit.score_limit < 1.0) {
      if (error != nullptr) *error = "invalid per-level safety limit";
      return false;
    }
    limit.level = static_cast<int>(level);
    limit.calibrated = calibrated;
    level_limits_.push_back(limit);
  }
  if (level_limits_.empty()) {
    if (error != nullptr) *error = "manifest has no level_limits";
    return false;
  }
  return true;
}

const RLSafetyLevelLimit* RLSafetyController::LevelLimit(int level) const {
  for (const RLSafetyLevelLimit& limit : level_limits_) {
    if (limit.level == level) return &limit;
  }
  return nullptr;
}

void RLSafetyController::UpdateHysteresis(bool classifiable, bool above,
                                          Hysteresis* state) {
  if (!classifiable) return;
  if (above) {
    state->below = 0;
    if (state->above < enter_windows_) ++state->above;
    if (state->above >= enter_windows_) state->breached = true;
  } else {
    state->above = 0;
    if (state->below < exit_windows_) ++state->below;
    if (state->below >= exit_windows_) state->breached = false;
  }
}

bool RLSafetyController::UpdateLatency(
    std::deque<LatencySample>* samples, Hysteresis* hysteresis,
    uint64_t count, double avg_ns, const RLLatencyHistogram& buckets,
    double avg_limit_ns, uint64_t p95_limit_ns, uint64_t* sample_count,
    bool* classifiable, double* rolling_average_ns,
    uint64_t* rolling_p95_ns) {
  samples->push_back({count, avg_ns * static_cast<double>(count), buckets});
  while (samples->size() > rolling_window_count_) samples->pop_front();
  uint64_t total_count = 0;
  double total_sum = 0.0;
  RLLatencyHistogram merged{};
  for (const LatencySample& sample : *samples) {
    total_count += sample.count;
    total_sum += sample.sum_ns;
    for (size_t bucket = 0; bucket < merged.size(); ++bucket) {
      merged[bucket] += sample.buckets[bucket];
    }
  }
  const uint64_t histogram_count = RLLatencyHistogramCount(merged);
  *sample_count = std::min(total_count, histogram_count);
  *classifiable = total_count >= minimum_samples_ &&
                  histogram_count >= minimum_samples_;
  const double average =
      total_count == 0 ? 0.0 : total_sum / static_cast<double>(total_count);
  const uint64_t p95 = RLLatencyP95(merged);
  *rolling_average_ns = average;
  *rolling_p95_ns = p95;
  UpdateHysteresis(*classifiable,
                   average > avg_limit_ns || p95 > p95_limit_ns,
                   hysteresis);
  return hysteresis->breached;
}

RLSLOBreachState RLSafetyController::Update(const RLStateV2& state) {
  RLSLOBreachState result;
  result.manifest_invalid = invalid_;
  if (!calibrated_) return result;
  const bool get = UpdateLatency(
      &get_samples_, &get_hysteresis_, state.get_latency_count,
      state.get_latency_avg_ns, state.get_latency_buckets, get_avg_limit_ns_,
      get_p95_limit_ns_, &result.get_samples, &result.get_classifiable,
      &result.get_average_ns, &result.get_p95_ns);
  const bool scan = UpdateLatency(
      &scan_samples_, &scan_hysteresis_, state.scan_latency_count,
      state.scan_latency_avg_ns, state.scan_latency_buckets, scan_avg_limit_ns_,
      scan_p95_limit_ns_, &result.scan_samples, &result.scan_classifiable,
      &result.scan_average_ns, &result.scan_p95_ns);
  result.write = UpdateLatency(
      &write_samples_, &write_hysteresis_, state.write_latency_count,
      state.write_latency_avg_ns, state.write_latency_buckets,
      write_avg_limit_ns_, write_p95_limit_ns_, &result.write_samples,
      &result.write_classifiable, &result.write_average_ns,
      &result.write_p95_ns);
  result.read = get || scan;
  result.guard_ready = result.get_classifiable && result.scan_classifiable &&
                       result.write_classifiable;
  const bool space_above = state.physical_sst_bytes > physical_sst_bytes_limit_;
  UpdateHysteresis(true, space_above, &space_hysteresis_);
  result.space = space_hysteresis_.breached;
  result.simultaneous_read_write = result.read && result.write;
  return result;
}

}  // namespace ROCKSDB_NAMESPACE
