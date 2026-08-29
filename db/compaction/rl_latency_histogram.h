#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

constexpr size_t kRLLatencyBucketCount = 64;
using RLLatencyHistogram = std::array<uint64_t, kRLLatencyBucketCount>;

inline uint64_t RLLatencyBucketUpperBound(size_t bucket) {
  return bucket >= kRLLatencyBucketCount - 1
             ? std::numeric_limits<uint64_t>::max()
             : (1ULL << (bucket + 1)) - 1;
}

inline uint64_t RLLatencyHistogramCount(
    const RLLatencyHistogram& histogram) {
  uint64_t count = 0;
  for (uint64_t value : histogram) count += value;
  return count;
}

inline uint64_t RLLatencyP95(const RLLatencyHistogram& histogram) {
  const uint64_t count = RLLatencyHistogramCount(histogram);
  if (count == 0) return 0;
  const uint64_t rank = (count * 95 + 99) / 100;
  uint64_t seen = 0;
  for (size_t bucket = 0; bucket < histogram.size(); ++bucket) {
    seen += histogram[bucket];
    if (seen >= rank) return RLLatencyBucketUpperBound(bucket);
  }
  return 0;
}

}  // namespace ROCKSDB_NAMESPACE
