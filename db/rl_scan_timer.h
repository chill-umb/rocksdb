//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork (PREREGISTRATION D-23 §3(a), D-24 §2; the interim binary):
// the scan set-up timer. Times a user iterator's set-up (DBImpl::NewIterator,
// entry to return, on every path) or its teardown (DBIter's destructor),
// on the DB's clock, into a nanosecond ticker; the set-up also counts the
// iterators so created. PATHWAYS D §1's c^0_sc is measured from it; the
// fixed part inside the first Seek is the calibration's intercept.

#pragma once

#include <cstdint>

#include "monitoring/statistics_impl.h"
#include "rocksdb/statistics.h"
#include "rocksdb/system_clock.h"

namespace ROCKSDB_NAMESPACE {

class RLScanTimer {
 public:
  // `count` is TICKER_ENUM_MAX for a timer that counts nothing.
  RLScanTimer(Statistics* stats, SystemClock* clock, Tickers nanos,
              Tickers count = TICKER_ENUM_MAX)
      : stats_(clock == nullptr ? nullptr : stats),
        clock_(clock),
        nanos_(nanos),
        count_(count),
        start_(stats_ == nullptr ? 0 : clock_->NowNanos()) {}

  ~RLScanTimer() {
    if (stats_ == nullptr) {
      return;
    }
    RecordTick(stats_, nanos_, clock_->NowNanos() - start_);
    if (count_ != TICKER_ENUM_MAX) {
      RecordTick(stats_, count_, 1);
    }
  }

  RLScanTimer(const RLScanTimer&) = delete;
  RLScanTimer& operator=(const RLScanTimer&) = delete;

 private:
  Statistics* const stats_;
  SystemClock* const clock_;
  const Tickers nanos_;
  const Tickers count_;
  const uint64_t start_;
};

}  // namespace ROCKSDB_NAMESPACE
