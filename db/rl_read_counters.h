//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork (PATHWAYS D §4, Gate N0 item 3; plan WP3): per-level read
// counters. Each kind counts at the same site as a global ticker, keyed by
// the level of the version being read:
//
//   kProbe       POINT_SST_PROBE                  FilePicker chose a file
//   kFilterPass  BLOOM_FILTER_FULL_POSITIVE       the whole-key filter passed
//   kFilterHit   BLOOM_FILTER_FULL_TRUE_POSITIVE  the key was in that file
//   kSeek        SORTED_RUN_SEEK                  a user seek of one run
//   kGetReopen   READ_TABLE_REOPEN (with the next)  a Get reopened a table
//   kIterReopen  READ_TABLE_REOPEN                  a user iterator did
//   kReopenNanos READ_TABLE_REOPEN_NANOS            the time those took
//
// so on the Get and user-iterator paths the per-level sums equal those
// tickers (MultiGet is not counted per level). A level's false-positive
// block reads are kFilterPass - kFilterHit; the hit's own block read goes to
// the shared hit-read bucket, which no level is charged.
//
// A reopen (PREREGISTRATION D-21) is TableCache::FindTable opening a table
// it found closed, keyed by the level the read passed in: the table cache
// holds at most open_files - 10 tables. Its time runs from the open to the
// cache insert, which closes the table the insert evicts. Opens by flushes,
// compactions and other callers are not counted: each new file's verifying
// open is a write cost, in c_w's job seconds (D-20 §2d).
//
// The table reader's own level is not used: it is the level the file was
// first opened at, stale after a trivial move.
//
// Process-wide relaxed atomics, like RLCompactionTelemetry: db_bench runs one
// DB per process. Tests call Reset() between cases.

#pragma once

#include <atomic>
#include <cstdint>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

enum class RLReadCounter : int {
  kProbe = 0,
  kFilterPass,
  kFilterHit,
  kSeek,
  kGetReopen,
  kIterReopen,
  kReopenNanos,
  // Hidden internal entries a user iterator stepped over (PREREGISTRATION
  // D-23 §3(a), D-24 §2), keyed by the level where the hidden entry lives:
  // NUMBER_ITER_SKIP's entries, counted on forward steps by DBIter and added
  // when it is destroyed, as that ticker is. A memtable's entries are not
  // counted per level.
  kHiddenStep,
  kCount
};

class RLReadCounters {
 public:
  static constexpr int kMaxLevels = 16;
  static constexpr int kKinds = static_cast<int>(RLReadCounter::kCount);

  // A level outside [0, kMaxLevels) is not counted: -1 marks a table
  // iterator built outside a version (ingestion, sst_dump, verification).
  static void Add(int level, RLReadCounter kind, uint64_t amount = 1) {
    if (level < 0 || level >= kMaxLevels) {
      return;
    }
    counters_[level][static_cast<int>(kind)].fetch_add(
        amount, std::memory_order_relaxed);
  }

  static uint64_t Get(int level, RLReadCounter kind) {
    if (level < 0 || level >= kMaxLevels) {
      return 0;
    }
    return counters_[level][static_cast<int>(kind)].load(
        std::memory_order_relaxed);
  }

  static void Reset() {
    for (auto& level : counters_) {
      for (auto& counter : level) {
        counter.store(0, std::memory_order_relaxed);
      }
    }
  }

 private:
  // Defined in db/rl_controller_host.cc.
  static std::atomic<uint64_t> counters_[kMaxLevels][kKinds];
};

}  // namespace ROCKSDB_NAMESPACE
