//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork: per-level read counters (PATHWAYS D §4, Gate N0 item 3;
// implementation plan WP3 and §6.2), and the reads' table reopens
// (PREREGISTRATION D-21).

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "db/db_test_util.h"
#include "db/rl_read_counters.h"
#include "port/stack_trace.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/statistics.h"
#include "rocksdb/table.h"
#include "test_util/sync_point.h"

namespace ROCKSDB_NAMESPACE {

class PerLevelReadCountersTest : public DBTestBase {
 public:
  PerLevelReadCountersTest()
      : DBTestBase("per_level_read_counters_test", /*env_do_fsync=*/false) {}

  static constexpr int kLevels = 4;
  using Counts =
      std::array<std::array<uint64_t, RLReadCounters::kKinds>, kLevels>;

  // Levels are filled by hand and never compacted on their own. Table
  // readers stay open, so a moved file keeps the level it was opened at.
  // With one_open_table, the table cache holds one table instead: a read of
  // any other table reopens it, and the version builder pins no reader (it
  // pins only while the cache is under a quarter full, and a quarter of one
  // is none). SanitizeOptions raises max_open_files to 20, so a sync point
  // sets it back to 11, a cache of 11 - 10 = 1 table (Debug builds only).
  void OpenTree(bool one_open_table = false) {
    if (one_open_table) {
      SyncPoint::GetInstance()->SetCallBack(
          "SanitizeOptions::AfterChangeMaxOpenFiles",
          [](void* arg) { *static_cast<int*>(arg) = 11; });
      SyncPoint::GetInstance()->EnableProcessing();
    }
    Options options;
    options.env = env_;
    options.create_if_missing = true;
    options.num_levels = kLevels;
    options.compaction_style = kCompactionStyleLevel;
    options.level_compaction_dynamic_level_bytes = false;
    options.disable_auto_compactions = true;
    options.compression = kNoCompression;
    options.write_buffer_size = 64 << 20;
    options.max_open_files = one_open_table ? 11 : -1;
    options.statistics = CreateDBStatistics();
    BlockBasedTableOptions table_options;
    table_options.filter_policy.reset(NewBloomFilterPolicy(10));
    options.table_factory.reset(NewBlockBasedTableFactory(table_options));
    stats_ = options.statistics;
    RLReadCounters::Reset();
    DestroyAndReopen(options);
    if (one_open_table) {
      SyncPoint::GetInstance()->DisableProcessing();
      SyncPoint::GetInstance()->ClearAllCallBacks();
      ASSERT_EQ(1u, dbfull()->TEST_table_cache()->GetCapacity());
    }
  }

  // Closes every table no reader holds, so the next read reopens it.
  void CloseTables() { dbfull()->TEST_table_cache()->EraseUnRefEntries(); }

  // One flushed file holding `keys` (value "v" + key), moved to `level`.
  // Fill deeper levels first: each move must find the levels it crosses
  // empty, so it is trivial.
  void FileAt(int level, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
      ASSERT_OK(Put(key, "v" + key));
    }
    ASSERT_OK(Flush());
    if (level > 0) {
      MoveFilesToLevel(level);
    }
  }

  // Every file spans a..z, so a Get for any key between probes every level
  // until it finds the key. `filler` keys make the Bloom filter large enough
  // to have its nominal false-positive rate.
  void BuildCoveringTree(int filler) {
    for (int level = kLevels - 1; level >= 0; --level) {
      std::vector<std::string> keys = {"a", "k" + std::to_string(level), "z"};
      for (int i = 0; i < filler; ++i) {
        char key[32];
        snprintf(key, sizeof(key), "f%d-%05d", level, i);
        keys.emplace_back(key);
      }
      FileAt(level, keys);
    }
    ASSERT_EQ("1,1,1,1", FilesPerLevel());
  }

  static Counts Read() {
    Counts counts{};
    for (int level = 0; level < kLevels; ++level) {
      for (int kind = 0; kind < RLReadCounters::kKinds; ++kind) {
        counts[level][kind] =
            RLReadCounters::Get(level, static_cast<RLReadCounter>(kind));
      }
    }
    return counts;
  }

  static Counts Delta(const Counts& before) {
    Counts after = Read();
    for (int level = 0; level < kLevels; ++level) {
      for (int kind = 0; kind < RLReadCounters::kKinds; ++kind) {
        after[level][kind] -= before[level][kind];
      }
    }
    return after;
  }

  static uint64_t At(const Counts& counts, int level, RLReadCounter kind) {
    return counts[level][static_cast<int>(kind)];
  }

  // Counters and tickers both start at zero in OpenTree, and the tree
  // above L3 is empty, so each sum must equal its ticker exactly.
  void ExpectSumsEqualTickers() {
    const Counts counts = Read();
    const std::pair<RLReadCounter, Tickers> pairs[] = {
        {RLReadCounter::kProbe, POINT_SST_PROBE},
        {RLReadCounter::kFilterPass, BLOOM_FILTER_FULL_POSITIVE},
        {RLReadCounter::kFilterHit, BLOOM_FILTER_FULL_TRUE_POSITIVE},
        {RLReadCounter::kSeek, SORTED_RUN_SEEK},
        {RLReadCounter::kReopenNanos, READ_TABLE_REOPEN_NANOS},
    };
    for (const auto& [kind, ticker] : pairs) {
      uint64_t sum = 0;
      for (int level = 0; level < kLevels; ++level) {
        sum += At(counts, level, kind);
      }
      EXPECT_EQ(stats_->getTickerCount(ticker), sum)
          << "kind " << static_cast<int>(kind);
    }
    // One ticker counts both kinds of reopen.
    uint64_t reopens = 0;
    for (int level = 0; level < kLevels; ++level) {
      reopens += At(counts, level, RLReadCounter::kGetReopen) +
                 At(counts, level, RLReadCounter::kIterReopen);
    }
    EXPECT_EQ(stats_->getTickerCount(READ_TABLE_REOPEN), reopens);
  }

  // The four read kinds, before the reopens.
  static constexpr RLReadCounter kReadKinds[] = {
      RLReadCounter::kProbe, RLReadCounter::kFilterPass,
      RLReadCounter::kFilterHit, RLReadCounter::kSeek};

  uint64_t FileNumberAt(int level) {
    std::vector<LiveFileMetaData> files;
    db_->GetLiveFilesMetaData(&files);
    uint64_t number = 0;
    for (const auto& file : files) {
      if (file.level == level) {
        EXPECT_EQ(0u, number) << "more than one file at L" << level;
        number = file.file_number;
      }
    }
    return number;
  }

  std::shared_ptr<Statistics> stats_;
};

TEST_F(PerLevelReadCountersTest, GetCountsAtEachCoveringLevelDownToTheHit) {
  OpenTree();
  BuildCoveringTree(/*filler=*/0);
  for (int j = 0; j < kLevels; ++j) {
    SCOPED_TRACE("key at L" + std::to_string(j));
    const std::string key = "k" + std::to_string(j);
    const Counts before = Read();
    ASSERT_EQ("v" + key, Get(key));
    const Counts d = Delta(before);
    for (int level = 0; level < kLevels; ++level) {
      SCOPED_TRACE("L" + std::to_string(level));
      if (level < j) {
        // Probed; the filter may pass (a false positive) but never hits.
        EXPECT_EQ(1u, At(d, level, RLReadCounter::kProbe));
        EXPECT_LE(At(d, level, RLReadCounter::kFilterPass), 1u);
        EXPECT_EQ(0u, At(d, level, RLReadCounter::kFilterHit));
      } else if (level == j) {
        EXPECT_EQ(1u, At(d, level, RLReadCounter::kProbe));
        EXPECT_EQ(1u, At(d, level, RLReadCounter::kFilterPass));
        EXPECT_EQ(1u, At(d, level, RLReadCounter::kFilterHit));
      } else {
        EXPECT_EQ(0u, At(d, level, RLReadCounter::kProbe));
        EXPECT_EQ(0u, At(d, level, RLReadCounter::kFilterPass));
        EXPECT_EQ(0u, At(d, level, RLReadCounter::kFilterHit));
      }
      EXPECT_EQ(0u, At(d, level, RLReadCounter::kSeek));
    }
  }
  ExpectSumsEqualTickers();
}

TEST_F(PerLevelReadCountersTest, FalsePositivesArePassesWithoutHits) {
  OpenTree();
  // ~3000 keys per file at 10 bits per key: about 1% false positives, so
  // 2000 absent keys give some at every level. The filter is deterministic.
  BuildCoveringTree(/*filler=*/3000);
  constexpr int kLookups = 2000;
  const Counts before = Read();
  for (int i = 0; i < kLookups; ++i) {
    char key[32];
    snprintf(key, sizeof(key), "m-absent-%05d", i);
    ASSERT_EQ("NOT_FOUND", Get(key));
  }
  const Counts d = Delta(before);
  uint64_t passes = 0;
  for (int level = 0; level < kLevels; ++level) {
    SCOPED_TRACE("L" + std::to_string(level));
    EXPECT_EQ(static_cast<uint64_t>(kLookups),
              At(d, level, RLReadCounter::kProbe));
    EXPECT_EQ(0u, At(d, level, RLReadCounter::kFilterHit));
    EXPECT_LT(At(d, level, RLReadCounter::kFilterPass),
              static_cast<uint64_t>(kLookups) / 10);
    passes += At(d, level, RLReadCounter::kFilterPass);
  }
  EXPECT_GT(passes, 0u);
  ExpectSumsEqualTickers();
}

TEST_F(PerLevelReadCountersTest, TrivialMoveCountsAtTheNewLevel) {
  OpenTree();
  // Flushed at L0, where its table reader is opened (max_open_files = -1),
  // then relinked to L1 and later L2. The reader keeps level 0 throughout.
  FileAt(1, {"a", "k", "z"});
  const uint64_t file_number = FileNumberAt(1);
  ASSERT_NE(0u, file_number);

  auto read_once = [&]() {
    const Counts before = Read();
    EXPECT_EQ("vk", Get("k"));
    std::unique_ptr<Iterator> it(db_->NewIterator(ReadOptions()));
    it->Seek("k");
    EXPECT_TRUE(it->Valid());
    EXPECT_OK(it->status());
    return Delta(before);
  };
  auto expect_only_at = [&](const Counts& d, int where) {
    for (int level = 0; level < kLevels; ++level) {
      SCOPED_TRACE("L" + std::to_string(level));
      const uint64_t want = level == where ? 1 : 0;
      for (RLReadCounter kind : kReadKinds) {
        EXPECT_EQ(want, At(d, level, kind))
            << "kind " << static_cast<int>(kind);
      }
      // Every table stays open (max_open_files -1): nothing is reopened.
      EXPECT_EQ(0u, At(d, level, RLReadCounter::kGetReopen));
      EXPECT_EQ(0u, At(d, level, RLReadCounter::kIterReopen));
      EXPECT_EQ(0u, At(d, level, RLReadCounter::kReopenNanos));
    }
  };

  expect_only_at(read_once(), 1);

  ASSERT_OK(dbfull()->TEST_CompactRange(1, nullptr, nullptr));
  ASSERT_EQ("0,0,1", FilesPerLevel());
  ASSERT_EQ(file_number, FileNumberAt(2));  // relinked, not rewritten
  expect_only_at(read_once(), 2);
  ExpectSumsEqualTickers();
}

TEST_F(PerLevelReadCountersTest, SeekCountsOncePerSortedRun) {
  OpenTree();
  FileAt(2, {"a", "z"});
  FileAt(1, {"b", "c"});
  FileAt(1, {"d", "e"});  // a second L1 file, so a scan crosses a boundary
  FileAt(0, {"m"});
  FileAt(0, {"n"});
  ASSERT_EQ("2,2,1", FilesPerLevel());

  std::unique_ptr<Iterator> it(db_->NewIterator(ReadOptions()));
  Counts before = Read();
  it->Seek("b");
  std::string last;
  for (; it->Valid() && it->key().ToString() <= "e"; it->Next()) {
    last = it->key().ToString();
  }
  ASSERT_OK(it->status());
  ASSERT_EQ("e", last);  // crossed from L1's first file into its second
  Counts d = Delta(before);
  // Each L0 file is its own run; L1's crossing adds nothing.
  EXPECT_EQ(2u, At(d, 0, RLReadCounter::kSeek));
  EXPECT_EQ(1u, At(d, 1, RLReadCounter::kSeek));
  EXPECT_EQ(1u, At(d, 2, RLReadCounter::kSeek));
  EXPECT_EQ(0u, At(d, 3, RLReadCounter::kSeek));

  // No L1 file at or after "x": L1 is not seeked. Every L0 file still is.
  before = Read();
  it->Seek("x");
  ASSERT_TRUE(it->Valid());
  ASSERT_EQ("z", it->key().ToString());
  d = Delta(before);
  EXPECT_EQ(2u, At(d, 0, RLReadCounter::kSeek));
  EXPECT_EQ(0u, At(d, 1, RLReadCounter::kSeek));
  EXPECT_EQ(1u, At(d, 2, RLReadCounter::kSeek));
  for (int level = 0; level < kLevels; ++level) {
    EXPECT_EQ(0u, At(d, level, RLReadCounter::kProbe));
  }
  it.reset();
  ExpectSumsEqualTickers();
}

// D-21: a Get that finds a table closed reopens it, counted and timed at the
// level being probed. Each probed level reopens once; levels below the hit
// are not probed.
TEST_F(PerLevelReadCountersTest, GetReopensAtEachProbedLevel) {
  OpenTree(/*one_open_table=*/true);
  BuildCoveringTree(/*filler=*/0);
  for (int j = 0; j < kLevels; ++j) {
    SCOPED_TRACE("key at L" + std::to_string(j));
    const std::string key = "k" + std::to_string(j);
    CloseTables();
    const Counts before = Read();
    const uint64_t opens_before = stats_->getTickerCount(NO_FILE_OPENS);
    ASSERT_EQ("v" + key, Get(key));
    const Counts d = Delta(before);
    uint64_t reopens = 0;
    for (int level = 0; level < kLevels; ++level) {
      SCOPED_TRACE("L" + std::to_string(level));
      const uint64_t want = level <= j ? 1 : 0;
      EXPECT_EQ(want, At(d, level, RLReadCounter::kGetReopen));
      EXPECT_EQ(0u, At(d, level, RLReadCounter::kIterReopen));
      if (want) {
        EXPECT_GT(At(d, level, RLReadCounter::kReopenNanos), 0u);
      } else {
        EXPECT_EQ(0u, At(d, level, RLReadCounter::kReopenNanos));
      }
      reopens += At(d, level, RLReadCounter::kGetReopen);
    }
    // These reads opened nothing but the tables they reopened.
    EXPECT_EQ(reopens, stats_->getTickerCount(NO_FILE_OPENS) - opens_before);
  }
  // A table that is still open is not reopened.
  CloseTables();
  ASSERT_EQ("vk0", Get("k0"));
  const Counts before = Read();
  ASSERT_EQ("vk0", Get("k0"));
  EXPECT_EQ(0u, At(Delta(before), 0, RLReadCounter::kGetReopen));
  ExpectSumsEqualTickers();
}

// D-21: a user iterator reopens each table it opens: every L0 file when it
// is built, and at each other level the file it is positioned in, again
// when a scan crosses into the level's next file (which seeks no new run).
TEST_F(PerLevelReadCountersTest, IteratorReopensEachTableItOpens) {
  OpenTree(/*one_open_table=*/true);
  FileAt(2, {"a", "z"});
  FileAt(1, {"b", "c"});
  FileAt(1, {"d", "e"});
  FileAt(0, {"m"});
  FileAt(0, {"n"});
  ASSERT_EQ("2,2,1", FilesPerLevel());

  CloseTables();
  const Counts before = Read();
  std::unique_ptr<Iterator> it(db_->NewIterator(ReadOptions()));
  it->Seek("b");
  std::string last;
  for (; it->Valid() && it->key().ToString() <= "e"; it->Next()) {
    last = it->key().ToString();
  }
  ASSERT_OK(it->status());
  ASSERT_EQ("e", last);
  const Counts d = Delta(before);
  EXPECT_EQ(2u, At(d, 0, RLReadCounter::kIterReopen));
  EXPECT_EQ(2u, At(d, 1, RLReadCounter::kIterReopen));
  EXPECT_EQ(1u, At(d, 2, RLReadCounter::kIterReopen));
  EXPECT_EQ(0u, At(d, 3, RLReadCounter::kIterReopen));
  EXPECT_EQ(1u, At(d, 1, RLReadCounter::kSeek));
  for (int level = 0; level < kLevels; ++level) {
    EXPECT_EQ(0u, At(d, level, RLReadCounter::kGetReopen));
    EXPECT_EQ(At(d, level, RLReadCounter::kIterReopen) > 0,
              At(d, level, RLReadCounter::kReopenNanos) > 0);
  }
  it.reset();
  ExpectSumsEqualTickers();
}

// D-21 (and D-20 §2d): opens by flushes and compactions are write costs, in
// c_w's job seconds, and are not the reads' reopens.
TEST_F(PerLevelReadCountersTest, JobOpensAreNotReadReopens) {
  OpenTree(/*one_open_table=*/true);
  const uint64_t opens_before = stats_->getTickerCount(NO_FILE_OPENS);
  FileAt(0, {"a", "m"});
  FileAt(0, {"b", "n"});
  CloseTables();
  ASSERT_OK(db_->CompactRange(CompactRangeOptions(), nullptr, nullptr));
  ASSERT_EQ("0,1", FilesPerLevel());
  // Each flush and the compaction opened their outputs, and the compaction
  // reopened its closed inputs.
  EXPECT_GE(stats_->getTickerCount(NO_FILE_OPENS) - opens_before, 5u);
  const Counts counts = Read();
  for (int level = 0; level < kLevels; ++level) {
    for (RLReadCounter kind :
         {RLReadCounter::kGetReopen, RLReadCounter::kIterReopen,
          RLReadCounter::kReopenNanos}) {
      EXPECT_EQ(0u, At(counts, level, kind))
          << "L" << level << " kind " << static_cast<int>(kind);
    }
  }
  EXPECT_EQ(0u, stats_->getTickerCount(READ_TABLE_REOPEN));
  EXPECT_EQ(0u, stats_->getTickerCount(READ_TABLE_REOPEN_NANOS));
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
