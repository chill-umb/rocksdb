//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork: per-level read counters (PATHWAYS D §4, Gate N0 item 3;
// implementation plan WP3 and §6.2).

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
  void OpenTree() {
    Options options;
    options.env = env_;
    options.create_if_missing = true;
    options.num_levels = kLevels;
    options.compaction_style = kCompactionStyleLevel;
    options.level_compaction_dynamic_level_bytes = false;
    options.disable_auto_compactions = true;
    options.compression = kNoCompression;
    options.write_buffer_size = 64 << 20;
    options.max_open_files = -1;
    options.statistics = CreateDBStatistics();
    BlockBasedTableOptions table_options;
    table_options.filter_policy.reset(NewBloomFilterPolicy(10));
    options.table_factory.reset(NewBlockBasedTableFactory(table_options));
    stats_ = options.statistics;
    RLReadCounters::Reset();
    DestroyAndReopen(options);
  }

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
    };
    for (const auto& [kind, ticker] : pairs) {
      uint64_t sum = 0;
      for (int level = 0; level < kLevels; ++level) {
        sum += At(counts, level, kind);
      }
      EXPECT_EQ(stats_->getTickerCount(ticker), sum)
          << "kind " << static_cast<int>(kind);
    }
  }

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
      for (int kind = 0; kind < RLReadCounters::kKinds; ++kind) {
        EXPECT_EQ(want, d[level][kind]) << "kind " << kind;
      }
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

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
