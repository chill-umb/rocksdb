//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork: the level_target_multipliers option (PATHWAYS Pathway A,
// A-Impl-1 to A-Impl-4 and A-Impl-7; ACT-1; implementation plan §6.2).

#include <cstdlib>
#include <string>
#include <vector>

#include "db/column_family.h"
#include "db/db_test_util.h"
#include "db/version_set.h"
#include "port/stack_trace.h"
#include "rocksdb/convenience.h"
#include "rocksdb/utilities/options_util.h"
#include "util/cast_util.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

class LevelTargetMultipliersTest : public DBTestBase {
 public:
  LevelTargetMultipliersTest()
      : DBTestBase("level_target_multipliers_test", /*env_do_fsync=*/false) {}

  static constexpr uint64_t kBase = 1 << 20;  // L1's static target, 1 MiB

  Options BaseOptions() {
    Options options;
    options.env = env_;
    options.create_if_missing = true;
    options.num_levels = 4;
    options.compaction_style = kCompactionStyleLevel;
    options.level_compaction_dynamic_level_bytes = false;
    options.disable_auto_compactions = true;  // scores are still computed
    options.compression = kNoCompression;
    options.max_bytes_for_level_base = kBase;
    options.max_bytes_for_level_multiplier = 10;
    options.level0_file_num_compaction_trigger = 4;
    options.write_buffer_size = 64 << 20;  // flush only when asked
    return options;
  }

  VersionStorageInfo* Storage() {
    auto* cfh = static_cast_with_check<ColumnFamilyHandleImpl>(
        db_->DefaultColumnFamily());
    return cfh->cfd()->current()->storage_info();
  }

  // One flushed file of `keys` keys with `value_size`-byte values, moved to
  // `level`. Callers fill deeper levels first and use disjoint key ranges,
  // so each move is trivial and levels keep exactly what was put there.
  void WriteFileTo(int level, int first_key, int keys, int value_size) {
    Random rnd(301 + first_key);
    for (int i = 0; i < keys; ++i) {
      ASSERT_OK(Put(Key(first_key + i), rnd.RandomString(value_size)));
    }
    ASSERT_OK(Flush());
    if (level > 0) {
      MoveFilesToLevel(level);
    }
  }

  // A tree with L2 ~ 2 MiB, L1 ~ 1.5 MiB (above its 1 MiB target) and two
  // small L0 files.
  void BuildTree() {
    WriteFileTo(2, 0, 200, 10 << 10);
    WriteFileTo(1, 10000, 150, 10 << 10);
    WriteFileTo(0, 20000, 10, 1 << 10);
    WriteFileTo(0, 30000, 10, 1 << 10);
    const uint64_t l1 = Storage()->NumLevelBytes(1);
    ASSERT_GT(l1, kBase);
    ASSERT_LT(l1, 2 * kBase);
    ASSERT_EQ(2, NumTableFilesAtLevel(0));
  }

  double ScoreOf(int level) {
    VersionStorageInfo* vs = Storage();
    for (int i = 0; i <= vs->MaxInputLevel(); ++i) {
      if (vs->CompactionScoreLevel(i) == level) {
        return vs->CompactionScore(i);
      }
    }
    ADD_FAILURE() << "no score for level " << level;
    return -1;
  }

  uint64_t CompensatedBytes(int level) {
    uint64_t bytes = 0;
    for (const FileMetaData* f : Storage()->LevelFiles(level)) {
      bytes += f->compensated_file_size;
    }
    return bytes;
  }

  Status Set(const std::string& value) {
    return db_->SetOptions({{"level_target_multipliers", value}});
  }

  std::vector<double> Current() {
    return db_->GetOptions().level_target_multipliers;
  }
};

TEST_F(LevelTargetMultipliersTest, SetOptionsScalesTargetsAndScoresAtOnce) {
  Reopen(BaseOptions());
  BuildTree();
  VersionStorageInfo* before = Storage();
  const double l1_before = ScoreOf(1);
  ASSERT_GT(l1_before, 1.0);  // due at m = 1

  // No write follows: SetOptions itself must publish the new targets.
  ASSERT_OK(Set("1:2:0.5:1"));
  ASSERT_EQ(Current(), (std::vector<double>{1, 2, 0.5, 1}));
  VersionStorageInfo* vs = Storage();
  ASSERT_NE(before, vs);  // SetOptions appended a new version

  ASSERT_EQ(vs->MaxBytesForLevel(0), vs->BaseMaxBytesForLevel(0));
  ASSERT_EQ(vs->MaxBytesForLevel(1), 2 * vs->BaseMaxBytesForLevel(1));
  ASSERT_EQ(vs->MaxBytesForLevel(2), vs->BaseMaxBytesForLevel(2) / 2);
  ASSERT_EQ(vs->MaxBytesForLevel(3), vs->BaseMaxBytesForLevel(3));
  for (int level = 1; level <= 2; ++level) {
    ASSERT_DOUBLE_EQ(ScoreOf(level),
                     static_cast<double>(CompensatedBytes(level)) /
                         vs->MaxBytesForLevel(level));
  }
  ASSERT_LT(ScoreOf(1), 1.0);  // deferred by the multiplier alone
  ASSERT_DOUBLE_EQ(ScoreOf(1), l1_before / 2);
}

TEST_F(LevelTargetMultipliersTest, L0ScoreAndTargetNeverScaled) {
  Reopen(BaseOptions());
  BuildTree();
  const double l0 = ScoreOf(0);
  const uint64_t l0_target = Storage()->MaxBytesForLevel(0);
  // 0.5 on L1 would halve L0's byte trigger if the multiplier were routed
  // through max_bytes_for_level_base (A-Impl-1).
  for (const char* value : {"1:0.5:1:1", "1:2:2:2", "1:1:1:1"}) {
    SCOPED_TRACE(value);
    ASSERT_OK(Set(value));
    ASSERT_DOUBLE_EQ(ScoreOf(0), l0);
    ASSERT_EQ(Storage()->MaxBytesForLevel(0), l0_target);
  }
}

TEST_F(LevelTargetMultipliersTest, AllOnesMatchesAbsent) {
  Reopen(BaseOptions());
  BuildTree();
  std::vector<double> scores;
  std::vector<uint64_t> targets;
  for (int level = 0; level <= Storage()->MaxInputLevel(); ++level) {
    scores.push_back(ScoreOf(level));
  }
  for (int level = 0; level < 4; ++level) {
    targets.push_back(Storage()->MaxBytesForLevel(level));
  }
  const uint64_t pending = Storage()->estimated_compaction_needed_bytes();

  ASSERT_OK(Set("1:1:1:1"));
  for (int level = 0; level <= Storage()->MaxInputLevel(); ++level) {
    ASSERT_DOUBLE_EQ(ScoreOf(level), scores[level]);
  }
  for (int level = 0; level < 4; ++level) {
    ASSERT_EQ(Storage()->MaxBytesForLevel(level), targets[level]);
  }
  ASSERT_EQ(Storage()->estimated_compaction_needed_bytes(), pending);
}

TEST_F(LevelTargetMultipliersTest, PendingEstimateUsesScaledTargets) {
  Reopen(BaseOptions());
  WriteFileTo(2, 0, 200, 10 << 10);
  WriteFileTo(1, 10000, 150, 10 << 10);  // L1 ~ 1.5 MiB, L0 empty
  const uint64_t native = Storage()->estimated_compaction_needed_bytes();
  ASSERT_GT(native, 0u);

  ASSERT_OK(Set("1:2:1:1"));  // L1's target 2 MiB: nothing pending
  ASSERT_EQ(Storage()->estimated_compaction_needed_bytes(), 0u);

  ASSERT_OK(Set("1:0.5:1:1"));  // target 0.5 MiB: more pending than native
  ASSERT_GT(Storage()->estimated_compaction_needed_bytes(), native);
}

TEST_F(LevelTargetMultipliersTest, InvalidVectorsRejectedBySetOptions) {
  Options options = BaseOptions();
  options.max_bytes_for_level_multiplier =
      2;  // T = 2, so shrinking is possible
  Reopen(options);
  ASSERT_OK(Set("1:1.5:1:1"));
  const std::vector<double> valid = Current();
  for (const char* value : {
           "1:1:1",      // wrong size
           "1:1:1:1:1",  // wrong size
           "1.5:1:1:1",  // L0 scaled
           "1:nan:1:1",  // not finite
           "1:inf:1:1",  // not finite
           "1:2.5:1:1",  // above 2.0
           "1:0.4:1:1",  // below 0.5
           "1:2:0.5:1",  // L2 target 0.5 * 2 = 1 < 2: shrinks going down
           "1:x:1:1",    // unparsable
       }) {
    SCOPED_TRACE(value);
    ASSERT_NOK(Set(value));
    ASSERT_EQ(Current(), valid);  // never clamped, never half-applied
  }
  ASSERT_OK(Set("1:2:1:2"));  // 1 * 2 >= 2 and 2 * 2 >= 1: allowed at T = 2
}

TEST_F(LevelTargetMultipliersTest, InvalidVectorsRejectedAtOpen) {
  Options options = BaseOptions();
  options.level_target_multipliers = {1, 1, 1};  // wrong size
  ASSERT_TRUE(TryReopen(options).IsInvalidArgument());

  options = BaseOptions();
  options.level_target_multipliers = {1, 1.5, 1, 1};
  options.level_compaction_dynamic_level_bytes = true;
  ASSERT_TRUE(TryReopen(options).IsInvalidArgument());

  options = BaseOptions();
  options.level_target_multipliers = {1, 1.5, 1, 1};
  options.compaction_style = kCompactionStyleUniversal;
  ASSERT_TRUE(TryReopen(options).IsInvalidArgument());

  options = BaseOptions();
  options.max_bytes_for_level_multiplier = 2;
  options.level_target_multipliers = {1, 2, 0.5, 1};  // shrinks going down
  ASSERT_TRUE(TryReopen(options).IsInvalidArgument());

  options = BaseOptions();
  options.level_target_multipliers = {1, 1.5, 0.8, 1};
  ASSERT_OK(TryReopen(options));
  ASSERT_EQ(Storage()->MaxBytesForLevel(1),
            static_cast<uint64_t>(1.5L * Storage()->BaseMaxBytesForLevel(1)));
}

TEST_F(LevelTargetMultipliersTest, SetOptionsRejectedUnderDynamicOrUniversal) {
  Options options = BaseOptions();
  options.level_compaction_dynamic_level_bytes = true;
  Reopen(options);
  ASSERT_NOK(Set("1:1.5:1:1"));

  options = BaseOptions();
  options.compaction_style = kCompactionStyleUniversal;
  DestroyAndReopen(options);
  ASSERT_NOK(Set("1:1.5:1:1"));
}

TEST_F(LevelTargetMultipliersTest, PersistsInOptionsFileAndReopens) {
  Reopen(BaseOptions());
  ASSERT_OK(Set("1:1.5:0.8:1"));
  const std::vector<double> expected{1, 1.5, 0.8, 1};
  Close();

  ConfigOptions config_options;
  config_options.env = env_;
  DBOptions db_options;
  std::vector<ColumnFamilyDescriptor> cf_descs;
  ASSERT_OK(LoadLatestOptions(config_options, dbname_, &db_options, &cf_descs));
  ASSERT_EQ(cf_descs.size(), 1u);
  ASSERT_EQ(cf_descs[0].options.level_target_multipliers, expected);

  Options reopened(db_options, cf_descs[0].options);
  reopened.env = env_;
  Reopen(reopened);
  ASSERT_EQ(Current(), expected);
  ASSERT_EQ(Storage()->CapacityScale(1), 1.5);
  ASSERT_EQ(Storage()->CapacityScale(2), 0.8);
}

TEST_F(LevelTargetMultipliersTest, OldEnvironmentPathIsGone) {
  // RL_STATIC_CAPACITY_SCALES used to scale targets behind the options'
  // back; the option is now the only way in.
  ASSERT_EQ(0, setenv("RL_STATIC_CAPACITY_SCALES", "1,2,2,1", 1));
  Reopen(BaseOptions());
  BuildTree();
  for (int level = 0; level < 4; ++level) {
    ASSERT_EQ(Storage()->CapacityScale(level), 1.0);
    ASSERT_EQ(Storage()->MaxBytesForLevel(level),
              Storage()->BaseMaxBytesForLevel(level));
  }
  ASSERT_EQ(0, unsetenv("RL_STATIC_CAPACITY_SCALES"));
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
