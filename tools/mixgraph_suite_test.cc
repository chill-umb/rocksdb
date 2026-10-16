//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork: tests for tools/mixgraph_suite.h, the Programme 1 suite's
// mixgraph additions (docs/PREREGISTRATION.md D-29 in the parent
// repository).

#include "tools/mixgraph_suite.h"

#include <chrono>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "port/stack_trace.h"
#include "rocksdb/slice.h"
#include "test_util/testharness.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {
namespace mixgraph_suite {
namespace {

uint64_t TimeSeed() {
  return static_cast<uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
}

// A loaded key as Benchmark::GenerateKeyFromInt writes it with no prefix:
// the 8-byte big-endian ID, then kKeyPadByte up to key_size.
std::string LoadedKey(uint64_t id, size_t key_size) {
  std::string key(key_size, kKeyPadByte);
  for (size_t i = 0; i < kIdBytes; ++i) {
    key[i] = static_cast<char>((id >> ((7 - i) * 8)) & 0xFF);
  }
  return key;
}

std::string InsertedKey(uint64_t id, uint64_t c, size_t key_size) {
  std::string key = LoadedKey(id, key_size);
  WriteInsertSuffix(&key[0], key.size(), c);
  return key;
}

int Compare(const std::string& a, const std::string& b) {
  return Slice(a).compare(Slice(b));
}

}  // namespace

// The key-sign fix: every key lands in [0, num), including draws with the
// top bit set, which the released code turned into negative keys.
TEST(MixgraphSuiteTest, PowerLawKeyStaysInLoadedRange) {
  struct Case {
    uint64_t scrambled;
    int64_t num;
  };
  const Case cases[] = {
      {0, 1},
      {UINT64_MAX, 20000000},
      {uint64_t{1} << 63, 20000000},
      {(uint64_t{1} << 63) + 12345, 2500000},
      {19999999, 20000000},
      {20000000, 20000000},
  };
  for (const Case& c : cases) {
    const int64_t key = PowerLawKey(c.scrambled, c.num);
    EXPECT_GE(key, 0);
    EXPECT_LT(key, c.num);
    EXPECT_EQ(static_cast<uint64_t>(key),
              c.scrambled % static_cast<uint64_t>(c.num));
  }
  // The released expression goes negative for a top-bit draw.
  const uint64_t top_bit = (uint64_t{1} << 63) + 12345;
  EXPECT_LT(static_cast<int64_t>(top_bit) % int64_t{20000000}, 0);
}

TEST(MixgraphSuiteTest, PowerLawKeyRandomized) {
  const uint64_t seed = TimeSeed();
  SCOPED_TRACE("seed=" + std::to_string(seed));
  Random64 rnd(seed);
  for (int i = 0; i < 100000; ++i) {
    const int64_t num = static_cast<int64_t>(rnd.Uniform(50000000) + 1);
    const int64_t key = PowerLawKey(rnd.Next(), num);
    ASSERT_GE(key, 0);
    ASSERT_LT(key, num);
  }
}

// Coupled: the key draw's rand_v decides the type. Decoupled: the query
// stream decides it, whatever the key.
TEST(MixgraphSuiteTest, QueryTypeDrawFollowsTheMode) {
  EXPECT_EQ(QueryTypeDraw(false, 4321, 99), 4321);
  EXPECT_EQ(QueryTypeDraw(false, 4321, UINT64_MAX), 4321);
  EXPECT_EQ(QueryTypeDraw(true, 4321, 99), 49);
  EXPECT_GE(QueryTypeDraw(true, 4321, UINT64_MAX), 0);
  EXPECT_EQ(QueryTypeDraw(true, 0, 1000), QueryTypeDraw(true, 777, 1000));
}

// For one fixed key (one rand_v), the decoupled draw reproduces the mix:
// QueryDecider's position is |draw| % range, with Assoc's cumulative
// thresholds 806 / 965 / 1000. The coupled draw gives that key one type.
TEST(MixgraphSuiteTest, DecoupledTypesFollowTheMixForOneKey) {
  const uint64_t seed = TimeSeed();
  SCOPED_TRACE("seed=" + std::to_string(seed));
  Random64 query_stream(StreamSeed(static_cast<int64_t>(seed), 0,
                                   kQueryStreamSalt));
  const int64_t thresholds[3] = {806, 965, 1000};
  const auto type_of = [&](int64_t draw) {
    const int64_t pos = draw % 1000;
    int t = 0;
    while (pos >= thresholds[t]) {
      ++t;
    }
    return t;
  };
  const int64_t rand_v = 812;  // a Put position in the coupled comb
  const int n = 1000000;
  int counts[3] = {0, 0, 0};
  std::set<int> coupled_types;
  for (int i = 0; i < n; ++i) {
    ++counts[type_of(QueryTypeDraw(true, rand_v, query_stream.Next()))];
    coupled_types.insert(type_of(QueryTypeDraw(false, rand_v, 0)));
  }
  const double expected[3] = {0.806, 0.159, 0.035};
  for (int t = 0; t < 3; ++t) {
    // Binomial standard deviations at n = 1e6 are under 0.0004.
    EXPECT_NEAR(static_cast<double>(counts[t]) / n, expected[t], 0.003);
  }
  EXPECT_EQ(coupled_types, std::set<int>({1}));
}

// Coupled: the key draw's u. Decoupled: strictly inside (0, 1) whatever the
// draw, so ParetoCdfInversion never sees u = 0.
TEST(MixgraphSuiteTest, SizeDrawFollowsTheModeAndStaysInside) {
  EXPECT_DOUBLE_EQ(SizeDraw(false, 0.25, 12345), 0.25);
  EXPECT_DOUBLE_EQ(SizeDraw(false, 0.0, UINT64_MAX), 0.0);
  for (uint64_t draw : {uint64_t{0}, uint64_t{1}, uint64_t{4095},
                        uint64_t{1} << 63, UINT64_MAX}) {
    SCOPED_TRACE("draw=" + std::to_string(draw));
    const double u = SizeDraw(true, 0.25, draw);
    EXPECT_GT(u, 0.0);
    EXPECT_LT(u, 1.0);
  }
  EXPECT_DOUBLE_EQ(SizeDraw(true, 0.0, 777), SizeDraw(true, 0.9, 777));
}

// Decoupled draws are uniform: every tenth of (0, 1) gets about a tenth,
// and the mean is about a half.
TEST(MixgraphSuiteTest, SizeDrawIsUniform) {
  const uint64_t seed = TimeSeed();
  SCOPED_TRACE("seed=" + std::to_string(seed));
  Random64 size_stream(
      StreamSeed(static_cast<int64_t>(seed), 0, kSizeStreamSalt));
  const int n = 1000000;
  int deciles[10] = {};
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    const double u = SizeDraw(true, 0.5, size_stream.Next());
    ++deciles[static_cast<int>(u * 10)];
    sum += u;
  }
  for (int b = 0; b < 10; ++b) {
    // A tenth is 100,000 with a standard deviation of 300.
    EXPECT_NEAR(deciles[b], n / 10, 2000) << "decile " << b;
  }
  EXPECT_NEAR(sum / n, 0.5, 0.002);
}

TEST(MixgraphSuiteTest, StreamSeedsDifferByThreadAndSalt) {
  const int64_t base = 301;
  std::set<uint64_t> seeds = {
      StreamSeed(base, 0, kQueryStreamSalt),
      StreamSeed(base, 1, kQueryStreamSalt),
      StreamSeed(base, 0, kPresentKeyStreamSalt),
      StreamSeed(base, 0, kSizeStreamSalt),
      StreamSeed(base + 1, 0, kQueryStreamSalt),
  };
  EXPECT_EQ(seeds.size(), 5u);
  EXPECT_EQ(StreamSeed(base, 0, kQueryStreamSalt),
            StreamSeed(base, 0, kQueryStreamSalt));
}

// An inserted key sorts after the loaded key with its ID and before the
// next loaded ID, and inserted keys at one ID sort by insert number.
TEST(MixgraphSuiteTest, InsertedKeysSortBetweenLoadedKeys) {
  struct Case {
    uint64_t id;
    uint64_t c;
  };
  const Case cases[] = {
      {0, 1},
      {0, UINT64_MAX},
      {41, 0x3030303030303030ULL},  // all-padding bytes after the marker
      {19999999, 36000000},
      {0x00FFFFFFFFFFFFFEULL, 7},
  };
  for (size_t key_size : {kInsertKeyMinSize, size_t{64}}) {
    for (const Case& c : cases) {
      SCOPED_TRACE("key_size=" + std::to_string(key_size) +
                   " id=" + std::to_string(c.id) +
                   " c=" + std::to_string(c.c));
      const std::string loaded = LoadedKey(c.id, key_size);
      const std::string inserted = InsertedKey(c.id, c.c, key_size);
      const std::string next = LoadedKey(c.id + 1, key_size);
      ASSERT_EQ(inserted.size(), key_size);
      EXPECT_LT(Compare(loaded, inserted), 0);
      EXPECT_LT(Compare(inserted, next), 0);
      if (c.c < UINT64_MAX) {
        EXPECT_LT(Compare(inserted, InsertedKey(c.id, c.c + 1, key_size)), 0);
      }
    }
  }
}

TEST(MixgraphSuiteTest, InsertedKeysAreDistinctAndInRange) {
  const uint64_t seed = TimeSeed();
  SCOPED_TRACE("seed=" + std::to_string(seed));
  const uint64_t num = 2500;
  std::set<std::string> keys;
  for (uint64_t id = 0; id < num; ++id) {
    keys.insert(LoadedKey(id, 64));
  }
  const uint64_t inserts = 20000;
  for (uint64_t c = 1; c <= inserts; ++c) {
    const uint64_t id = InsertedKeyId(seed, c, num);
    ASSERT_LT(id, num);
    ASSERT_TRUE(keys.insert(InsertedKey(id, c, 64)).second) << "c=" << c;
  }
  EXPECT_EQ(keys.size(), num + inserts);
}

// Inserted keys land uniformly over the loaded IDs: 1M inserts over 1,000
// IDs, about 1,000 each (standard deviation about 32).
TEST(MixgraphSuiteTest, InsertedKeyIdsAreUniform) {
  const uint64_t seed = TimeSeed();
  SCOPED_TRACE("seed=" + std::to_string(seed));
  const uint64_t num = 1000;
  std::vector<int> counts(num, 0);
  for (uint64_t c = 1; c <= 1000000; ++c) {
    ++counts[InsertedKeyId(seed, c, num)];
  }
  for (uint64_t id = 0; id < num; ++id) {
    EXPECT_NEAR(counts[id], 1000, 200) << "id=" << id;
  }
}

TEST(MixgraphSuiteTest, PickPresentKeyBoundaries) {
  const uint64_t num = 100;
  const uint64_t seed = 7;
  const PresentKey first = PickPresentKey(0, num, 5, seed);
  EXPECT_FALSE(first.inserted);
  EXPECT_EQ(first.id, 0u);
  const PresentKey last_loaded = PickPresentKey(99, num, 5, seed);
  EXPECT_FALSE(last_loaded.inserted);
  EXPECT_EQ(last_loaded.id, 99u);
  const PresentKey first_inserted = PickPresentKey(100, num, 5, seed);
  EXPECT_TRUE(first_inserted.inserted);
  EXPECT_EQ(first_inserted.c, 1u);
  EXPECT_EQ(first_inserted.id, InsertedKeyId(seed, 1, num));
  const PresentKey last_inserted = PickPresentKey(104, num, 5, seed);
  EXPECT_TRUE(last_inserted.inserted);
  EXPECT_EQ(last_inserted.c, 5u);
  // Wraps: 105 % 105 is loaded key 0.
  EXPECT_FALSE(PickPresentKey(105, num, 5, seed).inserted);
  // Nothing inserted yet: always a loaded key.
  for (uint64_t draw : {uint64_t{0}, uint64_t{150}, UINT64_MAX}) {
    const PresentKey k = PickPresentKey(draw, num, 0, seed);
    EXPECT_FALSE(k.inserted);
    EXPECT_EQ(k.id, draw % num);
  }
}

// Uniform over the keys present: with 1,000 loaded and 3,000 inserted, a
// quarter of the picks are loaded keys, and every insert number shows up.
TEST(MixgraphSuiteTest, PickPresentKeyIsUniform) {
  const uint64_t seed = TimeSeed();
  SCOPED_TRACE("seed=" + std::to_string(seed));
  Random64 rnd(seed);
  const uint64_t num = 1000;
  const uint64_t inserted = 3000;
  const int n = 1000000;
  int loaded = 0;
  std::vector<int> per_c(inserted + 1, 0);
  for (int i = 0; i < n; ++i) {
    const PresentKey k = PickPresentKey(rnd.Next(), num, inserted, seed);
    if (k.inserted) {
      ASSERT_GE(k.c, 1u);
      ASSERT_LE(k.c, inserted);
      ASSERT_EQ(k.id, InsertedKeyId(seed, k.c, num));
      ++per_c[k.c];
    } else {
      ASSERT_LT(k.id, num);
      ++loaded;
    }
  }
  EXPECT_NEAR(static_cast<double>(loaded) / n, 0.25, 0.003);
  for (uint64_t c = 1; c <= inserted; ++c) {
    EXPECT_GT(per_c[c], 0) << "c=" << c;
  }
}

// D-29's W5 phases, and an insert field.
TEST(MixgraphSuiteTest, ParsePhasesAcceptsTheSuiteSpecs) {
  std::vector<Phase> phases;
  std::string error;
  ASSERT_TRUE(ParsePhases(
      "13333333:0.10:0.90:0;13333333:0.50:0.50:0;13333334:0.90:0.10:0",
      &phases, &error))
      << error;
  ASSERT_EQ(phases.size(), 3u);
  EXPECT_EQ(TotalOps(phases), 40000000u);
  EXPECT_EQ(phases[0].ops, 13333333u);
  EXPECT_DOUBLE_EQ(phases[0].get, 0.10);
  EXPECT_DOUBLE_EQ(phases[0].put, 0.90);
  EXPECT_DOUBLE_EQ(phases[2].get, 0.90);
  EXPECT_DOUBLE_EQ(phases[2].seek, 0.0);
  EXPECT_DOUBLE_EQ(phases[2].insert, 0.0);

  ASSERT_TRUE(ParsePhases("40000000:0.1:0:0:0.9", &phases, &error)) << error;
  ASSERT_EQ(phases.size(), 1u);
  EXPECT_DOUBLE_EQ(phases[0].insert, 0.9);
  ASSERT_TRUE(ParsePhases("5:.5:0.5:0", &phases, &error)) << error;
  EXPECT_DOUBLE_EQ(phases[0].get, 0.5);
}

TEST(MixgraphSuiteTest, ParsePhasesRejectsMalformedSpecs) {
  const char* bad[] = {
      "",                     // nothing
      "10:0:0:0",             // ratios sum to 0
      "0:1:0:0",              // no operations
      "x:1:0:0",              // count not a number
      "+5:1:0:0",             // signed count
      "10:1:0",               // too few fields
      "10:1:0:0:0:0",         // too many fields
      "10:-1:2:0",            // negative ratio
      "10:a:1:0",             // ratio not a number
      "10:nan:1:0",           // not finite
      "10:inf:1:0",           // not finite
      "10:1e999:0:0",         // overflows
      "10: 1:0:0",            // space
      "10:1:0:0;",            // empty trailing phase
      ";10:1:0:0",            // empty leading phase
      "10:1:0:0;;10:1:0:0",   // empty middle phase
      "10:1.0.0:0:0",         // trailing junk
      "99999999999999999999:1:0:0",  // count overflows
  };
  for (const char* spec : bad) {
    SCOPED_TRACE(std::string("spec='") + spec + "'");
    std::vector<Phase> phases;
    std::string error;
    EXPECT_FALSE(ParsePhases(spec, &phases, &error));
    EXPECT_FALSE(error.empty());
  }
}

}  // namespace mixgraph_suite
}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
