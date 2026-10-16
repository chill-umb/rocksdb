//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork: the mixgraph additions of the Programme 1 workload suite
// (docs/PREREGISTRATION.md D-29 in the parent repository). They are kept
// apart from db_bench's Benchmark state so tools/mixgraph_suite_test.cc can
// test them directly; db_bench_tool.cc's MixGraph calls them.
//
// - Decoupling: the query type is drawn from its own random stream, not from
//   the number that picks the key (Cao et al., FAST 2020, §7.2).
// - Decoupled sizes: so are a Put's value size and a Seek's scan length.
// - The power-law key-sign fix.
// - Inserts of new keys (W4): a new key keeps a loaded key's 8-byte ID and
//   sorts directly after it.
// - Phases by operation count (W5).

#pragma once

#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {
namespace mixgraph_suite {

// Benchmark::GenerateKeyFromInt (no prefix) writes a key as its 8-byte
// big-endian ID followed by `kKeyPadByte` up to --key_size.
constexpr size_t kIdBytes = 8;
constexpr char kKeyPadByte = '0';
// An inserted key's byte 8. It is above kKeyPadByte, so the key sorts after
// the loaded key with the same ID and before the next loaded ID.
constexpr char kInsertMarker = '1';
// ID, marker and the 8-byte big-endian insert number.
constexpr size_t kInsertKeyMinSize = kIdBytes + 1 + 8;

// Salts for the suite's random streams. Each stream is seeded from --seed and
// the thread, never drawn from the thread's own stream, so db_bench's draws
// are unchanged when the suite's options are off.
constexpr uint64_t kQueryStreamSalt = 0x51f15e5d0c0ffee1ULL;
constexpr uint64_t kPresentKeyStreamSalt = 0x7e57ab1e5eed0002ULL;
constexpr uint64_t kInsertIdSalt = 0x1d5ca77e2ed00003ULL;
constexpr uint64_t kSizeStreamSalt = 0x5123d1ce5eed0004ULL;

// splitmix64's finaliser: a bijection on 64-bit values that mixes every bit.
inline uint64_t Mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline uint64_t StreamSeed(int64_t seed_base, int tid, uint64_t salt) {
  return Mix64(static_cast<uint64_t>(seed_base) ^
               Mix64(static_cast<uint64_t>(tid) ^ salt));
}

// The power-law key-sign fix (D-29 §1). db_bench took Random64's output as
// int64 before `% num`, so about half the keys came out negative and fell
// outside the loaded key space. The fix takes the output unsigned.
inline int64_t PowerLawKey(uint64_t scrambled, int64_t num) {
  assert(num > 0);
  return static_cast<int64_t>(scrambled % static_cast<uint64_t>(num));
}

// The number QueryDecider::GetType reads. Coupled (db_bench as released):
// the key draw's rand_v, so the key's draw fixes the type. Decoupled
// (--mixgraph_decouple_query): a draw from the query's own stream. GetType
// takes |n| % range, so the draw is kept non-negative.
inline int64_t QueryTypeDraw(bool decouple, int64_t rand_v,
                             uint64_t query_draw) {
  return decouple ? static_cast<int64_t>(query_draw >> 1) : rand_v;
}

// The u that a Put's value size and a Seek's scan length are drawn from (both
// through ParetoCdfInversion). Coupled (db_bench as released): key_u, the
// key draw's rand_v / num, so a key's draw also fixes its sizes. Decoupled
// (--mixgraph_decouple_sizes): a uniform value in (0, 1) from the sizes'
// own stream; it is never 0, where ParetoCdfInversion's pow(u, -k) diverges.
inline double SizeDraw(bool decouple, double key_u, uint64_t size_draw) {
  // 52 bits plus a half step is exact in a double, so the result lies in
  // [2^-53, 1 - 2^-53].
  return decouple ? (static_cast<double>(size_draw >> 12) + 0.5) * 0x1.0p-52
                  : key_u;
}

// The loaded ID that inserted key `c` (1-based, in insert order) sorts
// after: uniform over [0, num), and a function of `c`, so no list of inserted
// keys is kept.
inline uint64_t InsertedKeyId(uint64_t insert_seed, uint64_t c, uint64_t num) {
  assert(num > 0);
  return Mix64(insert_seed ^ Mix64(c ^ kInsertIdSalt)) % num;
}

// Turns a key that holds a loaded ID (bytes [0, 8), padding after) into
// inserted key `c`: the marker at byte 8, then `c` big-endian.
inline void WriteInsertSuffix(char* key, size_t key_size, uint64_t c) {
  assert(key_size >= kInsertKeyMinSize);
  (void)key_size;
  key[kIdBytes] = kInsertMarker;
  for (size_t i = 0; i < 8; ++i) {
    key[kIdBytes + 1 + i] = static_cast<char>((c >> ((7 - i) * 8)) & 0xFF);
  }
}

// A key present in the database: one of the `num` loaded keys, or one of the
// first `inserted` inserted keys.
struct PresentKey {
  bool inserted;
  uint64_t id;  // the loaded ID (bytes [0, 8))
  uint64_t c;   // the insert number, when inserted
};

// Uniform over the num + inserted keys present, from one random draw.
inline PresentKey PickPresentKey(uint64_t draw, uint64_t num,
                                 uint64_t inserted, uint64_t insert_seed) {
  assert(num > 0);
  const uint64_t x = draw % (num + inserted);
  if (x < num) {
    return PresentKey{false, x, 0};
  }
  const uint64_t c = x - num + 1;
  return PresentKey{true, InsertedKeyId(insert_seed, c, num), c};
}

// One phase of --mix_phases: its operation count and its query mix.
struct Phase {
  uint64_t ops = 0;
  double get = 0.0;
  double put = 0.0;
  double seek = 0.0;
  double insert = 0.0;
};

// Parses --mix_phases: "ops:get:put:seek[:insert]" phases joined by ';'.
// Every ops is a positive integer, every ratio is finite and >= 0, and each
// phase's ratios sum to more than 0. Returns false with `error` set on any
// malformed input.
inline bool ParsePhases(const std::string& spec, std::vector<Phase>* phases,
                        std::string* error) {
  phases->clear();
  if (spec.empty()) {
    *error = "empty phase list";
    return false;
  }
  size_t start = 0;
  while (true) {
    const size_t end = spec.find(';', start);
    const std::string item =
        spec.substr(start, end == std::string::npos ? end : end - start);
    std::vector<std::string> fields;
    size_t f = 0;
    while (true) {
      const size_t colon = item.find(':', f);
      fields.push_back(item.substr(
          f, colon == std::string::npos ? colon : colon - f));
      if (colon == std::string::npos) {
        break;
      }
      f = colon + 1;
    }
    const std::string where = "phase " + std::to_string(phases->size() + 1);
    if (fields.size() != 4 && fields.size() != 5) {
      *error = where + ": expected ops:get:put:seek[:insert], got '" + item +
               "'";
      return false;
    }
    Phase p;
    {
      const std::string& s = fields[0];
      if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) {
        *error = where + ": operation count '" + s + "' is not a positive "
                 "integer";
        return false;
      }
      errno = 0;
      p.ops = std::strtoull(s.c_str(), nullptr, 10);
      if (errno != 0 || p.ops == 0) {
        *error = where + ": operation count '" + s + "' is not a positive "
                 "integer";
        return false;
      }
    }
    double* ratios[4] = {&p.get, &p.put, &p.seek, &p.insert};
    double sum = 0.0;
    for (size_t i = 1; i < fields.size(); ++i) {
      const std::string& s = fields[i];
      // A plain decimal: no sign, space, "nan" or "inf".
      const bool plain = !s.empty() && (std::isdigit(static_cast<unsigned char>(
                                            s[0])) != 0 ||
                                        s[0] == '.');
      char* stop = nullptr;
      errno = 0;
      const double v = plain ? std::strtod(s.c_str(), &stop) : -1.0;
      if (!plain || errno != 0 || stop != s.c_str() + s.size() ||
          !(v >= 0.0) || v > 1e300) {
        *error = where + ": ratio '" + s + "' is not a finite number >= 0";
        return false;
      }
      *ratios[i - 1] = v;
      sum += v;
    }
    if (!(sum > 0.0)) {
      *error = where + ": its ratios sum to 0";
      return false;
    }
    phases->push_back(p);
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return true;
}

inline uint64_t TotalOps(const std::vector<Phase>& phases) {
  uint64_t total = 0;
  for (const Phase& p : phases) {
    total += p.ops;
  }
  return total;
}

}  // namespace mixgraph_suite
}  // namespace ROCKSDB_NAMESPACE
