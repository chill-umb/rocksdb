//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork, Programme 1 (plan WP2; PATHWAYS H §6): the interface a
// separately loaded compaction-controller plugin sees. The DB side
// implements RLControllerHost (db/rl_controller_host.{h,cc}); the plugin
// (controller/, librl_controller.so) implements the two C entry points at the
// end of this file, which the host finds with dlsym.
//
// A pure interface: standard-library types only and no RocksDB library
// symbols, so the plugin and its tests build against this header with a fake
// host and without linking RocksDB. Standard-library objects cross the
// boundary, so the host and the plugin must be built with the same compiler,
// the same libstdc++ and the same ROCKSDB_NAMESPACE. The Release library is
// built without RTTI and the plugin with it; nothing that crosses depends on
// RTTI (no dynamic_cast or typeid across the boundary).
//
// Threads. The plugin calls the host from its own thread and from the
// threads that call rl_controller_create and rl_controller_destroy, never
// while holding the DB mutex; the host never calls into the plugin while
// holding it. The job callback runs on RocksDB's background threads, without
// the DB mutex, and must return quickly. Host methods must not throw.
//
// Loading. The host dlopens the plugin, creates it at the first measured
// operation (n_w) and destroys it after the drain. It may dlclose only after
// rl_controller_destroy has returned. If rl_controller_create returns
// nullptr, the host logs it and runs with every multiplier 1 and the
// configured L0 trigger, as native.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

// Options fixed for the run, read once when the plugin is created.
struct RLHostOptions {
  int num_levels = 0;
  // T, max_bytes_for_level_multiplier. Under the static ladder (PATHWAYS A1)
  // level i >= 1 has the nominal target C_i = base_level_bytes * T^(i-1).
  double level_multiplier = 0;
  // max_bytes_for_level_multiplier_additional. The fork's ordering check
  // multiplies T by these; the plugin refuses any entry other than 1.
  std::vector<int> level_multiplier_additional;
  uint64_t base_level_bytes = 0;  // C_1, max_bytes_for_level_base
  uint64_t write_buffer_size = 0;
  int l0_trigger = 0;  // the configured level0_file_num_compaction_trigger
  int l0_slowdown_trigger = 0;  // level0_slowdown_writes_trigger (K_slow)
  int l0_stop_trigger = 0;      // level0_stop_writes_trigger
};

// One level of a snapshot.
struct RLLevelSnapshot {
  uint64_t bytes = 0;             // SST bytes at the level
  uint64_t bytes_compacting = 0;  // of which inputs to a running compaction
  // MaxBytesForLevel(i), multiplier applied; at L0 max_bytes_for_level_base,
  // which L0's file-count score does not use.
  uint64_t target_bytes = 0;
  int num_files = 0;
  int num_files_compacting = 0;
  double score = 0;               // RocksDB's compaction score
  uint64_t due_since_micros = 0;  // when the score last reached 1; 0 if below
};

// The tree as of the last score computation, except the fields marked
// "live", which Snapshot() reads when it is called.
struct RLTreeSnapshot {
  uint64_t generation = 0;  // increases with every publication
  uint64_t t_micros = 0;    // steady_clock microseconds at publication
  std::vector<RLLevelSnapshot> levels;  // num_levels entries
  // Levels by descending score, as RocksDB ranks them for picking.
  std::vector<int> score_order;
  // Start level of the running compaction; -1 if the slot is idle. Taken from
  // the host's own compaction begin and completed events, which bracket the
  // job exactly, not from the being-compacted flags, which reach a snapshot
  // only at score recomputes.
  int running_start_level = -1;
  // EstimateCompactionBytesNeeded, which reads the scaled targets
  // (A-Impl-3).
  uint64_t pending_compaction_bytes = 0;
  uint64_t live_sst_bytes = 0;  // H, rocksdb.live-sst-files-size
  // Live: rocksdb.cur-size-active-mem-table.
  uint64_t active_memtable_bytes = 0;
  // The values in effect: num_levels multipliers with [0] = 1, and the L0
  // compaction trigger. Rare recomputes (a failed or deferred compaction,
  // CompactFiles) score with the options captured when their job was picked,
  // so for a moment l0_trigger can be the one before the last Apply.
  std::vector<double> level_target_multipliers;
  int l0_trigger = 0;
  // Live: the write controller is stopping writes.
  bool write_stopped = false;
};

// Operation tickers. The operation count of PATHWAYS §1.1 is total().
struct RLOpCounts {
  uint64_t keys_written = 0;   // NUMBER_KEYS_WRITTEN
  uint64_t keys_read = 0;      // NUMBER_KEYS_READ
  uint64_t seeks = 0;          // NUMBER_DB_SEEK
  uint64_t bytes_written = 0;  // BYTES_WRITTEN, user bytes
  uint64_t total() const { return keys_written + keys_read + seeks; }
};

// One level's read counters, in the order of RLReadCounter
// (db/rl_read_counters.h). False-positive block reads are
// filter_passes - filter_hits; the hit's own read is the shared bucket's.
// Reopens are tables a Get or a user iterator found closed (D-21).
struct RLLevelReadCounts {
  uint64_t probes = 0;         // kProbe, POINT_SST_PROBE
  uint64_t filter_passes = 0;  // kFilterPass, BLOOM_FILTER_FULL_POSITIVE
  uint64_t filter_hits = 0;    // kFilterHit, BLOOM_FILTER_FULL_TRUE_POSITIVE
  uint64_t seeks = 0;          // kSeek, SORTED_RUN_SEEK
  uint64_t get_reopens = 0;    // kGetReopen, READ_TABLE_REOPEN with the next
  uint64_t iter_reopens = 0;   // kIterReopen
  uint64_t reopen_nanos = 0;   // kReopenNanos, READ_TABLE_REOPEN_NANOS
};

// A job record, with the fields of the host log's job_begin and job_end
// lines. A compaction sends one record when it starts and one when it ends;
// a flush sends one when it ends.
struct RLJobRecord {
  enum class Kind { kCompactionBegin, kCompactionEnd, kFlushEnd };
  Kind kind = Kind::kCompactionBegin;
  int job_id = 0;
  int start_level = -1;  // -1 for a flush
  int output_level = 0;
  int reason = 0;         // CompactionReason as an integer; 0 for a flush
  bool trivial = false;   // a trivial move: files relinked, nothing written
  uint64_t s = 0;         // input bytes at the start level
  uint64_t o = 0;         // input bytes at the output level
  uint64_t x = 0;         // output bytes, at the end; a flush's file bytes
  uint64_t op = 0;        // operation count when the record was made
  uint64_t t_micros = 0;  // steady_clock microseconds, the same
  // When the start level's score last reached 1 before the job was picked;
  // 0 if it was not due or the job was not picked automatically.
  uint64_t due_since_micros = 0;
  bool ok = true;  // at the end: the job succeeded
};

class RLControllerHost {
 public:
  virtual ~RLControllerHost() = default;

  virtual RLHostOptions Options() const = 0;

  // The latest snapshot, published at every score computation, with its live
  // fields read now. Reading it takes no DB mutex, except rarely to release a
  // SuperVersion replaced meanwhile. Never null.
  virtual std::shared_ptr<const RLTreeSnapshot> Snapshot() const = 0;

  virtual RLOpCounts OpCounts() const = 0;
  uint64_t OpCount() const { return OpCounts().total(); }

  // Fills num_levels entries.
  virtual void ReadCounters(std::vector<RLLevelReadCounts>* out) const = 0;

  // Replaces the job callback; an empty function removes it. It waits for a
  // call of the previous callback that is still running, and destroys the
  // previous std::function before returning (its code lives in the plugin's
  // library). After it returns, the previous callback is never called again.
  virtual void SetJobCallback(
      std::function<void(const RLJobRecord&)> callback) = 0;

  // One DB::SetOptions call carrying level_target_multipliers = m
  // (num_levels entries, m[0] = 1) and level0_file_num_compaction_trigger =
  // k0. Each multiplier must reach RocksDB exactly (serialised with
  // max_digits10), because the fork's validation compares them. Returns
  // false, with a reason, when refused or when SetOptions fails; the options
  // in effect are then unchanged. The plugin calls it at most once per its
  // configured interval (A-Impl-5); in between, RocksDB keeps the values last
  // accepted, which can differ from the targets the plugin logs.
  //
  // Drain ordering. The host holds one mutex around Apply and around its
  // switch to the drain. At the drain it sets Draining() first and then,
  // under the same mutex, makes its own SetOptions with every multiplier 1
  // and the configured trigger. So an Apply either finishes before the
  // drain's SetOptions or is refused; it never lands after it.
  virtual bool Apply(const std::vector<double>& m, int k0,
                     std::string* error) = 0;

  // True from the start of the drain (plan WP4); Apply is then refused.
  virtual bool Draining() const = 0;
};

// The entry points, looked up by these names.
using RLControllerCreateFn = void* (*)(RLControllerHost* host,
                                       const char* config_path);
using RLControllerDestroyFn = void (*)(void* controller);

}  // namespace ROCKSDB_NAMESPACE

extern "C" {
// Creates a controller for `host`, configured by the JSON file at
// `config_path`, and starts its thread. Returns nullptr only when no
// controller could be made at all; a controller whose configuration is
// invalid still returns, applies m = 1 and the configured trigger, and logs
// the fallback (A-Impl-8). The host must outlive the controller.
void* rl_controller_create(ROCKSDB_NAMESPACE::RLControllerHost* host,
                           const char* config_path);

// Stops the controller's thread, closes its logs and frees it. The
// controller makes no host call after this returns.
void rl_controller_destroy(void* controller);
}
