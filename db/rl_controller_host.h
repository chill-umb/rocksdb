//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork, Programme 1: the host side of the controller (plan WP2-WP4):
// the host log (WP4), which every arm writes, native included; and the
// controller host with its plugin loader (WP2), on controller arms only.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rocksdb/listener.h"
#include "rocksdb/rl_controller_host.h"
#include "rocksdb/statistics.h"
#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {

class ColumnFamilyData;
class DB;
class DBImpl;

// Operations served since the statistics object was created: keys written
// (Puts, Deletes, Merges), Gets and iterator seeks. Plan WP2's OpCount().
uint64_t RLOperationCount(const Statistics& stats);

// The host log: one JSON object per line, flushed as written. Records
// (plan WP4; PATHWAYS Gate N0 items 3-5, D §4, G.4, H §5):
//
//   header     schema, num_levels, and both clocks at open
//   h          H = live SST bytes, with the operation count, after every
//              flush and compaction install (the only installs that change H)
//   job_begin  a compaction's job id, start and output level, trivial-move
//              flag, S and O (input bytes at the start and output level),
//              and when its start level became due (0 if it was not due, or
//              the job was manual)
//   job_end    the same, plus X (output bytes) and ok
//   stamp      a named phase point: operation count, H, cumulative stall
//              micros (the counter behind the internal-stats "Cumulative
//              stall" line), the per-level read counters and every ticker
//
// Times are steady_clock microseconds (`t_us`), the clock of the due-since
// values; the header and stamps also carry wall-clock micros (`wall_us`),
// the event log's clock. Operation counts come from the DB's statistics, so
// the DB must be opened with this object's statistics.
//
// Thread-safe: listener callbacks and stamps serialise on one mutex, and
// each line's operation count and H are read under it, so the log's
// operation counts never decrease. Callbacks run without the DB mutex.
class RLHostLog : public EventListener {
 public:
  // Creates (truncating) `path` and writes the header.
  static Status Open(const std::string& path,
                     std::shared_ptr<Statistics> statistics, int num_levels,
                     std::shared_ptr<RLHostLog>* result);
  ~RLHostLog() override;

  RLHostLog(const RLHostLog&) = delete;
  RLHostLog& operator=(const RLHostLog&) = delete;

  const char* Name() const override { return "RLHostLog"; }
  void OnFlushCompleted(DB* db, const FlushJobInfo& info) override;
  void OnCompactionBegin(DB* db, const CompactionJobInfo& info) override;
  void OnCompactionCompleted(DB* db, const CompactionJobInfo& info) override;

  // Writes a stamp named `name`. `extra` is appended to the object as-is and
  // must be empty or start with a comma (",\"ok\":1"). Returns an error if
  // this or any earlier write failed: a log with a hole must not be scored.
  Status Stamp(DB* db, const std::string& name, const std::string& extra = "");

 private:
  RLHostLog(FILE* file, std::shared_ptr<Statistics> statistics, int num_levels);

  // Requires mu_. The line must not end in a newline.
  void WriteLine(const std::string& line);
  void JobRecord(const CompactionJobInfo& info, bool end);
  void HSample(DB* db, const char* cause, int job_id);

  // Guards everything below, and the order of lines in the file.
  std::mutex mu_;
  FILE* file_;
  const std::shared_ptr<Statistics> statistics_;
  const int num_levels_;
  // A line failed to write, or a property it needed could not be read.
  bool write_failed_ = false;
};

// The controller host (plan WP2): RLControllerHost
// (include/rocksdb/rl_controller_host.h) on the DB's default column family.
// It is also an EventListener, registered when the DB opens, so it has seen
// every job, including those still running when the plugin starts; Attach()
// binds it to the open DB at n_w. db_bench runs one DB with one column
// family.
//
// Snapshot() reads the tree snapshot that the column family's pressure
// observer publishes at every score computation, under the DB mutex, once
// Attach() has switched it on; Snapshot() itself takes no DB mutex. Apply()
// and BeginDrain() serialise on one mutex, so an Apply either finishes
// before the drain's own SetOptions or is refused.
class RLControllerHostImpl : public RLControllerHost, public EventListener {
 public:
  explicit RLControllerHostImpl(std::shared_ptr<Statistics> statistics);

  RLControllerHostImpl(const RLControllerHostImpl&) = delete;
  RLControllerHostImpl& operator=(const RLControllerHostImpl&) = delete;

  // Binds the host to `db`'s default column family, reads the options fixed
  // for the run and publishes the first snapshot. Once, without the DB
  // mutex, before the plugin is created; `db` must outlive the plugin.
  Status Attach(DB* db);

  // The drain (plan WP4): refuses every later Apply, then sets every
  // multiplier to 1 and the L0 trigger to its value at Attach() in one
  // SetOptions, or makes no call when those are already in effect.
  Status BeginDrain();

  // RLControllerHost.
  RLHostOptions Options() const override;
  std::shared_ptr<const RLTreeSnapshot> Snapshot() const override;
  RLOpCounts OpCounts() const override;
  void ReadCounters(std::vector<RLLevelReadCounts>* out) const override;
  void SetJobCallback(
      std::function<void(const RLJobRecord&)> callback) override;
  bool Apply(const std::vector<double>& m, int k0, std::string* error) override;
  bool Draining() const override {
    return draining_.load(std::memory_order_acquire);
  }

  // EventListener.
  const char* Name() const override { return "RLControllerHost"; }
  void OnTableFileCreated(const TableFileCreationInfo& info) override;
  void OnFlushCompleted(DB* db, const FlushJobInfo& info) override;
  void OnCompactionBegin(DB* db, const CompactionJobInfo& info) override;
  void OnCompactionCompleted(DB* db, const CompactionJobInfo& info) override;

 private:
  // Requires apply_mu_. One SetOptions carrying both options.
  Status SetOptionsLocked(const std::vector<double>& m, int k0);
  void Deliver(const RLJobRecord& record);
  RLJobRecord CompactionRecord(const CompactionJobInfo& info, bool end) const;

  const std::shared_ptr<Statistics> statistics_;
  // Set by Attach(), read-only afterwards.
  DB* db_ = nullptr;
  DBImpl* db_impl_ = nullptr;
  ColumnFamilyData* cfd_ = nullptr;
  RLHostOptions options_;

  std::mutex apply_mu_;
  std::atomic<bool> draining_{false};

  // The job callback; held while it runs.
  std::mutex callback_mu_;
  std::function<void(const RLJobRecord&)> callback_;

  // Running compactions (job id -> start level), and the bytes of the files
  // each running flush has created (job id -> bytes).
  mutable std::mutex jobs_mu_;
  std::map<int, int> running_;
  std::map<int, uint64_t> flush_bytes_;
};

// A loaded controller plugin (plan WP2): dlopens the library, looks up
// rl_controller_create and rl_controller_destroy, and creates the controller
// for `host`. Destroying it destroys the controller first and only then
// closes the library.
class RLControllerPlugin {
 public:
  // An error if the library or a symbol is missing. A library whose
  // rl_controller_create returns nullptr loads with created() false; the
  // host then runs as native.
  static Status Load(const std::string& path, const std::string& config_path,
                     RLControllerHost* host,
                     std::unique_ptr<RLControllerPlugin>* result);
  ~RLControllerPlugin();

  RLControllerPlugin(const RLControllerPlugin&) = delete;
  RLControllerPlugin& operator=(const RLControllerPlugin&) = delete;

  bool created() const { return controller_ != nullptr; }

 private:
  RLControllerPlugin(void* library, RLControllerDestroyFn destroy,
                     void* controller)
      : library_(library), destroy_(destroy), controller_(controller) {}

  void* library_;
  RLControllerDestroyFn destroy_;
  void* controller_;
};

}  // namespace ROCKSDB_NAMESPACE
