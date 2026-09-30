//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork, Programme 1: the host side of the controller (plan WP2-WP4).
// So far the host log (WP4), which every arm writes, native included. The
// snapshot view, batched SetOptions and plugin loading of WP2 join here with
// the plugin (plan §7 step 8).

#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

#include "rocksdb/listener.h"
#include "rocksdb/statistics.h"
#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {

class DB;

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

}  // namespace ROCKSDB_NAMESPACE
