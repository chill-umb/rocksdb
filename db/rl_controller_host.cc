//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/rl_controller_host.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>

#include "db/compaction/compaction_pressure_observer.h"
#include "db/rl_read_counters.h"
#include "rocksdb/db.h"
#include "rocksdb/system_clock.h"

namespace ROCKSDB_NAMESPACE {

std::atomic<uint64_t> RLReadCounters::counters_[RLReadCounters::kMaxLevels]
                                               [RLReadCounters::kKinds] = {};

uint64_t RLOperationCount(const Statistics& stats) {
  return stats.getTickerCount(NUMBER_KEYS_WRITTEN) +
         stats.getTickerCount(NUMBER_KEYS_READ) +
         stats.getTickerCount(NUMBER_DB_SEEK);
}

namespace {

void AppendField(std::string* line, const char* key, uint64_t value) {
  line->append(",\"").append(key).append("\":").append(std::to_string(value));
}

std::string Begin(const char* type) {
  return std::string("{\"type\":\"") + type + "\"";
}

}  // namespace

Status RLHostLog::Open(const std::string& path,
                       std::shared_ptr<Statistics> statistics, int num_levels,
                       std::shared_ptr<RLHostLog>* result) {
  if (statistics == nullptr) {
    return Status::InvalidArgument(
        "RLHostLog needs the DB's statistics for its operation counts");
  }
  FILE* file = fopen(path.c_str(), "w");
  if (file == nullptr) {
    return Status::IOError("RLHostLog: cannot create " + path, strerror(errno));
  }
  std::shared_ptr<RLHostLog> log(
      new RLHostLog(file, std::move(statistics), num_levels));
  std::string line = Begin("header");
  AppendField(&line, "schema", 1);
  AppendField(&line, "num_levels", static_cast<uint64_t>(num_levels));
  AppendField(&line, "t_us", CompactionPressureObserver::NowMicros());
  AppendField(&line, "wall_us", SystemClock::Default()->NowMicros());
  {
    std::lock_guard<std::mutex> lock(log->mu_);
    log->WriteLine(line);
    if (log->write_failed_) {
      return Status::IOError("RLHostLog: cannot write " + path);
    }
  }
  *result = std::move(log);
  return Status::OK();
}

RLHostLog::RLHostLog(FILE* file, std::shared_ptr<Statistics> statistics,
                     int num_levels)
    : file_(file),
      statistics_(std::move(statistics)),
      num_levels_(
          std::min(std::max(num_levels, 1), RLReadCounters::kMaxLevels)) {}

RLHostLog::~RLHostLog() { fclose(file_); }

void RLHostLog::WriteLine(const std::string& line) {
  if (fputs(line.c_str(), file_) == EOF || fputs("}\n", file_) == EOF ||
      fflush(file_) != 0) {
    write_failed_ = true;
  }
}

void RLHostLog::HSample(DB* db, const char* cause, int job_id) {
  uint64_t h = 0;
  if (!db->GetAggregatedIntProperty(DB::Properties::kLiveSstFilesSize, &h)) {
    write_failed_ = true;
  }
  std::string line = Begin("h");
  line.append(",\"cause\":\"").append(cause).append("\"");
  AppendField(&line, "job", static_cast<uint64_t>(job_id));
  AppendField(&line, "t_us", CompactionPressureObserver::NowMicros());
  AppendField(&line, "op", RLOperationCount(*statistics_));
  AppendField(&line, "h", h);
  WriteLine(line);
}

void RLHostLog::JobRecord(const CompactionJobInfo& info, bool end) {
  std::string line = Begin(end ? "job_end" : "job_begin");
  AppendField(&line, "job", static_cast<uint64_t>(info.job_id));
  AppendField(&line, "cf", info.cf_id);
  AppendField(&line, "t_us", CompactionPressureObserver::NowMicros());
  AppendField(&line, "op", RLOperationCount(*statistics_));
  AppendField(&line, "start_level",
              static_cast<uint64_t>(info.base_input_level));
  AppendField(&line, "output_level", static_cast<uint64_t>(info.output_level));
  AppendField(&line, "reason", static_cast<uint64_t>(info.compaction_reason));
  AppendField(&line, "trivial",
              info.stats.num_input_files_trivially_moved > 0 ? 1 : 0);
  AppendField(&line, "s", info.rl_start_level_input_bytes);
  AppendField(&line, "o", info.rl_output_level_input_bytes);
  AppendField(&line, "due_since_us", info.rl_start_level_due_since_micros);
  if (end) {
    AppendField(&line, "x", info.stats.total_output_bytes);
    AppendField(&line, "ok", info.status.ok() ? 1 : 0);
  }
  WriteLine(line);
}

void RLHostLog::OnFlushCompleted(DB* db, const FlushJobInfo& info) {
  std::lock_guard<std::mutex> lock(mu_);
  HSample(db, "flush", info.job_id);
}

void RLHostLog::OnCompactionBegin(DB* /*db*/, const CompactionJobInfo& info) {
  std::lock_guard<std::mutex> lock(mu_);
  JobRecord(info, /*end=*/false);
}

void RLHostLog::OnCompactionCompleted(DB* db, const CompactionJobInfo& info) {
  std::lock_guard<std::mutex> lock(mu_);
  JobRecord(info, /*end=*/true);
  if (info.status.ok()) {
    HSample(db, "compaction", info.job_id);
  }
}

Status RLHostLog::Stamp(DB* db, const std::string& name,
                        const std::string& extra) {
  std::lock_guard<std::mutex> lock(mu_);
  uint64_t h = 0;
  std::map<std::string, std::string> db_stats;
  if (!db->GetAggregatedIntProperty(DB::Properties::kLiveSstFilesSize, &h) ||
      !db->GetMapProperty(DB::Properties::kDBStats, &db_stats) ||
      db_stats.count("db.user_write_stall_micros") == 0) {
    write_failed_ = true;
  }
  std::string line = Begin("stamp");
  line.append(",\"name\":\"").append(name).append("\"");
  AppendField(&line, "t_us", CompactionPressureObserver::NowMicros());
  AppendField(&line, "wall_us", SystemClock::Default()->NowMicros());
  AppendField(&line, "op", RLOperationCount(*statistics_));
  AppendField(&line, "h", h);
  AppendField(&line, "stall_micros",
              std::strtoull(db_stats["db.user_write_stall_micros"].c_str(),
                            nullptr, 10));
  // levels[i] = [probe, filter pass, filter hit, seek] (RLReadCounter order).
  line.append(",\"levels\":[");
  for (int level = 0; level < num_levels_; ++level) {
    line.append(level == 0 ? "[" : ",[");
    for (int kind = 0; kind < RLReadCounters::kKinds; ++kind) {
      line.append(kind == 0 ? "" : ",")
          .append(std::to_string(
              RLReadCounters::Get(level, static_cast<RLReadCounter>(kind))));
    }
    line.append("]");
  }
  line.append("],\"tickers\":{");
  for (size_t i = 0; i < TickersNameMap.size(); ++i) {
    line.append(i == 0 ? "\"" : ",\"")
        .append(TickersNameMap[i].second)
        .append("\":")
        .append(std::to_string(
            statistics_->getTickerCount(TickersNameMap[i].first)));
  }
  line.append("}").append(extra);
  WriteLine(line);
  return write_failed_ ? Status::IOError("RLHostLog: a record was lost")
                       : Status::OK();
}

}  // namespace ROCKSDB_NAMESPACE
