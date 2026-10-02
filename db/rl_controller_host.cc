//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/rl_controller_host.h"

#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>

#include "db/column_family.h"
#include "db/compaction/compaction_pressure_observer.h"
#include "db/db_impl/db_impl.h"
#include "db/memtable.h"
#include "db/rl_read_counters.h"
#include "db/write_controller.h"
#include "rocksdb/db.h"
#include "rocksdb/system_clock.h"
#include "util/cast_util.h"

namespace ROCKSDB_NAMESPACE {

// RLLevelReadCounts' fields are in RLReadCounter's order.
static_assert(static_cast<int>(RLReadCounter::kProbe) == 0 &&
                  static_cast<int>(RLReadCounter::kFilterPass) == 1 &&
                  static_cast<int>(RLReadCounter::kFilterHit) == 2 &&
                  static_cast<int>(RLReadCounter::kSeek) == 3 &&
                  static_cast<int>(RLReadCounter::kGetReopen) == 4 &&
                  static_cast<int>(RLReadCounter::kIterReopen) == 5 &&
                  static_cast<int>(RLReadCounter::kReopenNanos) == 6 &&
                  RLReadCounters::kKinds == 7,
              "RLLevelReadCounts follows RLReadCounter");

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
  // Schema 2 (D-21): each levels row gains the reopen counters.
  AppendField(&line, "schema", 2);
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
  // levels[i] = [probe, filter pass, filter hit, seek, get reopen, iterator
  // reopen, reopen nanos] (RLReadCounter order; schema 2, D-21).
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

RLControllerHostImpl::RLControllerHostImpl(
    std::shared_ptr<Statistics> statistics)
    : statistics_(std::move(statistics)) {}

Status RLControllerHostImpl::Attach(DB* db) {
  if (db_ != nullptr) {
    return Status::InvalidArgument("RLControllerHost: already attached");
  }
  if (statistics_ == nullptr) {
    return Status::InvalidArgument(
        "RLControllerHost needs the DB's statistics for its operation counts");
  }
  // A plain DB: checked in Debug builds only, since the Release library is
  // built without RTTI (USE_RTTI=AUTO), where dynamic_cast does not compile.
  auto* impl = static_cast_with_check<DBImpl>(db->GetRootDB());
  // `Options` alone names RLControllerHost::Options() in this class.
  const ROCKSDB_NAMESPACE::Options options = db->GetOptions();
  if (options.num_levels < 1 ||
      options.num_levels > RLReadCounters::kMaxLevels) {
    return Status::NotSupported("RLControllerHost: num_levels outside [1, " +
                                std::to_string(RLReadCounters::kMaxLevels) +
                                "]");
  }
  options_.num_levels = options.num_levels;
  options_.level_multiplier = options.max_bytes_for_level_multiplier;
  // One entry per level, the only ones RocksDB reads: the default vector has
  // seven entries whatever num_levels is, and a missing entry means 1.
  options_.level_multiplier_additional =
      options.max_bytes_for_level_multiplier_additional;
  options_.level_multiplier_additional.resize(options.num_levels, 1);
  options_.base_level_bytes = options.max_bytes_for_level_base;
  options_.write_buffer_size = options.write_buffer_size;
  options_.l0_trigger = options.level0_file_num_compaction_trigger;
  options_.l0_slowdown_trigger = options.level0_slowdown_writes_trigger;
  options_.l0_stop_trigger = options.level0_stop_writes_trigger;
  db_ = db;
  db_impl_ = impl;
  cfd_ =
      static_cast_with_check<ColumnFamilyHandleImpl>(db->DefaultColumnFamily())
          ->cfd();
  {
    InstrumentedMutexLock lock(impl->mutex());
    cfd_->EnableRLTreeSnapshots();
  }
  const auto view = cfd_->compaction_pressure_view();
  if (view == nullptr || view->LoadTree() == nullptr) {
    return Status::NotSupported(
        "RLControllerHost: the column family publishes no snapshot");
  }
  return Status::OK();
}

RLHostOptions RLControllerHostImpl::Options() const { return options_; }

std::shared_ptr<const RLTreeSnapshot> RLControllerHostImpl::Snapshot() const {
  std::shared_ptr<const RLTreeSnapshot> published;
  if (cfd_ != nullptr) {
    const auto view = cfd_->compaction_pressure_view();
    if (view != nullptr) {
      published = view->LoadTree();
    }
  }
  auto snapshot = published == nullptr
                      ? std::make_shared<RLTreeSnapshot>()
                      : std::make_shared<RLTreeSnapshot>(*published);
  {
    // The job listed first began first. A job's install still marks its
    // inputs as being compacted, so the flags in the published version are
    // not used for this.
    std::lock_guard<std::mutex> lock(jobs_mu_);
    snapshot->running_start_level =
        running_.empty() ? -1 : running_.begin()->second;
  }
  if (db_impl_ != nullptr) {
    // Live fields, without the DB mutex: the thread-local SuperVersion holds
    // the active memtable, and the stop count is atomic.
    SuperVersion* sv = db_impl_->GetAndRefSuperVersion(cfd_);
    snapshot->active_memtable_bytes =
        static_cast_with_check<MemTable>(sv->mem)->ApproximateMemoryUsageFast();
    db_impl_->ReturnAndCleanupSuperVersion(cfd_, sv);
    snapshot->write_stopped = db_impl_->write_controller().IsStopped();
  }
  return snapshot;
}

RLOpCounts RLControllerHostImpl::OpCounts() const {
  RLOpCounts counts;
  if (statistics_ != nullptr) {
    counts.keys_written = statistics_->getTickerCount(NUMBER_KEYS_WRITTEN);
    counts.keys_read = statistics_->getTickerCount(NUMBER_KEYS_READ);
    counts.seeks = statistics_->getTickerCount(NUMBER_DB_SEEK);
    counts.bytes_written = statistics_->getTickerCount(BYTES_WRITTEN);
  }
  return counts;
}

void RLControllerHostImpl::ReadCounters(
    std::vector<RLLevelReadCounts>* out) const {
  out->assign(options_.num_levels, RLLevelReadCounts());
  for (int level = 0; level < options_.num_levels; ++level) {
    RLLevelReadCounts& counts = (*out)[level];
    counts.probes = RLReadCounters::Get(level, RLReadCounter::kProbe);
    counts.filter_passes =
        RLReadCounters::Get(level, RLReadCounter::kFilterPass);
    counts.filter_hits = RLReadCounters::Get(level, RLReadCounter::kFilterHit);
    counts.seeks = RLReadCounters::Get(level, RLReadCounter::kSeek);
    counts.get_reopens = RLReadCounters::Get(level, RLReadCounter::kGetReopen);
    counts.iter_reopens =
        RLReadCounters::Get(level, RLReadCounter::kIterReopen);
    counts.reopen_nanos =
        RLReadCounters::Get(level, RLReadCounter::kReopenNanos);
  }
}

void RLControllerHostImpl::SetJobCallback(
    std::function<void(const RLJobRecord&)> callback) {
  // Waits for a call in flight; the previous function is destroyed here.
  std::lock_guard<std::mutex> lock(callback_mu_);
  callback_ = std::move(callback);
}

void RLControllerHostImpl::Deliver(const RLJobRecord& record) {
  std::lock_guard<std::mutex> lock(callback_mu_);
  if (callback_) {
    callback_(record);
  }
}

Status RLControllerHostImpl::SetOptionsLocked(const std::vector<double>& m,
                                              int k0) {
  // Every digit a double needs, so the fork's validation sees the exact
  // values the plugin checked (std::to_string would round to six places).
  std::string vector;
  char buffer[32];
  for (size_t i = 0; i < m.size(); ++i) {
    snprintf(buffer, sizeof(buffer), "%.17g", m[i]);
    vector.append(i == 0 ? "" : ":").append(buffer);
  }
  return db_->SetOptions(
      db_->DefaultColumnFamily(),
      {{"level_target_multipliers", vector},
       {"level0_file_num_compaction_trigger", std::to_string(k0)}});
}

bool RLControllerHostImpl::Apply(const std::vector<double>& m, int k0,
                                 std::string* error) {
  std::lock_guard<std::mutex> lock(apply_mu_);
  Status s;
  if (draining_.load(std::memory_order_acquire)) {
    s = Status::Aborted("RLControllerHost: the drain has started");
  } else if (db_ == nullptr) {
    s = Status::InvalidArgument("RLControllerHost: not attached");
  } else if (m.size() != static_cast<size_t>(options_.num_levels)) {
    s = Status::InvalidArgument(
        "RLControllerHost: " + std::to_string(m.size()) + " multipliers for " +
        std::to_string(options_.num_levels) + " levels");
  } else {
    s = SetOptionsLocked(m, k0);
  }
  if (!s.ok() && error != nullptr) {
    *error = s.ToString();
  }
  return s.ok();
}

Status RLControllerHostImpl::BeginDrain() {
  std::lock_guard<std::mutex> lock(apply_mu_);
  draining_.store(true, std::memory_order_release);
  if (db_ == nullptr) {
    return Status::OK();
  }
  const ROCKSDB_NAMESPACE::Options in_effect = db_->GetOptions();
  const bool native =
      in_effect.level0_file_num_compaction_trigger == options_.l0_trigger &&
      std::all_of(in_effect.level_target_multipliers.begin(),
                  in_effect.level_target_multipliers.end(),
                  [](double m) { return m == 1.0; });
  if (native) {
    return Status::OK();
  }
  return SetOptionsLocked(std::vector<double>(options_.num_levels, 1.0),
                          options_.l0_trigger);
}

RLJobRecord RLControllerHostImpl::CompactionRecord(
    const CompactionJobInfo& info, bool end) const {
  RLJobRecord record;
  record.kind = end ? RLJobRecord::Kind::kCompactionEnd
                    : RLJobRecord::Kind::kCompactionBegin;
  record.job_id = info.job_id;
  record.start_level = info.base_input_level;
  record.output_level = info.output_level;
  record.reason = static_cast<int>(info.compaction_reason);
  record.trivial = info.stats.num_input_files_trivially_moved > 0;
  record.s = info.rl_start_level_input_bytes;
  record.o = info.rl_output_level_input_bytes;
  record.x = end ? info.stats.total_output_bytes : 0;
  record.op = statistics_ == nullptr ? 0 : RLOperationCount(*statistics_);
  record.t_micros = CompactionPressureObserver::NowMicros();
  record.due_since_micros = info.rl_start_level_due_since_micros;
  record.ok = !end || info.status.ok();
  return record;
}

void RLControllerHostImpl::OnTableFileCreated(
    const TableFileCreationInfo& info) {
  if (info.reason != TableFileCreationReason::kFlush || !info.status.ok()) {
    return;
  }
  std::lock_guard<std::mutex> lock(jobs_mu_);
  flush_bytes_[info.job_id] += info.file_size;
}

void RLControllerHostImpl::OnFlushCompleted(DB* /*db*/,
                                            const FlushJobInfo& info) {
  RLJobRecord record;
  record.kind = RLJobRecord::Kind::kFlushEnd;
  record.job_id = info.job_id;
  record.start_level = -1;
  record.output_level = 0;
  {
    // The file was created, and counted, before the flush installed it.
    std::lock_guard<std::mutex> lock(jobs_mu_);
    const auto it = flush_bytes_.find(info.job_id);
    if (it != flush_bytes_.end()) {
      record.x = it->second;
      flush_bytes_.erase(it);
    }
  }
  record.op = statistics_ == nullptr ? 0 : RLOperationCount(*statistics_);
  record.t_micros = CompactionPressureObserver::NowMicros();
  Deliver(record);
}

void RLControllerHostImpl::OnCompactionBegin(DB* /*db*/,
                                             const CompactionJobInfo& info) {
  {
    std::lock_guard<std::mutex> lock(jobs_mu_);
    running_[info.job_id] = info.base_input_level;
  }
  Deliver(CompactionRecord(info, /*end=*/false));
}

void RLControllerHostImpl::OnCompactionCompleted(
    DB* /*db*/, const CompactionJobInfo& info) {
  {
    std::lock_guard<std::mutex> lock(jobs_mu_);
    running_.erase(info.job_id);
  }
  Deliver(CompactionRecord(info, /*end=*/true));
}

// ponytail: dlopen from libc, which needs glibc 2.34 or later (the node's
// toolchain floor already implies it); link ${CMAKE_DL_LIBS} for older ones.
Status RLControllerPlugin::Load(const std::string& path,
                                const std::string& config_path,
                                RLControllerHost* host,
                                std::unique_ptr<RLControllerPlugin>* result) {
  dlerror();
  void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    const char* reason = dlerror();
    return Status::IOError("cannot load the controller plugin " + path,
                           reason == nullptr ? "" : reason);
  }
  auto create = reinterpret_cast<RLControllerCreateFn>(
      dlsym(library, "rl_controller_create"));
  auto destroy = reinterpret_cast<RLControllerDestroyFn>(
      dlsym(library, "rl_controller_destroy"));
  if (create == nullptr || destroy == nullptr) {
    dlclose(library);
    return Status::InvalidArgument(
        path + " lacks rl_controller_create or rl_controller_destroy");
  }
  void* controller = create(host, config_path.c_str());
  result->reset(new RLControllerPlugin(library, destroy, controller));
  return Status::OK();
}

RLControllerPlugin::~RLControllerPlugin() {
  if (controller_ != nullptr) {
    destroy_(controller_);
  }
  dlclose(library_);
}

}  // namespace ROCKSDB_NAMESPACE
