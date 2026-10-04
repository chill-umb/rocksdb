//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork: the controller host (implementation plan WP2-WP4, §6.2):
// the host log (operation count, job records, H samples, stamps) and the
// controller host (snapshot, counters, job callback, Apply, the drain, and
// loading a plugin). The plugin case loads the library named by
// RL_TEST_PLUGIN, which 01b_build_test_trees.sh sets to the Debug plugin it
// builds first.

#include "db/rl_controller_host.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "db/compaction/compaction_pressure_observer.h"
#include "db/db_test_util.h"
#include "db/rl_read_counters.h"
#include "port/stack_trace.h"
#include "rocksdb/statistics.h"
#include "test_util/sync_point.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {
namespace {

// The number after "key": in a host-log line (all its values are flat
// unsigned integers, except `levels`, `tickers`, `type`, `name`, `cause`).
uint64_t Field(const std::string& line, const std::string& key) {
  const std::string needle = "\"" + key + "\":";
  const size_t at = line.find(needle);
  EXPECT_NE(std::string::npos, at) << key << " missing from " << line;
  if (at == std::string::npos) {
    return 0;
  }
  return std::strtoull(line.c_str() + at + needle.size(), nullptr, 10);
}

// A stamp's `levels`: one [probe, pass, hit, seek, get reopen, iterator
// reopen, reopen nanos] row per level.
std::vector<std::array<uint64_t, RLReadCounters::kKinds>> Levels(
    const std::string& line) {
  std::vector<std::array<uint64_t, RLReadCounters::kKinds>> rows;
  const std::string needle = "\"levels\":[";
  size_t pos = line.find(needle);
  EXPECT_NE(std::string::npos, pos);
  if (pos == std::string::npos) {
    return rows;
  }
  pos += needle.size();
  while (pos < line.size() && line[pos] == '[') {
    std::array<uint64_t, RLReadCounters::kKinds> row{};
    const char* p = line.c_str() + pos + 1;
    for (auto& value : row) {
      char* end = nullptr;
      value = std::strtoull(p, &end, 10);
      p = end + 1;  // past ',' or the row's ']'
    }
    rows.push_back(row);
    pos = static_cast<size_t>(p - line.c_str());
    if (pos < line.size() && line[pos] == ',') {
      ++pos;
    }
  }
  return rows;
}

}  // namespace

class RLControllerHostTest : public DBTestBase {
 public:
  RLControllerHostTest()
      : DBTestBase("rl_controller_host_test", /*env_do_fsync=*/false),
        log_path_(dbname_ + ".host_log.jsonl") {}

  ~RLControllerHostTest() override { std::remove(log_path_.c_str()); }

  Options BaseOptions() {
    Options options;
    options.env = env_;
    options.create_if_missing = true;
    options.num_levels = 4;
    options.compaction_style = kCompactionStyleLevel;
    options.level_compaction_dynamic_level_bytes = false;
    options.disable_auto_compactions = true;
    options.compression = kNoCompression;
    options.write_buffer_size = 64 << 20;  // flush only when asked
    return options;
  }

  void OpenWithHostLog(Options options, uint64_t snapshot_stride = 0) {
    stats_ = CreateDBStatistics();
    options.statistics = stats_;
    ASSERT_OK(RLHostLog::Open(log_path_, stats_, options.num_levels,
                              &host_log_, snapshot_stride));
    options.listeners.push_back(host_log_);
    RLReadCounters::Reset();
    DestroyAndReopen(options);
  }

  void WriteKeys(int first, int last) {
    Random rnd(first);
    for (int i = first; i < last; ++i) {
      ASSERT_OK(Put(Key(i), rnd.RandomString(100)));
    }
  }

  // Lines of the host log, in file order, of one type ("" for all).
  std::vector<std::string> Records(const std::string& type = "") {
    std::ifstream in(log_path_);
    std::vector<std::string> lines;
    std::string line;
    const std::string prefix = "{\"type\":\"" + type + "\"";
    while (std::getline(in, line)) {
      if (type.empty() || line.compare(0, prefix.size(), prefix) == 0) {
        lines.push_back(line);
      }
    }
    return lines;
  }

  std::string JobRecord(const std::string& type, uint64_t job) {
    for (const auto& line : Records(type)) {
      if (Field(line, "job") == job) {
        return line;
      }
    }
    ADD_FAILURE() << "no " << type << " for job " << job;
    return "";
  }

  std::map<int, uint64_t> LevelBytes() {
    std::vector<LiveFileMetaData> files;
    db_->GetLiveFilesMetaData(&files);
    std::map<int, uint64_t> bytes;
    for (const auto& file : files) {
      bytes[file.level] += file.size;
    }
    return bytes;
  }

  uint64_t LiveSstBytes() {
    uint64_t bytes = 0;
    EXPECT_TRUE(db_->GetIntProperty(DB::Properties::kLiveSstFilesSize, &bytes));
    return bytes;
  }

  const std::string log_path_;
  std::shared_ptr<Statistics> stats_;
  std::shared_ptr<RLHostLog> host_log_;
};

TEST_F(RLControllerHostTest, OpenRefusesMissingStatisticsAndBadPath) {
  std::shared_ptr<RLHostLog> log;
  EXPECT_TRUE(RLHostLog::Open(log_path_, nullptr, 4, &log).IsInvalidArgument());
  EXPECT_TRUE(RLHostLog::Open(dbname_ + "/no/such/dir/host_log.jsonl",
                              CreateDBStatistics(), 4, &log)
                  .IsIOError());
  EXPECT_EQ(nullptr, log);
}

TEST_F(RLControllerHostTest, HeaderComesFirst) {
  OpenWithHostLog(BaseOptions());
  const auto lines = Records();
  ASSERT_FALSE(lines.empty());
  EXPECT_EQ(0u, lines[0].rfind("{\"type\":\"header\"", 0));
  // Schema 3 (D-23 §3(a), D-24 §2): the fg names and the snapshot stride.
  EXPECT_EQ(3u, Field(lines[0], "schema"));
  EXPECT_EQ(4u, Field(lines[0], "num_levels"));
  EXPECT_NE(std::string::npos,
            lines[0].find("\"fg\":[\"rocksdb.point.sst.probe\","
                          "\"rocksdb.bloom.filter.full.positive\","
                          "\"rocksdb.sorted.run.seek\","
                          "\"rocksdb.read.table.reopen\","
                          "\"rocksdb.number.db.next.found\","
                          "\"rocksdb.number.iter.skip\","
                          "\"rocksdb.number.keys.read\","
                          "\"rocksdb.number.db.seek\","
                          "\"rocksdb.number.keys.written\","
                          "\"rocksdb.read.table.reopen.nanos\","
                          "\"rocksdb.rl.scan.setup.nanos\"]"))
      << lines[0];
  EXPECT_EQ(0u, Field(lines[0], "stride"));
  EXPECT_GT(Field(lines[0], "t_us"), 0u);
  EXPECT_GT(Field(lines[0], "wall_us"), 0u);
}

// D-23 §3(a): a flush writes flush_begin, then flush_end and its h sample,
// with its file's bytes; job records and flush records carry the fg
// counters, which never decrease.
TEST_F(RLControllerHostTest, FlushRecordsAndForegroundCounters) {
  OpenWithHostLog(BaseOptions());
  WriteKeys(0, 50);
  ASSERT_OK(Flush());
  const auto lines = Records();
  int begin = -1, end = -1, h = -1;
  for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
    if (lines[i].rfind("{\"type\":\"flush_begin\"", 0) == 0) {
      begin = i;
    } else if (lines[i].rfind("{\"type\":\"flush_end\"", 0) == 0) {
      end = i;
    } else if (lines[i].rfind("{\"type\":\"h\"", 0) == 0 && end >= 0 &&
               h < 0) {
      h = i;
    }
  }
  ASSERT_GE(begin, 0);
  ASSERT_GT(end, begin);
  ASSERT_EQ(end + 1, h);
  EXPECT_EQ(Field(lines[begin], "job"), Field(lines[end], "job"));
  EXPECT_EQ(Field(lines[end], "job"), Field(lines[h], "job"));
  EXPECT_EQ(1u, Field(lines[end], "files"));
  EXPECT_EQ(LiveSstBytes(), Field(lines[end], "x"));
  EXPECT_EQ(50u, Field(lines[end], "op"));
  // fg: 11 entries; entry 8 is the Puts, equal to op here.
  for (int i : {begin, end}) {
    const size_t at = lines[i].find("\"fg\":[");
    ASSERT_NE(std::string::npos, at);
    const std::string list =
        lines[i].substr(at + 6, lines[i].find(']', at) - at - 6);
    EXPECT_EQ(10, std::count(list.begin(), list.end(), ',')) << list;
  }
  const std::string fg_end =
      lines[end].substr(lines[end].find("\"fg\":["));
  int commas = 0;
  size_t pos = 6;
  while (commas < 8) {
    pos = fg_end.find(',', pos) + 1;
    ++commas;
  }
  EXPECT_EQ(50u, std::strtoull(fg_end.c_str() + pos, nullptr, 10));
}

// D-23 §3(a): with a stride, a snap whenever the operation count has advanced
// by at least the stride; with none, no snap.
TEST_F(RLControllerHostTest, SnapshotsFollowTheStride) {
  OpenWithHostLog(BaseOptions(), /*snapshot_stride=*/100);
  EXPECT_EQ(100u, Field(Records("header")[0], "stride"));
  for (int batch = 0; batch < 10; ++batch) {
    WriteKeys(batch * 100, (batch + 1) * 100);
    // The poller wakes every millisecond.
    env_->SleepForMicroseconds(20000);
  }
  const auto snaps = Records("snap");
  ASSERT_GE(snaps.size(), 9u);
  uint64_t last = 0;
  for (const auto& line : snaps) {
    const uint64_t op = Field(line, "op");
    EXPECT_GE(op, last + 100) << line;
    EXPECT_NE(std::string::npos, line.find("\"fg\":[")) << line;
    last = op;
  }
  host_log_.reset();  // joins the poller
}

TEST_F(RLControllerHostTest, NoSnapshotsWithoutAStride) {
  OpenWithHostLog(BaseOptions());
  WriteKeys(0, 300);
  env_->SleepForMicroseconds(20000);
  EXPECT_TRUE(Records("snap").empty());
}

TEST_F(RLControllerHostTest, OperationCountIsKeysWrittenGetsAndSeeks) {
  OpenWithHostLog(BaseOptions());
  EXPECT_EQ(0u, RLOperationCount(*stats_));
  for (int i = 0; i < 10; ++i) {
    ASSERT_OK(Put(Key(i), "v"));
  }
  for (int i = 0; i < 7; ++i) {
    Get(Key(i * 3));  // found or not, each Get is one operation
  }
  std::unique_ptr<Iterator> it(db_->NewIterator(ReadOptions()));
  it->Seek(Key(3));
  it->SeekForPrev(Key(5));
  it->Seek(Key(8));
  for (int i = 0; i < 4 && it->Valid(); ++i) {
    it->Next();  // a scan's steps are not operations
  }
  ASSERT_OK(it->status());
  EXPECT_EQ(10u + 7u + 3u, RLOperationCount(*stats_));
}

TEST_F(RLControllerHostTest, MergeRecordCarriesSourceOverlapAndOutputBytes) {
  OpenWithHostLog(BaseOptions());
  WriteKeys(0, 100);
  ASSERT_OK(Flush());
  MoveFilesToLevel(2);
  WriteKeys(50, 150);
  ASSERT_OK(Flush());
  MoveFilesToLevel(1);
  ASSERT_EQ("0,1,1", FilesPerLevel());
  auto before = LevelBytes();

  ASSERT_OK(dbfull()->TEST_CompactRange(1, nullptr, nullptr, nullptr,
                                        /*disallow_trivial_move=*/true));
  ASSERT_EQ("0,0,1", FilesPerLevel());
  const uint64_t output = LevelBytes()[2];

  const std::string end = Records("job_end").back();
  const uint64_t job = Field(end, "job");
  const std::string begin = JobRecord("job_begin", job);
  for (const auto& line : {begin, end}) {
    SCOPED_TRACE(line);
    EXPECT_EQ(1u, Field(line, "start_level"));
    EXPECT_EQ(2u, Field(line, "output_level"));
    EXPECT_EQ(0u, Field(line, "trivial"));
    EXPECT_EQ(before[1], Field(line, "s"));
    EXPECT_EQ(before[2], Field(line, "o"));
    EXPECT_EQ(0u, Field(line, "due_since_us"));  // manual
  }
  EXPECT_EQ(output, Field(end, "x"));
  EXPECT_EQ(1u, Field(end, "ok"));
  EXPECT_LE(Field(begin, "t_us"), Field(end, "t_us"));
  EXPECT_LE(Field(begin, "op"), Field(end, "op"));
}

TEST_F(RLControllerHostTest, TrivialMoveIsFlaggedWithTheBytesItMoves) {
  OpenWithHostLog(BaseOptions());
  WriteKeys(0, 100);
  ASSERT_OK(Flush());
  const uint64_t moved = LevelBytes()[0];
  ASSERT_GT(moved, 0u);

  ASSERT_OK(dbfull()->TEST_CompactRange(0, nullptr, nullptr));
  ASSERT_EQ("0,1", FilesPerLevel());

  const std::string end = Records("job_end").back();
  const std::string begin = JobRecord("job_begin", Field(end, "job"));
  for (const auto& line : {begin, end}) {
    SCOPED_TRACE(line);
    EXPECT_EQ(0u, Field(line, "start_level"));
    EXPECT_EQ(1u, Field(line, "output_level"));
    EXPECT_EQ(1u, Field(line, "trivial"));
    EXPECT_EQ(moved, Field(line, "s"));
    EXPECT_EQ(0u, Field(line, "o"));
  }
  EXPECT_EQ(0u, Field(end, "x"));
  EXPECT_EQ(1u, Field(end, "ok"));
  EXPECT_EQ(moved, LevelBytes()[1]);
}

TEST_F(RLControllerHostTest, AutomaticJobRecordsWhenItsLevelBecameDue) {
  Options options = BaseOptions();
  options.disable_auto_compactions = false;
  options.level0_file_num_compaction_trigger = 2;
  OpenWithHostLog(options);
  WriteKeys(0, 50);
  ASSERT_OK(Flush());
  WriteKeys(25, 75);
  ASSERT_OK(Flush());  // L0 reaches its trigger: score 1, due
  ASSERT_OK(dbfull()->TEST_WaitForCompact());

  const uint64_t opened = Field(Records("header")[0], "t_us");
  std::vector<std::string> from_l0;
  for (const auto& line : Records("job_begin")) {
    if (Field(line, "start_level") == 0) {
      from_l0.push_back(line);
    }
  }
  ASSERT_FALSE(from_l0.empty());
  const std::string& begin = from_l0.front();
  SCOPED_TRACE(begin);
  EXPECT_GE(Field(begin, "due_since_us"), opened);
  EXPECT_LE(Field(begin, "due_since_us"), Field(begin, "t_us"));
}

TEST_F(RLControllerHostTest, HSamplesFollowInstallsAndEndAtLiveSstBytes) {
  Options options = BaseOptions();
  options.disable_auto_compactions = false;
  options.level0_file_num_compaction_trigger = 2;
  OpenWithHostLog(options);
  for (int round = 0; round < 4; ++round) {
    WriteKeys(round * 30, round * 30 + 60);  // overlapping runs
    ASSERT_OK(Flush());
  }
  ASSERT_OK(dbfull()->TEST_WaitForCompact());

  int flushes = 0;
  int compactions = 0;
  for (const auto& line : Records("h")) {
    if (line.find("\"cause\":\"flush\"") != std::string::npos) {
      ++flushes;
    } else if (line.find("\"cause\":\"compaction\"") != std::string::npos) {
      ++compactions;
    }
  }
  int completed = 0;
  for (const auto& line : Records("job_end")) {
    completed += Field(line, "ok") == 1 ? 1 : 0;
  }
  EXPECT_EQ(4, flushes);
  EXPECT_GT(completed, 0);
  EXPECT_EQ(completed, compactions);
  EXPECT_EQ(Records("job_begin").size(), Records("job_end").size());
  EXPECT_EQ(LiveSstBytes(), Field(Records("h").back(), "h"));

  uint64_t last_op = 0;
  for (const auto& line : Records()) {
    if (line.rfind("{\"type\":\"header\"", 0) == 0) {
      continue;
    }
    const uint64_t op = Field(line, "op");
    EXPECT_LE(last_op, op) << line;
    last_op = op;
  }
  EXPECT_EQ(RLOperationCount(*stats_), last_op);
}

TEST_F(RLControllerHostTest, StampCarriesCountersTickersAndStall) {
  // A table cache of one table (max_open_files 11, set back by a sync point
  // after SanitizeOptions raises it to 20; Debug builds only), closed before
  // the reads, so that the stamp's reopen columns are not all zero.
  Options options = BaseOptions();
  options.max_open_files = 11;
  SyncPoint::GetInstance()->SetCallBack(
      "SanitizeOptions::AfterChangeMaxOpenFiles",
      [](void* arg) { *static_cast<int*>(arg) = 11; });
  SyncPoint::GetInstance()->EnableProcessing();
  OpenWithHostLog(options);
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  WriteKeys(0, 100);
  ASSERT_OK(Flush());
  dbfull()->TEST_table_cache()->EraseUnRefEntries();
  for (int i = 0; i < 150; i += 7) {
    Get(Key(i));
  }
  {
    std::unique_ptr<Iterator> it(db_->NewIterator(ReadOptions()));
    it->Seek(Key(40));
    ASSERT_OK(it->status());
  }
  ASSERT_OK(host_log_->Stamp(db_.get(), "check", ",\"ok\":1"));

  const std::string line = Records("stamp").back();
  EXPECT_NE(std::string::npos, line.find("\"name\":\"check\""));
  EXPECT_EQ(1u, Field(line, "ok"));
  EXPECT_EQ(RLOperationCount(*stats_), Field(line, "op"));
  EXPECT_EQ(LiveSstBytes(), Field(line, "h"));
  std::map<std::string, std::string> db_stats;
  ASSERT_TRUE(db_->GetMapProperty(DB::Properties::kDBStats, &db_stats));
  EXPECT_EQ(std::strtoull(db_stats["db.user_write_stall_micros"].c_str(),
                          nullptr, 10),
            Field(line, "stall_micros"));

  const auto levels = Levels(line);
  ASSERT_EQ(4u, levels.size());
  const std::pair<RLReadCounter, Tickers> pairs[] = {
      {RLReadCounter::kProbe, POINT_SST_PROBE},
      {RLReadCounter::kFilterPass, BLOOM_FILTER_FULL_POSITIVE},
      {RLReadCounter::kFilterHit, BLOOM_FILTER_FULL_TRUE_POSITIVE},
      {RLReadCounter::kSeek, SORTED_RUN_SEEK},
      {RLReadCounter::kGetReopen, READ_TABLE_REOPEN},
      {RLReadCounter::kReopenNanos, READ_TABLE_REOPEN_NANOS},
  };
  for (const auto& [kind, ticker] : pairs) {
    uint64_t sum = 0;
    for (const auto& row : levels) {
      sum += row[static_cast<int>(kind)];
      // One ticker counts both kinds of reopen.
      if (kind == RLReadCounter::kGetReopen) {
        sum += row[static_cast<int>(RLReadCounter::kIterReopen)];
      }
    }
    std::string name;
    for (const auto& entry : TickersNameMap) {
      if (entry.first == ticker) {
        name = entry.second;
      }
    }
    SCOPED_TRACE(name);
    EXPECT_EQ(stats_->getTickerCount(ticker), Field(line, name));
    EXPECT_EQ(Field(line, name), sum);
  }
  EXPECT_GT(levels[0][static_cast<int>(RLReadCounter::kProbe)], 0u);
  EXPECT_GT(levels[0][static_cast<int>(RLReadCounter::kSeek)], 0u);
  EXPECT_GT(levels[0][static_cast<int>(RLReadCounter::kGetReopen)], 0u);
  EXPECT_GT(levels[0][static_cast<int>(RLReadCounter::kReopenNanos)], 0u);
}

// The controller host (plan WP2), registered as a listener when the DB
// opens and attached afterwards, as db_bench does.
class RLControllerHostImplTest : public RLControllerHostTest {
 public:
  void OpenWithHost(Options options) {
    stats_ = CreateDBStatistics();
    options.statistics = stats_;
    host_ = std::make_shared<RLControllerHostImpl>(stats_);
    options.listeners.push_back(host_);
    RLReadCounters::Reset();
    DestroyAndReopen(options);
    ASSERT_OK(host_->Attach(db_.get()));
  }

  // A tree with files at L0, L1 and L2.
  void BuildTree() {
    WriteKeys(0, 100);
    ASSERT_OK(Flush());
    MoveFilesToLevel(2);
    WriteKeys(50, 150);
    ASSERT_OK(Flush());
    MoveFilesToLevel(1);
    WriteKeys(200, 260);
    ASSERT_OK(Flush());
    ASSERT_EQ("1,1,1", FilesPerLevel());
  }

  std::shared_ptr<RLControllerHostImpl> host_;
};

TEST_F(RLControllerHostImplTest, AttachNeedsStatisticsAndRunsOnce) {
  RLControllerHostImpl without(nullptr);
  EXPECT_TRUE(without.Attach(db_.get()).IsInvalidArgument());
  OpenWithHost(BaseOptions());
  EXPECT_TRUE(host_->Attach(db_.get()).IsInvalidArgument());
}

TEST_F(RLControllerHostImplTest, OptionsAreTheRunsOwn) {
  Options options = BaseOptions();
  options.max_bytes_for_level_multiplier = 6;
  options.max_bytes_for_level_base = 16 << 20;
  options.level0_file_num_compaction_trigger = 3;
  options.level0_slowdown_writes_trigger = 17;
  options.level0_stop_writes_trigger = 30;
  OpenWithHost(options);
  const RLHostOptions o = host_->Options();
  EXPECT_EQ(4, o.num_levels);
  EXPECT_EQ(6.0, o.level_multiplier);
  EXPECT_EQ(std::vector<int>(4, 1), o.level_multiplier_additional);
  EXPECT_EQ(16u << 20, o.base_level_bytes);
  EXPECT_EQ(64u << 20, o.write_buffer_size);
  EXPECT_EQ(3, o.l0_trigger);
  EXPECT_EQ(17, o.l0_slowdown_trigger);
  EXPECT_EQ(30, o.l0_stop_trigger);
}

TEST_F(RLControllerHostImplTest, SnapshotDescribesTheTree) {
  OpenWithHost(BaseOptions());
  BuildTree();
  ASSERT_OK(Put(Key(1000), "in the memtable"));
  const auto bytes = LevelBytes();
  const auto snapshot = host_->Snapshot();
  ASSERT_EQ(4u, snapshot->levels.size());
  for (int level = 0; level < 4; ++level) {
    SCOPED_TRACE(level);
    const RLLevelSnapshot& l = snapshot->levels[level];
    EXPECT_EQ(bytes.count(level) ? bytes.at(level) : 0u, l.bytes);
    EXPECT_EQ(NumTableFilesAtLevel(level), l.num_files);
    EXPECT_EQ(0u, l.bytes_compacting);
    EXPECT_EQ(0, l.num_files_compacting);
  }
  // Static ladder: C_1 = max_bytes_for_level_base, C_i = C_1 T^(i-1).
  const uint64_t base = Options().max_bytes_for_level_base;
  EXPECT_EQ(base, snapshot->levels[1].target_bytes);
  EXPECT_EQ(base * 10, snapshot->levels[2].target_bytes);
  // L0's score is its file count over the trigger.
  EXPECT_EQ(1.0 / Options().level0_file_num_compaction_trigger,
            snapshot->levels[0].score);
  EXPECT_EQ(0u, snapshot->levels[0].due_since_micros);
  EXPECT_EQ(3u, snapshot->score_order.size());
  EXPECT_EQ(0, snapshot->score_order[0]);  // the highest score
  EXPECT_EQ(LiveSstBytes(), snapshot->live_sst_bytes);
  uint64_t pending = 0;
  ASSERT_TRUE(db_->GetIntProperty(
      DB::Properties::kEstimatePendingCompactionBytes, &pending));
  EXPECT_EQ(pending, snapshot->pending_compaction_bytes);
  EXPECT_EQ(std::vector<double>(4, 1.0), snapshot->level_target_multipliers);
  EXPECT_EQ(Options().level0_file_num_compaction_trigger, snapshot->l0_trigger);
  EXPECT_EQ(-1, snapshot->running_start_level);
  uint64_t memtable = 0;
  ASSERT_TRUE(
      db_->GetIntProperty(DB::Properties::kCurSizeActiveMemTable, &memtable));
  EXPECT_EQ(memtable, snapshot->active_memtable_bytes);
  EXPECT_GT(snapshot->active_memtable_bytes, 0u);
  EXPECT_FALSE(snapshot->write_stopped);
  EXPECT_GT(snapshot->generation, 0u);

  // A later install publishes a later snapshot.
  ASSERT_OK(Flush());
  EXPECT_GT(host_->Snapshot()->generation, snapshot->generation);
  EXPECT_EQ(2, host_->Snapshot()->levels[0].num_files);
}

// A tree whose highest score is L1's, over its target: the order is by score,
// not by level; L1 is due since the install that made it due; the pending
// estimate is the property's and not zero.
TEST_F(RLControllerHostImplTest, SnapshotRanksByScoreAndMarksDueLevels) {
  Options options = BaseOptions();
  // L1 (800 keys, ~90 KiB) is over its target; L0 (one ~7 KiB file) scores
  // by its file count, 1/4; L2 (100 keys) is far below its 640 KiB.
  options.max_bytes_for_level_base = 64 << 10;
  OpenWithHost(options);
  const uint64_t start = CompactionPressureObserver::NowMicros();
  WriteKeys(0, 100);
  ASSERT_OK(Flush());
  MoveFilesToLevel(2);
  WriteKeys(1000, 1800);
  ASSERT_OK(Flush());
  MoveFilesToLevel(1);
  WriteKeys(200, 260);
  ASSERT_OK(Flush());
  ASSERT_EQ("1,1,1", FilesPerLevel());

  const auto snapshot = host_->Snapshot();
  ASSERT_EQ(4u, snapshot->levels.size());
  EXPECT_GT(snapshot->levels[1].score, 1.0);
  EXPECT_EQ(0.25, snapshot->levels[0].score);
  EXPECT_LT(snapshot->levels[2].score, 0.25);
  EXPECT_EQ(std::vector<int>({1, 0, 2}), snapshot->score_order);
  EXPECT_GE(snapshot->levels[1].due_since_micros, start);
  EXPECT_LE(snapshot->levels[1].due_since_micros, snapshot->t_micros);
  EXPECT_EQ(0u, snapshot->levels[0].due_since_micros);
  EXPECT_EQ(0u, snapshot->levels[2].due_since_micros);
  uint64_t pending = 0;
  ASSERT_TRUE(db_->GetIntProperty(
      DB::Properties::kEstimatePendingCompactionBytes, &pending));
  EXPECT_GT(pending, 0u);
  EXPECT_EQ(pending, snapshot->pending_compaction_bytes);
}

TEST_F(RLControllerHostImplTest, CountersAreTheTickers) {
  OpenWithHost(BaseOptions());
  BuildTree();
  for (int i = 0; i < 300; i += 7) {
    Get(Key(i));
  }
  {
    std::unique_ptr<Iterator> it(db_->NewIterator(ReadOptions()));
    it->Seek(Key(40));
    ASSERT_OK(it->status());
  }
  const RLOpCounts ops = host_->OpCounts();
  EXPECT_EQ(RLOperationCount(*stats_), ops.total());
  EXPECT_EQ(RLOperationCount(*stats_), host_->OpCount());
  EXPECT_EQ(stats_->getTickerCount(NUMBER_KEYS_WRITTEN), ops.keys_written);
  EXPECT_EQ(stats_->getTickerCount(NUMBER_KEYS_READ), ops.keys_read);
  EXPECT_EQ(stats_->getTickerCount(NUMBER_DB_SEEK), ops.seeks);
  EXPECT_EQ(stats_->getTickerCount(BYTES_WRITTEN), ops.bytes_written);

  std::vector<RLLevelReadCounts> reads;
  host_->ReadCounters(&reads);
  ASSERT_EQ(4u, reads.size());
  RLLevelReadCounts sum;
  for (const auto& level : reads) {
    sum.probes += level.probes;
    sum.filter_passes += level.filter_passes;
    sum.filter_hits += level.filter_hits;
    sum.seeks += level.seeks;
  }
  EXPECT_EQ(stats_->getTickerCount(POINT_SST_PROBE), sum.probes);
  EXPECT_EQ(stats_->getTickerCount(BLOOM_FILTER_FULL_POSITIVE),
            sum.filter_passes);
  EXPECT_EQ(stats_->getTickerCount(BLOOM_FILTER_FULL_TRUE_POSITIVE),
            sum.filter_hits);
  EXPECT_EQ(stats_->getTickerCount(SORTED_RUN_SEEK), sum.seeks);
  EXPECT_GT(sum.probes, 0u);
  EXPECT_GT(sum.seeks, 0u);
}

TEST_F(RLControllerHostImplTest, JobRecordsReachTheCallbackUntilRemoved) {
  OpenWithHost(BaseOptions());
  std::vector<RLJobRecord> records;
  std::mutex mu;
  host_->SetJobCallback([&](const RLJobRecord& record) {
    std::lock_guard<std::mutex> lock(mu);
    records.push_back(record);
  });
  // Read inside the merge: the picker's score recompute has published its
  // inputs as being compacted.
  std::shared_ptr<const RLTreeSnapshot> inside_job;
  SyncPoint::GetInstance()->SetCallBack(
      "CompactionJob::Run():Start",
      [&](void* /*arg*/) { inside_job = host_->Snapshot(); });
  SyncPoint::GetInstance()->EnableProcessing();

  BuildTree();  // three flushes and three trivial moves
  const uint64_t l0_file = LevelBytes()[0];
  auto before = LevelBytes();
  ASSERT_OK(dbfull()->TEST_CompactRange(1, nullptr, nullptr, nullptr,
                                        /*disallow_trivial_move=*/true));
  const uint64_t output = LevelBytes()[2];
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_NE(nullptr, inside_job);
  EXPECT_EQ(1, inside_job->running_start_level);
  EXPECT_EQ(before[1], inside_job->levels[1].bytes_compacting);
  EXPECT_EQ(before[2], inside_job->levels[2].bytes_compacting);
  EXPECT_EQ(1, inside_job->levels[1].num_files_compacting);
  EXPECT_EQ(1, inside_job->levels[2].num_files_compacting);
  EXPECT_EQ(0u, inside_job->levels[0].bytes_compacting);
  const auto after_job = host_->Snapshot();
  EXPECT_EQ(-1, after_job->running_start_level);
  EXPECT_EQ(0u, after_job->levels[2].bytes_compacting);

  // D-23 §3(a): a flush sends a begin record too.
  std::vector<RLJobRecord> flush_begins, flushes, begins, ends;
  {
    std::lock_guard<std::mutex> lock(mu);
    for (const auto& r : records) {
      (r.kind == RLJobRecord::Kind::kFlushBegin        ? flush_begins
       : r.kind == RLJobRecord::Kind::kFlushEnd        ? flushes
       : r.kind == RLJobRecord::Kind::kCompactionBegin ? begins
                                                       : ends)
          .push_back(r);
    }
  }
  ASSERT_EQ(3u, flush_begins.size());
  ASSERT_EQ(3u, flushes.size());
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(flush_begins[i].job_id, flushes[i].job_id);
    EXPECT_LE(flush_begins[i].op, flushes[i].op);
    EXPECT_LE(flush_begins[i].steps.puts, flushes[i].steps.puts);
  }
  EXPECT_EQ(l0_file, flushes.back().x);  // the flushed file's bytes
  for (const auto& r : flushes) {
    EXPECT_EQ(-1, r.start_level);
    EXPECT_EQ(0, r.output_level);
    EXPECT_GT(r.x, 0u);
    EXPECT_TRUE(r.ok);
  }
  ASSERT_EQ(4u, begins.size());
  ASSERT_EQ(4u, ends.size());
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(begins[i].trivial);
    EXPECT_TRUE(ends[i].trivial);
    EXPECT_EQ(begins[i].job_id, ends[i].job_id);
  }
  const RLJobRecord& merge = ends[3];
  EXPECT_EQ(begins[3].job_id, merge.job_id);
  EXPECT_FALSE(merge.trivial);
  EXPECT_EQ(1, merge.start_level);
  EXPECT_EQ(2, merge.output_level);
  EXPECT_EQ(before[1], merge.s);
  EXPECT_EQ(before[2], merge.o);
  EXPECT_EQ(output, merge.x);
  EXPECT_TRUE(merge.ok);
  EXPECT_EQ(0u, merge.due_since_micros);  // manual
  EXPECT_LE(begins[3].op, merge.op);
  EXPECT_LE(begins[3].t_micros, merge.t_micros);
  EXPECT_EQ(RLOperationCount(*stats_), merge.op);
  // The step counters, read at the same event as op.
  EXPECT_EQ(merge.op,
            merge.steps.puts + merge.steps.gets + merge.steps.scans);
  EXPECT_EQ(stats_->getTickerCount(NUMBER_KEYS_WRITTEN), merge.steps.puts);

  host_->SetJobCallback(nullptr);
  ASSERT_OK(Put(Key(500), "v"));
  ASSERT_OK(Flush());
  std::lock_guard<std::mutex> lock(mu);
  EXPECT_EQ(14u, records.size());
}

// D-23 §3(a): StepCounts() is the fg tickers, read now; the per-level
// counters carry the hidden steps.
TEST_F(RLControllerHostImplTest, StepCountsAreTheForegroundTickers) {
  OpenWithHost(BaseOptions());
  BuildTree();
  ASSERT_EQ("v", Get(Key(0)).substr(0, 1));
  {
    std::unique_ptr<Iterator> it(db_->NewIterator(ReadOptions()));
    int n = 0;
    for (it->SeekToFirst(); it->Valid() && n < 10; it->Next()) {
      ++n;
    }
  }
  const RLStepCounts steps = host_->StepCounts();
  EXPECT_EQ(stats_->getTickerCount(POINT_SST_PROBE), steps.probes);
  EXPECT_EQ(stats_->getTickerCount(BLOOM_FILTER_FULL_POSITIVE),
            steps.block_probes);
  EXPECT_EQ(stats_->getTickerCount(SORTED_RUN_SEEK), steps.run_seeks);
  EXPECT_EQ(stats_->getTickerCount(READ_TABLE_REOPEN), steps.reopens);
  EXPECT_EQ(stats_->getTickerCount(NUMBER_DB_NEXT_FOUND), steps.nexts_found);
  EXPECT_EQ(stats_->getTickerCount(NUMBER_ITER_SKIP), steps.iter_skips);
  EXPECT_EQ(stats_->getTickerCount(NUMBER_KEYS_READ), steps.gets);
  EXPECT_EQ(stats_->getTickerCount(NUMBER_DB_SEEK), steps.scans);
  EXPECT_EQ(stats_->getTickerCount(NUMBER_KEYS_WRITTEN), steps.puts);
  EXPECT_GT(steps.puts, 0u);
  EXPECT_GT(steps.nexts_found, 0u);
  std::vector<RLLevelReadCounts> counts;
  host_->ReadCounters(&counts);
  uint64_t hidden = 0;
  for (const auto& level : counts) {
    hidden += level.hidden_steps;
  }
  EXPECT_LE(hidden, steps.iter_skips);
}

TEST_F(RLControllerHostImplTest, ApplyIsOneSetOptionsSeenInTheNextSnapshot) {
  OpenWithHost(BaseOptions());
  BuildTree();
  const uint64_t generation = host_->Snapshot()->generation;
  // 4/3 needs all 17 digits: six (std::to_string) would not read back as it.
  const double m1 = 4.0 / 3;
  std::string error;
  ASSERT_TRUE(host_->Apply({1.0, m1, 1.5, 2.0}, 3, &error)) << error;
  const Options in_effect = db_->GetOptions();
  EXPECT_EQ(std::vector<double>({1.0, m1, 1.5, 2.0}),
            in_effect.level_target_multipliers);
  EXPECT_EQ(3, in_effect.level0_file_num_compaction_trigger);

  const auto snapshot = host_->Snapshot();
  EXPECT_GT(snapshot->generation, generation);
  EXPECT_EQ(in_effect.level_target_multipliers,
            snapshot->level_target_multipliers);
  EXPECT_EQ(3, snapshot->l0_trigger);
  const uint64_t base = Options().max_bytes_for_level_base;
  EXPECT_EQ(static_cast<uint64_t>(static_cast<long double>(base) * m1),
            snapshot->levels[1].target_bytes);
  EXPECT_EQ(1.0 / 3, snapshot->levels[0].score);

  // Refused, with the options in effect unchanged: a wrong length, and a
  // vector the fork's validation rejects (entry 0 must be 1).
  EXPECT_FALSE(host_->Apply({1.0, 1.0}, 4, &error));
  EXPECT_NE(std::string::npos, error.find("2 multipliers for 4 levels"));
  error.clear();
  EXPECT_FALSE(host_->Apply({0.9, 1.0, 1.0, 1.0}, 4, &error));
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(in_effect.level_target_multipliers,
            db_->GetOptions().level_target_multipliers);
  EXPECT_FALSE(host_->Draining());
}

TEST_F(RLControllerHostImplTest, TheDrainRestoresNativeAndRefusesApply) {
  OpenWithHost(BaseOptions());
  std::string error;
  ASSERT_TRUE(host_->Apply({1.0, 1.5, 1.0, 1.0}, 6, &error)) << error;
  ASSERT_OK(host_->BeginDrain());
  EXPECT_TRUE(host_->Draining());
  const Options in_effect = db_->GetOptions();
  EXPECT_EQ(std::vector<double>(4, 1.0), in_effect.level_target_multipliers);
  EXPECT_EQ(Options().level0_file_num_compaction_trigger,
            in_effect.level0_file_num_compaction_trigger);
  EXPECT_EQ(std::vector<double>(4, 1.0),
            host_->Snapshot()->level_target_multipliers);
  EXPECT_FALSE(host_->Apply({1.0, 1.5, 1.0, 1.0}, 6, &error));
  EXPECT_NE(std::string::npos, error.find("drain"));
  EXPECT_EQ(std::vector<double>(4, 1.0),
            db_->GetOptions().level_target_multipliers);
}

TEST_F(RLControllerHostImplTest, ANativeDrainMakesNoSetOptionsCall) {
  OpenWithHost(BaseOptions());
  // Every SetOptions writes a new OPTIONS file (A-Impl-5).
  const uint64_t options_file =
      dbfull()->GetVersionSet()->options_file_number();
  std::string error;
  ASSERT_TRUE(host_->Apply({1.0, 1.0, 1.0, 1.0}, 4, &error)) << error;
  const uint64_t after_apply = dbfull()->GetVersionSet()->options_file_number();
  EXPECT_GT(after_apply, options_file);
  ASSERT_OK(host_->BeginDrain());
  EXPECT_FALSE(host_->Apply({1.0, 1.5, 1.0, 1.0}, 6, &error));
  EXPECT_EQ(after_apply, dbfull()->GetVersionSet()->options_file_number());
}

TEST_F(RLControllerHostImplTest, APluginLoadsRunsHoldOnlyAndUnloads) {
  // Fails rather than skips, so tier 2 cannot pass without loading a plugin.
  const char* plugin = std::getenv("RL_TEST_PLUGIN");
  ASSERT_TRUE(plugin != nullptr && *plugin != '\0')
      << "RL_TEST_PLUGIN is not set (01b_build_test_trees.sh sets it to the "
         "Debug librl_controller.so)";
  std::unique_ptr<RLControllerPlugin> loaded;
  EXPECT_TRUE(RLControllerPlugin::Load(dbname_ + "/no_such_plugin.so", "",
                                       nullptr, &loaded)
                  .IsIOError());
  // A library without the entry points.
  EXPECT_TRUE(RLControllerPlugin::Load("libc.so.6", "", nullptr, &loaded)
                  .IsInvalidArgument());
  EXPECT_EQ(nullptr, loaded);

  OpenWithHost(BaseOptions());
  const std::string config = dbname_ + ".plugin.json";
  const std::string decisions = dbname_ + ".decisions.jsonl";
  const std::string transitions = dbname_ + ".transitions.jsonl";
  {
    std::ofstream out(config);
    out << "{\"mode\":\"hold-only\",\"m_min\":0.5,\"m_max\":2,\"k0_min\":2,"
           "\"k0_cap\":8,\"epsilon\":0.1,\"phi_min\":0.55,\"alpha\":1.5,"
           "\"kappa_d\":0.25,\"kappa_a\":1,\"k\":10,\"b_max\":1,"
           "\"beta_w\":1,\"beta_r\":10,\"beta_s\":1,\"c_w\":1e-9,"
           "\"c_f\":1e-9,\"c_blk\":1e-6,\"c_sk\":1e-6,\"c_open\":1e-5,"
           "\"c_s\":1e-17,"
           "\"q_bar\":1000,\"setoptions_min_interval_ms\":100,"
           "\"decision_log\":\""
        << decisions << "\",\"transition_log\":\"" << transitions << "\"}";
  }
  // Every SetOptions writes a new OPTIONS file (A-Impl-5).
  const uint64_t options_file =
      dbfull()->GetVersionSet()->options_file_number();
  ASSERT_OK(RLControllerPlugin::Load(plugin, config, host_.get(), &loaded));
  ASSERT_TRUE(loaded->created());
  BuildTree();
  for (int i = 0; i < 200; ++i) {
    Get(Key(i));
  }
  ASSERT_OK(host_->BeginDrain());
  loaded.reset();  // destroys the controller, then unloads the library

  std::ifstream in(decisions);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) {
    lines.push_back(line);
  }
  ASSERT_GE(lines.size(), 2u);
  EXPECT_EQ(0u, lines.front().rfind("{\"type\":\"start\"", 0));
  EXPECT_EQ(0u, lines.back().rfind("{\"type\":\"stop\"", 0));
  for (const auto& line : lines) {
    EXPECT_EQ(std::string::npos, line.find("\"type\":\"apply\"")) << line;
    EXPECT_EQ(std::string::npos, line.find("\"type\":\"fallback\"")) << line;
  }
  // Hold-only and a native drain changed nothing: the option is still the
  // empty vector the DB opened with, and no OPTIONS file was written.
  EXPECT_TRUE(db_->GetOptions().level_target_multipliers.empty());
  EXPECT_EQ(options_file, dbfull()->GetVersionSet()->options_file_number());

  // A config the plugin refuses (epsilon must be below 1) still makes a
  // controller, which falls back (A-Impl-8) and says so in its log.
  {
    std::ifstream good(config);
    std::stringstream text;
    text << good.rdbuf();
    std::string bad = text.str();
    bad.replace(bad.find("\"epsilon\":0.1"), 13, "\"epsilon\":2.0");
    std::ofstream(config) << bad;
  }
  ASSERT_OK(RLControllerPlugin::Load(plugin, config, host_.get(), &loaded));
  ASSERT_TRUE(loaded->created());
  loaded.reset();
  std::ifstream fallback_log(decisions);
  bool fell_back = false;
  for (std::string line; std::getline(fallback_log, line);) {
    fell_back = fell_back || line.rfind("{\"type\":\"fallback\"", 0) == 0;
  }
  EXPECT_TRUE(fell_back);
  std::remove(config.c_str());
  std::remove(decisions.c_str());
  std::remove(transitions.c_str());
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
