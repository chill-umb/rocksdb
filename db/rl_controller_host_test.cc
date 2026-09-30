//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Research fork: the controller host (implementation plan WP2-WP4, §6.2).
// So far the host log's parts: the operation count, job records, H samples
// and stamps. The snapshot, Apply and plugin cases join with WP2.

#include "db/rl_controller_host.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "db/db_test_util.h"
#include "db/rl_read_counters.h"
#include "port/stack_trace.h"
#include "rocksdb/statistics.h"
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

// A stamp's `levels`: one [probe, pass, hit, seek] row per level.
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

  void OpenWithHostLog(Options options) {
    stats_ = CreateDBStatistics();
    options.statistics = stats_;
    ASSERT_OK(
        RLHostLog::Open(log_path_, stats_, options.num_levels, &host_log_));
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
  EXPECT_EQ(1u, Field(lines[0], "schema"));
  EXPECT_EQ(4u, Field(lines[0], "num_levels"));
  EXPECT_GT(Field(lines[0], "t_us"), 0u);
  EXPECT_GT(Field(lines[0], "wall_us"), 0u);
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
  OpenWithHostLog(BaseOptions());
  WriteKeys(0, 100);
  ASSERT_OK(Flush());
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
  };
  for (const auto& [kind, ticker] : pairs) {
    uint64_t sum = 0;
    for (const auto& row : levels) {
      sum += row[static_cast<int>(kind)];
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
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
