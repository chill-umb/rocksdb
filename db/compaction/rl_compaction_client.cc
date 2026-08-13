#include "db/compaction/rl_compaction_client.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ROCKSDB_NAMESPACE {

namespace {

// Read the socket path from the environment, with a safe default.
std::string SocketPathFromEnv() {
  const char* env = std::getenv("RL_COMPACTION_SOCKET_PATH");
  if (env) return std::string(env);
  const char* home = std::getenv("HOME");
  return home ? std::string(home) + "/lsm_dqn/rl_compaction.sock"
              : std::string("/tmp/rl_compaction.sock");
}

int SocketTimeoutMsFromEnv() {
  const char* env = std::getenv("RL_COMPACTION_SOCKET_TIMEOUT_MS");
  if (!env) return 100;
  int timeout_ms = std::atoi(env);
  return timeout_ms > 0 ? timeout_ms : 100;
}

// Minimal JSON formatting (no external dependency).
void AppendUInt64Array(std::ostringstream& os,
                       const std::vector<uint64_t>& values) {
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) os << ",";
    os << values[i];
  }
  os << "]";
}

void AppendCandidateState(std::ostringstream& os,
                          const RLCandidateState& c) {
  os << "{\"snapshot_epoch\":" << c.snapshot_epoch
     << ",\"source_file_number\":" << c.source_file_number
     << ",\"source_level\":" << c.source_level
     << ",\"output_level\":" << c.output_level
     << ",\"source_bytes\":" << c.source_bytes
     << ",\"expanded_source_bytes\":" << c.expanded_source_bytes
     << ",\"expanded_source_files\":";
  AppendUInt64Array(os, c.expanded_source_files);
  os << ",\"overlap_files\":";
  AppendUInt64Array(os, c.overlap_files);
  os << ",\"overlap_bytes\":" << c.overlap_bytes
     << ",\"estimated_read_bytes\":" << c.estimated_read_bytes
     << ",\"estimated_write_bytes\":" << c.estimated_write_bytes
     << ",\"overlap_ratio\":" << c.overlap_ratio
     << ",\"num_entries\":" << c.num_entries
     << ",\"num_deletions\":" << c.num_deletions
     << ",\"compensated_size\":" << c.compensated_size
     << ",\"projected_source_fullness\":"
     << c.projected_source_fullness
     << ",\"projected_output_fullness\":"
     << c.projected_output_fullness
     << ",\"empties_source_level\":"
     << (c.empties_source_level ? "true" : "false")
     << ",\"priority_rank\":" << c.priority_rank
     << ",\"conflict\":" << (c.conflict ? "true" : "false") << "}";
}

void AppendLevelState(std::ostringstream& os, const RLLevelState& l) {
  os << "{\"level\":" << l.level << ",\"files\":" << l.files
     << ",\"bytes\":" << l.bytes << ",\"score\":" << l.score
     << ",\"target_bytes\":" << l.target_bytes
     << ",\"next_level_files\":" << l.next_level_files
     << ",\"next_level_bytes\":" << l.next_level_bytes
     << ",\"next_level_score\":" << l.next_level_score
     << ",\"next_level_target_bytes\":" << l.next_level_target_bytes
     << ",\"overlap_bytes\":" << l.overlap_bytes
     << ",\"bytes_in\":" << l.bytes_in
     << ",\"bytes_read_out\":" << l.bytes_read_out
     << ",\"bytes_written_out\":" << l.bytes_written_out
     << ",\"compactions_from\":" << l.compactions_from
     << ",\"compactions_scheduled\":" << l.compactions_scheduled
     << ",\"compactions_forced\":" << l.compactions_forced
     << ",\"prev_action_executed\":" << l.prev_action_executed
     << ",\"prev_action_overridden\":"
     << (l.prev_action_overridden ? "true" : "false")
     << ",\"prev_compaction_picked\":"
     << (l.prev_compaction_picked ? "true" : "false")
     << ",\"prev_decision_id\":" << l.prev_decision_id
     << ",\"prev_snapshot_epoch\":" << l.prev_snapshot_epoch
     << ",\"prev_candidate_file_number\":"
     << l.prev_candidate_file_number
     << ",\"prev_scheduling_result\":" << l.prev_scheduling_result
     << ",\"prev_completion_result\":" << l.prev_completion_result
     << ",\"prev_completed_decision_id\":"
     << l.prev_completed_decision_id
     << ",\"prev_completed_candidate_file_number\":"
     << l.prev_completed_candidate_file_number
     << ",\"prev_override_reason\":" << l.prev_override_reason
     << ",\"prev_transition_valid\":"
     << (l.prev_transition_valid ? "true" : "false")
     << ",\"defer_count\":" << l.defer_count
     << ",\"default_needed\":" << (l.default_needed ? "true" : "false")
     << ",\"is_last\":" << (l.is_last ? "true" : "false")
     << ",\"candidates\":[";
  for (size_t i = 0; i < l.candidates.size(); ++i) {
    if (i > 0) os << ",";
    AppendCandidateState(os, l.candidates[i]);
  }
  os << "]}";
}

std::string FormatStateV2(const RLStateV2& s) {
  std::ostringstream os;
  os.precision(6);
  os << std::fixed;
  os << "{\"version\":" << s.protocol_version
     << ",\"snapshot_epoch\":" << s.snapshot_epoch
     << ",\"pending_compaction_bytes\":" << s.pending_compaction_bytes
     << ",\"flushed_bytes\":" << s.flushed_bytes
     << ",\"compaction_bytes_read\":" << s.compaction_bytes_read
     << ",\"compaction_bytes_written\":" << s.compaction_bytes_written
     << ",\"compactions_completed\":" << s.compactions_completed
     << ",\"stall_count\":" << s.stall_count
     << ",\"stop_count\":" << s.stop_count
     << ",\"fallback_count\":" << s.fallback_count
     << ",\"l0_compaction_trigger\":" << s.l0_compaction_trigger
     << ",\"l0_slowdown_trigger\":" << s.l0_slowdown_trigger
     << ",\"l0_stop_trigger\":" << s.l0_stop_trigger
     << ",\"l0_delay_trigger_count\":" << s.l0_delay_trigger_count
     << ",\"interval_micros\":" << s.interval_micros
     << ",\"keys_read\":" << s.keys_read << ",\"seeks\":" << s.seeks
     << ",\"get_hit_l0\":" << s.get_hit_l0
     << ",\"get_hit_l1\":" << s.get_hit_l1
     << ",\"get_hit_l2_and_up\":" << s.get_hit_l2_and_up
     << ",\"bloom_useful\":" << s.bloom_useful
     << ",\"non_last_level_read_count\":" << s.non_last_level_read_count
     << ",\"last_level_read_count\":" << s.last_level_read_count
     << ",\"user_logical_write_bytes\":" << s.user_logical_write_bytes
     << ",\"point_sst_probes\":" << s.point_sst_probes
     << ",\"scan_returned_entries\":" << s.scan_returned_entries
     << ",\"scan_internal_skipped\":" << s.scan_internal_skipped
     << ",\"scan_sorted_run_seeks\":" << s.scan_sorted_run_seeks
     << ",\"physical_sst_bytes\":" << s.physical_sst_bytes
     << ",\"live_logical_bytes\":" << s.live_logical_bytes
     << ",\"stall_duration_micros\":" << s.stall_duration_micros
     << ",\"get_latency_count\":" << s.get_latency_count
     << ",\"get_latency_avg_ns\":" << s.get_latency_avg_ns
     << ",\"get_latency_p95_ns\":" << s.get_latency_p95_ns
     << ",\"scan_latency_count\":" << s.scan_latency_count
     << ",\"scan_latency_avg_ns\":" << s.scan_latency_avg_ns
     << ",\"scan_latency_p95_ns\":" << s.scan_latency_p95_ns
     << ",\"write_latency_count\":" << s.write_latency_count
     << ",\"write_latency_avg_ns\":" << s.write_latency_avg_ns
     << ",\"write_latency_p95_ns\":" << s.write_latency_p95_ns
     << ",\"done\":" << (s.done ? "true" : "false") << ",\"levels\":[";
  for (size_t i = 0; i < s.levels.size(); ++i) {
    if (i > 0) os << ",";
    AppendLevelState(os, s.levels[i]);
  }
  os << "]}";
  return os.str();
}

// Naive JSON array extractor: finds "key": [a, b, ...] and returns the ints.
// Whitespace-tolerant around ':' and '[' — json.dumps and hand-rolled
// serializers differ here, and an intolerant needle silently rejected every
// response once (causing a permanent fallback to leveled). Returns an empty
// vector if the key is missing or the array is malformed.
std::vector<int> ParseIntArrayField(const std::string& json, const char* key) {
  std::vector<int> out;
  const std::string needle = std::string("\"") + key + "\"";
  size_t pos = json.find(needle);
  if (pos == std::string::npos) return out;
  pos += needle.size();
  auto skip_ws = [&json](size_t p) {
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) ++p;
    return p;
  };
  pos = skip_ws(pos);
  if (pos >= json.size() || json[pos] != ':') return out;
  pos = skip_ws(pos + 1);
  if (pos >= json.size() || json[pos] != '[') return out;
  ++pos;
  while (pos < json.size() && json[pos] != ']') {
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\t' || json[pos] == ',')) {
      ++pos;
    }
    if (pos >= json.size() || json[pos] == ']') break;
    if (json[pos] != '-' && (json[pos] < '0' || json[pos] > '9')) {
      return {};  // malformed
    }
    out.push_back(std::atoi(json.c_str() + pos));
    while (pos < json.size() && json[pos] != ',' && json[pos] != ']') ++pos;
  }
  return out;
}

uint64_t ParseUInt64Field(const std::string& json, const char* key,
                          bool* found) {
  *found = false;
  const std::string needle = std::string("\"") + key + "\"";
  size_t pos = json.find(needle);
  if (pos == std::string::npos) return 0;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return 0;
  ++pos;
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  if (pos >= json.size() || json[pos] < '0' || json[pos] > '9') return 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(json.c_str() + pos, &end, 10);
  if (end == json.c_str() + pos) return 0;
  *found = true;
  return static_cast<uint64_t>(value);
}

std::vector<uint64_t> ParseUInt64ArrayField(const std::string& json,
                                            const char* key) {
  std::vector<uint64_t> out;
  const std::string needle = std::string("\"") + key + "\"";
  size_t pos = json.find(needle);
  if (pos == std::string::npos) return out;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return out;
  pos = json.find('[', pos + 1);
  if (pos == std::string::npos) return out;
  ++pos;
  while (pos < json.size() && json[pos] != ']') {
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\t' || json[pos] == ',')) {
      ++pos;
    }
    if (pos >= json.size() || json[pos] == ']') break;
    if (json[pos] < '0' || json[pos] > '9') return {};
    char* end = nullptr;
    const unsigned long long value =
        std::strtoull(json.c_str() + pos, &end, 10);
    if (end == json.c_str() + pos) return {};
    out.push_back(static_cast<uint64_t>(value));
    pos = static_cast<size_t>(end - json.c_str());
    while (pos < json.size() && json[pos] != ',' && json[pos] != ']') ++pos;
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

RLCompactionClient& RLCompactionClient::Get() {
  static RLCompactionClient instance;
  return instance;
}

RLCompactionClient::RLCompactionClient()
    : socket_path_(SocketPathFromEnv()),
      socket_timeout_ms_(SocketTimeoutMsFromEnv()) {}

RLCompactionClient::~RLCompactionClient() { Disconnect(); }

// ---------------------------------------------------------------------------
// Connection management
// ---------------------------------------------------------------------------

bool RLCompactionClient::Connect() {
  if (fd_ >= 0) return true;

  fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd_ < 0) return false;

  if (!SetSocketTimeouts()) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }

  ::sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (socket_path_.size() >= sizeof(addr.sun_path)) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }
  std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

  if (!ConnectWithTimeout(&addr)) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }
  return true;
}

bool RLCompactionClient::SetSocketTimeouts() {
  if (fd_ < 0) return false;
  timeval tv{};
  tv.tv_sec = socket_timeout_ms_ / 1000;
  tv.tv_usec = (socket_timeout_ms_ % 1000) * 1000;
  if (::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
    return false;
  }
  if (::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) {
    return false;
  }
  return true;
}

bool RLCompactionClient::ConnectWithTimeout(::sockaddr_un* addr) {
  const int old_flags = ::fcntl(fd_, F_GETFL, 0);
  if (old_flags < 0) {
    return false;
  }
  if (::fcntl(fd_, F_SETFL, old_flags | O_NONBLOCK) < 0) {
    return false;
  }

  int rc = ::connect(fd_, reinterpret_cast<::sockaddr*>(addr), sizeof(*addr));
  if (rc == 0) {
    ::fcntl(fd_, F_SETFL, old_flags);
    return true;
  }
  if (errno != EINPROGRESS) {
    ::fcntl(fd_, F_SETFL, old_flags);
    return false;
  }

  pollfd pfd{};
  pfd.fd = fd_;
  pfd.events = POLLOUT;
  rc = ::poll(&pfd, 1, socket_timeout_ms_);
  if (rc <= 0) {
    ::fcntl(fd_, F_SETFL, old_flags);
    return false;
  }

  int so_error = 0;
  socklen_t len = sizeof(so_error);
  if (::getsockopt(fd_, SOL_SOCKET, SO_ERROR, &so_error, &len) != 0 ||
      so_error != 0) {
    ::fcntl(fd_, F_SETFL, old_flags);
    return false;
  }

  return ::fcntl(fd_, F_SETFL, old_flags) == 0;
}

void RLCompactionClient::Disconnect() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

bool RLCompactionClient::SendLine(const std::string& line) {
  std::string msg = line + "\n";
  const char* ptr = msg.c_str();
  ssize_t remaining = static_cast<ssize_t>(msg.size());
  while (remaining > 0) {
#ifdef MSG_NOSIGNAL
    ssize_t n = ::send(fd_, ptr, static_cast<size_t>(remaining), MSG_NOSIGNAL);
#else
    ssize_t n = ::write(fd_, ptr, static_cast<size_t>(remaining));
#endif
    if (n <= 0) return false;
    ptr += n;
    remaining -= n;
  }
  return true;
}

std::string RLCompactionClient::RecvLine() {
  std::string result;
  char c;
  while (true) {
    ssize_t n = ::read(fd_, &c, 1);
    if (n <= 0) return {};
    if (c == '\n') break;
    result += c;
    // Protocol v3 carries up to eight candidates per level; bound malformed
    // peers without truncating a legitimate multi-level observation.
    if (result.size() > 1024 * 1024) return {};
  }
  return result;
}

// ---------------------------------------------------------------------------
// Main entry point called by compaction picker
// ---------------------------------------------------------------------------

RLMultiQueryResult RLCompactionClient::QueryActions(const RLStateV2& state) {
  RLMultiQueryResult result;
  if (state.levels.empty()) {
    return result;
  }

  std::unique_lock<std::mutex> lock(mu_);

  // Lazy connect; retry if connection was lost.
  if (!IsConnected() && !Connect()) {
    return result;
  }

  const std::string msg = FormatStateV2(state);
  if (!SendLine(msg)) {
    Disconnect();
    return result;
  }

  const std::string response = RecvLine();
  if (response.empty()) {
    Disconnect();
    return result;
  }

  std::vector<int> actions = ParseIntArrayField(response, "actions");
  if (actions.size() != state.levels.size()) {
    // Mis-sized or malformed response: treat as unavailable so the caller
    // falls back to its local policy rather than misrouting actions.
    return result;
  }

  result.actions.reserve(actions.size());
  for (int a : actions) {
    if (a < 0 || a > 1) a = static_cast<int>(RLAction::kDoNothing);
    result.actions.push_back(static_cast<RLAction>(a));
  }
  if (state.protocol_version >= 3) {
    bool decision_found = false;
    bool epoch_found = false;
    result.decision_id =
        ParseUInt64Field(response, "decision_id", &decision_found);
    result.snapshot_epoch =
        ParseUInt64Field(response, "snapshot_epoch", &epoch_found);
    result.candidate_file_numbers =
        ParseUInt64ArrayField(response, "candidate_file_numbers");
    if (!decision_found || result.decision_id == 0 || !epoch_found ||
        result.snapshot_epoch != state.snapshot_epoch ||
        result.candidate_file_numbers.size() != state.levels.size()) {
      return RLMultiQueryResult();
    }
    size_t compact_actions = 0;
    for (size_t i = 0; i < result.actions.size(); ++i) {
      const uint64_t selected = result.candidate_file_numbers[i];
      if (result.actions[i] == RLAction::kDoNothing) {
        if (selected != 0) return RLMultiQueryResult();
        continue;
      }
      ++compact_actions;
      if (selected == 0) {
        // The only non-candidate v3 action is the deterministic tuned-level
        // safety fallback, and it is meaningful only for a due level.
        if (!state.levels[i].default_needed) return RLMultiQueryResult();
        continue;
      }
      bool previewed_and_valid = false;
      for (const RLCandidateState& candidate : state.levels[i].candidates) {
        if (candidate.source_file_number == selected && !candidate.conflict &&
            candidate.snapshot_epoch == state.snapshot_epoch) {
          previewed_and_valid = true;
          break;
        }
      }
      if (!previewed_and_valid) return RLMultiQueryResult();
    }
    if (compact_actions > 1) return RLMultiQueryResult();
  } else {
    result.candidate_file_numbers.assign(state.levels.size(), 0);
  }
  result.ok = true;
  return result;
}

}  // namespace ROCKSDB_NAMESPACE
