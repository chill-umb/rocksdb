#include "db/compaction/rl_compaction_client.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

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

// Minimal JSON number formatting (no external dependency).
std::string FormatState(const RLState& s) {
  std::ostringstream os;
  os.precision(6);
  os << std::fixed;
  os << "{\"f0\":"     << s.f0
     << ",\"df0\":"    << s.df0
     << ",\"s0\":"     << s.s0
     << ",\"pcb\":"    << s.pcb
     << ",\"stall\":"  << s.stall
     << ",\"bw\":"     << s.bw
     << ",\"reward\":" << s.reward
     << ",\"done\":"   << (s.done ? "true" : "false")
     << "}";
  return os.str();
}

// Naive JSON field extractor: finds "key":N and returns N as int.
// Returns `fallback` if not found.
int ParseIntField(const std::string& json, const char* key, int fallback) {
  std::string needle = std::string("\"") + key + "\":";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return fallback;
  pos += needle.size();
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  if (pos >= json.size()) return fallback;
  return std::atoi(json.c_str() + pos);
}

}  // namespace

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

RLCompactionClient& RLCompactionClient::Get() {
  static RLCompactionClient instance;
  return instance;
}

RLCompactionClient::RLCompactionClient() : socket_path_(SocketPathFromEnv()) {}

RLCompactionClient::~RLCompactionClient() { Disconnect(); }

// ---------------------------------------------------------------------------
// Connection management
// ---------------------------------------------------------------------------

bool RLCompactionClient::Connect() {
  if (fd_ >= 0) return true;

  fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd_ < 0) return false;

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

  if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }
  return true;
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
    ssize_t n = ::write(fd_, ptr, static_cast<size_t>(remaining));
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
    if (result.size() > 4096) return {};  // guard against runaway reads
  }
  return result;
}

// ---------------------------------------------------------------------------
// Main entry point called by compaction picker
// ---------------------------------------------------------------------------

RLAction RLCompactionClient::QueryAction(const RLState& state) {
  std::unique_lock<std::mutex> lock(mu_);

  // Lazy connect; retry if connection was lost.
  if (!IsConnected() && !Connect()) {
    // Server not running — fail safe: let RocksDB decide normally.
    return RLAction::kCompactNow;
  }

  const std::string msg = FormatState(state);
  if (!SendLine(msg)) {
    Disconnect();
    return RLAction::kCompactNow;
  }

  const std::string response = RecvLine();
  if (response.empty()) {
    Disconnect();
    return RLAction::kCompactNow;
  }

  int action = ParseIntField(response, "action", static_cast<int>(RLAction::kCompactNow));
  if (action < 0 || action > 2) action = static_cast<int>(RLAction::kCompactNow);
  return static_cast<RLAction>(action);
}

}  // namespace ROCKSDB_NAMESPACE
