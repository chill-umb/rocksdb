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

// Minimal JSON number formatting (no external dependency).
std::string FormatState(const RLState& s) {
  std::ostringstream os;
  os.precision(6);
  os << std::fixed;
  os << "{\"f0\":" << s.f0 << ",\"df0\":" << s.df0 << ",\"s0\":" << s.s0
     << ",\"pcb\":" << s.pcb << ",\"stall\":" << s.stall << ",\"bw\":" << s.bw
     << ",\"reward\":" << s.reward
     << ",\"done\":" << (s.done ? "true" : "false") << "}";
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
    if (result.size() > 4096) return {};  // guard against runaway reads
  }
  return result;
}

// ---------------------------------------------------------------------------
// Main entry point called by compaction picker
// ---------------------------------------------------------------------------

RLQueryResult RLCompactionClient::QueryAction(const RLState& state) {
  std::unique_lock<std::mutex> lock(mu_);

  // Lazy connect; retry if connection was lost.
  if (!IsConnected() && !Connect()) {
    return {false, RLAction::kDoNothing};
  }

  const std::string msg = FormatState(state);
  if (!SendLine(msg)) {
    Disconnect();
    return {false, RLAction::kDoNothing};
  }

  const std::string response = RecvLine();
  if (response.empty()) {
    Disconnect();
    return {false, RLAction::kDoNothing};
  }

  int action = ParseIntField(response, "action",
                             static_cast<int>(RLAction::kCompactNow));
  if (action < 0 || action > 2)
    action = static_cast<int>(RLAction::kCompactNow);
  return {true, static_cast<RLAction>(action)};
}

}  // namespace ROCKSDB_NAMESPACE
