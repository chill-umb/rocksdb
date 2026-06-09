#pragma once

#include <sys/un.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

// Actions the RL agent can return for L0 compaction decisions.
enum class RLAction : int {
  kDoNothing = 0,
  kCompactNow = 1,
  kDelay = 2,
};

struct RLQueryResult {
  bool ok;
  RLAction action;
};

// Normalized state vector sent to the Python RL server each step.
struct RLState {
  double f0;      // L0 file count / 20.0            [0, 1]
  double df0;     // delta L0 file count / 20.0       [-1, 1]
  double s0;      // L0 compaction score, clamped      [0, 1]
  double pcb;     // pending compaction bytes / 10 GB [0, 1]
  double stall;   // real stall observed in last step  {0, 1}
  double bw;      // flushed bytes / 64 MB             [0, 1]
  double reward;  // reward for the previous action
  bool done;      // episode done flag
};

// Singleton client that communicates with the Python RL server over a
// Unix domain socket.  The C++ compaction picker calls QueryAction()
// on each L0 evaluation; the server returns an action and trains online.
//
// Thread-safe: a single mutex serialises socket I/O.
class RLCompactionClient {
 public:
  // Returns the process-wide singleton, lazily initialised.
  static RLCompactionClient& Get();

  // Sends `state` to the Python server and returns the action. `ok=false`
  // means the caller should use its local fallback policy.
  RLQueryResult QueryAction(const RLState& state);

  bool IsConnected() const { return fd_ >= 0; }

 private:
  RLCompactionClient();
  ~RLCompactionClient();

  // Non-copyable.
  RLCompactionClient(const RLCompactionClient&) = delete;
  RLCompactionClient& operator=(const RLCompactionClient&) = delete;

  bool Connect();
  void Disconnect();
  bool SetSocketTimeouts();
  bool ConnectWithTimeout(::sockaddr_un* addr);

  // Send a newline-terminated string; returns false on error.
  bool SendLine(const std::string& line);

  // Read until '\n'; returns empty string on error.
  std::string RecvLine();

  std::string socket_path_;
  int socket_timeout_ms_{100};
  int fd_{-1};
  mutable std::mutex mu_;
};

}  // namespace ROCKSDB_NAMESPACE
