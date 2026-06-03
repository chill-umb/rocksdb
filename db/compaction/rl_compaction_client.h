#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

// Actions the RL agent can return for L0 compaction decisions.
enum class RLAction : int {
  kDoNothing  = 0,
  kCompactNow = 1,
  kDelay      = 2,
};

// Normalized state vector sent to the Python RL server each step.
struct RLState {
  double f0;       // L0 file count / 20.0            [0, 1]
  double df0;      // delta L0 file count / 20.0       [-1, 1]
  double s0;       // L0 compaction score / 4.0        [0, 1]
  double pcb;      // pending compaction bytes / 10 GB [0, 1]
  double stall;    // binary stall indicator            {0, 1}
  double bw;       // recent write pressure (proxy)    [0, 1]
  double reward;   // reward for the previous action
  bool   done;     // episode done flag
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

  // Sends `state` to the Python server and returns the action.
  // Falls back to kCompactNow on any I/O error so the DB stays safe.
  RLAction QueryAction(const RLState& state);

  bool IsConnected() const { return fd_ >= 0; }

 private:
  RLCompactionClient();
  ~RLCompactionClient();

  // Non-copyable.
  RLCompactionClient(const RLCompactionClient&) = delete;
  RLCompactionClient& operator=(const RLCompactionClient&) = delete;

  bool Connect();
  void Disconnect();

  // Send a newline-terminated string; returns false on error.
  bool SendLine(const std::string& line);

  // Read until '\n'; returns empty string on error.
  std::string RecvLine();

  std::string socket_path_;
  int         fd_{-1};
  mutable std::mutex mu_;
};

}  // namespace ROCKSDB_NAMESPACE
