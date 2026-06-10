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
};

struct RLQueryResult {
  bool ok;
  RLAction action;
};

// Raw state vector sent to the Python RL server each step.
// RocksDB only reports observable L0 state and recent telemetry deltas. The
// Python agent owns normalization, reward computation, and learning.
struct RLState {
  int l0_files;
  uint64_t l0_size_bytes;
  double l0_score;
  int l0_delay_trigger_count;
  int l0_compaction_trigger;
  int l0_slowdown_trigger;
  int l0_stop_trigger;
  uint64_t pending_compaction_bytes;
  uint64_t flushed_bytes;
  uint64_t compaction_bytes_read;
  uint64_t compaction_bytes_written;
  uint64_t compactions_completed;
  uint64_t l0_compactions_completed;
  uint64_t l0_compactions_scheduled;
  uint64_t stall_count;
  uint64_t stop_count;
  bool default_l0_compaction_needed;
  bool done;
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
