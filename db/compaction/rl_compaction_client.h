#pragma once

#include <sys/un.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

// Actions the RL agent can return for a per-level compaction decision.
enum class RLAction : int {
  kDoNothing = 0,
  kCompactNow = 1,
};

// Raw observable state for one LSM level. RocksDB only reports observables;
// the Python agent owns normalization, reward computation, and learning.
struct RLLevelState {
  int level = 0;
  int files = 0;
  uint64_t bytes = 0;
  double score = 0.0;           // RocksDB compaction score for this level
  uint64_t target_bytes = 0;    // MaxBytesForLevel (0 for L0: use triggers)
  int next_level_files = 0;     // 0 when is_last
  uint64_t next_level_bytes = 0;
  double next_level_score = 0.0;
  uint64_t next_level_target_bytes = 0;
  uint64_t overlap_bytes = 0;   // bytes in level+1 overlapping this level
  // Telemetry deltas since the previous RL query, scoped to this level.
  uint64_t bytes_in = 0;            // arrived via flush/compaction output
  uint64_t bytes_read_out = 0;      // compaction reads with this base level
  uint64_t bytes_written_out = 0;   // compaction writes with this base level
  uint64_t compactions_from = 0;    // completed compactions from this level
  uint64_t compactions_scheduled = 0;
  bool default_needed = false;  // would RocksDB's own trigger fire (score>=1)
  bool is_last = false;         // no next level below this one
};

// Full request: global state + one entry per candidate level.
struct RLStateV2 {
  uint64_t pending_compaction_bytes = 0;
  uint64_t flushed_bytes = 0;
  uint64_t compaction_bytes_read = 0;
  uint64_t compaction_bytes_written = 0;
  uint64_t compactions_completed = 0;
  uint64_t stall_count = 0;
  uint64_t stop_count = 0;
  int l0_compaction_trigger = 4;
  int l0_slowdown_trigger = 20;
  int l0_stop_trigger = 36;
  int l0_delay_trigger_count = 0;
  bool done = false;
  std::vector<RLLevelState> levels;
};

// Result of a multi-level query: one action per requested level, in request
// order. `ok=false` means the caller should use its local fallback policy.
struct RLMultiQueryResult {
  bool ok = false;
  std::vector<RLAction> actions;
};

// Singleton client that communicates with the Python RL server over a
// Unix domain socket.  The C++ compaction picker calls QueryActions()
// on each decision evaluation; the server returns per-level actions and
// trains its per-level agents online.
//
// Thread-safe: a single mutex serialises socket I/O.
class RLCompactionClient {
 public:
  // Returns the process-wide singleton, lazily initialised.
  static RLCompactionClient& Get();

  // Sends `state` to the Python server and returns one action per level in
  // `state.levels` (same order). `ok=false` on connection failure, timeout,
  // or a malformed/mis-sized response.
  RLMultiQueryResult QueryActions(const RLStateV2& state);

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
