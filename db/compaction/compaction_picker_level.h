//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once

#include "db/compaction/compaction_picker.h"

namespace ROCKSDB_NAMESPACE {

// Non-mutating description of the compaction RocksDB would construct when a
// particular source SST is selected. Policy wrappers use this so preview and
// actuation share clean-cut expansion and conflict checks.
struct LevelCompactionCandidate {
  int source_level = 0;
  int output_level = 0;
  uint64_t source_file_number = 0;
  uint64_t source_bytes = 0;
  uint64_t expanded_source_bytes = 0;
  uint64_t overlap_bytes = 0;
  uint64_t estimated_read_bytes = 0;
  uint64_t estimated_write_bytes = 0;
  uint64_t num_entries = 0;
  uint64_t num_deletions = 0;
  uint64_t compensated_size = 0;
  double projected_source_fullness = 0.0;
  double projected_output_fullness = 0.0;
  bool empties_source_level = false;
  bool conflict = false;
  int priority_rank = 0;
  std::vector<uint64_t> expanded_source_files;
  std::vector<uint64_t> overlap_files;
};

// Picking compactions for leveled compaction. See wiki page
// https://github.com/facebook/rocksdb/wiki/Leveled-Compaction
// for description of Leveled compaction.
class LevelCompactionPicker : public CompactionPicker {
 public:
  LevelCompactionPicker(const ImmutableOptions& ioptions,
                        const InternalKeyComparator* icmp)
      : CompactionPicker(ioptions, icmp) {}
  Compaction* PickCompaction(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options,
      const std::vector<SequenceNumber>& /* existing_snapshots */,
      const SnapshotChecker* /* snapshot_checker */,
      VersionStorageInfo* vstorage, LogBuffer* log_buffer,
      const std::string& full_history_ts_low,
      bool /*require_max_output_level*/ = false) override;

  bool NeedsCompaction(const VersionStorageInfo* vstorage) const override;

 protected:
  // Pick from a specific input level while reusing leveled compaction's normal
  // file selection, overlap expansion, clean-cut checks, and output-level
  // setup. Policy wrappers can use this when they override only the trigger
  // decision, not RocksDB's compaction construction mechanics.
  Compaction* PickCompactionFromLevel(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options, VersionStorageInfo* vstorage,
      LogBuffer* log_buffer, const std::string& full_history_ts_low,
      int forced_start_level, double forced_start_level_score,
      CompactionReason compaction_reason);

  Compaction* PickCompactionFromFile(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options, VersionStorageInfo* vstorage,
      LogBuffer* log_buffer, const std::string& full_history_ts_low,
      int forced_start_level, uint64_t source_file_number,
      double forced_start_level_score, CompactionReason compaction_reason);

  std::vector<LevelCompactionCandidate> PreviewCompactionCandidates(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options, VersionStorageInfo* vstorage,
      LogBuffer* log_buffer, const std::string& full_history_ts_low, int level,
      size_t limit);
};

}  // namespace ROCKSDB_NAMESPACE
