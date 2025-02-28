// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/resource_coordinator/leveldb_site_characteristics_database.h"

#include <string>

#include "base/files/file_util.h"
#include "base/logging.h"
#include "base/memory/ptr_util.h"
#include "base/metrics/histogram_functions.h"
#include "base/metrics/histogram_macros.h"
#include "base/task_runner_util.h"
#include "base/threading/thread_restrictions.h"
#include "chrome/browser/resource_coordinator/utils.h"
#include "third_party/leveldatabase/env_chromium.h"
#include "third_party/leveldatabase/leveldb_chrome.h"
#include "third_party/leveldatabase/src/include/leveldb/write_batch.h"

namespace resource_coordinator {

namespace {

const char kInitStatusHistogramLabel[] =
    "ResourceCoordinator.LocalDB.DatabaseInit";
const char kInitStatusAfterRepairHistogramLabel[] =
    "ResourceCoordinator.LocalDB.DatabaseInitAfterRepair";
const char kInitStatusAfterDeleteHistogramLabel[] =
    "ResourceCoordinator.LocalDB.DatabaseInitAfterDelete";

enum class InitStatus {
  kInitStatusOk,
  kInitStatusCorruption,
  kInitStatusIOError,
  kInitStatusUnknownError,
  kInitStatusMax
};

// Report the database's initialization status metrics.
void ReportInitStatus(const char* histogram_name,
                      const leveldb::Status& status) {
  if (status.ok()) {
    base::UmaHistogramEnumeration(histogram_name, InitStatus::kInitStatusOk,
                                  InitStatus::kInitStatusMax);
  } else if (status.IsCorruption()) {
    base::UmaHistogramEnumeration(histogram_name,
                                  InitStatus::kInitStatusCorruption,
                                  InitStatus::kInitStatusMax);
  } else if (status.IsIOError()) {
    base::UmaHistogramEnumeration(histogram_name,
                                  InitStatus::kInitStatusIOError,
                                  InitStatus::kInitStatusMax);
  } else {
    base::UmaHistogramEnumeration(histogram_name,
                                  InitStatus::kInitStatusUnknownError,
                                  InitStatus::kInitStatusMax);
  }
}

// Attempt to repair the database stored in |db_path|.
bool RepairDatabase(const std::string& db_path) {
  leveldb_env::Options options;
  options.reuse_logs = false;
  options.max_open_files = 0;
  bool repair_succeeded = leveldb::RepairDB(db_path, options).ok();
  UMA_HISTOGRAM_BOOLEAN("ResourceCoordinator.LocalDB.DatabaseRepair",
                        repair_succeeded);
  return repair_succeeded;
}

bool ShouldAttemptDbRepair(const leveldb::Status& status) {
  // A corrupt database might be repaired (some data might be loss but it's
  // better than losing everything).
  if (status.IsCorruption())
    return true;
  // An I/O error might be caused by a missing manifest, it's sometime possible
  // to repair this (some data might be loss).
  if (status.IsIOError())
    return true;

  return false;
}

}  // namespace

// Helper class used to run all the blocking operations posted by
// LocalSiteCharacteristicDatabase on a TaskScheduler sequence with the
// |MayBlock()| trait.
//
// Instances of this class should only be destructed once all the posted tasks
// have been run, in practice it means that they should ideally be stored in a
// std::unique_ptr<AsyncHelper, base::OnTaskRunnerDeleter>.
class LevelDBSiteCharacteristicsDatabase::AsyncHelper {
 public:
  explicit AsyncHelper(const base::FilePath& db_path) : db_path_(db_path) {
    DETACH_FROM_SEQUENCE(sequence_checker_);
    // Setting |sync| to false might cause some data loss if the system crashes
    // but it'll make the write operations faster (no data will be lost if only
    // the process crashes).
    write_options_.sync = false;
  }
  ~AsyncHelper() = default;

  // Open the database from |db_path_| after creating it if it didn't exist.
  void OpenOrCreateDatabase();

  // Implementations of the DB manipulation functions of
  // LevelDBSiteCharacteristicsDatabase that run on a blocking sequence.
  base::Optional<SiteCharacteristicsProto> ReadSiteCharacteristicsFromDB(
      const url::Origin& origin);
  void WriteSiteCharacteristicsIntoDB(
      const url::Origin& origin,
      const SiteCharacteristicsProto& site_characteristic_proto);
  void RemoveSiteCharacteristicsFromDB(
      const std::vector<url::Origin>& site_origin);
  void ClearDatabase();

  bool DBIsInitialized() { return db_ != nullptr; }

 private:
  // The on disk location of the database.
  const base::FilePath db_path_;
  // The connection to the LevelDB database.
  std::unique_ptr<leveldb::DB> db_;
  // The options to be used for all database read operations.
  leveldb::ReadOptions read_options_;
  // The options to be used for all database write operations.
  leveldb::WriteOptions write_options_;

  SEQUENCE_CHECKER(sequence_checker_);
  DISALLOW_COPY_AND_ASSIGN(AsyncHelper);
};

void LevelDBSiteCharacteristicsDatabase::AsyncHelper::OpenOrCreateDatabase() {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  DCHECK(!db_) << "Database already open";
  base::AssertBlockingAllowed();

  // Report the on disk size of the database if it already exists.
  if (base::DirectoryExists(db_path_)) {
    int64_t db_ondisk_size_in_bytes = base::ComputeDirectorySize(db_path_);
    UMA_HISTOGRAM_MEMORY_KB("ResourceCoordinator.LocalDB.OnDiskSize",
                            db_ondisk_size_in_bytes / 1024);
  }

  leveldb_env::Options options;
  options.create_if_missing = true;
  leveldb::Status status =
      leveldb_env::OpenDB(options, db_path_.AsUTF8Unsafe(), &db_);

  ReportInitStatus(kInitStatusHistogramLabel, status);

  if (status.ok())
    return;

  if (!ShouldAttemptDbRepair(status))
    return;

  if (RepairDatabase(db_path_.AsUTF8Unsafe())) {
    status = leveldb_env::OpenDB(options, db_path_.AsUTF8Unsafe(), &db_);
    ReportInitStatus(kInitStatusAfterRepairHistogramLabel, status);
    if (status.ok())
      return;
  }

  // Delete the database and try to open it one last time.
  if (leveldb_chrome::DeleteDB(db_path_, options).ok()) {
    status = leveldb_env::OpenDB(options, db_path_.AsUTF8Unsafe(), &db_);
    ReportInitStatus(kInitStatusAfterDeleteHistogramLabel, status);
    if (!status.ok())
      db_.reset();
  }
}

base::Optional<SiteCharacteristicsProto>
LevelDBSiteCharacteristicsDatabase::AsyncHelper::ReadSiteCharacteristicsFromDB(
    const url::Origin& origin) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  base::AssertBlockingAllowed();

  if (!db_)
    return base::nullopt;

  std::string protobuf_value;
  leveldb::Status s = db_->Get(
      read_options_, SerializeOriginIntoDatabaseKey(origin), &protobuf_value);
  base::Optional<SiteCharacteristicsProto> site_characteristic_proto;
  if (s.ok()) {
    site_characteristic_proto = SiteCharacteristicsProto();
    if (!site_characteristic_proto->ParseFromString(protobuf_value)) {
      site_characteristic_proto = base::nullopt;
      LOG(ERROR) << "Error while trying to parse a SiteCharacteristicsProto "
                 << "protobuf.";
    }
  }
  return site_characteristic_proto;
}

void LevelDBSiteCharacteristicsDatabase::AsyncHelper::
    WriteSiteCharacteristicsIntoDB(
        const url::Origin& origin,
        const SiteCharacteristicsProto& site_characteristic_proto) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  base::AssertBlockingAllowed();

  if (!db_)
    return;

  leveldb::Status s =
      db_->Put(write_options_, SerializeOriginIntoDatabaseKey(origin),
               site_characteristic_proto.SerializeAsString());
  if (!s.ok()) {
    LOG(ERROR) << "Error while inserting an element in the site characteristic "
               << "database: " << s.ToString();
  }
}

void LevelDBSiteCharacteristicsDatabase::AsyncHelper::
    RemoveSiteCharacteristicsFromDB(
        const std::vector<url::Origin>& site_origins) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  base::AssertBlockingAllowed();

  if (!db_)
    return;

  leveldb::WriteBatch batch;
  for (const auto& iter : site_origins)
    batch.Delete(SerializeOriginIntoDatabaseKey(iter));
  leveldb::Status status = db_->Write(write_options_, &batch);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to remove some entries from the site "
                 << "characteristics database: " << status.ToString();
  }
}

void LevelDBSiteCharacteristicsDatabase::AsyncHelper::ClearDatabase() {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  base::AssertBlockingAllowed();
  if (!db_)
    return;

  leveldb_env::Options options;
  db_.reset();
  leveldb::Status status = leveldb::DestroyDB(db_path_.AsUTF8Unsafe(), options);
  if (status.ok()) {
    OpenOrCreateDatabase();
  } else {
    LOG(WARNING) << "Failed to destroy the site characteristics database: "
                 << status.ToString();
  }
}

LevelDBSiteCharacteristicsDatabase::LevelDBSiteCharacteristicsDatabase(
    const base::FilePath& db_path)
    : blocking_task_runner_(base::CreateSequencedTaskRunnerWithTraits(
          // The |BLOCK_SHUTDOWN| trait is required to ensure that a clearing of
          // the database won't be skipped.
          {base::MayBlock(), base::TaskShutdownBehavior::BLOCK_SHUTDOWN})),
      async_helper_(new AsyncHelper(db_path),
                    base::OnTaskRunnerDeleter(blocking_task_runner_)) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);

  blocking_task_runner_->PostTask(
      FROM_HERE, base::BindOnce(&LevelDBSiteCharacteristicsDatabase::
                                    AsyncHelper::OpenOrCreateDatabase,
                                base::Unretained(async_helper_.get())));
}

LevelDBSiteCharacteristicsDatabase::~LevelDBSiteCharacteristicsDatabase() =
    default;

void LevelDBSiteCharacteristicsDatabase::ReadSiteCharacteristicsFromDB(
    const url::Origin& origin,
    LocalSiteCharacteristicsDatabase::ReadSiteCharacteristicsFromDBCallback
        callback) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);

  // Trigger the asynchronous task and make it run the callback on this thread
  // once it returns.
  base::PostTaskAndReplyWithResult(
      blocking_task_runner_.get(), FROM_HERE,
      base::BindOnce(&LevelDBSiteCharacteristicsDatabase::AsyncHelper::
                         ReadSiteCharacteristicsFromDB,
                     base::Unretained(async_helper_.get()), origin),
      base::BindOnce(std::move(callback)));
}

void LevelDBSiteCharacteristicsDatabase::WriteSiteCharacteristicsIntoDB(
    const url::Origin& origin,
    const SiteCharacteristicsProto& site_characteristic_proto) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  blocking_task_runner_->PostTask(
      FROM_HERE, base::BindOnce(&LevelDBSiteCharacteristicsDatabase::
                                    AsyncHelper::WriteSiteCharacteristicsIntoDB,
                                base::Unretained(async_helper_.get()), origin,
                                std::move(site_characteristic_proto)));
}

void LevelDBSiteCharacteristicsDatabase::RemoveSiteCharacteristicsFromDB(
    const std::vector<url::Origin>& site_origins) {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  blocking_task_runner_->PostTask(
      FROM_HERE,
      base::BindOnce(&LevelDBSiteCharacteristicsDatabase::AsyncHelper::
                         RemoveSiteCharacteristicsFromDB,
                     base::Unretained(async_helper_.get()),
                     std::move(site_origins)));
}

void LevelDBSiteCharacteristicsDatabase::ClearDatabase() {
  DCHECK_CALLED_ON_VALID_SEQUENCE(sequence_checker_);
  blocking_task_runner_->PostTask(
      FROM_HERE,
      base::BindOnce(
          &LevelDBSiteCharacteristicsDatabase::AsyncHelper::ClearDatabase,
          base::Unretained(async_helper_.get())));
}

bool LevelDBSiteCharacteristicsDatabase::DatabaseIsInitializedForTesting() {
  return async_helper_->DBIsInitialized();
}

}  // namespace resource_coordinator
