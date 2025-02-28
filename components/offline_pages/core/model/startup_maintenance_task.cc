// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/offline_pages/core/model/startup_maintenance_task.h"

#include <string>
#include <vector>

#include "base/bind.h"
#include "base/files/file_enumerator.h"
#include "base/files/file_util.h"
#include "base/metrics/histogram_functions.h"
#include "base/metrics/histogram_macros.h"
#include "base/numerics/safe_conversions.h"
#include "base/trace_event/trace_event.h"
#include "components/offline_pages/core/archive_manager.h"
#include "components/offline_pages/core/client_policy_controller.h"
#include "components/offline_pages/core/offline_page_client_policy.h"
#include "components/offline_pages/core/offline_page_metadata_store.h"
#include "components/offline_pages/core/offline_store_utils.h"
#include "sql/connection.h"
#include "sql/statement.h"
#include "sql/transaction.h"

namespace offline_pages {

namespace {

#define OFFLINE_PAGES_TABLE_NAME "offlinepages_v1"

struct PageInfo {
  int64_t offline_id;
  base::FilePath file_path;
};

std::vector<PageInfo> GetPageInfosByNamespaces(
    const std::vector<std::string>& temp_namespaces,
    sql::Connection* db) {
  std::vector<PageInfo> result;

  static const char kSql[] =
      "SELECT offline_id, file_path"
      " FROM " OFFLINE_PAGES_TABLE_NAME " WHERE client_namespace = ?";

  for (const auto& temp_namespace : temp_namespaces) {
    sql::Statement statement(db->GetCachedStatement(SQL_FROM_HERE, kSql));
    statement.BindString(0, temp_namespace);
    while (statement.Step()) {
      result.push_back(
          {statement.ColumnInt64(0),
           store_utils::FromDatabaseFilePath(statement.ColumnString(1))});
    }
  }

  return result;
}

std::set<base::FilePath> GetAllArchives(const base::FilePath& archives_dir) {
  std::set<base::FilePath> result;
  base::FileEnumerator file_enumerator(archives_dir, false,
                                       base::FileEnumerator::FILES,
                                       FILE_PATH_LITERAL("*.mhtml"));
  for (auto archive_path = file_enumerator.Next(); !archive_path.empty();
       archive_path = file_enumerator.Next()) {
    result.insert(archive_path);
  }
  return result;
}

bool DeletePagesByOfflineIds(const std::vector<int64_t>& offline_ids,
                             sql::Connection* db) {
  static const char kSql[] =
      "DELETE FROM " OFFLINE_PAGES_TABLE_NAME " WHERE offline_id = ?";

  for (const auto& offline_id : offline_ids) {
    sql::Statement statement(db->GetCachedStatement(SQL_FROM_HERE, kSql));
    statement.BindInt64(0, offline_id);
    if (!statement.Run())
      return false;
  }
  return true;
}

bool DeleteFiles(const std::vector<base::FilePath>& file_paths) {
  bool result = true;
  for (const auto& file_path : file_paths)
    result = base::DeleteFile(file_path, false) && result;
  return result;
}

// This method is clearing the private dir(the legacy dir).
// - For all files associated with temporary pages:
//   The strategy is if any temporary page
//   is still left behind in the legacy dir, delete them.
// - For all files associated with persistent pages:
//   Leave them as-is, since they might be still in use.
// - For all files without any associated DB entry:
//   Delete the files, since they're 'headless' and has no way to be accessed.
SyncOperationResult ClearLegacyPagesInPrivateDirSync(
    sql::Connection* db,
    const std::vector<std::string>& temporary_namespaces,
    const std::vector<std::string>& persistent_namespaces,
    const base::FilePath& private_dir) {
  // One large database transaction that will:
  // 1. Get temporary page infos from the database.
  // 2. Get persistent page infos from the database, in case they're in private
  //    dir.
  // 3. Get all file paths in private dir as a set F.
  // 4. For each temporary page info:
  //    - If its file path is in F, record its offline id for deletion.
  // 5. For each persistent page info:
  //    - If its file path is in F, remove it from F.
  // 6. Delete page entries by recorded offline ids, and delete the remaining
  //    files in F.
  sql::Transaction transaction(db);
  if (!transaction.Begin())
    return SyncOperationResult::TRANSACTION_BEGIN_ERROR;

  std::vector<PageInfo> temporary_page_infos =
      GetPageInfosByNamespaces(temporary_namespaces, db);
  std::vector<PageInfo> persistent_page_infos =
      GetPageInfosByNamespaces(persistent_namespaces, db);
  std::map<base::FilePath, PageInfo> path_to_page_info;

  std::set<base::FilePath> archive_paths = GetAllArchives(private_dir);
  std::vector<int64_t> offline_ids_to_delete;

  for (const auto& page_info : temporary_page_infos) {
    if (archive_paths.find(page_info.file_path) != archive_paths.end())
      offline_ids_to_delete.push_back(page_info.offline_id);
  }
  for (const auto& page_info : persistent_page_infos) {
    auto iter = archive_paths.find(page_info.file_path);
    if (iter != archive_paths.end())
      archive_paths.erase(iter);
  }

  // Try to delete the pages by offline ids collected above.
  // If there's any database related errors, the function will return failure,
  // and the database operations will be rolled back since the transaction will
  // not be committed.
  if (!DeletePagesByOfflineIds(offline_ids_to_delete, db))
    return SyncOperationResult::DB_OPERATION_ERROR;

  if (!transaction.Commit())
    return SyncOperationResult::TRANSACTION_COMMIT_ERROR;

  std::vector<base::FilePath> files_to_delete(archive_paths.begin(),
                                              archive_paths.end());
  if (!DeleteFiles(files_to_delete))
    return SyncOperationResult::FILE_OPERATION_ERROR;

  size_t headless_file_count =
      files_to_delete.size() - offline_ids_to_delete.size();
  if (headless_file_count > 0) {
    UMA_HISTOGRAM_COUNTS_1M(
        "OfflinePages.ConsistencyCheck.Legacy.DeletedHeadlessFileCount",
        headless_file_count);
  }

  return SyncOperationResult::SUCCESS;
}

SyncOperationResult CheckTemporaryPageConsistencySync(
    sql::Connection* db,
    const std::vector<std::string>& namespaces,
    const base::FilePath& archives_dir) {
  // One large database transaction that will:
  // 1. Get page infos by |namespaces| from the database.
  // 2. Decide which pages to delete.
  // 3. Delete metadata entries from the database.
  sql::Transaction transaction(db);
  if (!transaction.Begin())
    return SyncOperationResult::TRANSACTION_BEGIN_ERROR;

  std::vector<PageInfo> page_infos = GetPageInfosByNamespaces(namespaces, db);

  std::set<base::FilePath> page_info_paths;
  std::vector<int64_t> offline_ids_to_delete;
  for (const auto& page_info : page_infos) {
    // Get pages whose archive files does not exist and delete.
    if (!base::PathExists(page_info.file_path)) {
      offline_ids_to_delete.push_back(page_info.offline_id);
    } else {
      // Extract existing file paths from |page_infos| so that we can do a
      // faster matching later.
      page_info_paths.insert(page_info.file_path);
    }
  }

  if (offline_ids_to_delete.size() > 0) {
    // Try to delete the pages by offline ids collected above. If there's any
    // database related errors, the function will return false, and the database
    // operations will be rolled back since the transaction will not be
    // committed.
    if (!DeletePagesByOfflineIds(offline_ids_to_delete, db))
      return SyncOperationResult::DB_OPERATION_ERROR;
    UMA_HISTOGRAM_COUNTS_1M(
        "OfflinePages.ConsistencyCheck.Temporary.PagesMissingArchiveFileCount",
        base::saturated_cast<int32_t>(offline_ids_to_delete.size()));
  }

  if (!transaction.Commit())
    return SyncOperationResult::TRANSACTION_COMMIT_ERROR;

  // Delete any files in the temporary archive directory that no longer have
  // associated entries in the database.
  std::set<base::FilePath> archive_paths = GetAllArchives(archives_dir);
  std::vector<base::FilePath> files_to_delete;
  for (const auto& archive_path : archive_paths) {
    if (page_info_paths.find(archive_path) == page_info_paths.end())
      files_to_delete.push_back(archive_path);
  }

  if (files_to_delete.size() > 0) {
    if (!DeleteFiles(files_to_delete))
      return SyncOperationResult::FILE_OPERATION_ERROR;
    UMA_HISTOGRAM_COUNTS_1M(
        "OfflinePages.ConsistencyCheck.Temporary.PagesMissingDbEntryCount",
        static_cast<int32_t>(files_to_delete.size()));
  }

  return SyncOperationResult::SUCCESS;
}

void ReportStorageUsageSync(sql::Connection* db,
                            const std::vector<std::string>& namespaces) {
  static const char kSql[] =
      "SELECT sum(file_size) FROM " OFFLINE_PAGES_TABLE_NAME
      " WHERE client_namespace = ?";
  for (const auto& name_space : namespaces) {
    sql::Statement statement(db->GetCachedStatement(SQL_FROM_HERE, kSql));
    statement.BindString(0, name_space);
    int size_in_kib = 0;
    while (statement.Step()) {
      size_in_kib = base::saturated_cast<int>(statement.ColumnInt64(0) / 1024);
    }
    base::UmaHistogramCustomCounts(
        "OfflinePages.ClearStoragePreRunUsage2." + name_space, size_in_kib, 1,
        10000000, 50);
  }
}

bool StartupMaintenanceSync(
    const std::vector<std::string>& persistent_namespaces,
    const std::vector<std::string>& temporary_namespaces,
    const base::FilePath& temporary_archives_dir,
    const base::FilePath& private_archives_dir,
    sql::Connection* db) {
  // Clear temporary pages that are in legacy directory, which is also the
  // directory that serves as the 'private' directory.
  SyncOperationResult result = ClearLegacyPagesInPrivateDirSync(
      db, temporary_namespaces, persistent_namespaces, private_archives_dir);

  // Clear temporary pages in cache directory.
  result = CheckTemporaryPageConsistencySync(db, temporary_namespaces,
                                             temporary_archives_dir);
  UMA_HISTOGRAM_ENUMERATION("OfflinePages.ConsistencyCheck.Temporary.Result",
                            result, SyncOperationResult::RESULT_COUNT);

  // Report storage usage UMA, |temporary_namespaces| + |persistent_namespaces|
  // should be all namespaces. This is implicitly checked by the
  // TestReportStorageUsage unit test.
  ReportStorageUsageSync(db, temporary_namespaces);
  ReportStorageUsageSync(db, persistent_namespaces);

  return true;
}

}  // namespace

StartupMaintenanceTask::StartupMaintenanceTask(
    OfflinePageMetadataStore* store,
    ArchiveManager* archive_manager,
    ClientPolicyController* policy_controller)
    : store_(store),
      archive_manager_(archive_manager),
      policy_controller_(policy_controller),
      weak_ptr_factory_(this) {
  DCHECK(store_);
  DCHECK(archive_manager_);
  DCHECK(policy_controller_);
}

StartupMaintenanceTask::~StartupMaintenanceTask() = default;

void StartupMaintenanceTask::Run() {
  TRACE_EVENT_ASYNC_BEGIN0("offline_pages", "StartupMaintenanceTask running",
                           this);
  std::vector<std::string> all_namespaces =
      policy_controller_->GetAllNamespaces();
  std::vector<std::string> temporary_namespaces =
      policy_controller_->GetNamespacesRemovedOnCacheReset();
  std::vector<std::string> persistent_namespaces =
      policy_controller_->GetNamespacesForUserRequestedDownload();

  store_->Execute(
      base::BindOnce(&StartupMaintenanceSync, persistent_namespaces,
                     temporary_namespaces,
                     archive_manager_->GetTemporaryArchivesDir(),
                     archive_manager_->GetPrivateArchivesDir()),
      base::BindOnce(&StartupMaintenanceTask::OnStartupMaintenanceDone,
                     weak_ptr_factory_.GetWeakPtr()),
      false);
}

void StartupMaintenanceTask::OnStartupMaintenanceDone(bool result) {
  TRACE_EVENT_ASYNC_END1("offline_pages", "StartupMaintenanceTask running",
                         this, "result", result);
  TaskComplete();
}

}  // namespace offline_pages
