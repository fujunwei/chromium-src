// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/offline_pages/core/offline_page_metadata_store.h"

#include <stdint.h>

#include <memory>

#include "base/bind.h"
#include "base/files/file_path.h"
#include "base/files/scoped_temp_dir.h"
#include "base/memory/ref_counted.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/utf_string_conversions.h"
#include "base/test/bind_test_util.h"
#include "base/test/test_mock_time_task_runner.h"
#include "base/threading/thread_task_runner_handle.h"
#include "components/offline_pages/core/client_namespace_constants.h"
#include "components/offline_pages/core/model/offline_page_item_generator.h"
#include "components/offline_pages/core/offline_page_item.h"
#include "components/offline_pages/core/offline_page_metadata_store.h"
#include "components/offline_pages/core/offline_page_model.h"
#include "components/offline_pages/core/offline_page_thumbnail.h"
#include "components/offline_pages/core/offline_store_utils.h"
#include "sql/connection.h"
#include "sql/meta_table.h"
#include "sql/statement.h"
#include "sql/transaction.h"
#include "testing/gtest/include/gtest/gtest.h"

namespace offline_pages {

namespace {

#define OFFLINE_PAGES_TABLE_V1 "offlinepages_v1"

const char kTestClientNamespace[] = "CLIENT_NAMESPACE";
const char kTestURL[] = "https://example.com";
const char kOriginalTestURL[] = "https://example.com/foo";
const ClientId kTestClientId1(kTestClientNamespace, "1234");
const ClientId kTestClientId2(kTestClientNamespace, "5678");
const base::FilePath::CharType kFilePath[] =
    FILE_PATH_LITERAL("/offline_pages/example_com.mhtml");
int64_t kFileSize = 234567LL;
int64_t kOfflineId = 12345LL;
const char kTestRequestOrigin[] = "request.origin";
int64_t kTestSystemDownloadId = 42LL;
const char kTestDigest[] = "test-digest";
const base::Time kThumbnailExpiration = store_utils::FromDatabaseTime(42);

// Build a store with outdated schema to simulate the upgrading process.
void BuildTestStoreWithSchemaFromM52(const base::FilePath& file) {
  sql::Connection connection;
  ASSERT_TRUE(
      connection.Open(file.Append(FILE_PATH_LITERAL("OfflinePages.db"))));
  ASSERT_TRUE(connection.is_open());
  ASSERT_TRUE(connection.BeginTransaction());
  ASSERT_TRUE(connection.Execute("CREATE TABLE " OFFLINE_PAGES_TABLE_V1
                                 "(offline_id INTEGER PRIMARY KEY NOT NULL, "
                                 "creation_time INTEGER NOT NULL, "
                                 "file_size INTEGER NOT NULL, "
                                 "version INTEGER NOT NULL, "
                                 "last_access_time INTEGER NOT NULL, "
                                 "access_count INTEGER NOT NULL, "
                                 "status INTEGER NOT NULL DEFAULT 0, "
                                 "user_initiated INTEGER, "
                                 "client_namespace VARCHAR NOT NULL, "
                                 "client_id VARCHAR NOT NULL, "
                                 "online_url VARCHAR NOT NULL, "
                                 "offline_url VARCHAR NOT NULL DEFAULT '', "
                                 "file_path VARCHAR NOT NULL "
                                 ")"));
  ASSERT_TRUE(connection.CommitTransaction());
  sql::Statement statement(connection.GetUniqueStatement(
      "INSERT INTO " OFFLINE_PAGES_TABLE_V1
      "(offline_id, creation_time, file_size, version, "
      "last_access_time, access_count, client_namespace, "
      "client_id, online_url, file_path) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"));
  statement.BindInt64(0, kOfflineId);
  statement.BindInt(1, 0);
  statement.BindInt64(2, kFileSize);
  statement.BindInt(3, 0);
  statement.BindInt(4, 0);
  statement.BindInt(5, 1);
  statement.BindCString(6, kTestClientNamespace);
  statement.BindString(7, kTestClientId2.id);
  statement.BindCString(8, kTestURL);
  statement.BindString(9, base::FilePath(kFilePath).MaybeAsASCII());
  ASSERT_TRUE(statement.Run());
  ASSERT_TRUE(connection.DoesTableExist(OFFLINE_PAGES_TABLE_V1));
  ASSERT_FALSE(
      connection.DoesColumnExist(OFFLINE_PAGES_TABLE_V1, "expiration_time"));
}

void BuildTestStoreWithSchemaFromM53(const base::FilePath& file) {
  sql::Connection connection;
  ASSERT_TRUE(
      connection.Open(file.Append(FILE_PATH_LITERAL("OfflinePages.db"))));
  ASSERT_TRUE(connection.is_open());
  ASSERT_TRUE(connection.BeginTransaction());
  ASSERT_TRUE(connection.Execute("CREATE TABLE " OFFLINE_PAGES_TABLE_V1
                                 "(offline_id INTEGER PRIMARY KEY NOT NULL, "
                                 "creation_time INTEGER NOT NULL, "
                                 "file_size INTEGER NOT NULL, "
                                 "version INTEGER NOT NULL, "
                                 "last_access_time INTEGER NOT NULL, "
                                 "access_count INTEGER NOT NULL, "
                                 "status INTEGER NOT NULL DEFAULT 0, "
                                 "user_initiated INTEGER, "
                                 "expiration_time INTEGER NOT NULL DEFAULT 0, "
                                 "client_namespace VARCHAR NOT NULL, "
                                 "client_id VARCHAR NOT NULL, "
                                 "online_url VARCHAR NOT NULL, "
                                 "offline_url VARCHAR NOT NULL DEFAULT '', "
                                 "file_path VARCHAR NOT NULL "
                                 ")"));
  ASSERT_TRUE(connection.CommitTransaction());
  sql::Statement statement(connection.GetUniqueStatement(
      "INSERT INTO " OFFLINE_PAGES_TABLE_V1
      "(offline_id, creation_time, file_size, version, "
      "last_access_time, access_count, client_namespace, "
      "client_id, online_url, file_path, expiration_time) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"));
  statement.BindInt64(0, kOfflineId);
  statement.BindInt(1, 0);
  statement.BindInt64(2, kFileSize);
  statement.BindInt(3, 0);
  statement.BindInt(4, 0);
  statement.BindInt(5, 1);
  statement.BindCString(6, kTestClientNamespace);
  statement.BindString(7, kTestClientId2.id);
  statement.BindCString(8, kTestURL);
  statement.BindString(9, base::FilePath(kFilePath).MaybeAsASCII());
  statement.BindInt64(10, store_utils::ToDatabaseTime(base::Time::Now()));
  ASSERT_TRUE(statement.Run());
  ASSERT_TRUE(connection.DoesTableExist(OFFLINE_PAGES_TABLE_V1));
  ASSERT_FALSE(connection.DoesColumnExist(OFFLINE_PAGES_TABLE_V1, "title"));
}

void BuildTestStoreWithSchemaFromM54(const base::FilePath& file) {
  sql::Connection connection;
  ASSERT_TRUE(
      connection.Open(file.Append(FILE_PATH_LITERAL("OfflinePages.db"))));
  ASSERT_TRUE(connection.is_open());
  ASSERT_TRUE(connection.BeginTransaction());
  ASSERT_TRUE(connection.Execute("CREATE TABLE " OFFLINE_PAGES_TABLE_V1
                                 "(offline_id INTEGER PRIMARY KEY NOT NULL, "
                                 "creation_time INTEGER NOT NULL, "
                                 "file_size INTEGER NOT NULL, "
                                 "version INTEGER NOT NULL, "
                                 "last_access_time INTEGER NOT NULL, "
                                 "access_count INTEGER NOT NULL, "
                                 "status INTEGER NOT NULL DEFAULT 0, "
                                 "user_initiated INTEGER, "
                                 "expiration_time INTEGER NOT NULL DEFAULT 0, "
                                 "client_namespace VARCHAR NOT NULL, "
                                 "client_id VARCHAR NOT NULL, "
                                 "online_url VARCHAR NOT NULL, "
                                 "offline_url VARCHAR NOT NULL DEFAULT '', "
                                 "file_path VARCHAR NOT NULL, "
                                 "title VARCHAR NOT NULL DEFAULT ''"
                                 ")"));
  ASSERT_TRUE(connection.CommitTransaction());
  sql::Statement statement(connection.GetUniqueStatement(
      "INSERT INTO " OFFLINE_PAGES_TABLE_V1
      "(offline_id, creation_time, file_size, version, "
      "last_access_time, access_count, client_namespace, "
      "client_id, online_url, file_path, expiration_time, title) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"));
  statement.BindInt64(0, kOfflineId);
  statement.BindInt(1, 0);
  statement.BindInt64(2, kFileSize);
  statement.BindInt(3, 0);
  statement.BindInt(4, 0);
  statement.BindInt(5, 1);
  statement.BindCString(6, kTestClientNamespace);
  statement.BindString(7, kTestClientId2.id);
  statement.BindCString(8, kTestURL);
  statement.BindString(9, base::FilePath(kFilePath).MaybeAsASCII());
  statement.BindInt64(10, store_utils::ToDatabaseTime(base::Time::Now()));
  statement.BindString16(11, base::UTF8ToUTF16("Test title"));
  ASSERT_TRUE(statement.Run());
  ASSERT_TRUE(connection.DoesTableExist(OFFLINE_PAGES_TABLE_V1));
  ASSERT_TRUE(connection.DoesColumnExist(OFFLINE_PAGES_TABLE_V1, "version"));
  ASSERT_TRUE(connection.DoesColumnExist(OFFLINE_PAGES_TABLE_V1, "status"));
  ASSERT_TRUE(
      connection.DoesColumnExist(OFFLINE_PAGES_TABLE_V1, "user_initiated"));
  ASSERT_TRUE(
      connection.DoesColumnExist(OFFLINE_PAGES_TABLE_V1, "offline_url"));
}

void BuildTestStoreWithSchemaFromM55(const base::FilePath& file) {
  sql::Connection connection;
  ASSERT_TRUE(
      connection.Open(file.Append(FILE_PATH_LITERAL("OfflinePages.db"))));
  ASSERT_TRUE(connection.is_open());
  ASSERT_TRUE(connection.BeginTransaction());
  ASSERT_TRUE(connection.Execute("CREATE TABLE " OFFLINE_PAGES_TABLE_V1
                                 "(offline_id INTEGER PRIMARY KEY NOT NULL, "
                                 "creation_time INTEGER NOT NULL, "
                                 "file_size INTEGER NOT NULL, "
                                 "last_access_time INTEGER NOT NULL, "
                                 "access_count INTEGER NOT NULL, "
                                 "expiration_time INTEGER NOT NULL DEFAULT 0, "
                                 "client_namespace VARCHAR NOT NULL, "
                                 "client_id VARCHAR NOT NULL, "
                                 "online_url VARCHAR NOT NULL, "
                                 "file_path VARCHAR NOT NULL, "
                                 "title VARCHAR NOT NULL DEFAULT ''"
                                 ")"));
  ASSERT_TRUE(connection.CommitTransaction());
  sql::Statement statement(connection.GetUniqueStatement(
      "INSERT INTO " OFFLINE_PAGES_TABLE_V1
      "(offline_id, creation_time, file_size, "
      "last_access_time, access_count, client_namespace, "
      "client_id, online_url, file_path, expiration_time, title) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"));
  statement.BindInt64(0, kOfflineId);
  statement.BindInt(1, 0);
  statement.BindInt64(2, kFileSize);
  statement.BindInt(3, 0);
  statement.BindInt(4, 1);
  statement.BindCString(5, kTestClientNamespace);
  statement.BindString(6, kTestClientId2.id);
  statement.BindCString(7, kTestURL);
  statement.BindString(8, base::FilePath(kFilePath).MaybeAsASCII());
  statement.BindInt64(9, store_utils::ToDatabaseTime(base::Time::Now()));
  statement.BindString16(10, base::UTF8ToUTF16("Test title"));
  ASSERT_TRUE(statement.Run());
  ASSERT_TRUE(connection.DoesTableExist(OFFLINE_PAGES_TABLE_V1));
  ASSERT_TRUE(connection.DoesColumnExist(OFFLINE_PAGES_TABLE_V1, "title"));
  ASSERT_FALSE(
      connection.DoesColumnExist(OFFLINE_PAGES_TABLE_V1, "original_url"));
}

void BuildTestStoreWithSchemaFromM56(const base::FilePath& file) {
  sql::Connection connection;
  ASSERT_TRUE(
      connection.Open(file.Append(FILE_PATH_LITERAL("OfflinePages.db"))));
  ASSERT_TRUE(connection.is_open());
  ASSERT_TRUE(connection.BeginTransaction());
  ASSERT_TRUE(connection.Execute("CREATE TABLE " OFFLINE_PAGES_TABLE_V1
                                 "(offline_id INTEGER PRIMARY KEY NOT NULL, "
                                 "creation_time INTEGER NOT NULL, "
                                 "file_size INTEGER NOT NULL, "
                                 "last_access_time INTEGER NOT NULL, "
                                 "access_count INTEGER NOT NULL, "
                                 "expiration_time INTEGER NOT NULL DEFAULT 0, "
                                 "client_namespace VARCHAR NOT NULL, "
                                 "client_id VARCHAR NOT NULL, "
                                 "online_url VARCHAR NOT NULL, "
                                 "file_path VARCHAR NOT NULL, "
                                 "title VARCHAR NOT NULL DEFAULT '', "
                                 "original_url VARCHAR NOT NULL DEFAULT ''"
                                 ")"));
  ASSERT_TRUE(connection.CommitTransaction());
  sql::Statement statement(connection.GetUniqueStatement(
      "INSERT INTO " OFFLINE_PAGES_TABLE_V1
      "(offline_id, creation_time, file_size, "
      "last_access_time, access_count, client_namespace, "
      "client_id, online_url, file_path, expiration_time, title, original_url) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"));
  statement.BindInt64(0, kOfflineId);
  statement.BindInt(1, 0);
  statement.BindInt64(2, kFileSize);
  statement.BindInt(3, 0);
  statement.BindInt(4, 1);
  statement.BindCString(5, kTestClientNamespace);
  statement.BindString(6, kTestClientId2.id);
  statement.BindCString(7, kTestURL);
  statement.BindString(8, base::FilePath(kFilePath).MaybeAsASCII());
  statement.BindInt64(9, store_utils::ToDatabaseTime(base::Time::Now()));
  statement.BindString16(10, base::UTF8ToUTF16("Test title"));
  statement.BindCString(11, kOriginalTestURL);
  ASSERT_TRUE(statement.Run());
  ASSERT_TRUE(connection.DoesTableExist(OFFLINE_PAGES_TABLE_V1));
  ASSERT_TRUE(
      connection.DoesColumnExist(OFFLINE_PAGES_TABLE_V1, "expiration_time"));
}

void BuildTestStoreWithSchemaFromM57(const base::FilePath& file) {
  sql::Connection connection;
  ASSERT_TRUE(
      connection.Open(file.Append(FILE_PATH_LITERAL("OfflinePages.db"))));
  ASSERT_TRUE(connection.is_open());
  ASSERT_TRUE(connection.BeginTransaction());
  ASSERT_TRUE(connection.Execute("CREATE TABLE " OFFLINE_PAGES_TABLE_V1
                                 "(offline_id INTEGER PRIMARY KEY NOT NULL,"
                                 " creation_time INTEGER NOT NULL,"
                                 " file_size INTEGER NOT NULL,"
                                 " last_access_time INTEGER NOT NULL,"
                                 " access_count INTEGER NOT NULL,"
                                 " client_namespace VARCHAR NOT NULL,"
                                 " client_id VARCHAR NOT NULL,"
                                 " online_url VARCHAR NOT NULL,"
                                 " file_path VARCHAR NOT NULL,"
                                 " title VARCHAR NOT NULL DEFAULT '',"
                                 " original_url VARCHAR NOT NULL DEFAULT ''"
                                 ")"));
  ASSERT_TRUE(connection.CommitTransaction());
  sql::Statement statement(connection.GetUniqueStatement(
      "INSERT INTO " OFFLINE_PAGES_TABLE_V1
      "(offline_id, creation_time, file_size, "
      "last_access_time, access_count, client_namespace, "
      "client_id, online_url, file_path, title, original_url) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"));
  statement.BindInt64(0, kOfflineId);
  statement.BindInt(1, 0);
  statement.BindInt64(2, kFileSize);
  statement.BindInt(3, 0);
  statement.BindInt(4, 1);
  statement.BindCString(5, kTestClientNamespace);
  statement.BindString(6, kTestClientId2.id);
  statement.BindCString(7, kTestURL);
  statement.BindString(8, base::FilePath(kFilePath).MaybeAsASCII());
  statement.BindString16(9, base::UTF8ToUTF16("Test title"));
  statement.BindCString(10, kOriginalTestURL);
  ASSERT_TRUE(statement.Run());
  ASSERT_TRUE(connection.DoesTableExist(OFFLINE_PAGES_TABLE_V1));
  ASSERT_FALSE(
      connection.DoesColumnExist(OFFLINE_PAGES_TABLE_V1, "request_origin"));
}

void BuildTestStoreWithSchemaFromM61(const base::FilePath& file) {
  sql::Connection connection;
  ASSERT_TRUE(
      connection.Open(file.Append(FILE_PATH_LITERAL("OfflinePages.db"))));
  ASSERT_TRUE(connection.is_open());
  ASSERT_TRUE(connection.BeginTransaction());
  ASSERT_TRUE(connection.Execute("CREATE TABLE " OFFLINE_PAGES_TABLE_V1
                                 "(offline_id INTEGER PRIMARY KEY NOT NULL,"
                                 " creation_time INTEGER NOT NULL,"
                                 " file_size INTEGER NOT NULL,"
                                 " last_access_time INTEGER NOT NULL,"
                                 " access_count INTEGER NOT NULL,"
                                 " client_namespace VARCHAR NOT NULL,"
                                 " client_id VARCHAR NOT NULL,"
                                 " online_url VARCHAR NOT NULL,"
                                 " file_path VARCHAR NOT NULL,"
                                 " title VARCHAR NOT NULL DEFAULT '',"
                                 " original_url VARCHAR NOT NULL DEFAULT '',"
                                 " request_origin VARCHAR NOT NULL DEFAULT ''"
                                 ")"));
  ASSERT_TRUE(connection.CommitTransaction());
  sql::Statement statement(connection.GetUniqueStatement(
      "INSERT INTO " OFFLINE_PAGES_TABLE_V1
      "(offline_id, creation_time, file_size, "
      "last_access_time, access_count, client_namespace, "
      "client_id, online_url, file_path, title, original_url, "
      "request_origin) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"));
  statement.BindInt64(0, kOfflineId);
  statement.BindInt(1, 0);
  statement.BindInt64(2, kFileSize);
  statement.BindInt(3, 0);
  statement.BindInt(4, 1);
  statement.BindCString(5, kTestClientNamespace);
  statement.BindString(6, kTestClientId2.id);
  statement.BindCString(7, kTestURL);
  statement.BindString(8, base::FilePath(kFilePath).MaybeAsASCII());
  statement.BindString16(9, base::UTF8ToUTF16("Test title"));
  statement.BindCString(10, kOriginalTestURL);
  statement.BindString(11, kTestRequestOrigin);
  ASSERT_TRUE(statement.Run());
  ASSERT_TRUE(connection.DoesTableExist(OFFLINE_PAGES_TABLE_V1));
  ASSERT_FALSE(connection.DoesColumnExist(OFFLINE_PAGES_TABLE_V1, "digest"));
}

void InjectItemInM62Store(sql::Connection* db, const OfflinePageItem& item) {
  ASSERT_TRUE(db->BeginTransaction());
  sql::Statement statement(db->GetUniqueStatement(
      "INSERT INTO " OFFLINE_PAGES_TABLE_V1
      "(offline_id, creation_time, file_size, "
      "last_access_time, access_count, client_namespace, "
      "client_id, online_url, file_path, title, original_url, "
      "request_origin, system_download_id, file_missing_time, "
      "upgrade_attempt, digest) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"));
  statement.BindInt64(0, item.offline_id);
  statement.BindInt(1, store_utils::ToDatabaseTime(item.creation_time));
  statement.BindInt64(2, item.file_size);
  statement.BindInt(3, store_utils::ToDatabaseTime(item.last_access_time));
  statement.BindInt(4, item.access_count);
  statement.BindString(5, item.client_id.name_space);
  statement.BindString(6, item.client_id.id);
  statement.BindString(7, item.url.spec());
  statement.BindString(8, store_utils::ToDatabaseFilePath(item.file_path));
  statement.BindString16(9, item.title);
  statement.BindString(10, item.original_url.spec());
  statement.BindString(11, item.request_origin);
  statement.BindInt64(12, item.system_download_id);
  statement.BindInt(13, store_utils::ToDatabaseTime(item.file_missing_time));
  statement.BindInt(14, item.upgrade_attempt);
  statement.BindString(15, item.digest);
  ASSERT_TRUE(statement.Run());
  ASSERT_TRUE(db->CommitTransaction());
}

void BuildTestStoreWithSchemaFromM62(const base::FilePath& file) {
  sql::Connection connection;
  ASSERT_TRUE(
      connection.Open(file.Append(FILE_PATH_LITERAL("OfflinePages.db"))));
  ASSERT_TRUE(connection.is_open());
  ASSERT_TRUE(connection.BeginTransaction());
  ASSERT_TRUE(
      connection.Execute("CREATE TABLE " OFFLINE_PAGES_TABLE_V1
                         "(offline_id INTEGER PRIMARY KEY NOT NULL,"
                         " creation_time INTEGER NOT NULL,"
                         " file_size INTEGER NOT NULL,"
                         " last_access_time INTEGER NOT NULL,"
                         " access_count INTEGER NOT NULL,"
                         " system_download_id INTEGER NOT NULL DEFAULT 0,"
                         " file_missing_time INTEGER NOT NULL DEFAULT 0,"
                         " upgrade_attempt INTEGER NOT NULL DEFAULT 0,"
                         " client_namespace VARCHAR NOT NULL,"
                         " client_id VARCHAR NOT NULL,"
                         " online_url VARCHAR NOT NULL,"
                         " file_path VARCHAR NOT NULL,"
                         " title VARCHAR NOT NULL DEFAULT '',"
                         " original_url VARCHAR NOT NULL DEFAULT '',"
                         " request_origin VARCHAR NOT NULL DEFAULT '',"
                         " digest VARCHAR NOT NULL DEFAULT ''"
                         ")"));
  ASSERT_TRUE(connection.CommitTransaction());

  OfflinePageItemGenerator generator;
  generator.SetNamespace(kTestClientNamespace);
  generator.SetId(kTestClientId2.id);
  generator.SetUrl(GURL(kTestURL));
  generator.SetRequestOrigin(kTestRequestOrigin);
  generator.SetFileSize(kFileSize);
  OfflinePageItem test_item = generator.CreateItem();
  test_item.offline_id = kOfflineId;
  test_item.file_path = base::FilePath(kFilePath);
  InjectItemInM62Store(&connection, test_item);
}

void BuildTestStoreWithSchemaVersion1(const base::FilePath& file) {
  BuildTestStoreWithSchemaFromM62(file);
  sql::Connection connection;
  ASSERT_TRUE(
      connection.Open(file.Append(FILE_PATH_LITERAL("OfflinePages.db"))));
  ASSERT_TRUE(connection.is_open());
  ASSERT_TRUE(connection.BeginTransaction());
  sql::MetaTable meta_table;
  ASSERT_TRUE(meta_table.Init(&connection, 1, 1));
  ASSERT_TRUE(connection.CommitTransaction());

  OfflinePageItemGenerator generator;
  generator.SetUrl(GURL(kTestURL));
  generator.SetRequestOrigin(kTestRequestOrigin);
  generator.SetFileSize(kFileSize);

  generator.SetNamespace(kAsyncNamespace);
  InjectItemInM62Store(&connection, generator.CreateItem());
  generator.SetNamespace(kDownloadNamespace);
  InjectItemInM62Store(&connection, generator.CreateItem());
  generator.SetNamespace(kBrowserActionsNamespace);
  InjectItemInM62Store(&connection, generator.CreateItem());
  generator.SetNamespace(kNTPSuggestionsNamespace);
  InjectItemInM62Store(&connection, generator.CreateItem());
}

void BuildTestStoreWithSchemaVersion2(const base::FilePath& file) {
  BuildTestStoreWithSchemaVersion1(file);
  sql::Connection db;
  ASSERT_TRUE(db.Open(file.Append(FILE_PATH_LITERAL("OfflinePages.db"))));
  sql::MetaTable meta_table;
  ASSERT_TRUE(meta_table.Init(&db, OfflinePageMetadataStore::kCurrentVersion,
                              OfflinePageMetadataStore::kCompatibleVersion));
  static const char kSql[] =
      "CREATE TABLE page_thumbnails"
      " (offline_id INTEGER PRIMARY KEY NOT NULL,"
      " expiration INTEGER NOT NULL,"
      " thumbnail BLOB NOT NULL"
      ")";
  ASSERT_TRUE(db.Execute(kSql));
}

// Create an offline page item from a SQL result.  Expects complete rows with
// all columns present.
OfflinePageItem MakeOfflinePageItem(sql::Statement* statement) {
  int64_t id = statement->ColumnInt64(0);
  base::Time creation_time =
      store_utils::FromDatabaseTime(statement->ColumnInt64(1));
  int64_t file_size = statement->ColumnInt64(2);
  base::Time last_access_time =
      store_utils::FromDatabaseTime(statement->ColumnInt64(3));
  int access_count = statement->ColumnInt(4);
  int64_t system_download_id = statement->ColumnInt64(5);
  base::Time file_missing_time =
      store_utils::FromDatabaseTime(statement->ColumnInt64(6));
  int upgrade_attempt = statement->ColumnInt(7);
  ClientId client_id(statement->ColumnString(8), statement->ColumnString(9));
  GURL url(statement->ColumnString(10));
  base::FilePath path(
      store_utils::FromDatabaseFilePath(statement->ColumnString(11)));
  base::string16 title = statement->ColumnString16(12);
  GURL original_url(statement->ColumnString(13));
  std::string request_origin = statement->ColumnString(14);
  std::string digest = statement->ColumnString(15);

  OfflinePageItem item(url, id, client_id, path, file_size, creation_time);
  item.last_access_time = last_access_time;
  item.access_count = access_count;
  item.title = title;
  item.original_url = original_url;
  item.request_origin = request_origin;
  item.system_download_id = system_download_id;
  item.file_missing_time = file_missing_time;
  item.upgrade_attempt = upgrade_attempt;
  item.digest = digest;
  return item;
}

std::vector<OfflinePageItem> GetOfflinePagesSync(sql::Connection* db) {
  std::vector<OfflinePageItem> result;

  static const char kSql[] = "SELECT * FROM " OFFLINE_PAGES_TABLE_V1;
  sql::Statement statement(db->GetCachedStatement(SQL_FROM_HERE, kSql));

  while (statement.Step())
    result.push_back(MakeOfflinePageItem(&statement));

  if (!statement.Succeeded()) {
    result.clear();
  }
  return result;
}

class OfflinePageMetadataStoreTest : public testing::Test {
 public:
  OfflinePageMetadataStoreTest()
      : task_runner_(new base::TestMockTimeTaskRunner),
        task_runner_handle_(task_runner_) {
    EXPECT_TRUE(temp_directory_.CreateUniqueTempDir());
  }
  ~OfflinePageMetadataStoreTest() override{};

 protected:
  void TearDown() override {
    // Wait for all the pieces of the store to delete itself properly.
    PumpLoop();
  }

  std::unique_ptr<OfflinePageMetadataStore> BuildStore() {
    auto store = std::make_unique<OfflinePageMetadataStore>(
        base::ThreadTaskRunnerHandle::Get(), TempPath());
    PumpLoop();
    return store;
  }

  void PumpLoop() { task_runner_->RunUntilIdle(); }
  void FastForwardBy(base::TimeDelta delta) {
    task_runner_->FastForwardBy(delta);
  }

  base::TestMockTimeTaskRunner* task_runner() const {
    return task_runner_.get();
  }
  base::FilePath TempPath() const { return temp_directory_.GetPath(); }

  OfflinePageItem CheckThatStoreHasOneItem(OfflinePageMetadataStore* store) {
    std::vector<OfflinePageItem> pages = GetOfflinePages(store);
    EXPECT_EQ(1U, pages.size());
    return pages[0];
  }

  void CheckThatOfflinePageCanBeSaved(
      std::unique_ptr<OfflinePageMetadataStore> store) {
    size_t store_size = GetOfflinePages(store.get()).size();
    OfflinePageItem offline_page(GURL(kTestURL), 1234LL, kTestClientId1,
                                 base::FilePath(kFilePath), kFileSize);
    offline_page.title = base::UTF8ToUTF16("a title");
    offline_page.original_url = GURL(kOriginalTestURL);
    offline_page.system_download_id = kTestSystemDownloadId;
    offline_page.digest = kTestDigest;

    EXPECT_EQ(ItemActionStatus::SUCCESS,
              AddOfflinePage(store.get(), offline_page));

    // Close the store first to ensure file lock is removed.
    store.reset();
    store = BuildStore();
    std::vector<OfflinePageItem> pages = GetOfflinePages(store.get());
    ASSERT_EQ(store_size + 1, pages.size());
    if (store_size > 0 && pages[0].offline_id != offline_page.offline_id) {
      std::swap(pages[0], pages[1]);
    }
    EXPECT_EQ(offline_page, pages[0]);
  }

  void CheckThatPageThumbnailCanBeSaved(OfflinePageMetadataStore* store) {
    OfflinePageThumbnail thumbnail;
    thumbnail.offline_id = kOfflineId;
    thumbnail.expiration = kThumbnailExpiration;
    thumbnail.thumbnail = "content";

    AddThumbnail(store, thumbnail);
    std::vector<OfflinePageThumbnail> thumbnails = GetThumbnails(store);
    EXPECT_EQ(1UL, thumbnails.size());
    EXPECT_EQ(thumbnail, thumbnails[0]);
  }

  void VerifyMetaVersions() {
    sql::Connection connection;
    ASSERT_TRUE(connection.Open(temp_directory_.GetPath().Append(
        FILE_PATH_LITERAL("OfflinePages.db"))));
    ASSERT_TRUE(connection.is_open());
    EXPECT_TRUE(sql::MetaTable::DoesTableExist(&connection));
    sql::MetaTable meta_table;
    EXPECT_TRUE(meta_table.Init(&connection, 1, 1));

    EXPECT_EQ(OfflinePageMetadataStore::kCurrentVersion,
              meta_table.GetVersionNumber());
    EXPECT_EQ(OfflinePageMetadataStore::kCompatibleVersion,
              meta_table.GetCompatibleVersionNumber());
  }

  void LoadAndCheckStore() {
    auto store = std::make_unique<OfflinePageMetadataStore>(
        base::ThreadTaskRunnerHandle::Get(), TempPath());
    OfflinePageItem item = CheckThatStoreHasOneItem(store.get());
    CheckThatPageThumbnailCanBeSaved((OfflinePageMetadataStore*)store.get());
    CheckThatOfflinePageCanBeSaved(std::move(store));
    VerifyMetaVersions();
  }

  void LoadAndCheckStoreFromMetaVersion1AndUp() {
    // At meta version 1, more items were added to the database for testing,
    // which necessitates different checks.
    auto store = std::make_unique<OfflinePageMetadataStore>(
        base::ThreadTaskRunnerHandle::Get(), TempPath());
    std::vector<OfflinePageItem> pages = GetOfflinePages(store.get());
    EXPECT_EQ(5U, pages.size());

    // TODO(fgorski): Use persistent namespaces from the client policy
    // controller once an appropriate method is available.
    std::set<std::string> upgradeable_namespaces{
        kAsyncNamespace, kDownloadNamespace, kBrowserActionsNamespace,
        kNTPSuggestionsNamespace};

    for (const OfflinePageItem& page : pages) {
      if (upgradeable_namespaces.count(page.client_id.name_space) > 0)
        EXPECT_EQ(5, page.upgrade_attempt);
      else
        EXPECT_EQ(0, page.upgrade_attempt);
    }
    CheckThatPageThumbnailCanBeSaved((OfflinePageMetadataStore*)store.get());
    CheckThatOfflinePageCanBeSaved(std::move(store));
    VerifyMetaVersions();
  }

  template <typename T>
  T ExecuteSync(OfflinePageMetadataStore* store,
                base::OnceCallback<T(sql::Connection*)> run_callback,
                T default_value) {
    bool called = false;
    T result;
    auto result_callback = base::BindLambdaForTesting([&](T async_result) {
      result = std::move(async_result);
      called = true;
    });
    store->Execute<T>(std::move(run_callback), result_callback, default_value);
    PumpLoop();
    EXPECT_TRUE(called);
    return result;
  }

  void GetOfflinePagesAsync(
      OfflinePageMetadataStore* store,
      base::OnceCallback<void(std::vector<OfflinePageItem>)> callback) {
    auto run_callback = base::BindOnce(&GetOfflinePagesSync);
    store->Execute<std::vector<OfflinePageItem>>(std::move(run_callback),
                                                 std::move(callback), {});
  }

  std::vector<OfflinePageItem> GetOfflinePages(
      OfflinePageMetadataStore* store) {
    return ExecuteSync<std::vector<OfflinePageItem>>(
        store, base::BindOnce(&GetOfflinePagesSync), {});
  }

  ItemActionStatus AddOfflinePage(OfflinePageMetadataStore* store,
                                  const OfflinePageItem& item) {
    auto result_callback = base::BindLambdaForTesting([&](sql::Connection* db) {
      // Using 'INSERT OR FAIL' or 'INSERT OR ABORT' in the query below
      // causes debug builds to DLOG.
      static const char kSql[] =
          "INSERT OR IGNORE INTO " OFFLINE_PAGES_TABLE_V1
          " (offline_id, online_url, client_namespace, client_id, "
          "file_path, "
          "file_size, creation_time, last_access_time, access_count, "
          "title, original_url, request_origin, system_download_id, "
          "file_missing_time, upgrade_attempt, digest)"
          " VALUES "
          " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

      sql::Statement statement(db->GetCachedStatement(SQL_FROM_HERE, kSql));
      statement.BindInt64(0, item.offline_id);
      statement.BindString(1, item.url.spec());
      statement.BindString(2, item.client_id.name_space);
      statement.BindString(3, item.client_id.id);
      statement.BindString(4, store_utils::ToDatabaseFilePath(item.file_path));
      statement.BindInt64(5, item.file_size);
      statement.BindInt64(6, store_utils::ToDatabaseTime(item.creation_time));
      statement.BindInt64(7,
                          store_utils::ToDatabaseTime(item.last_access_time));
      statement.BindInt(8, item.access_count);
      statement.BindString16(9, item.title);
      statement.BindString(10, item.original_url.spec());
      statement.BindString(11, item.request_origin);
      statement.BindInt64(12, item.system_download_id);
      statement.BindInt64(13,
                          store_utils::ToDatabaseTime(item.file_missing_time));
      statement.BindInt(14, item.upgrade_attempt);
      statement.BindString(15, item.digest);

      if (!statement.Run())
        return ItemActionStatus::STORE_ERROR;
      if (db->GetLastChangeCount() == 0)
        return ItemActionStatus::ALREADY_EXISTS;
      return ItemActionStatus::SUCCESS;
    });
    return ExecuteSync<ItemActionStatus>(store, result_callback,
                                         ItemActionStatus::SUCCESS);
  }

  std::vector<OfflinePageThumbnail> GetThumbnails(
      OfflinePageMetadataStore* store) {
    std::vector<OfflinePageThumbnail> thumbnails;
    auto run_callback = base::BindLambdaForTesting([&](sql::Connection* db) {
      static const char kSql[] = "SELECT * FROM page_thumbnails";
      sql::Statement statement(db->GetCachedStatement(SQL_FROM_HERE, kSql));

      while (statement.Step()) {
        OfflinePageThumbnail thumb;
        thumb.offline_id = statement.ColumnInt64(0);
        thumb.expiration =
            store_utils::FromDatabaseTime(statement.ColumnInt64(1));
        statement.ColumnBlobAsString(2, &thumb.thumbnail);
        thumbnails.push_back(std::move(thumb));
      }

      EXPECT_TRUE(statement.Succeeded());
      return thumbnails;
    });
    return ExecuteSync<std::vector<OfflinePageThumbnail>>(store, run_callback,
                                                          {});
  }

  void AddThumbnail(OfflinePageMetadataStore* store,
                    const OfflinePageThumbnail& thumbnail) {
    std::vector<OfflinePageThumbnail> thumbnails;
    auto run_callback = base::BindLambdaForTesting([&](sql::Connection* db) {
      static const char kSql[] =
          "INSERT INTO page_thumbnails"
          " (offline_id, expiration, thumbnail) VALUES (?, ?, ?)";
      sql::Statement statement(db->GetCachedStatement(SQL_FROM_HERE, kSql));

      statement.BindInt64(0, thumbnail.offline_id);
      statement.BindInt64(1, store_utils::ToDatabaseTime(thumbnail.expiration));
      statement.BindString(2, thumbnail.thumbnail);
      EXPECT_TRUE(statement.Run());
      return thumbnails;
    });
    ExecuteSync<std::vector<OfflinePageThumbnail>>(store, run_callback, {});
  }

 protected:
  base::ScopedTempDir temp_directory_;
  scoped_refptr<base::TestMockTimeTaskRunner> task_runner_;
  base::ThreadTaskRunnerHandle task_runner_handle_;
};

// Loads empty store and makes sure that there are no offline pages stored in
// it.
TEST_F(OfflinePageMetadataStoreTest, LoadEmptyStore) {
  std::unique_ptr<OfflinePageMetadataStore> store(BuildStore());
  EXPECT_EQ(0U, GetOfflinePages(store.get()).size());
}

TEST_F(OfflinePageMetadataStoreTest, GetOfflinePagesFromInvalidStore) {
  std::unique_ptr<OfflinePageMetadataStore> store(BuildStore());

  // Because execute method is self-healing this part of the test expects a
  // positive results now.
  store->SetStateForTesting(StoreState::NOT_LOADED, false);
  EXPECT_EQ(0UL, GetOfflinePages(store.get()).size());
  EXPECT_EQ(StoreState::LOADED, store->GetStateForTesting());

  store->SetStateForTesting(StoreState::FAILED_LOADING, false);
  EXPECT_EQ(0UL, GetOfflinePages(store.get()).size());
  EXPECT_EQ(StoreState::FAILED_LOADING, store->GetStateForTesting());

  store->SetStateForTesting(StoreState::FAILED_RESET, false);
  EXPECT_EQ(0UL, GetOfflinePages(store.get()).size());
  EXPECT_EQ(StoreState::FAILED_RESET, store->GetStateForTesting());

  store->SetStateForTesting(StoreState::LOADED, true);
  EXPECT_EQ(0UL, GetOfflinePages(store.get()).size());

  store->SetStateForTesting(StoreState::NOT_LOADED, true);
  EXPECT_EQ(0UL, GetOfflinePages(store.get()).size());

  store->SetStateForTesting(StoreState::FAILED_LOADING, false);
  EXPECT_EQ(0UL, GetOfflinePages(store.get()).size());

  store->SetStateForTesting(StoreState::FAILED_RESET, false);
  EXPECT_EQ(0UL, GetOfflinePages(store.get()).size());
}

// Loads a store which has an outdated schema.
// These tests would crash if it's not handling correctly when we're loading
// old version stores.
TEST_F(OfflinePageMetadataStoreTest, LoadVersion52Store) {
  BuildTestStoreWithSchemaFromM52(TempPath());
  LoadAndCheckStore();
}
TEST_F(OfflinePageMetadataStoreTest, LoadVersion53Store) {
  BuildTestStoreWithSchemaFromM53(TempPath());
  LoadAndCheckStore();
}
TEST_F(OfflinePageMetadataStoreTest, LoadVersion54Store) {
  BuildTestStoreWithSchemaFromM54(TempPath());
  LoadAndCheckStore();
}
TEST_F(OfflinePageMetadataStoreTest, LoadVersion55Store) {
  BuildTestStoreWithSchemaFromM55(TempPath());
  LoadAndCheckStore();
}
TEST_F(OfflinePageMetadataStoreTest, LoadVersion56Store) {
  BuildTestStoreWithSchemaFromM56(TempPath());
  LoadAndCheckStore();
}
TEST_F(OfflinePageMetadataStoreTest, LoadVersion57Store) {
  BuildTestStoreWithSchemaFromM57(TempPath());
  LoadAndCheckStore();
}
TEST_F(OfflinePageMetadataStoreTest, LoadVersion61Store) {
  BuildTestStoreWithSchemaFromM61(TempPath());
  LoadAndCheckStore();
}
TEST_F(OfflinePageMetadataStoreTest, LoadVersion62Store) {
  BuildTestStoreWithSchemaFromM62(TempPath());
  LoadAndCheckStore();
}

TEST_F(OfflinePageMetadataStoreTest, LoadStoreWithMetaVersion1) {
  BuildTestStoreWithSchemaVersion1(TempPath());
  LoadAndCheckStoreFromMetaVersion1AndUp();
}

TEST_F(OfflinePageMetadataStoreTest, LoadStoreWithMetaVersion2) {
  BuildTestStoreWithSchemaVersion2(TempPath());
  LoadAndCheckStoreFromMetaVersion1AndUp();
}

// Adds metadata of an offline page into a store and then opens the store
// again to make sure that stored metadata survives store restarts.
TEST_F(OfflinePageMetadataStoreTest, AddOfflinePage) {
  CheckThatOfflinePageCanBeSaved(BuildStore());
}

TEST_F(OfflinePageMetadataStoreTest, AddSameOfflinePageTwice) {
  std::unique_ptr<OfflinePageMetadataStore> store(BuildStore());

  OfflinePageItem offline_page(GURL(kTestURL), 1234LL, kTestClientId1,
                               base::FilePath(kFilePath), kFileSize);
  offline_page.title = base::UTF8ToUTF16("a title");

  EXPECT_EQ(ItemActionStatus::SUCCESS,
            AddOfflinePage(store.get(), offline_page));

  EXPECT_EQ(ItemActionStatus::ALREADY_EXISTS,
            AddOfflinePage(store.get(), offline_page));
}

// Adds metadata of multiple offline pages into a store and removes some.
TEST_F(OfflinePageMetadataStoreTest, AddRemoveMultipleOfflinePages) {
  std::unique_ptr<OfflinePageMetadataStore> store(BuildStore());

  // Add an offline page.
  OfflinePageItem offline_page_1(GURL(kTestURL), 12345LL, kTestClientId1,
                                 base::FilePath(kFilePath), kFileSize);
  EXPECT_EQ(ItemActionStatus::SUCCESS,
            AddOfflinePage(store.get(), offline_page_1));

  // Add anther offline page.
  base::FilePath file_path_2 =
      base::FilePath(FILE_PATH_LITERAL("//other.page.com.mhtml"));
  OfflinePageItem offline_page_2(GURL("https://other.page.com"), 5678LL,
                                 kTestClientId2, file_path_2, 12345,
                                 base::Time::Now(), kTestRequestOrigin);
  offline_page_2.original_url = GURL("https://example.com/bar");
  offline_page_2.system_download_id = kTestSystemDownloadId;
  offline_page_2.digest = kTestDigest;

  EXPECT_EQ(ItemActionStatus::SUCCESS,
            AddOfflinePage(store.get(), offline_page_2));

  // Get all pages from the store.
  std::vector<OfflinePageItem> pages = GetOfflinePages(store.get());
  EXPECT_EQ(2U, pages.size());

  // Close and reload the store.
  store.reset();
  store = BuildStore();
  pages = GetOfflinePages(store.get());
  ASSERT_EQ(2U, pages.size());
  EXPECT_EQ(offline_page_2, pages[0]);
}

TEST_F(OfflinePageMetadataStoreTest, StoreCloses) {
  std::unique_ptr<OfflinePageMetadataStore> store(BuildStore());
  GetOfflinePages(store.get());

  EXPECT_TRUE(task_runner()->HasPendingTask());
  EXPECT_LT(base::TimeDelta(), task_runner()->NextPendingTaskDelay());

  FastForwardBy(OfflinePageMetadataStore::kClosingDelay);
  PumpLoop();
  EXPECT_EQ(StoreState::NOT_LOADED, store->GetStateForTesting());

  // Ensure that next call to the store will actually reinitialize it.
  EXPECT_EQ(0U, GetOfflinePages(store.get()).size());
  EXPECT_EQ(StoreState::LOADED, store->GetStateForTesting());
}

TEST_F(OfflinePageMetadataStoreTest, MultiplePendingCalls) {
  auto store = std::make_unique<OfflinePageMetadataStore>(
      base::ThreadTaskRunnerHandle::Get(), TempPath());
  EXPECT_FALSE(task_runner()->HasPendingTask());
  EXPECT_EQ(StoreState::NOT_LOADED, store->GetStateForTesting());

  // First call flips the state to initializing.
  // Subsequent calls should be pending until store is initialized.
  int callback_count = 0;
  auto get_complete =
      base::BindLambdaForTesting([&](std::vector<OfflinePageItem> pages) {
        ++callback_count;
        EXPECT_TRUE(pages.empty());
      });
  GetOfflinePagesAsync(store.get(), get_complete);
  EXPECT_EQ(StoreState::INITIALIZING, store->GetStateForTesting());

  GetOfflinePagesAsync(store.get(), get_complete);
  EXPECT_EQ(0U, GetOfflinePages(store.get()).size());
  EXPECT_EQ(StoreState::LOADED, store->GetStateForTesting());
  EXPECT_EQ(2, callback_count);
}

}  // namespace
}  // namespace offline_pages
