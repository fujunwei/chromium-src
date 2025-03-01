// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/resource_coordinator/leveldb_site_characteristics_database.h"

#include "base/bind.h"
#include "base/files/file_util.h"
#include "base/files/scoped_temp_dir.h"
#include "base/strings/stringprintf.h"
#include "base/test/metrics/histogram_tester.h"
#include "base/test/scoped_task_environment.h"
#include "base/test/test_file_util.h"
#include "build/build_config.h"
#include "chrome/browser/resource_coordinator/site_characteristics.pb.h"
#include "testing/gtest/include/gtest/gtest.h"
#include "third_party/leveldatabase/leveldb_chrome.h"
#include "url/gurl.h"

namespace resource_coordinator {

namespace {

class ScopedReadOnlyDirectory {
 public:
  explicit ScopedReadOnlyDirectory(const base::FilePath& root_dir);
  ~ScopedReadOnlyDirectory() {
    permission_restorer_.reset();
    EXPECT_TRUE(base::DeleteFile(read_only_path_, true));
  }

  const base::FilePath& GetReadOnlyPath() { return read_only_path_; }

 private:
  base::FilePath read_only_path_;
  std::unique_ptr<base::FilePermissionRestorer> permission_restorer_;
};

ScopedReadOnlyDirectory::ScopedReadOnlyDirectory(
    const base::FilePath& root_dir) {
  EXPECT_TRUE(base::CreateTemporaryDirInDir(
      root_dir, FILE_PATH_LITERAL("read_only_path"), &read_only_path_));
  permission_restorer_ =
      std::make_unique<base::FilePermissionRestorer>(read_only_path_);
#if defined(OS_WIN)
  base::DenyFilePermission(read_only_path_, GENERIC_WRITE);
#else  // defined(OS_WIN)
  EXPECT_TRUE(base::MakeFileUnwritable(read_only_path_));
#endif
  EXPECT_FALSE(base::PathIsWritable(read_only_path_));
}

// Initialize a SiteCharacteristicsProto object with a test value (the same
// value is used to initialize all fields).
void InitSiteCharacteristicProto(SiteCharacteristicsProto* proto,
                                 ::google::protobuf::int64 test_value) {
  proto->set_last_loaded(test_value);

  SiteCharacteristicsFeatureProto feature_proto;
  feature_proto.set_observation_duration(test_value);
  feature_proto.set_use_timestamp(test_value);

  proto->mutable_updates_favicon_in_background()->CopyFrom(feature_proto);
  proto->mutable_updates_title_in_background()->CopyFrom(feature_proto);
  proto->mutable_uses_notifications_in_background()->CopyFrom(feature_proto);
  proto->mutable_uses_audio_in_background()->CopyFrom(feature_proto);
}

}  // namespace

class LevelDBSiteCharacteristicsDatabaseTest : public ::testing::Test {
 public:
  LevelDBSiteCharacteristicsDatabaseTest() {}

  void SetUp() override {
    EXPECT_TRUE(temp_dir_.CreateUniqueTempDir());
    OpenDB();
  }

  void TearDown() override {
    db_.reset();
    WaitForAsyncOperationsToComplete();
    EXPECT_TRUE(temp_dir_.Delete());
  }

  void OpenDB() { OpenDB(temp_dir_.GetPath()); }

  void OpenDB(base::FilePath path) {
    db_ = std::make_unique<LevelDBSiteCharacteristicsDatabase>(path);
    WaitForAsyncOperationsToComplete();
    EXPECT_TRUE(db_);
  }

  const base::FilePath& GetTempPath() { return temp_dir_.GetPath(); }

 protected:
  // Try to read an entry from the database, returns true if the entry is
  // present and false otherwise. |receiving_proto| will receive the protobuf
  // corresponding to this entry on success.
  bool ReadFromDB(const url::Origin& origin,
                  SiteCharacteristicsProto* receiving_proto) {
    EXPECT_TRUE(receiving_proto);
    bool success = false;
    auto init_callback = base::BindOnce(
        [](SiteCharacteristicsProto* receiving_proto, bool* success,
           base::Optional<SiteCharacteristicsProto> proto_opt) {
          *success = proto_opt.has_value();
          if (proto_opt)
            receiving_proto->CopyFrom(proto_opt.value());
        },
        base::Unretained(receiving_proto), base::Unretained(&success));
    db_->ReadSiteCharacteristicsFromDB(origin, std::move(init_callback));
    WaitForAsyncOperationsToComplete();
    return success;
  }

  // Add some entries to the database and returns a vector with their origins.
  std::vector<url::Origin> AddDummyEntriesToDB() {
    const size_t kEntryCount = 10;
    std::vector<url::Origin> site_origins;
    for (size_t i = 0; i < kEntryCount; ++i) {
      SiteCharacteristicsProto proto_temp;
      std::string origin_str = base::StringPrintf("http://%zu.com", i);
      InitSiteCharacteristicProto(&proto_temp,
                                  static_cast<::google::protobuf::int64>(i));
      EXPECT_TRUE(proto_temp.IsInitialized());
      url::Origin origin = url::Origin::Create(GURL(origin_str));
      db_->WriteSiteCharacteristicsIntoDB(origin, proto_temp);
      site_origins.emplace_back(origin);
    }
    WaitForAsyncOperationsToComplete();
    return site_origins;
  }

  void WaitForAsyncOperationsToComplete() { task_env_.RunUntilIdle(); }

  const url::Origin kDummyOrigin = url::Origin::Create(GURL("http://foo.com"));

  base::test::ScopedTaskEnvironment task_env_;
  base::ScopedTempDir temp_dir_;
  std::unique_ptr<LevelDBSiteCharacteristicsDatabase> db_;
};

TEST_F(LevelDBSiteCharacteristicsDatabaseTest, InitAndStoreSiteCharacteristic) {
  // Initializing an entry that doesn't exist in the database should fail.
  SiteCharacteristicsProto early_read_proto;
  EXPECT_FALSE(ReadFromDB(kDummyOrigin, &early_read_proto));

  // Add an entry to the database and make sure that we can read it back.
  ::google::protobuf::int64 test_value = 42;
  SiteCharacteristicsProto stored_proto;
  InitSiteCharacteristicProto(&stored_proto, test_value);
  db_->WriteSiteCharacteristicsIntoDB(kDummyOrigin, stored_proto);
  SiteCharacteristicsProto read_proto;
  EXPECT_TRUE(ReadFromDB(kDummyOrigin, &read_proto));
  EXPECT_TRUE(read_proto.IsInitialized());
  EXPECT_EQ(stored_proto.SerializeAsString(), read_proto.SerializeAsString());
}

TEST_F(LevelDBSiteCharacteristicsDatabaseTest, RemoveEntries) {
  std::vector<url::Origin> site_origins = AddDummyEntriesToDB();

  // Remove half the origins from the database.
  std::vector<url::Origin> site_origins_to_remove(
      site_origins.begin(), site_origins.begin() + site_origins.size() / 2);
  db_->RemoveSiteCharacteristicsFromDB(site_origins_to_remove);

  WaitForAsyncOperationsToComplete();

  // Verify that the origins were removed correctly.
  SiteCharacteristicsProto proto_temp;
  for (const auto& iter : site_origins_to_remove)
    EXPECT_FALSE(ReadFromDB(iter, &proto_temp));

  for (auto iter = site_origins.begin() + site_origins.size() / 2;
       iter != site_origins.end(); ++iter) {
    EXPECT_TRUE(ReadFromDB(*iter, &proto_temp));
  }

  // Clear the database.
  db_->ClearDatabase();

  WaitForAsyncOperationsToComplete();

  // Verify that no origin remains.
  for (auto iter : site_origins)
    EXPECT_FALSE(ReadFromDB(iter, &proto_temp));
}

TEST_F(LevelDBSiteCharacteristicsDatabaseTest, DatabaseRecoveryTest) {
  std::vector<url::Origin> site_origins = AddDummyEntriesToDB();

  db_.reset();

  EXPECT_TRUE(leveldb_chrome::CorruptClosedDBForTesting(temp_dir_.GetPath()));

  base::HistogramTester histogram_tester;
  histogram_tester.ExpectTotalCount("ResourceCoordinator.LocalDB.DatabaseInit",
                                    0);
  // Open the corrupt DB and ensure that the appropriate histograms gets
  // updated.
  OpenDB();
  EXPECT_TRUE(db_->DatabaseIsInitializedForTesting());
  histogram_tester.ExpectUniqueSample(
      "ResourceCoordinator.LocalDB.DatabaseInit", 1 /* kInitStatusCorruption */,
      1);
  histogram_tester.ExpectUniqueSample(
      "ResourceCoordinator.LocalDB.DatabaseInitAfterRepair",
      0 /* kInitStatusOk */, 1);

  // TODO(sebmarchand): try to induce an I/O error by deleting one of the
  // manifest files.
}

// Ensure that there's no fatal failures if we try using the database after
// failing to open it (all the events will be ignored).
TEST_F(LevelDBSiteCharacteristicsDatabaseTest, DatabaseOpeningFailure) {
  db_.reset();
  ScopedReadOnlyDirectory read_only_dir(GetTempPath());

  OpenDB(read_only_dir.GetReadOnlyPath());
  EXPECT_FALSE(db_->DatabaseIsInitializedForTesting());

  SiteCharacteristicsProto proto_temp;
  EXPECT_FALSE(
      ReadFromDB(url::Origin::Create(GURL("https://foo.com")), &proto_temp));
  WaitForAsyncOperationsToComplete();
  db_->WriteSiteCharacteristicsIntoDB(
      url::Origin::Create(GURL("https://foo.com")), proto_temp);
  WaitForAsyncOperationsToComplete();
  db_->RemoveSiteCharacteristicsFromDB({});
  WaitForAsyncOperationsToComplete();
  db_->ClearDatabase();
  WaitForAsyncOperationsToComplete();
}

}  // namespace resource_coordinator
