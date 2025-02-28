// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/ntp_tiles/custom_links_manager_impl.h"

#include <stdint.h>

#include <memory>

#include "base/strings/utf_string_conversions.h"
#include "components/sync_preferences/testing_pref_service_syncable.h"
#include "testing/gtest/include/gtest/gtest.h"

using sync_preferences::TestingPrefServiceSyncable;

namespace ntp_tiles {

namespace {

struct TestCaseItem {
  const char* url;
  const char* title;
};

const TestCaseItem kTestCase1[] = {{"http://foo1.com/", "Foo1"}};
const TestCaseItem kTestCase2[] = {
    {"http://foo1.com/", "Foo1"},
    {"http://foo2.com/", "Foo2"},
};
const TestCaseItem kTestCaseMax[] = {
    {"http://foo1.com/", "Foo1"}, {"http://foo2.com/", "Foo2"},
    {"http://foo3.com/", "Foo3"}, {"http://foo4.com/", "Foo4"},
    {"http://foo5.com/", "Foo5"}, {"http://foo6.com/", "Foo6"},
    {"http://foo7.com/", "Foo7"}, {"http://foo8.com/", "Foo8"},
    {"http://foo9.com/", "Foo9"}, {"http://foo10.com/", "Foo10"},
};

const char kTestTitle[] = "Test";
const char kTestUrl[] = "http://test.com/";

void AddTile(NTPTilesVector* tiles, const char* url, const char* title) {
  NTPTile tile;
  tile.url = GURL(url);
  tile.title = base::UTF8ToUTF16(title);
  tiles->push_back(std::move(tile));
}

NTPTilesVector FillTestTiles(base::span<const TestCaseItem> test_cases) {
  NTPTilesVector tiles;
  for (const auto& test_case : test_cases) {
    AddTile(&tiles, test_case.url, test_case.title);
  }
  return tiles;
}

std::vector<CustomLinksManager::Link> FillTestLinks(
    base::span<const TestCaseItem> test_cases) {
  std::vector<CustomLinksManager::Link> links;
  for (const auto& test_case : test_cases) {
    links.emplace_back(CustomLinksManager::Link{
        GURL(test_case.url), base::UTF8ToUTF16(test_case.title)});
  }
  return links;
}

}  // namespace

class CustomLinksManagerImplTest : public testing::Test {
 public:
  CustomLinksManagerImplTest() {
    CustomLinksManagerImpl::RegisterProfilePrefs(prefs_.registry());
    custom_links_ = std::make_unique<CustomLinksManagerImpl>(&prefs_);
  }

 protected:
  sync_preferences::TestingPrefServiceSyncable prefs_;
  std::unique_ptr<CustomLinksManagerImpl> custom_links_;

  DISALLOW_COPY_AND_ASSIGN(CustomLinksManagerImplTest);
};

TEST_F(CustomLinksManagerImplTest, InitializeOnlyOnce) {
  NTPTilesVector initial_tiles = FillTestTiles(kTestCase1);
  NTPTilesVector new_tiles = FillTestTiles(kTestCase2);
  std::vector<CustomLinksManager::Link> initial_links =
      FillTestLinks(kTestCase1);
  std::vector<CustomLinksManager::Link> empty_links;

  ASSERT_FALSE(custom_links_->IsInitialized());
  ASSERT_TRUE(custom_links_->GetLinks().empty());

  // Initialize.
  EXPECT_TRUE(custom_links_->Initialize(initial_tiles));
  EXPECT_EQ(initial_links, custom_links_->GetLinks());

  // Try to initialize again. This should fail and leave the links intact.
  EXPECT_FALSE(custom_links_->Initialize(new_tiles));
  EXPECT_EQ(initial_links, custom_links_->GetLinks());
}

TEST_F(CustomLinksManagerImplTest, UninitializeDeletesOldLinks) {
  NTPTilesVector initial_tiles = FillTestTiles(kTestCase1);
  std::vector<CustomLinksManager::Link> initial_links =
      FillTestLinks(kTestCase1);

  // Initialize.
  ASSERT_TRUE(custom_links_->Initialize(initial_tiles));
  ASSERT_EQ(initial_links, custom_links_->GetLinks());

  custom_links_->Uninitialize();
  EXPECT_TRUE(custom_links_->GetLinks().empty());

  // Initialize with no links.
  EXPECT_TRUE(custom_links_->Initialize(NTPTilesVector()));
  EXPECT_TRUE(custom_links_->GetLinks().empty());
}

TEST_F(CustomLinksManagerImplTest, ReInitializeWithNewLinks) {
  NTPTilesVector initial_tiles = FillTestTiles(kTestCase1);
  NTPTilesVector new_tiles = FillTestTiles(kTestCase2);
  std::vector<CustomLinksManager::Link> initial_links =
      FillTestLinks(kTestCase1);
  std::vector<CustomLinksManager::Link> new_links = FillTestLinks(kTestCase2);

  // Initialize.
  ASSERT_TRUE(custom_links_->Initialize(initial_tiles));
  ASSERT_EQ(initial_links, custom_links_->GetLinks());

  custom_links_->Uninitialize();
  ASSERT_TRUE(custom_links_->GetLinks().empty());

  // Initialize with new links.
  EXPECT_TRUE(custom_links_->Initialize(new_tiles));
  EXPECT_EQ(new_links, custom_links_->GetLinks());
}

TEST_F(CustomLinksManagerImplTest, AddLink) {
  NTPTilesVector initial_tiles = FillTestTiles(kTestCase1);
  std::vector<CustomLinksManager::Link> initial_links =
      FillTestLinks(kTestCase1);
  std::vector<CustomLinksManager::Link> expected_links = initial_links;
  expected_links.emplace_back(
      CustomLinksManager::Link{GURL(kTestUrl), base::UTF8ToUTF16(kTestTitle)});

  // Initialize.
  ASSERT_TRUE(custom_links_->Initialize(initial_tiles));
  ASSERT_EQ(initial_links, custom_links_->GetLinks());

  // Add link.
  EXPECT_TRUE(
      custom_links_->AddLink(GURL(kTestUrl), base::UTF8ToUTF16(kTestTitle)));
  EXPECT_EQ(expected_links, custom_links_->GetLinks());
}

TEST_F(CustomLinksManagerImplTest, AddLinkWhenAtMaxLinks) {
  NTPTilesVector initial_tiles = FillTestTiles(kTestCaseMax);
  std::vector<CustomLinksManager::Link> initial_links =
      FillTestLinks(kTestCaseMax);

  // Initialize.
  ASSERT_TRUE(custom_links_->Initialize(initial_tiles));
  ASSERT_EQ(initial_links, custom_links_->GetLinks());

  // Try to add link. This should fail and not modify the list.
  EXPECT_FALSE(
      custom_links_->AddLink(GURL(kTestUrl), base::UTF8ToUTF16(kTestTitle)));
  EXPECT_EQ(initial_links, custom_links_->GetLinks());
}

TEST_F(CustomLinksManagerImplTest, AddDuplicateLink) {
  NTPTilesVector initial_tiles = FillTestTiles(kTestCase1);
  std::vector<CustomLinksManager::Link> initial_links =
      FillTestLinks(kTestCase1);

  // Initialize.
  ASSERT_TRUE(custom_links_->Initialize(initial_tiles));
  ASSERT_EQ(initial_links, custom_links_->GetLinks());

  // Try to add duplicate link. This should fail and not modify the list.
  EXPECT_FALSE(custom_links_->AddLink(GURL(kTestCase1[0].url),
                                      base::UTF8ToUTF16(kTestCase1[0].title)));
  EXPECT_EQ(initial_links, custom_links_->GetLinks());
}

TEST_F(CustomLinksManagerImplTest, DeleteLink) {
  NTPTilesVector initial_tiles;
  AddTile(&initial_tiles, kTestUrl, kTestTitle);
  std::vector<CustomLinksManager::Link> initial_links({CustomLinksManager::Link{
      GURL(kTestUrl), base::UTF8ToUTF16(kTestTitle)}});

  // Initialize.
  ASSERT_TRUE(custom_links_->Initialize(initial_tiles));
  ASSERT_EQ(initial_links, custom_links_->GetLinks());

  // Delete link.
  EXPECT_TRUE(custom_links_->DeleteLink(GURL(kTestUrl)));
  EXPECT_TRUE(custom_links_->GetLinks().empty());
}

TEST_F(CustomLinksManagerImplTest, DeleteLinkWhenUrlDoesNotExist) {
  NTPTilesVector initial_tiles;

  // Initialize.
  ASSERT_TRUE(custom_links_->Initialize(initial_tiles));
  ASSERT_TRUE(custom_links_->GetLinks().empty());

  // Try to delete link. This should fail and not modify the list.
  EXPECT_FALSE(custom_links_->DeleteLink(GURL(kTestUrl)));
  EXPECT_TRUE(custom_links_->GetLinks().empty());
}

TEST_F(CustomLinksManagerImplTest, UndoDeleteLink) {
  NTPTilesVector initial_tiles;
  AddTile(&initial_tiles, kTestUrl, kTestTitle);
  std::vector<CustomLinksManager::Link> expected_links(
      {CustomLinksManager::Link{GURL(kTestUrl),
                                base::UTF8ToUTF16(kTestTitle)}});

  // Initialize.
  ASSERT_TRUE(custom_links_->Initialize(initial_tiles));
  ASSERT_EQ(expected_links, custom_links_->GetLinks());

  // Try to undo delete before delete is called. This should fail and not modify
  // the list.
  EXPECT_FALSE(custom_links_->UndoDeleteLink());
  EXPECT_EQ(expected_links, custom_links_->GetLinks());

  // Delete link.
  ASSERT_TRUE(custom_links_->DeleteLink(GURL(kTestUrl)));
  ASSERT_TRUE(custom_links_->GetLinks().empty());

  // Undo delete link.
  EXPECT_TRUE(custom_links_->UndoDeleteLink());
  EXPECT_EQ(expected_links, custom_links_->GetLinks());
}

TEST_F(CustomLinksManagerImplTest, UndoDeleteLinkAfterAdd) {
  NTPTilesVector initial_tiles;
  std::vector<CustomLinksManager::Link> expected_links(
      {CustomLinksManager::Link{GURL(kTestUrl),
                                base::UTF8ToUTF16(kTestTitle)}});

  // Initialize.
  ASSERT_TRUE(custom_links_->Initialize(initial_tiles));
  ASSERT_TRUE(custom_links_->GetLinks().empty());

  // Add link.
  ASSERT_TRUE(
      custom_links_->AddLink(GURL(kTestUrl), base::UTF8ToUTF16(kTestTitle)));
  ASSERT_EQ(expected_links, custom_links_->GetLinks());

  // Delete link.
  ASSERT_TRUE(custom_links_->DeleteLink(GURL(kTestUrl)));
  ASSERT_TRUE(custom_links_->GetLinks().empty());

  // Undo delete link.
  EXPECT_TRUE(custom_links_->UndoDeleteLink());
  EXPECT_EQ(expected_links, custom_links_->GetLinks());
}

TEST_F(CustomLinksManagerImplTest, UndoDeleteLinkWhenAtMaxLinks) {
  NTPTilesVector initial_tiles = FillTestTiles(kTestCaseMax);
  std::vector<CustomLinksManager::Link> intial_links =
      FillTestLinks(kTestCaseMax);
  std::vector<CustomLinksManager::Link> links_after_delete(intial_links);
  links_after_delete.pop_back();
  std::vector<CustomLinksManager::Link> links_after_add(links_after_delete);
  links_after_add.emplace_back(
      CustomLinksManager::Link{GURL(kTestUrl), base::UTF8ToUTF16(kTestTitle)});

  // Initialize.
  ASSERT_TRUE(custom_links_->Initialize(initial_tiles));
  ASSERT_EQ(intial_links, custom_links_->GetLinks());

  // Delete link.
  ASSERT_TRUE(custom_links_->DeleteLink(GURL(kTestCaseMax[9].url)));
  ASSERT_EQ(links_after_delete, custom_links_->GetLinks());

  // Add link. Should be at max links.
  ASSERT_TRUE(
      custom_links_->AddLink(GURL(kTestUrl), base::UTF8ToUTF16(kTestTitle)));
  ASSERT_EQ(links_after_add, custom_links_->GetLinks());

  // Try to undo delete link. This should fail and not modify the list.
  EXPECT_FALSE(custom_links_->UndoDeleteLink());
  EXPECT_EQ(links_after_add, custom_links_->GetLinks());
}

}  // namespace ntp_tiles
