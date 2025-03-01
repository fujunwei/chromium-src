// Copyright 2013 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "ios/chrome/browser/ui/settings/passphrase_collection_view_controller_test.h"

#import <UIKit/UIKit.h>

#include <memory>

#include "base/strings/sys_string_conversions.h"
#include "components/browser_sync/profile_sync_service_mock.h"
#include "components/pref_registry/pref_registry_syncable.h"
#include "components/sync_preferences/pref_service_mock_factory.h"
#include "components/sync_preferences/pref_service_syncable.h"
#include "ios/chrome/browser/browser_state/chrome_browser_state.h"
#include "ios/chrome/browser/browser_state/test_chrome_browser_state.h"
#include "ios/chrome/browser/prefs/browser_prefs.h"
#include "ios/chrome/browser/signin/authentication_service_factory.h"
#import "ios/chrome/browser/signin/authentication_service_fake.h"
#include "ios/chrome/browser/sync/ios_chrome_profile_sync_test_util.h"
#include "ios/chrome/browser/sync/profile_sync_service_factory.h"
#include "ios/chrome/browser/sync/sync_setup_service.h"
#include "ios/chrome/browser/sync/sync_setup_service_factory.h"
#import "ios/chrome/browser/ui/settings/settings_navigation_controller.h"
#import "ios/public/provider/chrome/browser/signin/fake_chrome_identity_service.h"
#import "testing/gtest_mac.h"
#include "testing/platform_test.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

using testing::DefaultValue;
using testing::NiceMock;
using testing::Return;

std::unique_ptr<sync_preferences::PrefServiceSyncable> CreatePrefService() {
  sync_preferences::PrefServiceMockFactory factory;
  scoped_refptr<user_prefs::PrefRegistrySyncable> registry(
      new user_prefs::PrefRegistrySyncable);
  std::unique_ptr<sync_preferences::PrefServiceSyncable> prefs =
      factory.CreateSyncable(registry.get());
  RegisterBrowserStatePrefs(registry.get());
  return prefs;
}

std::unique_ptr<KeyedService>
PassphraseCollectionViewControllerTest::CreateNiceProfileSyncServiceMock(
    web::BrowserState* context) {
  browser_sync::ProfileSyncService::InitParams init_params =
      CreateProfileSyncServiceParamsForTest(
          nullptr, ios::ChromeBrowserState::FromBrowserState(context));
  return std::make_unique<NiceMock<browser_sync::ProfileSyncServiceMock>>(
      &init_params);
}

PassphraseCollectionViewControllerTest::PassphraseCollectionViewControllerTest()
    : CollectionViewControllerTest(),
      fake_sync_service_(NULL),
      default_auth_error_(GoogleServiceAuthError::NONE) {}

PassphraseCollectionViewControllerTest::
    ~PassphraseCollectionViewControllerTest() {}

void PassphraseCollectionViewControllerTest::SetUp() {
  CollectionViewControllerTest::SetUp();

  // Set up the default return values for non-trivial return types.
  DefaultValue<const GoogleServiceAuthError&>::Set(default_auth_error_);
  DefaultValue<syncer::SyncCycleSnapshot>::Set(default_sync_cycle_snapshot_);

  TestChromeBrowserState::Builder test_cbs_builder;
  test_cbs_builder.AddTestingFactory(
      AuthenticationServiceFactory::GetInstance(),
      AuthenticationServiceFake::CreateAuthenticationService);
  test_cbs_builder.SetPrefService(CreatePrefService());
  chrome_browser_state_ = test_cbs_builder.Build();

  fake_sync_service_ = static_cast<browser_sync::ProfileSyncServiceMock*>(
      ProfileSyncServiceFactory::GetInstance()->SetTestingFactoryAndUse(
          chrome_browser_state_.get(), CreateNiceProfileSyncServiceMock));
  ON_CALL(*fake_sync_service_, GetRegisteredDataTypes())
      .WillByDefault(Return(syncer::ModelTypeSet()));
  fake_sync_service_->Initialize();

  // Set up non-default return values for our sync service mock.
  ON_CALL(*fake_sync_service_, IsPassphraseRequired())
      .WillByDefault(Return(true));
  ON_CALL(*fake_sync_service_, GetState())
      .WillByDefault(Return(syncer::SyncService::State::ACTIVE));

  ios::FakeChromeIdentityService* identityService =
      ios::FakeChromeIdentityService::GetInstanceFromChromeProvider();
  identityService->AddIdentities(@[ @"identity1" ]);
  ChromeIdentity* identity =
      [identityService->GetAllIdentitiesSortedForDisplay() objectAtIndex:0];
  AuthenticationServiceFactory::GetForBrowserState(chrome_browser_state_.get())
      ->SignIn(identity, "");
}

void PassphraseCollectionViewControllerTest::SetUpNavigationController(
    UIViewController* test_controller) {
  dummy_controller_ = [[UIViewController alloc] init];
  nav_controller_ = [[SettingsNavigationController alloc]
      initWithRootViewController:dummy_controller_
                    browserState:chrome_browser_state_.get()
                        delegate:nil];
  [nav_controller_ pushViewController:test_controller animated:NO];
}
