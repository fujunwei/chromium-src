// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "ios/chrome/browser/ui/tab_grid/tab_grid_mediator.h"

#include <memory>

#include "base/scoped_observer.h"
#include "base/strings/sys_string_conversions.h"
#include "components/favicon/ios/web_favicon_driver.h"
#include "ios/chrome/browser/browser_state/chrome_browser_state.h"
#include "ios/chrome/browser/chrome_url_constants.h"
#import "ios/chrome/browser/chrome_url_util.h"
#include "ios/chrome/browser/experimental_flags.h"
#import "ios/chrome/browser/snapshots/snapshot_cache.h"
#import "ios/chrome/browser/snapshots/snapshot_cache_factory.h"
#import "ios/chrome/browser/snapshots/snapshot_tab_helper.h"
#import "ios/chrome/browser/tabs/tab_model.h"
#import "ios/chrome/browser/ui/tab_grid/grid/grid_consumer.h"
#import "ios/chrome/browser/ui/tab_grid/grid/grid_item.h"
#import "ios/chrome/browser/web/tab_id_tab_helper.h"
#include "ios/chrome/browser/web_state_list/web_state_list.h"
#import "ios/chrome/browser/web_state_list/web_state_list_observer_bridge.h"
#import "ios/chrome/browser/web_state_list/web_state_list_serialization.h"
#include "ios/chrome/browser/web_state_list/web_state_opener.h"
#include "ios/web/public/web_state/web_state.h"
#import "ios/web/public/web_state/web_state_observer_bridge.h"
#include "ui/gfx/image/image.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

namespace {
// Constructs a GridItem from a |web_state|.
GridItem* CreateItem(web::WebState* web_state) {
  TabIdTabHelper* tab_helper = TabIdTabHelper::FromWebState(web_state);
  GridItem* item = [[GridItem alloc] initWithIdentifier:tab_helper->tab_id()];
  // chrome://newtab (NTP) tabs have no title.
  if (!IsURLNtp(web_state->GetVisibleURL())) {
    item.title = base::SysUTF16ToNSString(web_state->GetTitle());
  }
  return item;
}

// Constructs an array of GridItems from a |web_state_list|.
NSArray* CreateItems(WebStateList* web_state_list) {
  NSMutableArray* items = [[NSMutableArray alloc] init];
  for (int i = 0; i < web_state_list->count(); i++) {
    web::WebState* web_state = web_state_list->GetWebStateAt(i);
    [items addObject:CreateItem(web_state)];
  }
  return [items copy];
}

// Returns the ID of the active tab in |web_state_list|.
NSString* GetActiveTabId(WebStateList* web_state_list) {
  web::WebState* web_state = web_state_list->GetActiveWebState();
  if (!web_state)
    return nil;
  TabIdTabHelper* tab_helper = TabIdTabHelper::FromWebState(web_state);
  return tab_helper->tab_id();
}

// Returns the index of the tab with |identifier| in |web_state_list|. Returns
// -1 if not found.
int GetIndexOfTabWithId(WebStateList* web_state_list, NSString* identifier) {
  for (int i = 0; i < web_state_list->count(); i++) {
    web::WebState* web_state = web_state_list->GetWebStateAt(i);
    TabIdTabHelper* tab_helper = TabIdTabHelper::FromWebState(web_state);
    if ([identifier isEqualToString:tab_helper->tab_id()])
      return i;
  }
  return -1;
}

// Returns the WebState with |identifier| in |web_state_list|. Returns |nullptr|
// if not found.
web::WebState* GetWebStateWithId(WebStateList* web_state_list,
                                 NSString* identifier) {
  for (int i = 0; i < web_state_list->count(); i++) {
    web::WebState* web_state = web_state_list->GetWebStateAt(i);
    TabIdTabHelper* tab_helper = TabIdTabHelper::FromWebState(web_state);
    if ([identifier isEqualToString:tab_helper->tab_id()])
      return web_state;
  }
  return nullptr;
}

}  // namespace

@interface TabGridMediator ()<CRWWebStateObserver, WebStateListObserving>
// The list from the tab model.
@property(nonatomic, assign) WebStateList* webStateList;
// The UI consumer to which updates are made.
@property(nonatomic, weak) id<GridConsumer> consumer;
// The saved session window just before close all tabs is called.
@property(nonatomic, strong) SessionWindowIOS* closedSessionWindow;
@end

@implementation TabGridMediator {
  // Observers for WebStateList.
  std::unique_ptr<WebStateListObserverBridge> _webStateListObserverBridge;
  std::unique_ptr<ScopedObserver<WebStateList, WebStateListObserver>>
      _scopedWebStateListObserver;
  // Observer for WebStates.
  std::unique_ptr<web::WebStateObserverBridge> _webStateObserverBridge;
  std::unique_ptr<ScopedObserver<web::WebState, web::WebStateObserver>>
      _scopedWebStateObserver;
}

// Public properties.
@synthesize tabModel = _tabModel;
// Private properties.
@synthesize webStateList = _webStateList;
@synthesize consumer = _consumer;
@synthesize closedSessionWindow = _closedSessionWindow;

- (instancetype)initWithConsumer:(id<GridConsumer>)consumer {
  if (self = [super init]) {
    _consumer = consumer;
    _webStateListObserverBridge =
        std::make_unique<WebStateListObserverBridge>(self);
    _scopedWebStateListObserver =
        std::make_unique<ScopedObserver<WebStateList, WebStateListObserver>>(
            _webStateListObserverBridge.get());
    _webStateObserverBridge =
        std::make_unique<web::WebStateObserverBridge>(self);
    _scopedWebStateObserver =
        std::make_unique<ScopedObserver<web::WebState, web::WebStateObserver>>(
            _webStateObserverBridge.get());
  }
  return self;
}

#pragma mark - Public properties

- (void)setTabModel:(TabModel*)tabModel {
  _scopedWebStateListObserver->RemoveAll();
  _scopedWebStateObserver->RemoveAll();
  _tabModel = tabModel;
  _webStateList = tabModel.webStateList;
  if (_webStateList) {
    _scopedWebStateListObserver->Add(_webStateList);
    for (int i = 0; i < self.webStateList->count(); i++) {
      web::WebState* webState = self.webStateList->GetWebStateAt(i);
      _scopedWebStateObserver->Add(webState);
    }
    [self populateConsumerItems];
  }
}

#pragma mark - WebStateListObserving

- (void)webStateList:(WebStateList*)webStateList
    didInsertWebState:(web::WebState*)webState
              atIndex:(int)index
           activating:(BOOL)activating {
  [self.consumer insertItem:CreateItem(webState)
                    atIndex:index
             selectedItemID:GetActiveTabId(webStateList)];
  _scopedWebStateObserver->Add(webState);
}

- (void)webStateList:(WebStateList*)webStateList
     didMoveWebState:(web::WebState*)webState
           fromIndex:(int)fromIndex
             toIndex:(int)toIndex {
  TabIdTabHelper* tabHelper = TabIdTabHelper::FromWebState(webState);
  [self.consumer moveItemWithID:tabHelper->tab_id() toIndex:toIndex];
}

- (void)webStateList:(WebStateList*)webStateList
    didReplaceWebState:(web::WebState*)oldWebState
          withWebState:(web::WebState*)newWebState
               atIndex:(int)index {
  TabIdTabHelper* tabHelper = TabIdTabHelper::FromWebState(oldWebState);
  [self.consumer replaceItemID:tabHelper->tab_id()
                      withItem:CreateItem(newWebState)];
  _scopedWebStateObserver->Remove(oldWebState);
  _scopedWebStateObserver->Add(newWebState);
}

- (void)webStateList:(WebStateList*)webStateList
    didDetachWebState:(web::WebState*)webState
              atIndex:(int)index {
  TabIdTabHelper* tabHelper = TabIdTabHelper::FromWebState(webState);
  NSString* itemID = tabHelper->tab_id();
  [self.consumer removeItemWithID:itemID
                   selectedItemID:GetActiveTabId(webStateList)];
  _scopedWebStateObserver->Remove(webState);
}

- (void)webStateList:(WebStateList*)webStateList
    didChangeActiveWebState:(web::WebState*)newWebState
                oldWebState:(web::WebState*)oldWebState
                    atIndex:(int)atIndex
                     reason:(int)reason {
  // If the selected index changes as a result of the last webstate being
  // detached, atIndex will be -1.
  if (atIndex == -1) {
    [self.consumer selectItemWithID:nil];
    return;
  }

  TabIdTabHelper* tabHelper = TabIdTabHelper::FromWebState(newWebState);
  [self.consumer selectItemWithID:tabHelper->tab_id()];
}

#pragma mark - CRWWebStateObserver

- (void)webStateDidChangeTitle:(web::WebState*)webState {
  // Assumption: the ID of the webState didn't change as a result of this load.
  SnapshotTabHelper::FromWebState(webState)->RemoveSnapshot();
  TabIdTabHelper* tabHelper = TabIdTabHelper::FromWebState(webState);
  NSString* itemID = tabHelper->tab_id();
  [self.consumer replaceItemID:itemID withItem:CreateItem(webState)];
}

#pragma mark - GridCommands

- (void)addNewItem {
  [self insertNewItemAtIndex:self.webStateList->count()];
}

- (void)insertNewItemAtIndex:(NSUInteger)index {
  DCHECK(self.tabModel.browserState);
  web::WebState::CreateParams params(self.tabModel.browserState);
  std::unique_ptr<web::WebState> webState = web::WebState::Create(params);
  self.webStateList->InsertWebState(
      base::checked_cast<int>(index), std::move(webState),
      (WebStateList::INSERT_FORCE_INDEX | WebStateList::INSERT_ACTIVATE),
      WebStateOpener());
  web::WebState::OpenURLParams openParams(
      GURL(kChromeUINewTabURL), web::Referrer(),
      WindowOpenDisposition::CURRENT_TAB, ui::PAGE_TRANSITION_TYPED, false);
  self.webStateList->GetWebStateAt(index)->OpenURL(openParams);
}

- (void)moveItemWithID:(NSString*)itemID toIndex:(NSUInteger)destinationIndex {
  int sourceIndex = GetIndexOfTabWithId(self.webStateList, itemID);
  if (sourceIndex >= 0)
    self.webStateList->MoveWebStateAt(sourceIndex, destinationIndex);
}

- (void)selectItemWithID:(NSString*)itemID {
  int index = GetIndexOfTabWithId(self.webStateList, itemID);
  if (index >= 0)
    self.webStateList->ActivateWebStateAt(index);
}

- (void)closeItemWithID:(NSString*)itemID {
  int index = GetIndexOfTabWithId(self.webStateList, itemID);
  if (index >= 0)
    self.webStateList->CloseWebStateAt(index, WebStateList::CLOSE_USER_ACTION);
}

- (void)closeAllItems {
  // This is a no-op if |webStateList| is already empty.
  self.webStateList->CloseAllWebStates(WebStateList::CLOSE_USER_ACTION);
}

- (void)saveAndCloseAllItems {
  if (self.webStateList->empty())
    return;
  // Tell the cache to mark these images for deletion, rather than immediately
  // deleting them.
  DCHECK(self.tabModel.browserState);
  SnapshotCache* cache =
      SnapshotCacheFactory::GetForBrowserState(self.tabModel.browserState);
  for (int i = 0; i < self.webStateList->count(); i++) {
    web::WebState* webState = self.webStateList->GetWebStateAt(i);
    TabIdTabHelper* tabHelper = TabIdTabHelper::FromWebState(webState);
    [cache markImageWithSessionID:tabHelper->tab_id()];
  }
  self.closedSessionWindow = SerializeWebStateList(self.webStateList);
  self.webStateList->CloseAllWebStates(WebStateList::CLOSE_USER_ACTION);
}

- (void)undoCloseAllItems {
  if (!self.closedSessionWindow)
    return;
  DCHECK(self.tabModel.browserState);
  web::WebState::CreateParams createParams(self.tabModel.browserState);
  DeserializeWebStateList(
      self.webStateList, self.closedSessionWindow,
      base::BindRepeating(&web::WebState::CreateWithStorageSession,
                          createParams));
  self.closedSessionWindow = nil;
  // Unmark all images for deletion since they are now active tabs again.
  ios::ChromeBrowserState* browserState = self.tabModel.browserState;
  [SnapshotCacheFactory::GetForBrowserState(browserState) unmarkAllImages];
}

- (void)discardSavedClosedItems {
  if (!self.closedSessionWindow)
    return;
  self.closedSessionWindow = nil;
  // Delete all marked images from the cache.
  DCHECK(self.tabModel.browserState);
  ios::ChromeBrowserState* browserState = self.tabModel.browserState;
  [SnapshotCacheFactory::GetForBrowserState(browserState) removeMarkedImages];
}

#pragma mark - GridImageDataSource

- (void)snapshotForIdentifier:(NSString*)identifier
                   completion:(void (^)(UIImage*))completion {
  web::WebState* webState = GetWebStateWithId(self.webStateList, identifier);
  if (webState) {
    SnapshotTabHelper::FromWebState(webState)->RetrieveColorSnapshot(
        ^(UIImage* image) {
          if (image)
            completion(image);
        });
  }
}

- (void)faviconForIdentifier:(NSString*)identifier
                  completion:(void (^)(UIImage*))completion {
  web::WebState* webState = GetWebStateWithId(self.webStateList, identifier);
  if (!webState) {
    return;
  }
  // NTP tabs get no favicon.
  if (IsURLNtp(webState->GetVisibleURL())) {
    return;
  }
  UIImage* defaultFavicon;
  if (experimental_flags::IsCollectionsUIRebootEnabled()) {
    defaultFavicon = [UIImage imageNamed:@"default_world_favicon"];
  }
  defaultFavicon = [UIImage imageNamed:@"default_favicon"];
  completion(defaultFavicon);

  favicon::FaviconDriver* faviconDriver =
      favicon::WebFaviconDriver::FromWebState(webState);
  if (faviconDriver) {
    gfx::Image favicon = faviconDriver->GetFavicon();
    if (!favicon.IsEmpty())
      completion(favicon.ToUIImage());
  }
}

#pragma mark - Private

// Calls |-populateItems:selectedItemID:| on the consumer.
- (void)populateConsumerItems {
  if (self.webStateList->count() > 0) {
    [self.consumer populateItems:CreateItems(self.webStateList)
                  selectedItemID:GetActiveTabId(self.webStateList)];
  }
}

@end
