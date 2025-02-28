// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "ios/chrome/browser/ui/settings/clear_browsing_data_table_view_controller.h"

#include "base/mac/foundation_util.h"
#include "ios/chrome/browser/browsing_data/browsing_data_remove_mask.h"
#import "ios/chrome/browser/ui/alert_coordinator/action_sheet_coordinator.h"
#import "ios/chrome/browser/ui/commands/application_commands.h"
#import "ios/chrome/browser/ui/settings/cells/table_view_clear_browsing_data_item.h"
#include "ios/chrome/browser/ui/settings/clear_browsing_data_local_commands.h"
#import "ios/chrome/browser/ui/settings/clear_browsing_data_manager.h"
#import "ios/chrome/browser/ui/settings/settings_collection_view_controller.h"
#import "ios/chrome/browser/ui/table_view/cells/table_view_cells_constants.h"
#import "ios/chrome/browser/ui/table_view/cells/table_view_text_button_item.h"
#import "ios/chrome/browser/ui/table_view/cells/table_view_text_link_item.h"
#import "ios/chrome/browser/ui/table_view/chrome_table_view_styler.h"
#include "ios/chrome/grit/ios_strings.h"
#include "ui/base/l10n/l10n_util.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

namespace {
// Separation space between sections.
const CGFloat kSeparationSpaceBetweenSections = 9;
}  // namespace

namespace ios {
class ChromeBrowserState;
}

@interface ClearBrowsingDataTableViewController ()<
    TableViewTextLinkCellDelegate,
    ClearBrowsingDataConsumer>

// TODO(crbug.com/850699): remove direct dependency and replace with
// delegate.
@property(nonatomic, readonly, strong) ClearBrowsingDataManager* dataManager;

// Browser state.
@property(nonatomic, assign) ios::ChromeBrowserState* browserState;

// Coordinator that managers a UIAlertController to clear browsing data.
@property(nonatomic, strong) ActionSheetCoordinator* actionSheetCoordinator;

// Reference to clear browsing data button for positioning popover confirmation
// dialog.
@property(nonatomic, strong) UIButton* clearBrowsingDataButton;

@end

@implementation ClearBrowsingDataTableViewController
@synthesize actionSheetCoordinator = _actionSheetCoordinator;
@synthesize browserState = _browserState;
@synthesize clearBrowsingDataButton = _clearBrowsingDataButton;
@synthesize dataManager = _dataManager;
@synthesize dispatcher = _dispatcher;
@synthesize localDispatcher = _localDispatcher;

#pragma mark - ViewController Lifecycle.

- (instancetype)initWithBrowserState:(ios::ChromeBrowserState*)browserState {
  self = [super initWithTableViewStyle:UITableViewStylePlain
                           appBarStyle:ChromeTableViewControllerStyleNoAppBar];
  if (self) {
    _browserState = browserState;
    _dataManager = [[ClearBrowsingDataManager alloc]
        initWithBrowserState:browserState
                    listType:ClearBrowsingDataListType::kListTypeTableView];
    _dataManager.consumer = self;
  }
  return self;
}

- (void)viewDidLoad {
  [super viewDidLoad];
  // TableView configuration
  self.tableView.estimatedRowHeight = 56;
  self.tableView.rowHeight = UITableViewAutomaticDimension;
  self.tableView.estimatedSectionHeaderHeight = 0;
  // Add a tableFooterView in order to disable separators at the bottom of the
  // tableView.
  self.tableView.tableFooterView = [[UIView alloc] init];
  self.styler.tableViewBackgroundColor = [UIColor clearColor];
  // Align cell separators with text label leading margin.
  [self.tableView
      setSeparatorInset:UIEdgeInsetsMake(0, kTableViewHorizontalSpacing, 0, 0)];

  // Navigation controller configuration.
  self.title = l10n_util::GetNSString(IDS_IOS_CLEAR_BROWSING_DATA_TITLE);
  // Adds the "Done" button and hooks it up to |dismiss|.
  UIBarButtonItem* dismissButton = [[UIBarButtonItem alloc]
      initWithBarButtonSystemItem:UIBarButtonSystemItemDone
                           target:self
                           action:@selector(dismiss)];
  dismissButton.accessibilityIdentifier = kSettingsDoneButtonId;
  self.navigationItem.rightBarButtonItem = dismissButton;

  [self loadModel];
}

- (void)loadModel {
  [super loadModel];
  [self.dataManager loadModel:self.tableViewModel];
}

- (void)dismiss {
  [self.localDispatcher dismissClearBrowsingDataWithCompletion:nil];
}

#pragma mark - UITableViewDataSource

- (UITableViewCell*)tableView:(UITableView*)tableView
        cellForRowAtIndexPath:(NSIndexPath*)indexPath {
  UITableViewCell* cellToReturn =
      [super tableView:tableView cellForRowAtIndexPath:indexPath];
  TableViewItem* item = [self.tableViewModel itemAtIndexPath:indexPath];
  switch (item.type) {
    case ItemTypeFooterSavedSiteData:
    case ItemTypeFooterClearSyncAndSavedSiteData:
    case ItemTypeFooterGoogleAccountAndMyActivity: {
      TableViewTextLinkCell* tableViewTextLinkCell =
          base::mac::ObjCCastStrict<TableViewTextLinkCell>(cellToReturn);
      [tableViewTextLinkCell setDelegate:self];
      tableViewTextLinkCell.selectionStyle = UITableViewCellSelectionStyleNone;
      // Hide the cell separator inset for footnotes.
      tableViewTextLinkCell.separatorInset =
          UIEdgeInsetsMake(0, tableViewTextLinkCell.bounds.size.width, 0, 0);
      break;
    }
    case ItemTypeClearBrowsingDataButton: {
      TableViewTextButtonCell* tableViewTextButtonCell =
          base::mac::ObjCCastStrict<TableViewTextButtonCell>(cellToReturn);
      tableViewTextButtonCell.selectionStyle =
          UITableViewCellSelectionStyleNone;
      [tableViewTextButtonCell.button
                 addTarget:self
                    action:@selector(showClearBrowsingDataAlertController:)
          forControlEvents:UIControlEventTouchUpInside];
      self.clearBrowsingDataButton = tableViewTextButtonCell.button;
      break;
    }
    case ItemTypeDataTypeBrowsingHistory:
    case ItemTypeDataTypeCookiesSiteData:
    case ItemTypeDataTypeCache:
    case ItemTypeDataTypeSavedPasswords:
    case ItemTypeDataTypeAutofill:
    default:
      break;
  }
  return cellToReturn;
}

#pragma mark - UITableViewDelegate

- (void)tableView:(UITableView*)tableView
    didSelectRowAtIndexPath:(NSIndexPath*)indexPath {
  [self.tableView deselectRowAtIndexPath:indexPath animated:YES];
  TableViewItem* item = [self.tableViewModel itemAtIndexPath:indexPath];
  DCHECK(item);
  switch (item.type) {
    case ItemTypeDataTypeBrowsingHistory:
    case ItemTypeDataTypeCookiesSiteData:
    case ItemTypeDataTypeCache:
    case ItemTypeDataTypeSavedPasswords:
    case ItemTypeDataTypeAutofill: {
      TableViewClearBrowsingDataItem* clearBrowsingDataItem =
          base::mac::ObjCCastStrict<TableViewClearBrowsingDataItem>(item);
      clearBrowsingDataItem.checked = !clearBrowsingDataItem.checked;
      [self reconfigureCellsForItems:@[ clearBrowsingDataItem ]];
      break;
    }
    case ItemTypeClearBrowsingDataButton:
    case ItemTypeFooterGoogleAccount:
    case ItemTypeFooterGoogleAccountAndMyActivity:
    case ItemTypeFooterSavedSiteData:
    case ItemTypeFooterClearSyncAndSavedSiteData:
    case ItemTypeTimeRange:
    default:
      break;
  }
}

- (CGFloat)tableView:(UITableView*)tableView
    heightForFooterInSection:(NSInteger)section {
  return kSeparationSpaceBetweenSections;
}

#pragma mark - TableViewTextLinkCellDelegate

- (void)tableViewTextLinkCell:(TableViewTextLinkCell*)cell
            didRequestOpenURL:(const GURL&)URL {
  GURL copiedURL(URL);
  [self.localDispatcher openURL:copiedURL];
}

#pragma mark - ClearBrowsingDataConsumer

- (void)updateCellsForItem:(ListItem*)item {
  [self reconfigureCellsForItems:@[ item ]];
}

- (void)removeBrowsingDataForBrowserState:(ios::ChromeBrowserState*)browserState
                               timePeriod:(browsing_data::TimePeriod)timePeriod
                               removeMask:(BrowsingDataRemoveMask)removeMask
                          completionBlock:(ProceduralBlock)completionBlock {
  [self.dispatcher removeBrowsingDataForBrowserState:browserState
                                          timePeriod:timePeriod
                                          removeMask:removeMask
                                     completionBlock:completionBlock];
}

#pragma mark - Private Helpers

- (void)showClearBrowsingDataAlertController:(id)sender {
  BrowsingDataRemoveMask dataTypeMaskToRemove =
      BrowsingDataRemoveMask::REMOVE_NOTHING;
  NSArray* dataTypeItems = [self.tableViewModel
      itemsInSectionWithIdentifier:SectionIdentifierDataTypes];
  for (TableViewClearBrowsingDataItem* dataTypeItem in dataTypeItems) {
    DCHECK([dataTypeItem isKindOfClass:[TableViewClearBrowsingDataItem class]]);
    if (dataTypeItem.checked) {
      dataTypeMaskToRemove = dataTypeMaskToRemove | dataTypeItem.dataTypeMask;
    }
  }
  // Get button's position in coordinate system of table view.
  DCHECK_EQ(self.clearBrowsingDataButton, sender);
  CGRect clearBrowsingDataButtonRect = [self.clearBrowsingDataButton
      convertRect:self.clearBrowsingDataButton.bounds
           toView:self.tableView];
  self.actionSheetCoordinator = [self.dataManager
      actionSheetCoordinatorWithDataTypesToRemove:dataTypeMaskToRemove
                               baseViewController:self
                                       sourceRect:clearBrowsingDataButtonRect
                                       sourceView:self.tableView];
  if (self.actionSheetCoordinator) {
    [self.actionSheetCoordinator start];
  }
}

@end
