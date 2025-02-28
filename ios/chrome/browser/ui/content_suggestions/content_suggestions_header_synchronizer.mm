// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "ios/chrome/browser/ui/content_suggestions/content_suggestions_header_synchronizer.h"

#include "base/ios/ios_util.h"
#import "base/mac/foundation_util.h"
#import "ios/chrome/browser/ui/content_suggestions/cells/content_suggestions_cell.h"
#import "ios/chrome/browser/ui/content_suggestions/cells/content_suggestions_most_visited_action_cell.h"
#import "ios/chrome/browser/ui/content_suggestions/cells/content_suggestions_most_visited_cell.h"
#import "ios/chrome/browser/ui/content_suggestions/content_suggestions_collection_controlling.h"
#import "ios/chrome/browser/ui/content_suggestions/content_suggestions_collection_utils.h"
#import "ios/chrome/browser/ui/content_suggestions/content_suggestions_header_controlling.h"
#import "ios/chrome/browser/ui/content_suggestions/content_suggestions_view_controller.h"
#import "ios/chrome/browser/ui/uikit_ui_util.h"
#include "ios/web/public/features.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

namespace {
const CGFloat kShiftTilesDownAnimationDuration = 0.2;
const CGFloat kShiftTilesUpAnimationDuration = 0.25;

UIEdgeInsets SafeAreaInsetsForViewWithinNTP(UIView* view) {
  UIEdgeInsets insets = SafeAreaInsetsForView(view);
  if ((IsUIRefreshPhase1Enabled() ||
       base::FeatureList::IsEnabled(
           web::features::kBrowserContainerFullscreen)) &&
      !base::ios::IsRunningOnIOS11OrLater()) {
    // TODO(crbug.com/826369) Replace this when the NTP is contained by the
    // BVC with |self.collectionController.topLayoutGuide.length|.
    insets = UIEdgeInsetsMake(StatusBarHeight(), 0, 0, 0);
  }
  return insets;
}

}  // namespace

@interface ContentSuggestionsHeaderSynchronizer ()<UIGestureRecognizerDelegate>

@property(nonatomic, weak, readonly) UICollectionView* collectionView;
// |YES| if the fakebox header should be animated on scroll.
@property(nonatomic, assign) BOOL shouldAnimateHeader;
@property(nonatomic, weak) id<ContentSuggestionsCollectionControlling>
    collectionController;
@property(nonatomic, weak) id<ContentSuggestionsHeaderControlling>
    headerController;
@property(nonatomic, assign) CFTimeInterval shiftTileStartTime;
@property(nonatomic, strong) ProceduralBlock shiftUpCompletionBlock;

// Tap gesture recognizer when the omnibox is focused.
@property(nonatomic, strong) UITapGestureRecognizer* tapGestureRecognizer;
@end

@implementation ContentSuggestionsHeaderSynchronizer

@synthesize collectionController = _collectionController;
@synthesize headerController = _headerController;
@synthesize shouldAnimateHeader = _shouldAnimateHeader;
@synthesize shiftTileStartTime = _shiftTileStartTime;
@synthesize tapGestureRecognizer = _tapGestureRecognizer;
@synthesize collectionShiftingOffset = _collectionShiftingOffset;
@synthesize shiftUpCompletionBlock = _shiftUpCompletionBlock;

- (instancetype)
initWithCollectionController:
    (id<ContentSuggestionsCollectionControlling>)collectionController
            headerController:
                (id<ContentSuggestionsHeaderControlling>)headerController {
  self = [super init];
  if (self) {
    _shiftTileStartTime = -1;
    _shouldAnimateHeader = YES;

    _tapGestureRecognizer = [[UITapGestureRecognizer alloc]
        initWithTarget:self
                action:@selector(unfocusOmnibox)];
    [_tapGestureRecognizer setDelegate:self];

    _headerController = headerController;
    _collectionController = collectionController;

    _headerController.collectionSynchronizer = self;
    _collectionController.headerSynchronizer = self;

    _collectionShiftingOffset = 0;
  }
  return self;
}

#pragma mark - ContentSuggestionsCollectionSynchronizing

- (void)shiftTilesDown {
  [self.collectionView removeGestureRecognizer:self.tapGestureRecognizer];

  self.shouldAnimateHeader = YES;

  if (self.collectionShiftingOffset == 0 || self.collectionView.dragging) {
    self.collectionShiftingOffset = 0;
    [self updateFakeOmniboxOnCollectionScroll];
    return;
  }

  self.collectionController.scrolledToTop = NO;

  // CADisplayLink is used for this animation instead of the standard UIView
  // animation because the standard animation did not properly convert the
  // fakebox from its scrolled up mode to its scrolled down mode. Specifically,
  // calling |UICollectionView reloadData| adjacent to the standard animation
  // caused the fakebox's views to jump incorrectly. CADisplayLink avoids this
  // problem because it allows |shiftTilesDownAnimationDidFire| to directly
  // control each frame.
  CADisplayLink* link = [CADisplayLink
      displayLinkWithTarget:self
                   selector:@selector(shiftTilesDownAnimationDidFire:)];
  [link addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];
}

- (void)shiftTilesUpWithCompletionBlock:(ProceduralBlock)completion {
  // Add gesture recognizer to collection view when the omnibox is focused.
  [self.collectionView addGestureRecognizer:self.tapGestureRecognizer];

  if (self.collectionView.decelerating) {
    // Stop the scrolling if the scroll view is decelerating to prevent the
    // focus to be immediately lost.
    [self.collectionView setContentOffset:self.collectionView.contentOffset
                                 animated:NO];
  }

  CGFloat pinnedOffsetY = [self.headerController pinnedOffsetY];
  self.collectionShiftingOffset =
      MAX(0, pinnedOffsetY - self.collectionView.contentOffset.y);

  if (self.collectionController.scrolledToTop) {
    self.shouldAnimateHeader = NO;
    if (completion)
      completion();
    return;
  }

  self.collectionController.scrolledToTop = YES;
  self.shouldAnimateHeader = YES;
  self.shiftUpCompletionBlock = completion;

  // Layout the header for the constraints to be animated.
  [self.headerController layoutHeader];
  [self.collectionView.collectionViewLayout invalidateLayout];

  // Similar to -shiftTilesDown, also use a CADisplayLink so each contentOffset
  // tick forces an update for the fake omnibox. Otherwise the fakebox and its
  // label will be sized incorrectly when tapped.
  CADisplayLink* link = [CADisplayLink
      displayLinkWithTarget:self
                   selector:@selector(shiftTilesUpAnimationDidFire:)];
  [link addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSDefaultRunLoopMode];
}

- (void)invalidateLayout {
  [self updateFakeOmniboxOnNewWidth:self.collectionView.bounds.size.width];
  [self.collectionView.collectionViewLayout invalidateLayout];
}

#pragma mark - ContentSuggestionsHeaderSynchronizing

- (void)updateFakeOmniboxOnCollectionScroll {
  // Unfocus the omnibox when the scroll view is scrolled by the user (but not
  // when a scroll is triggered by layout/UIKit).
  if ([self.headerController isOmniboxFocused] && !self.shouldAnimateHeader &&
      self.collectionView.dragging) {
    [self.headerController unfocusOmnibox];
  }

  if (IsIPadIdiom() && !IsUIRefreshPhase1Enabled()) {
    return;
  }

  if (self.shouldAnimateHeader) {
    UIEdgeInsets insets = SafeAreaInsetsForViewWithinNTP(self.collectionView);
    [self.headerController
        updateFakeOmniboxForOffset:self.collectionView.contentOffset.y
                       screenWidth:self.collectionView.frame.size.width
                    safeAreaInsets:insets];
  }
}

- (void)updateFakeOmniboxOnNewWidth:(CGFloat)width {
  if (self.shouldAnimateHeader &&
      (IsUIRefreshPhase1Enabled() || !IsIPadIdiom())) {
    UIEdgeInsets insets = SafeAreaInsetsForViewWithinNTP(self.collectionView);
    [self.headerController
        updateFakeOmniboxForOffset:self.collectionView.contentOffset.y
                       screenWidth:width
                    safeAreaInsets:insets];
  } else {
    [self.headerController updateFakeOmniboxForWidth:width];
  }
}

- (void)updateConstraints {
  [self.headerController updateConstraints];
}

- (void)unfocusOmnibox {
  [self.headerController unfocusOmnibox];
}

- (CGFloat)pinnedOffsetY {
  return [self.headerController pinnedOffsetY];
}

- (CGFloat)headerHeight {
  return [self.headerController headerHeight];
}

#pragma mark - Private

// Convenience method to get the collection view of the suggestions.
- (UICollectionView*)collectionView {
  return [self.collectionController collectionView];
}

// Updates the collection view's scroll view offset for the next frame of the
// -shiftTilesUpWithCompletionBlock animation.
- (void)shiftTilesUpAnimationDidFire:(CADisplayLink*)link {
  // If this is the first frame of the animation, store the starting timestamp
  // and do nothing.
  if (self.shiftTileStartTime == -1) {
    self.shiftTileStartTime = link.timestamp;
    return;
  }

  CFTimeInterval timeElapsed = link.timestamp - self.shiftTileStartTime;
  double percentComplete = timeElapsed / kShiftTilesUpAnimationDuration;
  // Ensure that the percentage cannot be above 1.0.
  if (percentComplete > 1.0)
    percentComplete = 1.0;

  // Find how much the collection view should be scrolled up in the next frame.
  CGFloat pinnedOffsetY = [self.headerController pinnedOffsetY];
  CGFloat startingOffset = pinnedOffsetY - self.collectionShiftingOffset;
  CGFloat yOffset =
      startingOffset + percentComplete * (pinnedOffsetY - startingOffset);
  self.collectionView.contentOffset = CGPointMake(0, yOffset);

  if (percentComplete == 1.0) {
    [link invalidate];
    // Reset |shiftTileStartTime| to its sentinel value.
    self.shiftTileStartTime = -1;

    // Check to see if the collection are still scrolled to the top -- it's
    // possible (and difficult) to initiate a -shiftTilesDown before the
    // animation here completes.
    if (self.collectionController.scrolledToTop) {
      self.shouldAnimateHeader = NO;
      if (self.shiftUpCompletionBlock) {
        self.shiftUpCompletionBlock();
        self.shiftUpCompletionBlock = nil;
      }
    }
  }
}

// Updates the collection view's scroll view offset for the next frame of the
// shiftTilesDown animation.
- (void)shiftTilesDownAnimationDidFire:(CADisplayLink*)link {
  // If this is the first frame of the animation, store the starting timestamp
  // and do nothing.
  if (self.shiftTileStartTime == -1) {
    self.shiftTileStartTime = link.timestamp;
    return;
  }

  CFTimeInterval timeElapsed = link.timestamp - self.shiftTileStartTime;
  double percentComplete = timeElapsed / kShiftTilesDownAnimationDuration;
  // Ensure that the percentage cannot be above 1.0.
  if (percentComplete > 1.0)
    percentComplete = 1.0;

  // Find how much the collection view should be scrolled up in the next frame.
  CGFloat yOffset =
      (1.0 - percentComplete) * [self.headerController pinnedOffsetY] +
      percentComplete * ([self.headerController pinnedOffsetY] -
                         self.collectionShiftingOffset);
  self.collectionView.contentOffset = CGPointMake(0, yOffset);

  if (percentComplete == 1.0) {
    [link invalidate];
    self.collectionShiftingOffset = 0;
    // Reset |shiftTileStartTime| to its sentinel value.
    self.shiftTileStartTime = -1;
  }
}

#pragma mark - UIGestureRecognizerDelegate

- (BOOL)gestureRecognizer:(UIGestureRecognizer*)gestureRecognizer
       shouldReceiveTouch:(UITouch*)touch {
  BOOL isMostVisitedCell =
      content_suggestions::nearestAncestor(
          touch.view, [ContentSuggestionsMostVisitedCell class]) != nil;
  BOOL isMostVisitedActionCell =
      content_suggestions::nearestAncestor(
          touch.view, [ContentSuggestionsMostVisitedActionCell class]) != nil;
  BOOL isSuggestionCell =
      content_suggestions::nearestAncestor(
          touch.view, [ContentSuggestionsCell class]) != nil;
  return !isMostVisitedCell && !isMostVisitedActionCell && !isSuggestionCell;
}

- (UIView*)nearestAncestorOfView:(UIView*)view withClass:(Class)aClass {
  if (!view) {
    return nil;
  }
  if ([view isKindOfClass:aClass]) {
    return view;
  }
  return [self nearestAncestorOfView:[view superview] withClass:aClass];
}

@end
