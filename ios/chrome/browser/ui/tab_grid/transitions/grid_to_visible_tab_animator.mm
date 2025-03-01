// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "ios/chrome/browser/ui/tab_grid/transitions/grid_to_visible_tab_animator.h"

#import "ios/chrome/browser/ui/tab_grid/transitions/grid_transition_animation.h"
#import "ios/chrome/browser/ui/tab_grid/transitions/grid_transition_layout.h"
#import "ios/chrome/browser/ui/tab_grid/transitions/grid_transition_state_providing.h"
#import "ios/chrome/browser/ui/util/layout_guide_names.h"
#import "ios/chrome/browser/ui/util/named_guide.h"
#import "ios/chrome/browser/ui/util/property_animator_group.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

@interface GridToVisibleTabAnimator ()
@property(nonatomic, weak) id<GridTransitionStateProviding> stateProvider;
// Animation object for this transition.
@property(nonatomic, strong) GridTransitionAnimation* animation;
// Transition context passed into this object when the animation is started.
@property(nonatomic, weak) id<UIViewControllerContextTransitioning>
    transitionContext;
@end

@implementation GridToVisibleTabAnimator
@synthesize stateProvider = _stateProvider;
@synthesize animation = _animation;
@synthesize transitionContext = _transitionContext;

- (instancetype)initWithStateProvider:
    (id<GridTransitionStateProviding>)stateProvider {
  if ((self = [super init])) {
    _stateProvider = stateProvider;
  }
  return self;
}

- (NSTimeInterval)transitionDuration:
    (id<UIViewControllerContextTransitioning>)transitionContext {
  return 0.5;
}

- (void)animateTransition:
    (id<UIViewControllerContextTransitioning>)transitionContext {
  // Keep a pointer to the transition context for use in animation delegate
  // callbacks.
  self.transitionContext = transitionContext;

  // Get views and view controllers for this transition.
  UIView* containerView = [transitionContext containerView];
  UIViewController* presentedViewController = [transitionContext
      viewControllerForKey:UITransitionContextToViewControllerKey];
  UIView* presentedView =
      [transitionContext viewForKey:UITransitionContextToViewKey];

  // Add the presented view to the container. This isn't just for the
  // transition; this is how the presented view controller's view is added to
  // the view hierarchy.
  [containerView addSubview:presentedView];
  presentedView.frame =
      [transitionContext finalFrameForViewController:presentedViewController];
  presentedView.alpha = 0.0;

  // Get the layout of the grid for the transition.
  GridTransitionLayout* layout =
      [self.stateProvider layoutForTransitionContext:transitionContext];

  // Ask the state provider for the views to use when inserting the animation.
  UIView* proxyContainer =
      [self.stateProvider proxyContainerForTransitionContext:transitionContext];

  // Find the rect for the animating tab by getting the content area layout
  // guide.
  // Conceptually this transition is presenting a tab (a BVC). However,
  // currently the BVC instances are themselves contanted within a BVCContainer
  // view controller. This means that the |presentedView| is not the BVC's
  // view; rather it's the view of the view controller that contains the BVC.
  // Unfortunatley, the layout guide needed here is attached to the BVC's view,
  // which is the first (and only) subview of the BVCContainerViewController's
  // view.
  // TODO(crbug.com/860234) Clean up this arrangement.
  UIView* viewWithNamedGuides = presentedView.subviews[0];
  CGRect finalRect =
      [NamedGuide guideWithName:kContentAreaGuide view:viewWithNamedGuides]
          .layoutFrame;

  [layout.activeItem populateWithSnapshotsFromView:viewWithNamedGuides
                                        middleRect:finalRect];

  layout.expandedRect = [proxyContainer convertRect:viewWithNamedGuides.frame
                                           fromView:presentedView];

  NSTimeInterval duration = [self transitionDuration:transitionContext];
  // Create the animation view and insert it.
  self.animation = [[GridTransitionAnimation alloc]
      initWithLayout:layout
            duration:duration
           direction:GridAnimationDirectionExpanding];

  UIView* viewBehindProxies =
      [self.stateProvider proxyPositionForTransitionContext:transitionContext];
  [proxyContainer insertSubview:self.animation aboveSubview:viewBehindProxies];

  [self.animation.animator addCompletion:^(UIViewAnimatingPosition position) {
    BOOL finished = (position == UIViewAnimatingPositionEnd);
    [self gridTransitionAnimationDidFinish:finished];
  }];

  // Run the main animation.
  [self.animation.animator startAnimation];
}

- (void)gridTransitionAnimationDidFinish:(BOOL)finished {
  // Clean up the animation
  [self.animation removeFromSuperview];
  // If the transition was cancelled, remove the presented view.
  // If not, remove the grid view.
  UIView* gridView =
      [self.transitionContext viewForKey:UITransitionContextFromViewKey];
  UIView* presentedView =
      [self.transitionContext viewForKey:UITransitionContextToViewKey];
  if (self.transitionContext.transitionWasCancelled) {
    [presentedView removeFromSuperview];
  } else {
    // TODO(crbug.com/850507): Have the tab view animate itself in alongside
    // this transition instead of just setting the alpha here.
    presentedView.alpha = 1;
    [gridView removeFromSuperview];
  }
  // Mark the transition as completed.
  [self.transitionContext completeTransition:YES];
}

@end
