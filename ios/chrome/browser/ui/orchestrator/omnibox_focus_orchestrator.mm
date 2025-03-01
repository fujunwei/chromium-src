// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "ios/chrome/browser/ui/orchestrator/omnibox_focus_orchestrator.h"

#import "ios/chrome/browser/ui/orchestrator/location_bar_animatee.h"
#import "ios/chrome/browser/ui/orchestrator/toolbar_animatee.h"
#import "ios/chrome/common/material_timing.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

@implementation OmniboxFocusOrchestrator

@synthesize toolbarAnimatee = _toolbarAnimatee;
@synthesize locationBarAnimatee = _locationBarAnimatee;

- (void)transitionToStateOmniboxFocused:(BOOL)omniboxFocused
                        toolbarExpanded:(BOOL)toolbarExpanded
                               animated:(BOOL)animated {
  if (toolbarExpanded) {
    [self updateUIToExpandedState:animated];
  } else {
    [self updateUIToContractedState:animated];
  }

  // Make the rest of the animation happen on the next runloop when this
  // animation have calculated the final frame for the location bar.
  // This is necessary because expanding/contracting the toolbar is actually
  // changing the view layout. Therefore, the expand/contract animations are
  // actually moving views (through modifying the constraints). At the same time
  // the focus/defocus animation don't actually modify the view position, the
  // views remain in place, so it's better to animate them with transforms.
  // The cleanest way to compute and perform the transform animation together
  // with a constraint animation seems to be to let the constraint animation
  // start and compute the final frames, then perform the transform animation.
  dispatch_async(dispatch_get_main_queue(), ^{
    if (omniboxFocused) {
      [self focusOmniboxAnimated:animated];
    } else {
      [self defocusOmniboxAnimated:animated];
    }
  });
}

#pragma mark - Private

- (void)focusOmniboxAnimated:(BOOL)animated {
  // Cleans up after the animation.
  // The argument is necessary as this is used as |completion| in UIView
  // animateWithBlock: call.
  auto cleanup = ^(BOOL __unused complete) {
    [self.locationBarAnimatee setEditViewHidden:NO];
    [self.locationBarAnimatee setSteadyViewHidden:YES];
    [self.locationBarAnimatee resetTransforms];
  };

  if (animated) {
    // Prepare for animation.
    [self.locationBarAnimatee offsetEditViewToMatchSteadyView];
    // Make edit view transparent, but not hidden.
    [self.locationBarAnimatee setEditViewHidden:NO];
    [self.locationBarAnimatee setEditViewFaded:YES];

    CGFloat duration = ios::material::kDuration1;

    [UIView animateWithDuration:duration
                          delay:0
                        options:UIViewAnimationCurveEaseInOut
                     animations:^{
                       [self.locationBarAnimatee
                               resetEditViewOffsetAndOffsetSteadyViewToMatch];
                     }
                     completion:cleanup];

    // Fading the views happens with a different timing for a better visual
    // effect. The steady view looks like an ordinary label, and it fades before
    // the animation is complete. The edit view will be in pre-edit state, so it
    // looks like selected text. Since the selection is blue, it looks
    // overwhelming if faded in at the same time as the steady view. So it fades
    // in faster and later into the animation to look better.
    [UIView animateWithDuration:duration * 0.8
                          delay:duration * 0.1
                        options:UIViewAnimationCurveEaseInOut
                     animations:^{
                       [self.locationBarAnimatee setSteadyViewFaded:YES];
                     }
                     completion:nil];

    [UIView animateWithDuration:duration * 0.6
                          delay:duration * 0.4
                        options:UIViewAnimationCurveEaseInOut
                     animations:^{
                       [self.locationBarAnimatee setEditViewFaded:NO];
                     }
                     completion:nil];
  } else {
    cleanup(YES);
  }
}

- (void)defocusOmniboxAnimated:(BOOL)animated {
  // Cleans up after the animation.
  // The argument is necessary as this is used as |completion| in UIView
  // animateWithBlock: call.
  void (^cleanup)(BOOL _) = ^(BOOL _) {
    [self.locationBarAnimatee setEditViewHidden:YES];
    [self.locationBarAnimatee setSteadyViewHidden:NO];
    [self.locationBarAnimatee resetTransforms];
  };

  if (animated) {
    // Prepare for animation.
    [self.locationBarAnimatee offsetSteadyViewToMatchEditView];
    // Make steady view transparent, but not hidden.
    [self.locationBarAnimatee setSteadyViewHidden:NO];
    [self.locationBarAnimatee setSteadyViewFaded:YES];

    CGFloat duration = ios::material::kDuration1;

    [UIView animateWithDuration:duration
                          delay:0
                        options:UIViewAnimationCurveEaseInOut
                     animations:^{
                       [self.locationBarAnimatee
                               resetSteadyViewOffsetAndOffsetEditViewToMatch];
                     }
                     completion:cleanup];

    // These timings are explained in a comment in
    // focusOmniboxAnimated:shouldExpand:.
    [UIView animateWithDuration:duration * 0.8
                          delay:duration * 0.1
                        options:UIViewAnimationCurveEaseInOut
                     animations:^{
                       [self.locationBarAnimatee setEditViewFaded:YES];
                     }
                     completion:nil];

    [UIView animateWithDuration:duration * 0.6
                          delay:duration * 0.4
                        options:UIViewAnimationCurveEaseInOut
                     animations:^{
                       [self.locationBarAnimatee setSteadyViewFaded:NO];
                     }
                     completion:nil];

  } else {
    cleanup(YES);
  }
}

// Updates the UI elements reflect the toolbar expanded state, |animated| or
// not.
- (void)updateUIToExpandedState:(BOOL)animated {
  void (^expansion)() = ^{
    [self.toolbarAnimatee expandLocationBar];
    [self.toolbarAnimatee showCancelButton];
  };

  void (^hideControls)() = ^{
    [self.toolbarAnimatee hideControlButtons];
  };

  if (animated) {
    // Use UIView animateWithDuration instead of UIViewPropertyAnimator to
    // avoid UIKit bug. See https://crbug.com/856155.
    [UIView animateWithDuration:ios::material::kDuration1
                          delay:0
                        options:UIViewAnimationCurveEaseInOut
                     animations:expansion
                     completion:nil];

    [UIView animateWithDuration:ios::material::kDuration2
                          delay:0
                        options:UIViewAnimationCurveEaseInOut
                     animations:hideControls
                     completion:nil];
  } else {
    expansion();
    hideControls();
  }
}

// Updates the UI elements reflect the toolbar contracted state, |animated| or
// not.
- (void)updateUIToContractedState:(BOOL)animated {
  void (^contraction)() = ^{
    [self.toolbarAnimatee contractLocationBar];
  };

  void (^hideCancel)() = ^{
    [self.toolbarAnimatee hideCancelButton];
  };

  void (^showControls)() = ^{
    [self.toolbarAnimatee showControlButtons];
  };

  if (animated) {
    // Use UIView animateWithDuration instead of UIViewPropertyAnimator to
    // avoid UIKit bug. See https://crbug.com/856155.
    CGFloat totalDuration =
        ios::material::kDuration1 + ios::material::kDuration2;
    CGFloat relativeDurationAnimation1 =
        ios::material::kDuration1 / totalDuration;
    [UIView animateKeyframesWithDuration:totalDuration
        delay:0
        options:UIViewAnimationCurveEaseInOut
        animations:^{
          [UIView addKeyframeWithRelativeStartTime:0
                                  relativeDuration:relativeDurationAnimation1
                                        animations:^{
                                          contraction();
                                        }];
          [UIView
              addKeyframeWithRelativeStartTime:relativeDurationAnimation1
                              relativeDuration:1 - relativeDurationAnimation1
                                    animations:^{
                                      showControls();
                                    }];
        }
        completion:^(BOOL finished) {
          hideCancel();
        }];
  } else {
    contraction();
    showControls();
    hideCancel();
  }
}

@end
