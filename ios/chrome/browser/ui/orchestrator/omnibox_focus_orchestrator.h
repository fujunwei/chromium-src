// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef IOS_CHROME_BROWSER_UI_ORCHESTRATOR_OMNIBOX_FOCUS_ORCHESTRATOR_H_
#define IOS_CHROME_BROWSER_UI_ORCHESTRATOR_OMNIBOX_FOCUS_ORCHESTRATOR_H_

#import <UIKit/UIKit.h>

@protocol LocationBarAnimatee;
@protocol ToolbarAnimatee;

// Orchestrator for the animation occuring when the omnibox is
// focused/unfocused.
@interface OmniboxFocusOrchestrator : NSObject

// Toolbar animatee, orchestrated by this object.
@property(nonatomic, weak) id<ToolbarAnimatee> toolbarAnimatee;

@property(nonatomic, weak) id<LocationBarAnimatee> locationBarAnimatee;

// Updates the UI elements orchestrated by this object to reflect the
// |omniboxFocused| state, and the |toolbarExpanded| state, |animated| or not.
- (void)transitionToStateOmniboxFocused:(BOOL)omniboxFocused
                        toolbarExpanded:(BOOL)toolbarExpanded
                               animated:(BOOL)animated;

@end

#endif  // IOS_CHROME_BROWSER_UI_ORCHESTRATOR_OMNIBOX_FOCUS_ORCHESTRATOR_H_
