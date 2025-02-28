// Copyright 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_UI_OMNIBOX_ALTERNATE_NAV_INFOBAR_DELEGATE_H_
#define CHROME_BROWSER_UI_OMNIBOX_ALTERNATE_NAV_INFOBAR_DELEGATE_H_

#include <stddef.h>

#include <memory>

#include "base/macros.h"
#include "build/build_config.h"
#include "components/infobars/core/infobar_delegate.h"
#include "components/omnibox/browser/autocomplete_match.h"

class Profile;

namespace content {
class WebContents;
}

// This class creates an alternate nav infobar and delegate and adds the infobar
// to the infobar service for |web_contents|.
class AlternateNavInfoBarDelegate : public infobars::InfoBarDelegate {
 public:
  ~AlternateNavInfoBarDelegate() override;

  // Creates the delegate for omnibox navigations that have suggested URLs.
  // E.g. This will display a "Did you mean to go to http://test" infobar if the
  // user searches for "test" and there is a host called "test" in the network.
  static void CreateForOmniboxNavigation(content::WebContents* web_contents,
                                         const base::string16& text,
                                         const AutocompleteMatch& match,
                                         const GURL& search_url);

  // Creates the delegate for navigations involving internationalized domain
  // names (IDN) where the IDN looks similar to one of the top 10K domains.
  static void CreateForIDNNavigation(content::WebContents* web_contents,
                                     const base::string16& text,
                                     const GURL& suggested_url,
                                     const GURL& original_url);
  base::string16 GetMessageTextWithOffset(size_t* link_offset) const;
  base::string16 GetLinkText() const;
  GURL GetLinkURL() const;
  bool LinkClicked(WindowOpenDisposition disposition);

 private:
  AlternateNavInfoBarDelegate(Profile* profile,
                              const base::string16& text,
                              std::unique_ptr<AutocompleteMatch> match,
                              const GURL& destination_url,
                              const GURL& original_url);

  // Returns an alternate nav infobar that owns |delegate|.
  static std::unique_ptr<infobars::InfoBar> CreateInfoBar(
      std::unique_ptr<AlternateNavInfoBarDelegate> delegate);
#if defined(OS_MACOSX)
  // Temporary shim for Polychrome. See bottom of first comment in
  // https://crbug.com/804950 for details
  static std::unique_ptr<infobars::InfoBar> CreateInfoBarCocoa(
      std::unique_ptr<AlternateNavInfoBarDelegate> delegate);
#endif

  // InfoBarDelegate:
  infobars::InfoBarDelegate::InfoBarIdentifier GetIdentifier() const override;
  const gfx::VectorIcon& GetVectorIcon() const override;

  Profile* profile_;
  const base::string16 text_;

  // The autocomplete match to be used when deleting the corresponding shortcut.
  // Can be null when the event triggering the infobar was not an omnibox
  // navigation.
  std::unique_ptr<AutocompleteMatch> match_;

  // The URL to navigate to when the user clicks the link.
  const GURL destination_url_;

  // Original URL of the navigation. When the user clicks the suggested
  // navigation link, this will be removed from history.
  // For search navigations this is the search URL. For IDN navigations, this is
  // the URL that visually matches a top domain.
  const GURL original_url_;

  DISALLOW_COPY_AND_ASSIGN(AlternateNavInfoBarDelegate);
};

#endif  // CHROME_BROWSER_UI_OMNIBOX_ALTERNATE_NAV_INFOBAR_DELEGATE_H_
