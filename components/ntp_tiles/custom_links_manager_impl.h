// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef COMPONENTS_NTP_TILES_CUSTOM_LINKS_MANAGER_IMPL_H_
#define COMPONENTS_NTP_TILES_CUSTOM_LINKS_MANAGER_IMPL_H_

#include <utility>
#include <vector>

#include "base/macros.h"
#include "base/memory/weak_ptr.h"
#include "base/optional.h"
#include "components/ntp_tiles/custom_links_manager.h"
#include "components/ntp_tiles/custom_links_store.h"
#include "components/ntp_tiles/ntp_tile.h"

class PrefService;

namespace user_prefs {
class PrefRegistrySyncable;
}  // namespace user_prefs

namespace ntp_tiles {

// Non-test implementation of the CustomLinksManager interface.
class CustomLinksManagerImpl : public CustomLinksManager {
 public:
  // Restores the previous state of |current_links_| from prefs.
  explicit CustomLinksManagerImpl(PrefService* prefs);

  ~CustomLinksManagerImpl() override;

  // CustomLinksManager implementation.
  bool Initialize(const NTPTilesVector& tiles) override;
  void Uninitialize() override;
  bool IsInitialized() const override;

  const std::vector<Link>& GetLinks() const override;

  bool AddLink(const GURL& url, const base::string16& title) override;
  bool DeleteLink(const GURL& url) override;
  bool UndoDeleteLink() override;

  // Register preferences used by this class.
  static void RegisterProfilePrefs(
      user_prefs::PrefRegistrySyncable* user_prefs);

 private:
  void ClearLinks();
  // Returns an iterator into |custom_links_|.
  std::vector<Link>::iterator FindLinkWithUrl(const GURL& url);

  PrefService* const prefs_;
  CustomLinksStore store_;
  std::vector<Link> current_links_;
  // Contains the deleted link's data and the index it was located at.
  base::Optional<std::pair<size_t, Link>> prev_deleted_link_;

  base::WeakPtrFactory<CustomLinksManagerImpl> weak_ptr_factory_;

  DISALLOW_COPY_AND_ASSIGN(CustomLinksManagerImpl);
};

}  // namespace ntp_tiles

#endif  // COMPONENTS_NTP_TILES_CUSTOM_LINKS_MANAGER_IMPL_H_
