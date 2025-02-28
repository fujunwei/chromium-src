// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/web_applications/extensions/bookmark_app_installation_task.h"

#include <memory>
#include <utility>

#include "base/bind.h"
#include "base/callback.h"
#include "chrome/browser/web_applications/extensions/bookmark_app_data_retriever.h"

namespace extensions {

BookmarkAppInstallationTask::~BookmarkAppInstallationTask() = default;

void BookmarkAppInstallationTask::SetDataRetrieverForTesting(
    std::unique_ptr<BookmarkAppDataRetriever> data_retriever) {
  data_retriever_ = std::move(data_retriever);
}

BookmarkAppInstallationTask::BookmarkAppInstallationTask()
    : data_retriever_(std::make_unique<BookmarkAppDataRetriever>()) {}

}  // namespace extensions
