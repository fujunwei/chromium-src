// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_SEARCH_BACKGROUND_NTP_BACKGROUND_DATA_H_
#define CHROME_BROWSER_SEARCH_BACKGROUND_NTP_BACKGROUND_DATA_H_

#include <string>

#include "chrome/browser/search/background/ntp_background.pb.h"
#include "url/gurl.h"

enum class ErrorType {
  // Data retrieved successfully.
  NONE,

  // Network error occurred.
  NET_ERROR,

  // Response from backend couldn't be read.
  SERVICE_ERROR,

  // Unable to authenticate with Google Photos (INVALID_GAIA_CREDENTIALS).
  AUTH_ERROR
};

// Background images are organized into collections, according to a theme. This
// struct contains the data required to display information about a collection,
// including a representative image. The complete set of CollectionImages must
// be requested separately, by referencing the identifier for this collection.
struct CollectionInfo {
  CollectionInfo();
  CollectionInfo(const CollectionInfo&);
  CollectionInfo(CollectionInfo&&);
  ~CollectionInfo();

  CollectionInfo& operator=(const CollectionInfo&);
  CollectionInfo& operator=(CollectionInfo&&);

  static CollectionInfo CreateFromProto(
      const ntp::background::Collection& collection);

  // A unique identifier for the collection.
  std::string collection_id;
  // A human-readable name for the collection.
  std::string collection_name;
  // A representative image from the collection.
  GURL preview_image_url;
};

bool operator==(const CollectionInfo& lhs, const CollectionInfo& rhs);
bool operator!=(const CollectionInfo& lhs, const CollectionInfo& rhs);

// Represents an image within a collection. The associated collection_id may be
// used to get CollectionInfo.
struct CollectionImage {
  CollectionImage();
  CollectionImage(const CollectionImage&);
  CollectionImage(CollectionImage&&);
  ~CollectionImage();

  CollectionImage& operator=(const CollectionImage&);
  CollectionImage& operator=(CollectionImage&&);

  // default_image_options are applied to the image.image_url() if options
  // (specifying resolution, cropping, etc) are not already present.
  static CollectionImage CreateFromProto(
      const std::string& collection_id,
      const ntp::background::Image& image,
      const std::string& default_image_options);

  // A unique identifier for the collection the image is in.
  std::string collection_id;
  // A unique identifier for the image.
  uint64_t asset_id;
  // The thumbnail image URL, typically lower resolution than the image_url.
  GURL thumbnail_image_url;
  // The image URL.
  GURL image_url;
  // The attribution list for the image.
  std::vector<std::string> attribution;
  // A URL that can be accessed to find out more information about the image.
  GURL attribution_action_url;
};

bool operator==(const CollectionImage& lhs, const CollectionImage& rhs);
bool operator!=(const CollectionImage& lhs, const CollectionImage& rhs);

// This struct contains the data required to display information about a photo
// album, including a representative image. The photos in an album must be
// requested separately, by referencing the album_id and photo_container_id
// specified here.
struct AlbumInfo {
  AlbumInfo();
  AlbumInfo(const AlbumInfo&);
  AlbumInfo(AlbumInfo&&);
  ~AlbumInfo();

  AlbumInfo& operator=(const AlbumInfo&);
  AlbumInfo& operator=(AlbumInfo&&);

  static AlbumInfo CreateFromProto(const ntp::background::AlbumMetaData& album);

  // A unique identifier for the album. This is required when requesting the
  // album.
  int64_t album_id;
  // A generic photo container ID based on the photo provider. For Google
  // Photos, this corresponds to media keys for the collection. It is also
  // required when requesting the album.
  std::string photo_container_id;
  // A human-readable name for the album.
  std::string album_name;
  // A representative image from the album.
  GURL preview_image_url;
};

bool operator==(const AlbumInfo& lhs, const AlbumInfo& rhs);
bool operator!=(const AlbumInfo& lhs, const AlbumInfo& rhs);

// Represents a photo within an album.
struct AlbumPhoto {
  AlbumPhoto();
  // default_image_options are applied to the image.image_url() if options
  // (specifying resolution, cropping, etc) are not already present.
  AlbumPhoto(const std::string& album_id,
             const std::string& photo_container_id,
             const std::string& photo_url,
             const std::string& default_image_options);
  AlbumPhoto(const AlbumPhoto&);
  AlbumPhoto(AlbumPhoto&&);
  ~AlbumPhoto();

  AlbumPhoto& operator=(const AlbumPhoto&);
  AlbumPhoto& operator=(AlbumPhoto&&);

  // A unique identifier for the album. This is required when requesting the
  // album.
  std::string album_id;
  // A generic photo container ID based on the photo provider. For Google
  // Photos, this corresponds to media keys for the collection. It is also
  // required when requesting the album.
  std::string photo_container_id;
  // The thumbnail image URL, typically lower resolution than the photo_url.
  GURL thumbnail_photo_url;
  // The image URL.
  GURL photo_url;
};

bool operator==(const AlbumPhoto& lhs, const AlbumPhoto& rhs);
bool operator!=(const AlbumPhoto& lhs, const AlbumPhoto& rhs);

// Represents errors that occur when communicating with the Backdrop service and
// Google Photos.
struct ErrorInfo {
  ErrorInfo();
  ErrorInfo(const ErrorInfo&);
  ErrorInfo(ErrorInfo&&);
  ~ErrorInfo();

  ErrorInfo& operator=(const ErrorInfo&);
  ErrorInfo& operator=(ErrorInfo&&);

  void ClearError();

  // Network error number, listed at chrome://network-errors.
  int net_error;

  // Category of error that occured.
  ErrorType error_type;
};

#endif  // CHROME_BROWSER_SEARCH_BACKGROUND_NTP_BACKGROUND_DATA_H_
