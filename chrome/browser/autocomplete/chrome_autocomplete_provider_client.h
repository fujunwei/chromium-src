// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_AUTOCOMPLETE_CHROME_AUTOCOMPLETE_PROVIDER_CLIENT_H_
#define CHROME_BROWSER_AUTOCOMPLETE_CHROME_AUTOCOMPLETE_PROVIDER_CLIENT_H_

#include "base/macros.h"
#include "chrome/browser/autocomplete/chrome_autocomplete_scheme_classifier.h"
#include "chrome/browser/search_engines/ui_thread_search_terms_data.h"
#include "components/omnibox/browser/autocomplete_provider_client.h"

class Profile;

namespace content {
class StoragePartition;
}

class ChromeAutocompleteProviderClient : public AutocompleteProviderClient {
 public:
  explicit ChromeAutocompleteProviderClient(Profile* profile);
  ~ChromeAutocompleteProviderClient() override;

  // AutocompleteProviderClient:
  scoped_refptr<network::SharedURLLoaderFactory> GetURLLoaderFactory() override;
  PrefService* GetPrefs() override;
  const AutocompleteSchemeClassifier& GetSchemeClassifier() const override;
  AutocompleteClassifier* GetAutocompleteClassifier() override;
  history::HistoryService* GetHistoryService() override;
  scoped_refptr<history::TopSites> GetTopSites() override;
  bookmarks::BookmarkModel* GetBookmarkModel() override;
  history::URLDatabase* GetInMemoryDatabase() override;
  InMemoryURLIndex* GetInMemoryURLIndex() override;
  TemplateURLService* GetTemplateURLService() override;
  const TemplateURLService* GetTemplateURLService() const override;
  ContextualSuggestionsService* GetContextualSuggestionsService(
      bool create_if_necessary) const override;
  DocumentSuggestionsService* GetDocumentSuggestionsService(
      bool create_if_necessary) const override;
  const SearchTermsData& GetSearchTermsData() const override;
  scoped_refptr<ShortcutsBackend> GetShortcutsBackend() override;
  scoped_refptr<ShortcutsBackend> GetShortcutsBackendIfExists() override;
  std::unique_ptr<KeywordExtensionsDelegate> GetKeywordExtensionsDelegate(
      KeywordProvider* keyword_provider) override;
  std::string GetAcceptLanguages() const override;
  std::string GetEmbedderRepresentationOfAboutScheme() override;
  std::vector<base::string16> GetBuiltinURLs() override;
  std::vector<base::string16> GetBuiltinsToProvideAsUserTypes() override;
  // GetCurrentVisitTimestamp is only implemented for desktop users. For mobile
  // users, the function returns base::Time().
  base::Time GetCurrentVisitTimestamp() const override;
  bool IsOffTheRecord() const override;
  bool SearchSuggestEnabled() const override;
  bool IsTabUploadToGoogleActive() const override;
  bool IsAuthenticated() const override;
  void Classify(
      const base::string16& text,
      bool prefer_keyword,
      bool allow_exact_keyword_match,
      metrics::OmniboxEventProto::PageClassification page_classification,
      AutocompleteMatch* match,
      GURL* alternate_nav_url) override;
  void DeleteMatchingURLsForKeywordFromHistory(
      history::KeywordID keyword_id,
      const base::string16& term) override;
  void PrefetchImage(const GURL& url) override;
  void StartServiceWorker(const GURL& destination_url) override;
  void OnAutocompleteControllerResultReady(
      AutocompleteController* controller) override;
  bool IsTabOpenWithURL(const GURL& url,
                        const AutocompleteInput* input) override;

  // For testing.
  void set_storage_partition(content::StoragePartition* storage_partition) {
    storage_partition_ = storage_partition;
  }

  bool StrippedURLsAreEqual(const GURL& url1,
                            const GURL& url2,
                            const AutocompleteInput* input) const;

 private:
  Profile* profile_;
  ChromeAutocompleteSchemeClassifier scheme_classifier_;
  UIThreadSearchTermsData search_terms_data_;

  // Injectable storage partitiion, used for testing.
  content::StoragePartition* storage_partition_;

  DISALLOW_COPY_AND_ASSIGN(ChromeAutocompleteProviderClient);
};

#endif  // CHROME_BROWSER_AUTOCOMPLETE_CHROME_AUTOCOMPLETE_PROVIDER_CLIENT_H_
