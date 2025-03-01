// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package org.chromium.chrome.browser.preferences;

import android.support.annotation.NonNull;
import android.support.v7.widget.SearchView;
import android.view.MenuItem;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.ImageView;

import javax.annotation.Nullable;

/**
 * A helper class for applying the default search behavior to search items in Chromium settings.
 */
public class SearchUtils {
    /**
     * This interface allows to react to changed search queries when initialized with
     * {@link SearchUtils#initializeSearchView(MenuItem, String, QueryChangeListener)}.
     */
    public interface QueryChangeListener {
        /**
         * Called whenever the search query changes. This usually is immediately after a user types
         * and doesn't wait for submission of the whole query.
         * @param query Current query as entered by the user. Can be a partial query or empty.
         */
        void onQueryTextChange(String query);
    }

    /**
     * Initializes an Android default search item by setting listeners and default states of the
     * search icon, box and close icon.
     * @param searchItem The existing item that can trigger the search action view.
     * @param initialQuery The query that the search field should be opened with.
     * @param changeListener The listener to be notified when the user changes the query.
     */
    public static void initializeSearchView(@NonNull MenuItem searchItem,
            @Nullable String initialQuery, @NonNull QueryChangeListener changeListener) {
        SearchView searchView = getSearchView(searchItem);
        searchView.setImeOptions(EditorInfo.IME_FLAG_NO_FULLSCREEN);

        // Restore the search view if a query was recovered.
        if (initialQuery != null) {
            searchItem.expandActionView();
            searchView.setIconified(false);
            searchView.setQuery(initialQuery, false);
        }

        // Clicking the menu item hides the clear button and triggers search for an empty query.
        searchItem.setOnMenuItemClickListener((MenuItem m) -> {
            updateSearchClearButtonVisibility(searchItem, "");
            changeListener.onQueryTextChange("");
            return false; // Continue with the default action.
        });

        // Make the close button a clear button.
        searchView.findViewById(org.chromium.chrome.R.id.search_close_btn)
                .setOnClickListener((View v) -> {
                    searchView.setQuery("", false);
                    updateSearchClearButtonVisibility(searchItem, "");
                    changeListener.onQueryTextChange("");
                });

        // Ensure that a changed search view triggers the search - independent from use code path.
        searchView.setOnSearchClickListener(view -> {
            updateSearchClearButtonVisibility(searchItem, "");
            changeListener.onQueryTextChange("");
        });
        searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
            @Override
            public boolean onQueryTextSubmit(String query) {
                return true; // Consume event.
            }

            @Override
            public boolean onQueryTextChange(String query) {
                // TODO(fhorschig) Exit early if a tracked query indicates no changes.
                updateSearchClearButtonVisibility(searchItem, query);
                changeListener.onQueryTextChange(query);
                return true; // Consume event.
            }
        });
    }

    /**
     * Handles an item in {@link android.support.v4.app.Fragment#onOptionsItemSelected(MenuItem)} if
     * it is a search item and returns true. If it is not applicable, it returns false.
     * @param selectedItem The user-selected menu item.
     * @param searchItem The menu item known to contain the search view.
     * @param query The current search query.
     * @return Returns true if the item is a search item and could be handled. False otherwise.
     */
    public static boolean handleSearchNavigation(
            @NonNull MenuItem selectedItem, @NonNull MenuItem searchItem, @Nullable String query) {
        if (selectedItem.getItemId() != android.R.id.home || query == null) return false;
        SearchView searchView = (SearchView) searchItem.getActionView();
        searchView.setQuery(null, false);
        searchView.setIconified(true);
        searchItem.collapseActionView();
        return true;
    }

    /**
     * Shorthand to easily access a search item's action view.
     * @param searchItem The menu item containing search item.
     * @return The search view associated with the menu item.
     */
    public static SearchView getSearchView(MenuItem searchItem) {
        return (SearchView) searchItem.getActionView();
    }

    private static void updateSearchClearButtonVisibility(MenuItem searchItem, String query) {
        ImageView clearButton = findSearchClearButton(getSearchView(searchItem));
        clearButton.setVisibility(query == null || query.equals("") ? View.GONE : View.VISIBLE);
    }

    private static ImageView findSearchClearButton(SearchView searchView) {
        return searchView.findViewById(org.chromium.chrome.R.id.search_close_btn);
    }
}
