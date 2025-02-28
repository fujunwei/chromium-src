// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package org.chromium.chrome.browser.feed;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.FrameLayout;

import com.google.android.libraries.feed.api.scope.FeedProcessScope;
import com.google.android.libraries.feed.api.scope.FeedStreamScope;
import com.google.android.libraries.feed.api.stream.Stream;
import com.google.android.libraries.feed.host.stream.CardConfiguration;
import com.google.android.libraries.feed.host.stream.SnackbarApi;
import com.google.android.libraries.feed.host.stream.StreamConfiguration;

import org.chromium.base.ApiCompatibilityUtils;
import org.chromium.chrome.R;
import org.chromium.chrome.browser.ChromeActivity;
import org.chromium.chrome.browser.NativePageHost;
import org.chromium.chrome.browser.feed.action.FeedActionHandler;
import org.chromium.chrome.browser.ntp.ContextMenuManager;
import org.chromium.chrome.browser.ntp.ContextMenuManager.TouchEnabledDelegate;
import org.chromium.chrome.browser.ntp.NewTabPage;
import org.chromium.chrome.browser.ntp.NewTabPageLayout;
import org.chromium.chrome.browser.ntp.SnapScrollHelper;
import org.chromium.chrome.browser.ntp.snippets.SectionHeaderView;
import org.chromium.chrome.browser.profiles.Profile;
import org.chromium.chrome.browser.search_engines.TemplateUrlService;
import org.chromium.chrome.browser.snackbar.Snackbar;
import org.chromium.chrome.browser.snackbar.SnackbarManager;
import org.chromium.chrome.browser.suggestions.SuggestionsNavigationDelegateImpl;
import org.chromium.chrome.browser.tab.Tab;
import org.chromium.chrome.browser.tabmodel.TabModelSelector;
import org.chromium.chrome.browser.util.ViewUtils;
import org.chromium.chrome.browser.widget.displaystyle.UiConfig;

import java.util.Arrays;

/**
 * Provides a new tab page that displays an interest feed rendered list of content suggestions.
 */
public class FeedNewTabPage extends NewTabPage implements TouchEnabledDelegate {
    private final FeedNewTabPageMediator mMediator;
    private final StreamLifecycleManager mStreamLifecycleManager;
    private final Stream mStream;

    private FrameLayout mRootView;
    private SectionHeaderView mSectionHeaderView;
    private FeedImageLoader mImageLoader;

    private class BasicSnackbarApi implements SnackbarApi {
        @Override
        public void show(String message) {
            mNewTabPageManager.getSnackbarManager().showSnackbar(
                    Snackbar.make(message, new SnackbarManager.SnackbarController() {},
                            Snackbar.TYPE_ACTION, Snackbar.UMA_FEED_NTP_STREAM));
        }
    }

    private static class BasicStreamConfiguration implements StreamConfiguration {
        private final Resources mResources;
        private final int mPadding;

        public BasicStreamConfiguration(Resources resources) {
            mResources = resources;
            mPadding = mResources.getDimensionPixelSize(
                    R.dimen.content_suggestions_card_modern_margin);
        }

        @Override
        public int getPaddingStart() {
            return mPadding;
        }
        @Override
        public int getPaddingEnd() {
            return mPadding;
        }
        @Override
        public int getPaddingTop() {
            return 0;
        }
        @Override
        public int getPaddingBottom() {
            return 0;
        }
    }

    private static class BasicCardConfiguration implements CardConfiguration {
        private final Resources mResources;
        private final int mCornerRadius;
        private final Drawable mCardBackground;
        private final int mCardMarginBottom;

        public BasicCardConfiguration(Resources resources) {
            mResources = resources;
            mCornerRadius = mResources.getDimensionPixelSize(
                    R.dimen.content_suggestions_card_modern_corner_radius);
            mCardBackground = ApiCompatibilityUtils.getDrawable(
                    mResources, R.drawable.content_card_modern_background);
            mCardMarginBottom = mResources.getDimensionPixelSize(
                    R.dimen.content_suggestions_card_modern_margin);
        }

        @Override
        public int getDefaultCornerRadius() {
            return mCornerRadius;
        }

        @Override
        public Drawable getCardBackground() {
            return mCardBackground;
        }

        @Override
        public int getCardBottomMargin() {
            return mCardMarginBottom;
        }
    }

    /**
     * Constructs a new FeedNewTabPage.
     *
     * @param activity The containing {@link ChromeActivity}.
     * @param nativePageHost The host for this native page.
     * @param tabModelSelector The {@link TabModelSelector} for the containing activity.
     */
    public FeedNewTabPage(ChromeActivity activity, NativePageHost nativePageHost,
            TabModelSelector tabModelSelector) {
        super(activity, nativePageHost, tabModelSelector);

        FeedProcessScope feedProcessScope = FeedProcessScopeFactory.getFeedProcessScope();
        Tab tab = nativePageHost.getActiveTab();
        Profile profile = tab.getProfile();
        mImageLoader = new FeedImageLoader(profile, activity);
        SuggestionsNavigationDelegateImpl navigationDelegate =
                new SuggestionsNavigationDelegateImpl(
                        activity, profile, nativePageHost, tabModelSelector);
        FeedStreamScope streamScope =
                feedProcessScope
                        .createFeedStreamScopeBuilder(activity, mImageLoader,
                                new FeedActionHandler(navigationDelegate),
                                new BasicStreamConfiguration(activity.getResources()),
                                new BasicCardConfiguration(activity.getResources()),
                                new BasicSnackbarApi())
                        .build();

        mStream = streamScope.getStream();
        mStreamLifecycleManager = new StreamLifecycleManager(mStream, activity, tab);

        LayoutInflater inflater = LayoutInflater.from(activity);
        mNewTabPageLayout = (NewTabPageLayout) inflater.inflate(R.layout.new_tab_page_layout, null);
        mSectionHeaderView = (SectionHeaderView) inflater.inflate(
                R.layout.new_tab_page_snippets_expandable_header, null);

        mMediator = new FeedNewTabPageMediator(this,
                new SnapScrollHelper(mNewTabPageManager, mNewTabPageLayout, mStream.getView()));

        // Don't store a direct reference to the activity, because it might change later if the tab
        // is reparented.
        // TODO(twellington): Move this somewhere it can be shared with NewTabPageView?
        Runnable closeContextMenuCallback = () -> mTab.getActivity().closeContextMenu();
        ContextMenuManager contextMenuManager =
                new ContextMenuManager(mNewTabPageManager.getNavigationDelegate(),
                        this::setTouchEnabled, closeContextMenuCallback);
        mTab.getWindowAndroid().addContextMenuCloseListener(contextMenuManager);

        mNewTabPageLayout.initialize(mNewTabPageManager, mTab, mTileGroupDelegate,
                mSearchProviderHasLogo,
                TemplateUrlService.getInstance().isDefaultSearchEngineGoogle(), mMediator,
                contextMenuManager, new UiConfig(mRootView));

        mStream.getView().setBackgroundColor(Color.WHITE);
        mRootView.addView(mStream.getView());

        mStream.setHeaderViews(Arrays.asList(mNewTabPageLayout, mSectionHeaderView));
    }

    @Override
    protected void initializeMainView(Context context) {
        int topPadding = context.getResources().getDimensionPixelOffset(R.dimen.tab_strip_height);
        mRootView = new FrameLayout(context);
        mRootView.setLayoutParams(new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT));
        mRootView.setPadding(0, topPadding, 0, 0);
    }

    @Override
    public void destroy() {
        super.destroy();
        mMediator.destroy();
        mImageLoader.destroy();
        mStreamLifecycleManager.destroy();
    }

    @Override
    public View getView() {
        return mRootView;
    }

    @Override
    protected void restoreLastScrollPosition() {
        // This behavior is handled by the Feed library.
    }

    @Override
    protected void scrollToSuggestions() {
        // TODO(twellington): implement this method.
    }

    @Override
    public boolean shouldCaptureThumbnail() {
        // TODO(twellington): add more logic to this method that also takes into account other
        // UI changes that should trigger a thumbnail capture.
        return mNewTabPageLayout.shouldCaptureThumbnail();
    }

    @Override
    public void captureThumbnail(Canvas canvas) {
        mNewTabPageLayout.onPreCaptureThumbnail();
        ViewUtils.captureBitmap(mRootView, canvas);
    }

    /** @return The {@link Stream} that this class holds. */
    Stream getStream() {
        return mStream;
    }

    /** @return The {@link SectionHeaderView} for the Feed section header. */
    SectionHeaderView getSectionHeaderView() {
        return mSectionHeaderView;
    }

    // TouchEnabledDelegate interface.

    @Override
    public void setTouchEnabled(boolean enabled) {
        // TODO(twellington): implement this method.
    }
}
