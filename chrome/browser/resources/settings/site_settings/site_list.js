// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

/**
 * @fileoverview
 * 'site-list' shows a list of Allowed and Blocked sites for a given
 * category.
 */
Polymer({
  is: 'site-list',

  behaviors: [
    SiteSettingsBehavior,
    WebUIListenerBehavior,
    ListPropertyUpdateBehavior,
  ],

  properties: {
    /**
     * Some content types (like Location) do not allow the user to manually
     * edit the exception list from within Settings.
     * @private
     */
    readOnlyList: {
      type: Boolean,
      value: false,
    },

    /**
     * The site serving as the model for the currently open action menu.
     * @private {?SiteException}
     */
    actionMenuSite_: Object,

    /**
     * Whether the "edit exception" dialog should be shown.
     * @private
     */
    showEditExceptionDialog_: Boolean,

    /**
     * Array of sites to display in the widget.
     * @type {!Array<SiteException>}
     */
    sites: {
      type: Array,
      value: function() {
        return [];
      },
    },

    /**
     * The type of category this widget is displaying data for. Normally
     * either 'allow' or 'block', representing which sites are allowed or
     * blocked respectively.
     */
    categorySubtype: {
      type: String,
      value: settings.INVALID_CATEGORY_SUBTYPE,
    },

    /** @private */
    hasIncognito_: Boolean,

    /** @private */
    showAddSiteDialog_: Boolean,

    /**
     * Whether to show the Allow action in the action menu.
     * @private
     */
    showAllowAction_: Boolean,

    /**
     * Whether to show the Block action in the action menu.
     * @private
     */
    showBlockAction_: Boolean,

    /**
     * Whether to show the 'Clear on exit' action in the action
     * menu.
     * @private
     */
    showSessionOnlyAction_: Boolean,

    /**
     * All possible actions in the action menu.
     * @private
     */
    actions_: {
      readOnly: true,
      type: Object,
      values: {
        ALLOW: 'Allow',
        BLOCK: 'Block',
        RESET: 'Reset',
        SESSION_ONLY: 'SessionOnly',
      }
    },

    /** @private */
    lastFocused_: Object,
  },

  /**
   * The element to return focus to, when the currently active dialog is closed.
   * @private {?HTMLElement}
   */
  activeDialogAnchor_: null,

  observers: ['configureWidget_(category, categorySubtype)'],

  /** @override */
  ready: function() {
    this.addWebUIListener(
        'contentSettingSitePermissionChanged',
        this.siteWithinCategoryChanged_.bind(this));
    this.addWebUIListener(
        'onIncognitoStatusChanged', this.onIncognitoStatusChanged_.bind(this));
    this.browserProxy.updateIncognitoStatus();
  },

  /**
   * Called when a site changes permission.
   * @param {string} category The category of the site that changed.
   * @param {string} site The site that changed.
   * @private
   */
  siteWithinCategoryChanged_: function(category, site) {
    if (category == this.category)
      this.configureWidget_();
  },

  /**
   * Called for each site list when incognito is enabled or disabled. Only
   * called on change (opening N incognito windows only fires one message).
   * Another message is sent when the *last* incognito window closes.
   * @private
   */
  onIncognitoStatusChanged_: function(hasIncognito) {
    this.hasIncognito_ = hasIncognito;

    // The SESSION_ONLY list won't have any incognito exceptions. (Minor
    // optimization, not required).
    if (this.categorySubtype == settings.ContentSetting.SESSION_ONLY)
      return;

    // A change notification is not sent for each site. So we repopulate the
    // whole list when the incognito profile is created or destroyed.
    this.populateList_();
  },

  /**
   * Configures the action menu, visibility of the widget and shows the list.
   * @private
   */
  configureWidget_: function() {
    if (this.category == undefined)
      return;

    // The observer for All Sites fires before the attached/ready event, so
    // initialize this here.
    if (this.browserProxy_ === undefined) {
      this.browserProxy_ =
          settings.SiteSettingsPrefsBrowserProxyImpl.getInstance();
    }

    this.setUpActionMenu_();
    this.populateList_();

    // The Session permissions are only for cookies.
    if (this.categorySubtype == settings.ContentSetting.SESSION_ONLY) {
      this.$.category.hidden =
          this.category != settings.ContentSettingsTypes.COOKIES;
    }
  },

  /**
   * Whether there are any site exceptions added for this content setting.
   * @return {boolean}
   * @private
   */
  hasSites_: function() {
    return !!this.sites.length;
  },

  /**
   * A handler for the Add Site button.
   * @private
   */
  onAddSiteTap_: function() {
    assert(!this.readOnlyList);
    this.showAddSiteDialog_ = true;
  },

  /** @private */
  onAddSiteDialogClosed_: function() {
    this.showAddSiteDialog_ = false;
    cr.ui.focusWithoutInk(assert(this.$.addSite));
  },

  /**
   * Populate the sites list for display.
   * @private
   */
  populateList_: function() {
    this.browserProxy_.getExceptionList(this.category).then(exceptionList => {
      this.processExceptions_(exceptionList);
      this.closeActionMenu_();
    });
  },

  /**
   * Process the exception list returned from the native layer.
   * @param {!Array<RawSiteException>} exceptionList
   * @private
   */
  processExceptions_: function(exceptionList) {
    const sites =
        exceptionList
            .filter(
                site => site.setting != settings.ContentSetting.DEFAULT &&
                    site.setting == this.categorySubtype)
            .map(site => this.expandSiteException(site));
    this.updateList('sites', x => x.origin, sites);
  },

  /**
   * Set up the values to use for the action menu.
   * @private
   */
  setUpActionMenu_: function() {
    this.showAllowAction_ =
        this.categorySubtype != settings.ContentSetting.ALLOW;
    this.showBlockAction_ =
        this.categorySubtype != settings.ContentSetting.BLOCK;
    this.showSessionOnlyAction_ =
        this.categorySubtype != settings.ContentSetting.SESSION_ONLY &&
        this.category == settings.ContentSettingsTypes.COOKIES;
  },

  /**
   * @return {boolean} Whether to show the "Session Only" menu item for the
   *     currently active site.
   * @private
   */
  showSessionOnlyActionForSite_: function() {
    // It makes no sense to show "clear on exit" for exceptions that only apply
    // to incognito. It gives the impression that they might under some
    // circumstances not be cleared on exit, which isn't true.
    if (!this.actionMenuSite_ || this.actionMenuSite_.incognito)
      return false;

    return this.showSessionOnlyAction_;
  },

  /**
   * @param {!settings.ContentSetting} contentSetting
   * @private
   */
  setContentSettingForActionMenuSite_: function(contentSetting) {
    assert(this.actionMenuSite_);
    this.browserProxy.setCategoryPermissionForPattern(
        this.actionMenuSite_.origin, this.actionMenuSite_.embeddingOrigin,
        this.category, contentSetting, this.actionMenuSite_.incognito);
  },

  /** @private */
  onAllowTap_: function() {
    this.setContentSettingForActionMenuSite_(settings.ContentSetting.ALLOW);
    this.closeActionMenu_();
  },

  /** @private */
  onBlockTap_: function() {
    this.setContentSettingForActionMenuSite_(settings.ContentSetting.BLOCK);
    this.closeActionMenu_();
  },

  /** @private */
  onSessionOnlyTap_: function() {
    this.setContentSettingForActionMenuSite_(
        settings.ContentSetting.SESSION_ONLY);
    this.closeActionMenu_();
  },

  /** @private */
  onEditTap_: function() {
    // Close action menu without resetting |this.actionMenuSite_| since it is
    // bound to the dialog.
    /** @type {!CrActionMenuElement} */ (this.$$('cr-action-menu')).close();
    this.showEditExceptionDialog_ = true;
  },

  /** @private */
  onEditExceptionDialogClosed_: function() {
    this.showEditExceptionDialog_ = false;
    this.actionMenuSite_ = null;
    if (this.activeDialogAnchor_) {
      this.activeDialogAnchor_.focus();
      this.activeDialogAnchor_ = null;
    }
  },

  /** @private */
  onResetTap_: function() {
    const site = this.actionMenuSite_;
    assert(site);
    this.browserProxy.resetCategoryPermissionForPattern(
        site.origin, site.embeddingOrigin, this.category, site.incognito);
    this.closeActionMenu_();
  },

  /**
   * @param {!Event} e
   * @private
   */
  onShowActionMenu_: function(e) {
    this.activeDialogAnchor_ = /** @type {!HTMLElement} */ (e.detail.anchor);
    this.actionMenuSite_ = e.detail.model;
    /** @type {!CrActionMenuElement} */ (this.$$('cr-action-menu'))
        .showAt(this.activeDialogAnchor_);
  },

  /** @private */
  closeActionMenu_: function() {
    this.actionMenuSite_ = null;
    this.activeDialogAnchor_ = null;
    const actionMenu =
        /** @type {!CrActionMenuElement} */ (this.$$('cr-action-menu'));
    if (actionMenu.open)
      actionMenu.close();
  },
});
