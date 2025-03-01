// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

suite('cr-dialog', function() {
  function pressEnter(element) {
    MockInteractions.keyEventOn(element, 'keypress', 13, undefined, 'Enter');
  }

  setup(function() {
    PolymerTest.clearBody();
  });

  test('cr-dialog-open event fires when opened', function() {
    document.body.innerHTML = `
      <cr-dialog>
        <div slot="title">title</div>
        <div slot="body">body</div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    const whenFired = test_util.eventToPromise('cr-dialog-open', dialog);
    dialog.showModal();
    return whenFired;
  });

  test('close event bubbles', function() {
    document.body.innerHTML = `
      <cr-dialog>
        <div slot="title">title</div>
        <div slot="body">body</div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    dialog.showModal();
    const whenFired = test_util.eventToPromise('close', dialog);
    dialog.close();
    return whenFired.then(() => {
      assertEquals('success', dialog.getNative().returnValue);
    });
  });

  // cr-dialog has to catch and re-fire 'close' events fired from it's native
  // <dialog> child to force them to bubble in Shadow DOM V1. Ensure that this
  // mechanism does not interfere with nested <cr-dialog> 'close' events.
  test('close events not fired from <dialog> are not affected', function() {
    document.body.innerHTML = `
      <cr-dialog id="outer">
        <div slot="title">outer dialog title</div>
        <div slot="body">
          <cr-dialog id="inner">
            <div slot="title">inner dialog title</div>
            <div slot="body">body</div>
          </cr-dialog>
        </div>
      </cr-dialog>`;

    const outer = document.body.querySelector('#outer');
    assertTrue(!!outer);
    const inner = document.body.querySelector('#inner');
    assertTrue(!!inner);

    outer.showModal();
    inner.showModal();

    let whenFired = test_util.eventToPromise('close', window);
    inner.close();

    return whenFired
        .then(e => {
          // Check that the event's target is the inner dialog.
          assertEquals(inner, e.target);
          whenFired = test_util.eventToPromise('close', window);
          outer.close();
          return whenFired;
        })
        .then(e => {
          // Check that the event's target is the outer dialog.
          assertEquals(outer, e.target);
        });
  });

  test('cancel and close events bubbles when cancelled', function() {
    document.body.innerHTML = `
      <cr-dialog>
        <div slot="title">title</div>
        <div slot="body">body</div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    dialog.showModal();
    const whenCancelFired = test_util.eventToPromise('cancel', dialog);
    const whenCloseFired = test_util.eventToPromise('close', dialog);
    dialog.cancel();
    return Promise.all([whenCancelFired, whenCloseFired]).then(() => {
      assertEquals('', dialog.getNative().returnValue);
    });
  });

  test('focuses title on show', function() {
    document.body.innerHTML = `
      <cr-dialog>
        <div slot="title">title</div>
        <div slot="body"><button>button</button></div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    const button = document.body.querySelector('button');

    assertNotEquals(dialog, document.activeElement);
    assertNotEquals(button, document.activeElement);

    dialog.showModal();

    expectEquals(dialog, document.activeElement);
    expectNotEquals(button, document.activeElement);
  });

  test('enter keys should trigger action buttons once', function() {
    document.body.innerHTML = `
      <cr-dialog>
        <div slot="title">title</div>
        <div slot="body">
          <button class="action-button">button</button>
          <button id="other-button">other button</button>
        </div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    const actionButton = document.body.querySelector('.action-button');

    dialog.showModal();

    // MockInteractions triggers event listeners synchronously.
    let clickedCounter = 0;
    actionButton.addEventListener('click', function() {
      clickedCounter++;
    });

    // Enter key on the action button should only fire the click handler once.
    MockInteractions.tap(actionButton, 'keypress', 13, undefined, 'Enter');
    assertEquals(1, clickedCounter);

    // Enter keys on other buttons should be ignored.
    clickedCounter = 0;
    const otherButton = document.body.querySelector('#other-button');
    assertTrue(!!otherButton);
    pressEnter(otherButton);
    assertEquals(0, clickedCounter);

    // Enter keys on the close icon in the top-right corner should be ignored.
    pressEnter(dialog.getCloseButton());
    assertEquals(0, clickedCounter);
  });

  test('enter keys find the first non-hidden non-disabled button', function() {
    document.body.innerHTML = `
      <cr-dialog>
        <div slot="title">title</div>
        <div slot="body">
          <button id="hidden" class="action-button" hidden>hidden</button>
          <button class="action-button" disabled>disabled</button>
          <button class="action-button" disabled hidden>disabled hidden</button>
          <button id="active" class="action-button">active</button>
        </div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    const hiddenButton = document.body.querySelector('#hidden');
    const actionButton = document.body.querySelector('#active');
    dialog.showModal();

    // MockInteractions triggers event listeners synchronously.
    hiddenButton.addEventListener('click', function() {
      assertNotReached('Hidden button received a click.');
    });
    let clicked = false;
    actionButton.addEventListener('click', function() {
      clicked = true;
    });

    pressEnter(dialog);
    assertTrue(clicked);
  });

  test('enter keys from cr-inputs (only) are processed', function() {
    document.body.innerHTML = `
      <cr-dialog>
        <div slot="title">title</div>
        <div slot="body">
          <cr-input></cr-input>
          <foobar></foobar>
          <button class="action-button">active</button>
        </div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');

    const inputElement = document.body.querySelector('cr-input');
    const otherElement = document.body.querySelector('foobar');
    const actionButton = document.body.querySelector('.action-button');
    assertTrue(!!inputElement);
    assertTrue(!!otherElement);
    assertTrue(!!actionButton);

    // MockInteractions triggers event listeners synchronously.
    let clickedCounter = 0;
    actionButton.addEventListener('click', function() {
      clickedCounter++;
    });

    pressEnter(otherElement);
    assertEquals(0, clickedCounter);

    pressEnter(inputElement);
    assertEquals(1, clickedCounter);
  });

  test('focuses [autofocus] instead of title when present', function() {
    document.body.innerHTML = `
      <cr-dialog>
        <div slot="title">title</div>
        <div slot="body"><button autofocus>button</button></div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    const button = document.body.querySelector('button');

    assertNotEquals(dialog, document.activeElement);
    assertNotEquals(button, document.activeElement);

    dialog.showModal();

    expectNotEquals(dialog, document.activeElement);
    expectEquals(button, document.activeElement);
  });

  // Ensuring that intersectionObserver does not fire any callbacks before the
  // dialog has been opened.
  test('body scrollable border not added before modal shown', function(done) {
    document.body.innerHTML = `
      <cr-dialog>
        <div slot="title">title</div>
        <div slot="body">body</div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    assertFalse(dialog.open);
    const bodyContainer = dialog.$$('.body-container');
    assertTrue(!!bodyContainer);

    // Waiting for 1ms because IntersectionObserver fires one message loop after
    // dialog.attached.
    setTimeout(function() {
      assertFalse(bodyContainer.classList.contains('top-scrollable'));
      assertFalse(bodyContainer.classList.contains('bottom-scrollable'));
      done();
    }, 1);
  });

  test('dialog body scrollable border when appropriate', function(done) {
    document.body.innerHTML = `
      <cr-dialog>
        <div slot="title">title</div>
        <div slot="body">
          <div style="height: 100px">tall content</div>
        </div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    const bodyContainer = dialog.$$('.body-container');
    assertTrue(!!bodyContainer);

    dialog.showModal();  // Attach the dialog for the first time here.

    let observerCount = 0;

    // Needs to setup the observer before attaching, since InteractionObserver
    // calls callback before MutationObserver does.
    const observer = new MutationObserver(function(changes) {
      // Only care about class mutations.
      if (changes[0].attributeName != 'class')
        return;

      observerCount++;
      switch (observerCount) {
        case 1:  // Triggered when scrolled to bottom.
          assertFalse(bodyContainer.classList.contains('bottom-scrollable'));
          assertTrue(bodyContainer.classList.contains('top-scrollable'));
          bodyContainer.scrollTop = 0;
          break;
        case 2:  // Triggered when scrolled back to top.
          assertTrue(bodyContainer.classList.contains('bottom-scrollable'));
          assertFalse(bodyContainer.classList.contains('top-scrollable'));
          bodyContainer.scrollTop = 2;
          break;
        case 3:  // Triggered when finally scrolling to middle.
          assertTrue(bodyContainer.classList.contains('bottom-scrollable'));
          assertTrue(bodyContainer.classList.contains('top-scrollable'));
          observer.disconnect();
          done();
          break;
      }
    });
    observer.observe(bodyContainer, {attributes: true});

    // Height is normally set via CSS, but mixin doesn't work with innerHTML.
    bodyContainer.style.height = '60px';  // Element has "min-height: 60px".
    bodyContainer.scrollTop = 100;
  });

  test('dialog cannot be cancelled when `no-cancel` is set', function() {
    document.body.innerHTML = `
      <cr-dialog no-cancel>
        <div slot="title">title</div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    dialog.showModal();

    // The paper-icon-button-light is the hidden element which is the
    // parentElement of the button.
    assertTrue(dialog.getCloseButton().parentElement.hidden);

    // Hitting escape fires a 'cancel' event. Cancelling that event prevents the
    // dialog from closing.
    let e = new CustomEvent('cancel', {cancelable: true});
    dialog.getNative().dispatchEvent(e);
    assertTrue(e.defaultPrevented);

    dialog.noCancel = false;

    e = new CustomEvent('cancel', {cancelable: true});
    dialog.getNative().dispatchEvent(e);
    assertFalse(e.defaultPrevented);
  });

  test('dialog close button shown when showCloseButton is true', function() {
    document.body.innerHTML = `
      <cr-dialog show-close-button>
        <div slot="title">title</div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    dialog.showModal();
    assertTrue(dialog.open);

    // The paper-icon-button-light is the hidden element which is the
    // parentElement of the button.
    assertFalse(dialog.getCloseButton().parentElement.hidden);
    assertEquals(
        'block',
        window.getComputedStyle(dialog.getCloseButton().parentElement).display);
    dialog.getCloseButton().click();
    assertFalse(dialog.open);
  });

  test('dialog close button hidden when showCloseButton is false', function() {
    document.body.innerHTML = `
      <cr-dialog>
        <div slot="title">title</div>
      </cr-dialog>`;

    const dialog = document.body.querySelector('cr-dialog');
    dialog.showModal();

    // The paper-icon-button-light is the hidden element which is the
    // parentElement of the button.
    assertTrue(dialog.getCloseButton().parentElement.hidden);
    assertEquals(
        'none',
        window.getComputedStyle(dialog.getCloseButton().parentElement).display);
  });
});
