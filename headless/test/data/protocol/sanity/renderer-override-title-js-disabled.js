// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

(async function(testRunner) {
  let {page, session, dp} = await testRunner.startBlank(
      'Tests renderer: override title with JavaScript disabled.');

  let RendererTestHelper =
      await testRunner.loadScript('../helpers/renderer-test-helper.js');
  let {httpInterceptor, frameNavigationHelper, virtualTimeController} =
      await (new RendererTestHelper(testRunner, dp, page)).init();

  await dp.Emulation.setScriptExecutionDisabled({value: true});

  httpInterceptor.addResponse(
      `http://example.com/foobar`,
      `<html>
        <head>
          <title>JavaScript is off</title>
          <script language="JavaScript">
            function settitle() {
              document.title = 'JavaScript is on';
            }
            </script>
          </head>
        <body onload="settitle()">
          Hello, World!
        </body>
      </html>`);

  await virtualTimeController.grantInitialTime(500, 1000,
    null,
    async () => {
      testRunner.log(await session.evaluate('document.title'));
      testRunner.completeTest();
    }
  );

  await frameNavigationHelper.navigate('http://example.com/foobar');
})
