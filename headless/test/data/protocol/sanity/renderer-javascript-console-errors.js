// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

(async function(testRunner) {
  let {page, session, dp} = await testRunner.startBlank(
      'Tests renderer: verify JavaScript console errors reporting.');

  let RendererTestHelper =
      await testRunner.loadScript('../helpers/renderer-test-helper.js');
  let {httpInterceptor, frameNavigationHelper, virtualTimeController} =
      await (new RendererTestHelper(testRunner, dp, page)).init();

  httpInterceptor.addResponse(
      `http://example.com/foobar` ,
      `<html>
      <head>
        <script language="JavaScript">
          <![CDATA[
            function image() {
              window.open('<xsl:value-of select="/IMAGE/@href" />');
            }
          ]]>
        </script>
      </head>
      <body onload="func3()">
        <script type="text/javascript">
          func1()
        </script>
        <script type="text/javascript">
          func2();
        </script>
        <script type="text/javascript">
          console.log("Hello, Script!");
        </script>
      </body>
      </html>`);

  await dp.Runtime.enable();

  dp.Runtime.onConsoleAPICalled(data => {
    const text = data.params.args[0].value;
    testRunner.log(`${text}`);
  });

  dp.Runtime.onExceptionThrown(data => {
    const details = data.params.exceptionDetails;
    testRunner.log(
        `${details.text} ${details.exception.description.replace(/\n.*/, '')}`);
  });

  await virtualTimeController.grantInitialTime(500, 1000,
    null,
    async () => {
      testRunner.completeTest();
    }
  );

  await frameNavigationHelper.navigate('http://example.com/foobar');
})
