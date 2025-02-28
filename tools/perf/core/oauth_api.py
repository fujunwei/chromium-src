# Copyright 2017 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""API for generating OAuth2 access tokens from service account
keys predeployed to Chrome Ops bots via Puppet.
"""

import contextlib
import os
import subprocess
import tempfile


@contextlib.contextmanager
def with_access_token(
    service_account_json, prefix, token_expiration_in_minutes):
  """Yields an access token for the service account.

  Args:
    service_account_json: The path to the service account JSON file.
    prefix: the prefix of the token file
    token_expiration_in_minutes: the integer expiration time of the token
  """
  fd, path = tempfile.mkstemp(suffix='.json', prefix=prefix)
  try:
    args = ['luci-auth', 'token']
    if service_account_json:
      args += ['-service-account-json', service_account_json]
    args += ['-lifetime', '%im' % token_expiration_in_minutes]
    subprocess.check_call(args, stdout=fd)
    os.close(fd)
    fd = None
    yield path
  finally:
    if fd is not None:
      os.close(fd)
    os.remove(path)
