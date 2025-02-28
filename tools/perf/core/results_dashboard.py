#!/usr/bin/env vpython
# Copyright (c) 2013 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Functions for adding results to perf dashboard."""

# This file was copy-pasted over from:
# //build/scripts/slave/results_dashboard.py

import calendar
import datetime
import httplib
import json
import os
import subprocess
import sys
import traceback
import time
import tempfile
import urllib
import urllib2
import zlib

import httplib2

from telemetry.internal.util import external_modules

from core import path_util

psutil = external_modules.ImportOptionalModule('psutil')


# The paths in the results dashboard URLs for sending results.
SEND_RESULTS_PATH = '/add_point'
SEND_HISTOGRAMS_PATH = '/add_histograms'


ERROR_NO_OAUTH_TOKEN = (
    'No oauth token provided, cannot upload HistogramSet. Discarding.')


class SendResultException(Exception):
  pass


class SendResultsRetryException(SendResultException):
  pass


class SendResultsFatalException(SendResultException):
  pass


def SendResults(data, url, send_as_histograms=False, oauth_token=None,
                num_retries=3):
  """Sends results to the Chrome Performance Dashboard.

  This function tries to send the given data to the dashboard.

  Args:
    data: The data to try to send. Must be JSON-serializable.
    url: Performance Dashboard URL (including schema).
    send_as_histograms: True if result is to be sent to /add_histograms.
    oauth_token: string; used for flushing oauth uploads from cache. Note that
      client is responsible for making sure that the oauth_token doesn't expire
      when using this API.
    num_retries: Number of times to retry uploading to the perf dashboard upon
      recoverable error.
  """
  start = time.time()

  # Send all the results from this run and the previous cache to the
  # dashboard.
  errors, all_data_uploaded = _SendResultsToDashboard(
      data, url, oauth_token, is_histogramset=send_as_histograms,
      num_retries=num_retries)

  print 'Time spent sending results to %s: %s' % (url, time.time() - start)

  for err in errors:
    print err

  return all_data_uploaded


def _SendResultsToDashboard(
    dashboard_data, url, oauth_token, is_histogramset, num_retries):
  """Tries to send perf dashboard data to |url|.

  Args:
    perf_results_file_path: A file name.
    url: The instance URL to which to post results.
    oauth_token: An oauth token to use for histogram uploads. Might be None.
    num_retries: Number of time to retry uploading to the perf dashboard upon
      recoverable error.

  Returns:
    A tuple (errors, all_data_uploaded), whereas:
      errors is a list of error strings.
      all_data_uploaded is a boolean indicating whether all perf data was
        succesfully uploaded.
  """

  errors = []
  all_data_uploaded = False

  data_type = ('histogram' if is_histogramset else 'chartjson')

  for i in xrange(1, num_retries + 1):
    try:
      print 'Sending %s result to dashboard (attempt %i out of %i).' % (
          data_type, i, num_retries)
      if is_histogramset:
        # TODO(eakuefner): Remove this discard logic once all bots use
        # histograms.
        if oauth_token is None:
          raise SendResultsFatalException(ERROR_NO_OAUTH_TOKEN)

        _SendHistogramJson(url, json.dumps(dashboard_data), oauth_token)
      else:
        _SendResultsJson(url, json.dumps(dashboard_data))
      all_data_uploaded = True
      break
    except SendResultsRetryException as e:
      error = 'Error while uploading %s data: %s' % (data_type, str(e))
      errors.append(error)
    except SendResultsFatalException as e:
      error = 'Error uploading %s data: %s' % (data_type, str(e))
      errors.append(error)
      break
    except Exception:
      error = 'Unexpected error while uploading %s data: %s' % (
          data_type, traceback.format_exc())
      errors.append(error)
      break

  return errors, all_data_uploaded


def MakeHistogramSetWithDiagnostics(histograms_file,
                                    test_name, bot, buildername, buildnumber,
                                    revisions_dict, is_reference_build,
                                    perf_dashboard_machine_group):
  add_diagnostics_args = []
  add_diagnostics_args.extend([
      '--benchmarks', test_name,
      '--bots', bot,
      '--builds', buildnumber,
      '--masters', perf_dashboard_machine_group,
      '--is_reference_build', 'true' if is_reference_build else '',
  ])

  url = _MakeStdioUrl(test_name, buildername, buildnumber)
  if url:
    add_diagnostics_args.extend(['--log_urls_k', 'Buildbot stdio'])
    add_diagnostics_args.extend(['--log_urls_v', url])

  for k, v in revisions_dict.iteritems():
    add_diagnostics_args.extend((k, v))

  add_diagnostics_args.append(histograms_file)

  # Subprocess only accepts string args
  add_diagnostics_args = [str(v) for v in add_diagnostics_args]

  add_reserved_diagnostics_path = os.path.join(
      path_util.GetChromiumSrcDir(), 'third_party', 'catapult', 'tracing',
      'bin', 'add_reserved_diagnostics')

  tf = tempfile.NamedTemporaryFile(delete=False)
  tf.close()
  temp_histogram_output_file = tf.name

  cmd = ([sys.executable, add_reserved_diagnostics_path] +
         add_diagnostics_args + ['--output_path', temp_histogram_output_file])

  try:
    subprocess.check_call(cmd)
    # TODO: Handle reference builds
    with open(temp_histogram_output_file) as f:
      hs = json.load(f)
    return hs
  finally:
    os.remove(temp_histogram_output_file)


def MakeListOfPoints(charts, bot, test_name, buildername,
                     buildnumber, supplemental_columns,
                     perf_dashboard_machine_group,
                     revisions_dict=None):
  """Constructs a list of point dictionaries to send.

  The format output by this function is the original format for sending data
  to the perf dashboard.

  Args:
    charts: A dictionary of chart names to chart data, as generated by the
        log processor classes (see process_log_utils.GraphingLogProcessor).
    bot: A string which comes from perf_id, e.g. linux-release.
    test_name: A test suite name, e.g. sunspider.
    buildername: Builder name (for stdio links).
    buildnumber: Build number (for stdio links).
    supplemental_columns: A dictionary of extra data to send with a point.
    perf_dashboard_machine_group: Builder's perf machine group.

  Returns:
    A list of dictionaries in the format accepted by the perf dashboard.
    Each dictionary has the keys "master", "bot", "test", "value", "revision".
    The full details of this format are described at http://goo.gl/TcJliv.
  """
  results = []

  for chart_name, chart_data in sorted(charts.items()):
    point_id, revision_columns = _RevisionNumberColumns(
      revisions_dict if revisions_dict is not None else chart_data, prefix='r_')

    for trace_name, trace_values in sorted(chart_data['traces'].items()):
      is_important = trace_name in chart_data.get('important', [])
      test_path = _TestPath(test_name, chart_name, trace_name)
      result = {
          'master': perf_dashboard_machine_group,
          'bot': bot,
          'test': test_path,
          'revision': point_id,
          'supplemental_columns': {}
      }

      # Add the supplemental_columns values that were passed in after the
      # calculated revision column values so that these can be overwritten.
      result['supplemental_columns'].update(revision_columns)
      result['supplemental_columns'].update(
          _GetStdioUriColumn(test_name, buildername, buildnumber))
      result['supplemental_columns'].update(supplemental_columns)

      result['value'] = trace_values[0]
      result['error'] = trace_values[1]

      # Add other properties to this result dictionary if available.
      if chart_data.get('units'):
        result['units'] = chart_data['units']
      if is_important:
        result['important'] = True

      results.append(result)

  return results


def MakeDashboardJsonV1(chart_json, revision_dict, test_name, bot, buildername,
                        buildnumber, supplemental_dict, is_ref,
                        perf_dashboard_machine_group):
  """Generates Dashboard JSON in the new Telemetry format.

  See http://goo.gl/mDZHPl for more info on the format.

  Args:
    chart_json: A dict containing the telmetry output.
    revision_dict: Dictionary of revisions to include, include "rev",
        which determines the point ID.
    test_name: A test suite name, e.g. sunspider.
    bot: A string which comes from perf_id, e.g. linux-release.
    buildername: Builder name (for stdio links).
    buildnumber: Build number (for stdio links).
    supplemental_dict: A dictionary of extra data to send with a point;
        this includes revisions and annotation data.
    is_ref: True if this is a reference build, False otherwise.
    perf_dashboard_machine_group: Builder's perf machine group.

  Returns:
    A dictionary in the format accepted by the perf dashboard.
  """
  if not chart_json:
    print 'Error: No json output from telemetry.'
    print '@@@STEP_FAILURE@@@'

  point_id, versions = _RevisionNumberColumns(revision_dict, prefix='')

  supplemental = {}
  for key in supplemental_dict:
    if key.startswith('r_'):
      versions[key.replace('r_', '', 1)] = supplemental_dict[key]
    if key.startswith('a_'):
      supplemental[key.replace('a_', '', 1)] = supplemental_dict[key]

  supplemental.update(
      _GetStdioUriColumn(test_name, buildername, buildnumber))

  # TODO(sullivan): The android recipe sends "test_name.reference"
  # while the desktop one just sends "test_name" for ref builds. Need
  # to figure out why.
  # https://github.com/catapult-project/catapult/issues/2046
  test_name = test_name.replace('.reference', '')

  fields = {
      'master': perf_dashboard_machine_group,
      'bot': bot,
      'test_suite_name': test_name,
      'point_id': point_id,
      'supplemental': supplemental,
      'versions': versions,
      'chart_data': chart_json,
      'is_ref': is_ref,
  }
  return fields


def _MakeStdioUrl(test_name, buildername, buildnumber):
  """Returns a string url pointing to buildbot stdio log."""
  # TODO(780914): Link to logdog instead of buildbot.
  if not buildername or not buildnumber:
    return ''

  return '%sbuilders/%s/builds/%s/steps/%s/logs/stdio' % (
      _GetBuildBotUrl(),
      urllib.quote(buildername),
      urllib.quote(str(buildnumber)),
      urllib.quote(test_name))


def _GetStdioUriColumn(test_name, buildername, buildnumber):
  """Gets a supplemental column containing buildbot stdio link."""
  url = _MakeStdioUrl(test_name, buildername, buildnumber)
  if not url:
    return {}
  return _CreateLinkColumn('stdio_uri', 'Buildbot stdio', url)


def _CreateLinkColumn(name, label, url):
  """Returns a column containing markdown link to show on dashboard."""
  return {'a_' + name: '[%s](%s)' % (label, url)}


def _GetBuildBotUrl():
  """Gets the buildbot URL which contains hostname and master name."""
  return os.environ.get('BUILDBOT_BUILDBOTURL',
                        'http://build.chromium.org/p/chromium/')


def _GetTimestamp():
  """Get the Unix timestamp for the current time."""
  return int(calendar.timegm(datetime.datetime.utcnow().utctimetuple()))


def _RevisionNumberColumns(data, prefix):
  """Get the point id and revision-related columns from the given data.

  Args:
    data: A dict of information from one line of the log file.
    master: The name of the buildbot master.
    prefix: Prefix for revision type keys. 'r_' for non-telemetry json, '' for
        telemetry json.

  Returns:
    A tuple with the point id (which must be an int), and a dict of
    revision-related columns.
  """
  revision_supplemental_columns = {}

  # The dashboard requires points' x-values to be integers, and points are
  # ordered by these x-values. If data['rev'] can't be parsed as an int, assume
  # that it's a git commit hash and use timestamp as the x-value.
  try:
    revision = int(data['rev'])
    if revision and revision > 300000 and revision < 1000000:
      # Revision is the commit pos.
      # TODO(sullivan,qyearsley): use got_revision_cp when available.
      revision_supplemental_columns[prefix + 'commit_pos'] = revision
  except ValueError:
    # The dashboard requires ordered integer revision numbers. If the revision
    # is not an integer, assume it's a git hash and send a timestamp.
    revision = _GetTimestamp()
    revision_supplemental_columns[prefix + 'chromium'] = data['rev']

  # An explicit data['point_id'] overrides the default behavior.
  if 'point_id' in data:
    revision = int(data['point_id'])

  # For other revision data, add it if it's present and not undefined:
  for key in ['webrtc_git', 'v8_rev']:
    if key in data and data[key] != 'undefined':
      revision_supplemental_columns[prefix + key] = data[key]

  # If possible, also send the git hash.
  if 'git_revision' in data and data['git_revision'] != 'undefined':
    revision_supplemental_columns[prefix + 'chromium'] = data['git_revision']

  return revision, revision_supplemental_columns


def _TestPath(test_name, chart_name, trace_name):
  """Get the slash-separated test path to send.

  Args:
    test: Test name. Typically, this will be a top-level 'test suite' name.
    chart_name: Name of a chart where multiple trace lines are grouped. If the
        chart name is the same as the trace name, that signifies that this is
        the main trace for the chart.
    trace_name: The "trace name" is the name of an individual line on chart.

  Returns:
    A slash-separated list of names that corresponds to the hierarchy of test
    data in the Chrome Performance Dashboard; doesn't include master or bot
    name.
  """
  # For tests run on reference builds by builds/scripts/slave/telemetry.py,
  # "_ref" is appended to the trace name. On the dashboard, as long as the
  # result is on the right chart, it can just be called "ref".
  if trace_name == chart_name + '_ref':
    trace_name = 'ref'
  chart_name = chart_name.replace('_by_url', '')

  # No slashes are allowed in the trace name.
  trace_name = trace_name.replace('/', '_')

  # The results for "test/chart" and "test/chart/*" will all be shown on the
  # same chart by the dashboard. The result with path "test/path" is considered
  # the main trace for the chart.
  test_path = '%s/%s/%s' % (test_name, chart_name, trace_name)
  if chart_name == trace_name:
    test_path = '%s/%s' % (test_name, chart_name)
  return test_path


def _SendResultsJson(url, results_json):
  """Make a HTTP POST with the given JSON to the Performance Dashboard.

  Args:
    url: URL of Performance Dashboard instance, e.g.
        "https://chromeperf.appspot.com".
    results_json: JSON string that contains the data to be sent.

  Returns:
    None if successful, or an error string if there were errors.
  """
  # When data is provided to urllib2.Request, a POST is sent instead of GET.
  # The data must be in the application/x-www-form-urlencoded format.
  data = urllib.urlencode({'data': results_json})
  req = urllib2.Request(url + SEND_RESULTS_PATH, data)
  try:
    urllib2.urlopen(req)
  except (urllib2.HTTPError, urllib2.URLError, httplib.HTTPException):
    error = traceback.format_exc()

    if 'HTTPError: 400' in error:
      # If the remote app rejects the JSON, it's probably malformed,
      # so we don't want to retry it.
      raise SendResultsFatalException('Discarding JSON, error:\n%s' % error)
    raise SendResultsRetryException(error)

def _Httplib2Request(url, data, oauth_token):
  data = zlib.compress(data)
  headers = {
      'Authorization': 'Bearer %s' % oauth_token,
      'User-Agent': 'perf-uploader/1.0'
  }

  http = httplib2.Http()
  return http.request(
      url + SEND_HISTOGRAMS_PATH, method='POST', body=data, headers=headers)

def _SendHistogramJson(url, histogramset_json, oauth_token):
  """POST a HistogramSet JSON to the Performance Dashboard.

  Args:
    url: URL of Performance Dashboard instance, e.g.
        "https://chromeperf.appspot.com".
    histogramset_json: JSON string that contains a serialized HistogramSet.
    oauth_token: An oauth token to be used for this upload.

  Returns:
    None if successful, or an error string if there were errors.
  """
  try:
    response, _ = _Httplib2Request(url, histogramset_json, oauth_token)

    # A 500 is presented on an exception on the dashboard side, timeout,
    # exception, etc. The dashboard can also send back 400 and 403, we could
    # recover from 403 (auth error), but 400 is generally malformed data.
    if response.status == 403:
      raise SendResultsRetryException(traceback.format_exc())

    if response.status != 200:
      raise SendResultsFatalException('HTTP Response %d: %s' % (
          response.status, response.reason))
  except httplib2.HttpLib2Error:
    raise SendResultsRetryException(traceback.format_exc())
