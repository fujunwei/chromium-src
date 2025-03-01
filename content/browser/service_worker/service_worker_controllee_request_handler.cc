// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/service_worker/service_worker_controllee_request_handler.h"

#include <memory>
#include <set>
#include <string>

#include "base/trace_event/trace_event.h"
#include "components/offline_pages/buildflags/buildflags.h"
#include "content/browser/service_worker/service_worker_context_core.h"
#include "content/browser/service_worker/service_worker_metrics.h"
#include "content/browser/service_worker/service_worker_provider_host.h"
#include "content/browser/service_worker/service_worker_registration.h"
#include "content/browser/service_worker/service_worker_response_info.h"
#include "content/browser/service_worker/service_worker_url_job_wrapper.h"
#include "content/browser/service_worker/service_worker_url_request_job.h"
#include "content/common/navigation_subresource_loader_params.h"
#include "content/common/service_worker/service_worker_utils.h"
#include "content/public/browser/content_browser_client.h"
#include "content/public/browser/render_frame_host.h"
#include "content/public/browser/resource_request_info.h"
#include "content/public/browser/web_contents.h"
#include "content/public/common/browser_side_navigation_policy.h"
#include "content/public/common/content_client.h"
#include "net/base/load_flags.h"
#include "net/base/url_util.h"
#include "net/url_request/url_request.h"
#include "services/network/public/cpp/resource_request_body.h"
#include "services/network/public/cpp/resource_response_info.h"
#include "third_party/blink/public/common/service_worker/service_worker_utils.h"
#include "ui/base/page_transition_types.h"

#if BUILDFLAG(ENABLE_OFFLINE_PAGES)
#include "components/offline_pages/core/request_header/offline_page_header.h"
#endif  // BUILDFLAG(ENABLE_OFFLINE_PAGES)

namespace content {

namespace {

bool MaybeForwardToServiceWorker(ServiceWorkerURLJobWrapper* job,
                                 const ServiceWorkerVersion* version) {
  DCHECK(job);
  DCHECK(version);
  DCHECK_NE(version->fetch_handler_existence(),
            ServiceWorkerVersion::FetchHandlerExistence::UNKNOWN);
  if (version->fetch_handler_existence() ==
      ServiceWorkerVersion::FetchHandlerExistence::EXISTS) {
    job->ForwardToServiceWorker();
    return true;
  }

  job->FallbackToNetworkOrRenderer();
  return false;
}

#if BUILDFLAG(ENABLE_OFFLINE_PAGES)
// A web page, regardless of whether the service worker is used or not, could
// be downloaded with the offline snapshot captured. The user can then open
// the downloaded page which is identified by the presence of a specific
// offline header in the network request. In this case, we want to fall back
// in order for the subsequent offline page interceptor to bring up the
// offline snapshot of the page.
bool ShouldFallbackToLoadOfflinePage(
    const net::HttpRequestHeaders& extra_request_headers) {
  std::string offline_header_value;
  if (!extra_request_headers.GetHeader(offline_pages::kOfflinePageHeader,
                                       &offline_header_value)) {
    return false;
  }
  offline_pages::OfflinePageHeader offline_header(offline_header_value);
  return offline_header.reason ==
         offline_pages::OfflinePageHeader::Reason::DOWNLOAD;
}
#endif  // BUILDFLAG(ENABLE_OFFLINE_PAGES)

}  // namespace

ServiceWorkerControlleeRequestHandler::ServiceWorkerControlleeRequestHandler(
    base::WeakPtr<ServiceWorkerContextCore> context,
    base::WeakPtr<ServiceWorkerProviderHost> provider_host,
    base::WeakPtr<storage::BlobStorageContext> blob_storage_context,
    network::mojom::FetchRequestMode request_mode,
    network::mojom::FetchCredentialsMode credentials_mode,
    network::mojom::FetchRedirectMode redirect_mode,
    const std::string& integrity,
    bool keepalive,
    ResourceType resource_type,
    RequestContextType request_context_type,
    network::mojom::RequestContextFrameType frame_type,
    scoped_refptr<network::ResourceRequestBody> body)
    : ServiceWorkerRequestHandler(context,
                                  provider_host,
                                  blob_storage_context,
                                  resource_type),
      resource_type_(resource_type),
      is_main_resource_load_(
          ServiceWorkerUtils::IsMainResourceType(resource_type)),
      is_main_frame_load_(resource_type == RESOURCE_TYPE_MAIN_FRAME),
      request_mode_(request_mode),
      credentials_mode_(credentials_mode),
      redirect_mode_(redirect_mode),
      integrity_(integrity),
      keepalive_(keepalive),
      request_context_type_(request_context_type),
      frame_type_(frame_type),
      body_(body),
      force_update_started_(false),
      use_network_(false),
      weak_factory_(this) {}

ServiceWorkerControlleeRequestHandler::
    ~ServiceWorkerControlleeRequestHandler() {
  MaybeScheduleUpdate();

  if (is_main_resource_load_ && provider_host_)
    provider_host_->SetAllowAssociation(true);
}

void ServiceWorkerControlleeRequestHandler::MaybeScheduleUpdate() {
  if (!provider_host_ || !provider_host_->active_version())
    return;

  if (blink::ServiceWorkerUtils::IsServicificationEnabled()) {
    // For subresources: S13nServiceWorker doesn't come here.
    DCHECK(is_main_resource_load_);

    // For navigations, the update logic is taken care of
    // during navigation and waits for the HintToUpdateServiceWorker message.
    if (IsResourceTypeFrame(resource_type_))
      return;

    // Continue to the common non-S13nServiceWorker code for triggering update
    // for shared workers. The renderer doesn't yet send a
    // HintToUpdateServiceWorker message.
    // TODO(falken): Make the renderer send the message for shared worker,
    // to simplify the code.
  }

  // If DevTools forced an update, there is no need to update again.
  if (force_update_started_)
    return;

  if (is_main_resource_load_)
    provider_host_->active_version()->ScheduleUpdate();
  else
    provider_host_->active_version()->DeferScheduledUpdate();
}

net::URLRequestJob* ServiceWorkerControlleeRequestHandler::MaybeCreateJob(
    net::URLRequest* request,
    net::NetworkDelegate* network_delegate,
    ResourceContext* resource_context) {
  ClearJob();
  ServiceWorkerResponseInfo::ResetDataForRequest(request);

  if (!context_ || !provider_host_) {
    // We can't do anything other than to fall back to network.
    return nullptr;
  }

  // This may get called multiple times for original and redirect requests:
  // A. original request case: use_network_ is false, no previous location info.
  // B. redirect or restarted request case:
  //  a) use_network_ is false if the previous location was forwarded to SW.
  //  b) use_network_ is false if the previous location was fallback.
  //  c) use_network_ is true if additional restart was required to fall back.

  // Fall back to network. (Case B-c)
  if (use_network_) {
    // Once a subresource request has fallen back to the network once, it will
    // never be handled by a service worker. This is not true of main frame
    // requests.
    if (is_main_resource_load_)
      use_network_ = false;
    return nullptr;
  }

#if BUILDFLAG(ENABLE_OFFLINE_PAGES)
  // Fall back for the subsequent offline page interceptor to load the offline
  // snapshot of the page if required.
  if (ShouldFallbackToLoadOfflinePage(request->extra_request_headers()))
    return nullptr;
#endif  // BUILDFLAG(ENABLE_OFFLINE_PAGES)

  // It's for original request (A) or redirect case (B-a or B-b).
  std::unique_ptr<ServiceWorkerURLRequestJob> job(
      new ServiceWorkerURLRequestJob(
          request, network_delegate, provider_host_->client_uuid(),
          blob_storage_context_, resource_context, request_mode_,
          credentials_mode_, redirect_mode_, integrity_, keepalive_,
          resource_type_, request_context_type_, frame_type_, body_, this));
  url_job_ = std::make_unique<ServiceWorkerURLJobWrapper>(job->GetWeakPtr());

  resource_context_ = resource_context;

  if (is_main_resource_load_)
    PrepareForMainResource(request->url(), request->site_for_cookies());
  else
    PrepareForSubResource();

  if (url_job_->ShouldFallbackToNetwork()) {
    // If we know we can fallback to network at this point (in case
    // the storage lookup returned immediately), just destroy the job and return
    // NULL here to fallback to network.

    // If this is a subresource request, all subsequent requests should also use
    // the network.
    if (!is_main_resource_load_)
      use_network_ = true;

    job.reset();
    ClearJob();
  }

  return job.release();
}

void ServiceWorkerControlleeRequestHandler::MaybeCreateLoader(
    const network::ResourceRequest& resource_request,
    ResourceContext* resource_context,
    LoaderCallback callback) {
  DCHECK(blink::ServiceWorkerUtils::IsServicificationEnabled());
  DCHECK(is_main_resource_load_);
  ClearJob();

  if (!context_ || !provider_host_) {
    // We can't do anything other than to fall back to network.
    std::move(callback).Run({});
    return;
  }

  // In fallback cases we basically 'forward' the request, so we should
  // never see use_network_ gets true.
  DCHECK(!use_network_);

#if BUILDFLAG(ENABLE_OFFLINE_PAGES)
  // Fall back for the subsequent offline page interceptor to load the offline
  // snapshot of the page if required.
  if (ShouldFallbackToLoadOfflinePage(resource_request.headers)) {
    std::move(callback).Run({});
    return;
  }
#endif  // BUILDFLAG(ENABLE_OFFLINE_PAGES)

  url_job_ = std::make_unique<ServiceWorkerURLJobWrapper>(
      std::make_unique<ServiceWorkerNavigationLoader>(
          std::move(callback), this, resource_request,
          base::WrapRefCounted(context_->loader_factory_getter())));

  resource_context_ = resource_context;

  PrepareForMainResource(resource_request.url,
                         resource_request.site_for_cookies);

  if (url_job_->ShouldFallbackToNetwork()) {
    // We're falling back to the next NavigationLoaderInterceptor, forward
    // the request and clear job now.
    url_job_->FallbackToNetwork();
    ClearJob();
    return;
  }

  // We will asynchronously continue on DidLookupRegistrationForMainResource.
}

base::Optional<SubresourceLoaderParams>
ServiceWorkerControlleeRequestHandler::MaybeCreateSubresourceLoaderParams() {
  DCHECK(blink::ServiceWorkerUtils::IsServicificationEnabled());

  // We didn't create URLLoader for this request.
  if (!url_job_)
    return base::nullopt;

  // DidLookupRegistrationForMainResource() for the request didn't find
  // a matching service worker for this request, and
  // ServiceWorkerProviderHost::AssociateRegistration() was not called.
  if (!provider_host_ || !provider_host_->controller())
    return base::nullopt;

  // Otherwise let's send the controller service worker information along
  // with the navigation commit.
  SubresourceLoaderParams params;
  auto controller_info = mojom::ControllerServiceWorkerInfo::New();
  controller_info->mode = provider_host_->GetControllerMode();
  // Note that |controller_info->endpoint| is null if the controller has no
  // fetch event handler. In that case the renderer frame won't get the
  // controller pointer upon the navigation commit, and subresource loading will
  // not be intercepted. (It might get intercepted later if the controller
  // changes due to skipWaiting() so SetController is sent.)
  controller_info->endpoint =
      provider_host_->GetControllerServiceWorkerPtr().PassInterface();
  controller_info->client_id = provider_host_->client_uuid();
  base::WeakPtr<ServiceWorkerObjectHost> object_host =
      provider_host_->GetOrCreateServiceWorkerObjectHost(
          provider_host_->controller());
  if (object_host) {
    params.controller_service_worker_object_host = object_host;
    controller_info->object_info = object_host->CreateIncompleteObjectInfo();
  }
  params.controller_service_worker_info = std::move(controller_info);
  return base::Optional<SubresourceLoaderParams>(std::move(params));
}

void ServiceWorkerControlleeRequestHandler::PrepareForMainResource(
    const GURL& url,
    const GURL& site_for_cookies) {
  DCHECK(!JobWasCanceled());
  DCHECK(context_);
  DCHECK(provider_host_);
  TRACE_EVENT_ASYNC_BEGIN1(
      "ServiceWorker",
      "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
      url_job_.get(), "URL", url.spec());
  // The corresponding provider_host may already have associated a registration
  // in redirect case, unassociate it now.
  provider_host_->DisassociateRegistration();

  // Also prevent a registration from claiming this host while it's not
  // yet execution ready.
  // TODO(falken): Make an RAII helper instead of all these explcit allow and
  // disallow calls.
  provider_host_->SetAllowAssociation(false);

  stripped_url_ = net::SimplifyUrlForRequest(url);
  provider_host_->SetDocumentUrl(stripped_url_);
  provider_host_->SetTopmostFrameUrl(site_for_cookies);
  context_->storage()->FindRegistrationForDocument(
      stripped_url_, base::BindOnce(&self::DidLookupRegistrationForMainResource,
                                    weak_factory_.GetWeakPtr()));
}

void ServiceWorkerControlleeRequestHandler::
    DidLookupRegistrationForMainResource(
        blink::ServiceWorkerStatusCode status,
        scoped_refptr<ServiceWorkerRegistration> registration) {
  // The job may have been canceled before this was invoked.
  if (JobWasCanceled())
    return;

  if (provider_host_)
    provider_host_->SetAllowAssociation(true);

  if (status != blink::ServiceWorkerStatusCode::kOk) {
    url_job_->FallbackToNetwork();
    TRACE_EVENT_ASYNC_END1(
        "ServiceWorker",
        "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
        url_job_.get(), "Status", blink::ServiceWorkerStatusToString(status));
    return;
  }
  DCHECK(registration.get());

  if (!provider_host_) {
    url_job_->FallbackToNetwork();
    TRACE_EVENT_ASYNC_END1(
        "ServiceWorker",
        "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
        url_job_.get(), "Info", "No Provider");
    return;
  }

  if (!context_) {
    url_job_->FallbackToNetwork();
    TRACE_EVENT_ASYNC_END1(
        "ServiceWorker",
        "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
        url_job_.get(), "Info", "No Context");
    return;
  }

  if (!GetContentClient()->browser()->AllowServiceWorker(
          registration->pattern(), provider_host_->topmost_frame_url(),
          resource_context_, provider_host_->web_contents_getter())) {
    url_job_->FallbackToNetwork();
    TRACE_EVENT_ASYNC_END1(
        "ServiceWorker",
        "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
        url_job_.get(), "Info", "ServiceWorker is blocked");
    return;
  }

  if (!provider_host_->IsContextSecureForServiceWorker()) {
    // TODO(falken): Figure out a way to surface in the page's DevTools
    // console that the service worker was blocked for security.
    url_job_->FallbackToNetwork();
    TRACE_EVENT_ASYNC_END1(
        "ServiceWorker",
        "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
        url_job_.get(), "Info", "Insecure context");
    return;
  }

  const bool need_to_update = !force_update_started_ && registration &&
                              context_->force_update_on_page_load();
  if (need_to_update) {
    force_update_started_ = true;
    provider_host_->SetAllowAssociation(false);
    context_->UpdateServiceWorker(
        registration.get(), true /* force_bypass_cache */,
        true /* skip_script_comparison */,
        base::BindOnce(&self::DidUpdateRegistration, weak_factory_.GetWeakPtr(),
                       registration));
    return;
  }

  // Initiate activation of a waiting version. Usually a register job initiates
  // activation but that doesn't happen if the browser exits prior to activation
  // having occurred. This check handles that case.
  if (registration->waiting_version())
    registration->ActivateWaitingVersionWhenReady();

  scoped_refptr<ServiceWorkerVersion> active_version =
      registration->active_version();
  if (!active_version) {
    // Although there is no active version, a registration exists, so associate
    // it, so it can be used for .ready.
    // TODO(falken): There's no need to associate the registration just for
    // .ready. Change this to AddMatchingRegistration instead, and call it
    // unconditionally if there is a |provider_host_|.
    provider_host_->AssociateRegistration(registration.get(),
                                          false /* notify_controllerchange */);

    url_job_->FallbackToNetwork();
    TRACE_EVENT_ASYNC_END1(
        "ServiceWorker",
        "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
        url_job_.get(), "Info",
        "No active version, so falling back to network");
    return;
  }

  // TODO(falken): Change these to DCHECK if it holds.
  CHECK(active_version->status() == ServiceWorkerVersion::ACTIVATING ||
        active_version->status() == ServiceWorkerVersion::ACTIVATED);

  // Wait until it's activated before firing fetch events.
  if (active_version->status() == ServiceWorkerVersion::ACTIVATING) {
    provider_host_->SetAllowAssociation(false);
    registration->active_version()->RegisterStatusChangeCallback(base::BindOnce(
        &ServiceWorkerControlleeRequestHandler::
            ContinueWithInScopeMainResourceRequest,
        weak_factory_.GetWeakPtr(), registration, active_version));
    TRACE_EVENT_ASYNC_END1(
        "ServiceWorker",
        "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
        url_job_.get(), "Info", "Wait until finished SW activation");
    return;
  }

  ContinueWithInScopeMainResourceRequest(std::move(registration),
                                         std::move(active_version));
}

void ServiceWorkerControlleeRequestHandler::
    ContinueWithInScopeMainResourceRequest(
        scoped_refptr<ServiceWorkerRegistration> registration,
        scoped_refptr<ServiceWorkerVersion> active_version) {
  if (provider_host_)
    provider_host_->SetAllowAssociation(true);

  // The job may have been canceled before this was invoked. In that
  // case, |url_job_| can't be used, so return.
  if (JobWasCanceled()) {
    TRACE_EVENT_ASYNC_END1(
        "ServiceWorker",
        "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
        url_job_.get(), "Info", "The job was canceled");
    return;
  }

  if (!provider_host_) {
    url_job_->FallbackToNetwork();
    TRACE_EVENT_ASYNC_END1(
        "ServiceWorker",
        "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
        url_job_.get(), "Info",
        "The provider host is gone, so falling back to network");
    return;
  }

  if (active_version->status() != ServiceWorkerVersion::ACTIVATED) {
    // TODO(falken): Clean this up and clarify in what cases we come here. I
    // guess it's:
    // - strange system error cases where promoting from ACTIVATING to ACTIVATED
    //   failed (shouldn't happen)
    // - something calling Doom(), etc, making the active_version REDUNDANT
    // - a version called skipWaiting() during activation so the expected
    //   version is no longer the active one (shouldn't happen: skipWaiting()
    //   waits for the active version to finish activating).
    // In most cases, it sounds like falling back to network would not be right,
    // since it's still in-scope. We probably should do:
    //   1) If the provider host has an active version that is ACTIVATED, just
    //      use that, even if it wasn't the expected one.
    //   2) If the provider host has an active version that is not ACTIVATED,
    //      just fail the load. The correct thing is probably to re-try
    //      activating that version, but there's a risk of an infinite loop of
    //      retries.
    //   3) If the provider host does not have an active version, just fail the
    //      load.
    url_job_->FallbackToNetwork();
    TRACE_EVENT_ASYNC_END2(
        "ServiceWorker",
        "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
        url_job_.get(), "Info",
        "The expected active version is not ACTIVATED, so falling back to "
        "network",
        "Status",
        ServiceWorkerVersion::VersionStatusToString(active_version->status()));
    return;
  }

  provider_host_->AssociateRegistration(registration.get(),
                                        false /* notify_controllerchange */);

  // TODO(falken): Change these to DCHECK if it holds, or else figure out
  // how this happens.
  CHECK_EQ(active_version, registration->active_version());
  CHECK_EQ(active_version, provider_host_->controller());

  DCHECK_NE(active_version->fetch_handler_existence(),
            ServiceWorkerVersion::FetchHandlerExistence::UNKNOWN);
  ServiceWorkerMetrics::CountControlledPageLoad(
      active_version->site_for_uma(), stripped_url_, is_main_frame_load_);

  if (blink::ServiceWorkerUtils::IsServicificationEnabled() &&
      IsResourceTypeFrame(resource_type_)) {
    provider_host_->AddServiceWorkerToUpdate(active_version);
  }
  bool is_forwarded =
      MaybeForwardToServiceWorker(url_job_.get(), active_version.get());
  TRACE_EVENT_ASYNC_END1(
      "ServiceWorker",
      "ServiceWorkerControlleeRequestHandler::PrepareForMainResource",
      url_job_.get(), "Info",
      (is_forwarded) ? "Forwarded to the ServiceWorker"
                     : "Skipped the ServiceWorker which has no fetch handler");
}

void ServiceWorkerControlleeRequestHandler::DidUpdateRegistration(
    const scoped_refptr<ServiceWorkerRegistration>& original_registration,
    blink::ServiceWorkerStatusCode status,
    const std::string& status_message,
    int64_t registration_id) {
  DCHECK(force_update_started_);

  // The job may have been canceled before this was invoked.
  if (JobWasCanceled())
    return;

  if (!context_) {
    url_job_->FallbackToNetwork();
    return;
  }
  if (status != blink::ServiceWorkerStatusCode::kOk ||
      !original_registration->installing_version()) {
    // Update failed. Look up the registration again since the original
    // registration was possibly unregistered in the meantime.
    context_->storage()->FindRegistrationForDocument(
        stripped_url_,
        base::BindOnce(&self::DidLookupRegistrationForMainResource,
                       weak_factory_.GetWeakPtr()));
    return;
  }
  DCHECK_EQ(original_registration->id(), registration_id);
  scoped_refptr<ServiceWorkerVersion> new_version =
      original_registration->installing_version();
  new_version->ReportForceUpdateToDevTools();
  new_version->set_skip_waiting(true);
  new_version->RegisterStatusChangeCallback(base::BindOnce(
      &self::OnUpdatedVersionStatusChanged, weak_factory_.GetWeakPtr(),
      original_registration, new_version));
}

void ServiceWorkerControlleeRequestHandler::OnUpdatedVersionStatusChanged(
    const scoped_refptr<ServiceWorkerRegistration>& registration,
    const scoped_refptr<ServiceWorkerVersion>& version) {
  // The job may have been canceled before this was invoked.
  if (JobWasCanceled())
    return;

  if (!context_) {
    url_job_->FallbackToNetwork();
    return;
  }
  if (version->status() == ServiceWorkerVersion::ACTIVATED ||
      version->status() == ServiceWorkerVersion::REDUNDANT) {
    // When the status is REDUNDANT, the update failed (eg: script error), we
    // continue with the incumbent version.
    // In case unregister job may have run, look up the registration again.
    context_->storage()->FindRegistrationForDocument(
        stripped_url_,
        base::BindOnce(&self::DidLookupRegistrationForMainResource,
                       weak_factory_.GetWeakPtr()));
    return;
  }
  version->RegisterStatusChangeCallback(
      base::BindOnce(&self::OnUpdatedVersionStatusChanged,
                     weak_factory_.GetWeakPtr(), registration, version));
}

void ServiceWorkerControlleeRequestHandler::PrepareForSubResource() {
  DCHECK(!JobWasCanceled());
  DCHECK(context_);

  // When this request handler was created, the provider host had a controller
  // and hence an active version, but by the time MaybeCreateJob() is called
  // the active version may have been lost. This happens when
  // ServiceWorkerRegistration::DeleteVersion() was called to delete the worker
  // because a permanent failure occurred when trying to start it.
  //
  // As this is an exceptional case, just error out.
  // TODO(falken): Figure out if |active_version| can change to |controller| and
  // do it or document the findings.
  if (!provider_host_->active_version()) {
    url_job_->FailDueToLostController();
    return;
  }

  MaybeForwardToServiceWorker(url_job_.get(), provider_host_->active_version());
}

void ServiceWorkerControlleeRequestHandler::OnPrepareToRestart() {
  use_network_ = true;
  ClearJob();
}

ServiceWorkerVersion*
ServiceWorkerControlleeRequestHandler::GetServiceWorkerVersion(
    ServiceWorkerMetrics::URLRequestJobResult* result) {
  if (!provider_host_) {
    *result = ServiceWorkerMetrics::REQUEST_JOB_ERROR_NO_PROVIDER_HOST;
    return nullptr;
  }
  if (!provider_host_->active_version()) {
    *result = ServiceWorkerMetrics::REQUEST_JOB_ERROR_NO_ACTIVE_VERSION;
    return nullptr;
  }
  return provider_host_->active_version();
}

bool ServiceWorkerControlleeRequestHandler::RequestStillValid(
    ServiceWorkerMetrics::URLRequestJobResult* result) {
  // A null |provider_host_| probably means the tab was closed. The null value
  // would cause problems down the line, so bail out.
  if (!provider_host_) {
    *result = ServiceWorkerMetrics::REQUEST_JOB_ERROR_NO_PROVIDER_HOST;
    return false;
  }
  return true;
}

void ServiceWorkerControlleeRequestHandler::MainResourceLoadFailed() {
  DCHECK(provider_host_);
  // Detach the controller so subresource requests also skip the worker.
  provider_host_->NotifyControllerLost();
}

void ServiceWorkerControlleeRequestHandler::ClearJob() {
  url_job_.reset();
}

bool ServiceWorkerControlleeRequestHandler::JobWasCanceled() const {
  return !url_job_ || url_job_->WasCanceled();
}

}  // namespace content
