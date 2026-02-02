//! This module provides the core HTTP loading functionality, encompassing the entire lifecycle
//! of network requests from initiation to response handling. It manages various aspects
//! suchs as caching, redirection, cookie management, authentication, and integration with
//! developer tools. The implementation adheres to Fetch API specifications and
//! web security policies like Referrer Policy and Cross-Origin Resource Policy (CORP).

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::sync::{Arc as StdArc, Condvar, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_recursion::async_recursion;
use base::cross_process_instant::CrossProcessInstant;
use base::id::{HistoryStateId, PipelineId, WebViewId};
use crossbeam_channel::Sender;
use devtools_traits::{
    ChromeToDevtoolsControlMsg, DevtoolsControlMsg, HttpRequest as DevtoolsHttpRequest,
    HttpResponse as DevtoolsHttpResponse, NetworkEvent,
};
use embedder_traits::{
    EmbedderMsg, EmbedderProxy, PromptCredentialsInput, PromptDefinition, PromptOrigin,
};
use futures::{future, TryFutureExt, TryStreamExt};
use headers::authorization::Basic;
use headers::{
    AccessControlAllowCredentials, AccessControlAllowHeaders, AccessControlAllowMethods,
    AccessControlAllowOrigin, AccessControlMaxAge, AccessControlRequestHeaders,
    AccessControlRequestMethod, Authorization, CacheControl, ContentLength, HeaderMapExt,
    IfModifiedSince, LastModified, Pragma, Referer, UserAgent,
};
use http::header::{
    self, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_ENCODING, CONTENT_LANGUAGE, CONTENT_LOCATION,
    CONTENT_TYPE,
};
use http::{HeaderMap, Method, Request as HyperRequest, StatusCode};
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Frame};
use hyper::ext::ReasonPhrase;
use hyper::header::{HeaderName, TRANSFER_ENCODING};
use hyper::Response as HyperResponse;
use hyper_serde::Serde;
use hyper_util::client::legacy::Client;
use ipc_channel::ipc::{self, IpcSender};
use ipc_channel::router::ROUTER;
use log::{debug, error, info, log_enabled, warn};
use net_traits::http_status::HttpStatus;
use net_traits::pub_domains::reg_suffix;
use net_traits::request::Origin::Origin as SpecificOrigin;
use net_traits::request::{
    get_cors_unsafe_header_names, is_cors_non_wildcard_request_header_name,
    is_cors_safelisted_method, is_cors_safelisted_request_header, BodyChunkRequest,
    BodyChunkResponse, CacheMode, CredentialsMode, Destination, Initiator, Origin, RedirectMode,
    Referrer, Request, RequestBuilder, RequestMode, ResponseTainting, ServiceWorkersMode,
    Window as RequestWindow,
};
use net_traits::response::{HttpsState, Response, ResponseBody, ResponseType};
use net_traits::{
    CookieSource, FetchMetadata, NetworkError, RedirectEndValue, RedirectStartValue,
    ReferrerPolicy, ResourceAttribute, ResourceFetchTiming, ResourceTimeValue,
    DOCUMENT_ACCEPT_HEADER_VALUE,
};
use servo_arc::Arc;
use servo_url::{ImmutableOrigin, ServoUrl};
use tokio::sync::mpsc::{
    channel, unbounded_channel, Receiver as TokioReceiver, Sender as TokioSender,
    UnboundedReceiver, UnboundedSender,
};
use tokio_stream::wrappers::ReceiverStream;

use crate::async_runtime::HANDLE;
use crate::connector::{CertificateErrorOverrideManager, Connector};
use crate::cookie::ServoCookie;
use crate::cookie_storage::CookieStorage;
use crate::decoder::Decoder;
use crate::fetch::cors_cache::CorsCache;
use crate::fetch::fetch_params::FetchParams;
use crate::fetch::headers::{SecFetchDest, SecFetchMode, SecFetchSite, SecFetchUser};
use crate::fetch::methods::{main_fetch, Data, DoneChannel, FetchContext, Target};
use crate::hsts::HstsList;
use crate::http_cache::{CacheKey, HttpCache};
use crate::resource_thread::{AuthCache, AuthCacheEntry};

/// The various states an entry of the HttpCache can be in.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HttpCacheEntryState {
    /// The entry is fully up-to-date,
    /// there are no pending concurrent stores,
    /// and it is ready to construct cached responses.
    ReadyToConstruct,
    /// The entry is pending a number of concurrent stores.
    PendingStore(usize),
}

type HttpCacheState = Mutex<HashMap<CacheKey, Arc<(Mutex<HttpCacheEntryState>, Condvar)>>>;

pub struct HttpState {
    /// Manages the list of HTTP Strict Transport Security (HSTS) hosts.
    pub hsts_list: RwLock<HstsList>,
    /// Stores and manages HTTP cookies.
    pub cookie_jar: RwLock<CookieStorage>,
    /// Manages the HTTP cache for storing and retrieving responses.
    pub http_cache: RwLock<HttpCache>,
    /// A map of cache key to entry state,
    /// reflecting whether the cache entry is ready to read from,
    /// or whether a concurrent pending store should be awaited.
    pub http_cache_state: HttpCacheState,
    /// Caches authentication credentials for various origins.
    pub auth_cache: RwLock<AuthCache>,
    /// Stores history states, typically for navigation or session management.
    pub history_states: RwLock<HashMap<HistoryStateId, Vec<u8>>>,
    /// The Hyper client used for making HTTP requests.
    pub client: Client<Connector, crate::connector::BoxedBody>,
    /// Manages overrides for certificate errors.
    pub override_manager: CertificateErrorOverrideManager,
    /// Provides an interface for the embedder to interact with the HTTP loader.
    pub embedder_proxy: Mutex<EmbedderProxy>,
}

/// Step 13 of <https://fetch.spec.whatwg.org/#concept-fetch>.
/// Sets the default `Accept` header for a given HTTP request based on its destination.
///
/// This function implements Step 13 of the Fetch specification:
/// <https://fetch.spec.whatwg.org/#concept-fetch>.
pub(crate) fn set_default_accept(request: &mut Request) {
    // Block Logic: Checks if the request already contains an `Accept` header.
    // Invariant: If an `Accept` header is present, no default is set, preserving existing preference.
    if request.headers.contains_key(header::ACCEPT) {
        return;
    }

    // Block Logic: Determines the appropriate `Accept` header value based on the request's initiator and destination.
    // Invariant: A default `Accept` header value is chosen that best matches the expected content type for the request.
    let value = if request.initiator == Initiator::Prefetch {
        // Functional Utility: Uses a predefined `Accept` header value for prefetch requests.
        DOCUMENT_ACCEPT_HEADER_VALUE
    } else {
        match request.destination {
            // Functional Utility: Uses a predefined `Accept` header value for document, frame, or iframe destinations.
            Destination::Document | Destination::Frame | Destination::IFrame => {
                DOCUMENT_ACCEPT_HEADER_VALUE
            },
            // Functional Utility: Specifies `Accept` headers for image destinations.
            Destination::Image => {
                HeaderValue::from_static("image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5")
            },
            // Functional Utility: Specifies `Accept` headers for JSON destinations.
            Destination::Json => HeaderValue::from_static("application/json,*/*;q=0.5"),
            // Functional Utility: Specifies `Accept` headers for style destinations.
            Destination::Style => HeaderValue::from_static("text/css,*/*;q=0.1"),
            // Functional Utility: Provides a generic `Accept` header for other destinations.
            _ => HeaderValue::from_static("*/*"),
        }
    };

    // Functional Utility: Inserts the determined `Accept` header value into the request headers.
    request.headers.insert(header::ACCEPT, value);
}

/// Sets the default `Accept-Encoding` header if it's not already present.
fn set_default_accept_encoding(headers: &mut HeaderMap) {
    // Block Logic: Checks if the `Accept-Encoding` header is already present.
    // Invariant: If an `Accept-Encoding` header is present, no default is set, preserving existing preference.
    if headers.contains_key(header::ACCEPT_ENCODING) {
        return;
    }

    // TODO(eijebong): Change this once typed headers are done
    // Functional Utility: Inserts the default `Accept-Encoding` header value for common compression algorithms.
    headers.insert(
        header::ACCEPT_ENCODING,
        HeaderValue::from_static("gzip, deflate, br"),
    );
}

/// Implements the "no-referrer-when-downgrade" referrer policy.
///
/// This policy sends a full URL as a referrer when the protocol security level
/// stays the same or improves (HTTP to HTTPS), but sends no referrer when
/// the security level downgrades (HTTPS to HTTP).
/// See <https://w3c.github.io/webappsec-referrer-policy/#referrer-policy-no-referrer-when-downgrade>
fn no_referrer_when_downgrade(referrer_url: ServoUrl, current_url: ServoUrl) -> Option<ServoUrl> {
    // Block Logic: Implements Step 1 of the no-referrer-when-downgrade policy.
    // If the referrer URL is potentially trustworthy but the current URL is not,
    // then no referrer information is sent.
    // Invariant: Prevents sending referrer from a secure context to an insecure one.
    if referrer_url.is_potentially_trustworthy() && !current_url.is_potentially_trustworthy() {
        return None;
    }
    // Functional Utility: Implements Step 2: Strips sensitive information from the referrer URL.
    strip_url_for_use_as_referrer(referrer_url, false)
}

/// Implements the "strict-origin" referrer policy.
///
/// This policy sends the referrer's origin when the protocol security level
/// stays the same or improves, but sends no referrer when the security level
/// downgrades. It always strips paths and query strings.
/// See <https://w3c.github.io/webappsec-referrer-policy/#referrer-policy-strict-origin>
fn strict_origin(referrer_url: ServoUrl, current_url: ServoUrl) -> Option<ServoUrl> {
    // Block Logic: Implements Step 1 of the strict-origin policy.
    // If the referrer URL is potentially trustworthy but the current URL is not,
    // then no referrer information is sent.
    // Invariant: Prevents sending referrer from a secure context to an insecure one.
    if referrer_url.is_potentially_trustworthy() && !current_url.is_potentially_trustworthy() {
        return None;
    }
    // Functional Utility: Implements Step 2: Strips path and query information from the referrer URL,
    // sending only the origin.
    strip_url_for_use_as_referrer(referrer_url, true)
}

/// Implements the "strict-origin-when-cross-origin" referrer policy.
///
/// This policy sends a full URL when performing a same-origin request,
/// only the origin when performing a cross-origin request, and sends no
/// referrer when the security level downgrades.
/// See <https://w3c.github.io/webappsec-referrer-policy/#referrer-policy-strict-origin-when-cross-origin>
fn strict_origin_when_cross_origin(
    referrer_url: ServoUrl,
    current_url: ServoUrl,
) -> Option<ServoUrl> {
    // Block Logic: Implements Step 1 of the policy.
    // If the request is same-origin, send the full referrer URL.
    // Invariant: Full referrer is sent for same-origin requests.
    if referrer_url.origin() == current_url.origin() {
        return strip_url_for_use_as_referrer(referrer_url, false);
    }
    // Block Logic: Implements Step 2 of the policy.
    // If the referrer URL is potentially trustworthy but the current URL is not,
    // then no referrer information is sent.
    // Invariant: Prevents sending referrer from a secure context to an insecure one.
    if referrer_url.is_potentially_trustworthy() && !current_url.is_potentially_trustworthy() {
        return None;
    }
    // Functional Utility: Implements Step 3: For cross-origin requests, strips sensitive information,
    // sending only the origin.
    strip_url_for_use_as_referrer(referrer_url, true)
}

/// Determines if two given `ImmutableOrigin`s are "same site" according to the HTML specification.
///
/// This involves comparing their scheme and host values, with special handling for opaque origins.
/// See <https://html.spec.whatwg.org/multipage/#concept-site-same-site>
fn is_same_site(site_a: &ImmutableOrigin, site_b: &ImmutableOrigin) -> bool {
    // Block Logic: Implements Step 1.
    // If both origins are opaque and identical, they are considered same-site.
    // Invariant: Opaque origins are only same-site if they are exactly the same instance.
    if !site_a.is_tuple() && !site_b.is_tuple() && site_a == site_b {
        return true;
    }

    // Block Logic: Implements Step 2.
    // If either origin is opaque (but not both identical opaque origins, handled above),
    // they cannot be same-site with a tuple origin.
    // Invariant: Comparison continues only if both origins are tuples.
    let ImmutableOrigin::Tuple(scheme_a, host_a, _) = site_a else {
        return false;
    };
    let ImmutableOrigin::Tuple(scheme_b, host_b, _) = site_b else {
        return false;
    };

    // Block Logic: Implements Step 3.
    // If schemes differ, they are not same-site.
    // Invariant: Schemes must match for same-site determination.
    if scheme_a != scheme_b {
        return false;
    }

    // Block Logic: Implements Step 4.
    // If hosts differ, they are not same-site.
    // Invariant: Hosts must match for same-site determination.
    if host_a != host_b {
        return false;
    }

    // Functional Utility: Implements Step 5.
    // If all checks pass, they are same-site.
    true
}

/// Determines if two given `ImmutableOrigin`s are "schemelessly same site" according to the HTML specification.
///
/// This comparison primarily focuses on host and registrable domain, ignoring the scheme.
/// See <https://html.spec.whatwg.org/multipage/#schemelessly-same-site>
fn is_schemelessy_same_site(site_a: &ImmutableOrigin, site_b: &ImmutableOrigin) -> bool {
    // Block Logic: Implements Step 1.
    // If both origins are opaque and identical, they are considered schemelessly same-site.
    // Invariant: Opaque origins are only same-site if they are exactly the same instance.
    if !site_a.is_tuple() && !site_b.is_tuple() && site_a == site_b {
        true
    } else if site_a.is_tuple() && site_b.is_tuple() {
        // Block Logic: Implements Step 2.1.
        // Extracts and normalizes the host components of the origins.
        let host_a = site_a.host().map(|h| h.to_string()).unwrap_or_default();
        let host_b = site_b.host().map(|h| h.to_string()).unwrap_or_default();

        // Functional Utility: Retrieves the registrable domain for each host.
        let host_a_reg = reg_suffix(&host_a);
        let host_b_reg = reg_suffix(&host_b);

        // Block Logic: Implements Step 2.2-2.3.
        // Checks if hosts are identical OR if registrable domains are identical and non-empty.
        // Invariant: Comparison correctly identifies schemelessly same-site based on host or registrable domain.
        (site_a.host() == site_b.host() && host_a_reg.is_empty()) ||
            (host_a_reg == host_b_reg && !host_a_reg.is_empty())
    } else {
        // Functional Utility: Implements Step 3.
        // If one is a tuple origin and the other is opaque (and not identical opaque origins), they are not same-site.
        false
    }
}

/// Strips sensitive information from a URL before it is used as a referrer,
/// according to the rules defined in the Referrer Policy specification.
///
/// This function removes username, password, and fragment. It can optionally
/// strip path and query if `origin_only` is true or if the URL length exceeds a maximum.
/// See <https://w3c.github.io/webappsec-referrer-policy/#strip-url>
fn strip_url_for_use_as_referrer(mut url: ServoUrl, origin_only: bool) -> Option<ServoUrl> {
    const MAX_REFERRER_URL_LENGTH: usize = 4096;
    // Block Logic: Implements Step 2.
    // Referrers are not sent for local schemes.
    // Invariant: URLs with local schemes are not used as referrers.
    if url.is_local_scheme() {
        return None;
    }
    // Block Logic: Implements Steps 3-6.
    // Strips username, password, and fragment from the URL.
    // Optionally strips path and query if only the origin is required or if the URL is too long.
    // Invariant: Sensitive parts of the URL are removed.
    {
        let url = url.as_mut_url();
        let _ = url.set_username("");
        let _ = url.set_password(None);
        url.set_fragment(None);
        // Note: The result of serializing referrer url should not be
        // greater than 4096 as specified in Step 6 of
        // https://w3c.github.io/webappsec-referrer-policy/#determine-requests-referrer
        if origin_only || url.as_str().len() > MAX_REFERRER_URL_LENGTH {
            url.set_path("");
            url.set_query(None);
        }
    }
    // Functional Utility: Implements Step 7.
    // Returns the modified URL.
    Some(url)
}

/// Implements the "same-origin" referrer policy.
///
/// This policy sends a full URL when performing a same-origin request,
/// but sends no referrer when performing a cross-origin request.
/// See <https://w3c.github.io/webappsec-referrer-policy/#referrer-policy-same-origin>
fn same_origin(referrer_url: ServoUrl, current_url: ServoUrl) -> Option<ServoUrl> {
    // Block Logic: Implements Step 1 of the same-origin policy.
    // If the referrer's origin is the same as the current URL's origin,
    // the full referrer URL (stripped of sensitive parts) is sent.
    // Invariant: Full referrer is sent only for same-origin requests.
    if referrer_url.origin() == current_url.origin() {
        return strip_url_for_use_as_referrer(referrer_url, false);
    }
    // Functional Utility: Implements Step 2.
    // For cross-origin requests, no referrer information is sent.
    None
}

/// Implements the "origin-when-cross-origin" referrer policy.
///
/// This policy sends a full URL when performing a same-origin request,
/// and only the origin when performing a cross-origin request.
/// See <https://w3c.github.io/webappsec-referrer-policy/#referrer-policy-origin-when-cross-origin>
fn origin_when_cross_origin(referrer_url: ServoUrl, current_url: ServoUrl) -> Option<ServoUrl> {
    // Block Logic: Implements Step 1 of the origin-when-cross-origin policy.
    // If the referrer's origin is the same as the current URL's origin,
    // the full referrer URL (stripped of sensitive parts) is sent.
    // Invariant: Full referrer is sent for same-origin requests.
    if referrer_url.origin() == current_url.origin() {
        return strip_url_for_use_as_referrer(referrer_url, false);
    }
    // Functional Utility: Implements Step 2.
    // For cross-origin requests, only the origin of the referrer URL is sent.
    strip_url_for_use_as_referrer(referrer_url, true)
}

/// Determines the referrer URL to be used for a request based on the provided `ReferrerPolicy`.
///
/// This function acts as a dispatcher, applying the rules of the specified referrer policy.
/// See <https://w3c.github.io/webappsec-referrer-policy/#determine-requests-referrer>
pub fn determine_requests_referrer(
    referrer_policy: ReferrerPolicy,
    referrer_source: ServoUrl,
    current_url: ServoUrl,
) -> Option<ServoUrl> {
    // Block Logic: Matches the referrer policy to the appropriate handling function.
    // Invariant: The correct referrer URL (or None) is returned based on the active policy.
    match referrer_policy {
        // Functional Utility: For "EmptyString" or "NoReferrer" policies, no referrer is sent.
        ReferrerPolicy::EmptyString | ReferrerPolicy::NoReferrer => None,
        // Functional Utility: For "Origin" policy, only the origin of the referrer source is sent.
        ReferrerPolicy::Origin => strip_url_for_use_as_referrer(referrer_source, true),
        // Functional Utility: For "UnsafeUrl" policy, the full referrer source is sent.
        ReferrerPolicy::UnsafeUrl => strip_url_for_use_as_referrer(referrer_source, false),
        // Functional Utility: Dispatches to the `strict_origin` handler.
        ReferrerPolicy::StrictOrigin => strict_origin(referrer_source, current_url),
        // Functional Utility: Dispatches to the `strict_origin_when_cross_origin` handler.
        ReferrerPolicy::StrictOriginWhenCrossOrigin => {
            strict_origin_when_cross_origin(referrer_source, current_url)
        },
        // Functional Utility: Dispatches to the `same_origin` handler.
        ReferrerPolicy::SameOrigin => same_origin(referrer_source, current_url),
        // Functional Utility: Dispatches to the `origin_when_cross_origin` handler.
        ReferrerPolicy::OriginWhenCrossOrigin => {
            origin_when_cross_origin(referrer_source, current_url)
        },
        // Functional Utility: Dispatches to the `no_referrer_when_downgrade` handler.
        ReferrerPolicy::NoReferrerWhenDowngrade => {
            no_referrer_when_downgrade(referrer_source, current_url)
        },
    }
}

/// Sets the `Cookie` header in a request based on cookies stored in the `CookieStorage`.
///
/// It retrieves relevant cookies for the given URL, removes expired ones, and
/// appends them to the request headers.
fn set_request_cookies(
    url: &ServoUrl,
    headers: &mut HeaderMap,
    cookie_jar: &RwLock<CookieStorage>,
) {
    // Functional Utility: Acquires a write lock on the cookie jar to manage cookies.
    let mut cookie_jar = cookie_jar.write().unwrap();
    // Functional Utility: Removes any expired cookies relevant to the URL.
    cookie_jar.remove_expired_cookies_for_url(url);
    // Block Logic: If there are cookies for the given URL, they are formatted and added to the request headers.
    // Invariant: The `Cookie` header accurately reflects the non-expired HTTP-sourced cookies for the URL.
    if let Some(cookie_list) = cookie_jar.cookies_for_url(url, CookieSource::HTTP) {
        // Functional Utility: Inserts the formatted cookie string into the request headers.
        headers.insert(
            header::COOKIE,
            HeaderValue::from_bytes(cookie_list.as_bytes()).unwrap(),
        );
    }
}

/// Parses a raw cookie string and stores it in the `CookieStorage` for a given URL.
///
/// This function attempts to parse the `cookie_val` string into a `ServoCookie`
/// and, if successful, adds it to the cookie jar.
fn set_cookie_for_url(cookie_jar: &RwLock<CookieStorage>, request: &ServoUrl, cookie_val: &str) {
    // Functional Utility: Acquires a write lock on the cookie jar to add a new cookie.
    let mut cookie_jar = cookie_jar.write().unwrap();
    let source = CookieSource::HTTP;

    // Block Logic: Attempts to parse the raw cookie string into a `ServoCookie`.
    // If successful, the cookie is pushed into the cookie jar.
    // Invariant: Only validly parsed cookies are added to the storage.
    if let Some(cookie) = ServoCookie::from_cookie_string(cookie_val.into(), request, source) {
        // Functional Utility: Stores the parsed cookie in the cookie jar.
        cookie_jar.push(cookie, request, source);
    }
}

/// Extracts and processes `Set-Cookie` headers from an HTTP response to update the `CookieStorage`.
///
/// This function iterates through all `Set-Cookie` headers, converts their values to UTF-8,
/// and attempts to store each cookie in the cookie jar.
fn set_cookies_from_headers(
    url: &ServoUrl,
    headers: &HeaderMap,
    cookie_jar: &RwLock<CookieStorage>,
) {
    // Block Logic: Iterates over all `Set-Cookie` headers found in the response.
    // Invariant: Each `Set-Cookie` header is processed for potential storage.
    for cookie in headers.get_all(header::SET_COOKIE) {
        // Block Logic: Converts the header value to a UTF-8 string.
        // If successful, it attempts to set the cookie in the cookie jar.
        // Invariant: Only valid UTF-8 cookie strings are considered for storage.
        if let Ok(cookie_str) = std::str::from_utf8(cookie.as_bytes()) {
            // Functional Utility: Delegates to `set_cookie_for_url` to parse and store the cookie.
            set_cookie_for_url(cookie_jar, url, cookie_str);
        }
    }
}

/// Prepares a `ChromeToDevtoolsControlMsg::NetworkEvent` for an HTTP request
/// to be sent to DevTools.
///
/// This function constructs a `DevtoolsHttpRequest` with detailed information
/// about the network request, including timing and request body.
#[allow(clippy::too_many_arguments)]
fn prepare_devtools_request(
    request_id: String,
    url: ServoUrl,
    method: Method,
    headers: HeaderMap,
    body: Option<Vec<u8>>,
    pipeline_id: PipelineId,
    connect_time: Duration,
    send_time: Duration,
    is_xhr: bool,
) -> ChromeToDevtoolsControlMsg {
    // Functional Utility: Captures the current system time to mark when the request started.
    let started_date_time = SystemTime::now();
    // Functional Utility: Constructs the `DevtoolsHttpRequest` structure with all provided details.
    let request = DevtoolsHttpRequest {
        url,
        method,
        headers,
        body,
        pipeline_id,
        started_date_time,
        // Functional Utility: Calculates the timestamp as seconds since UNIX_EPOCH.
        time_stamp: started_date_time
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        connect_time,
        send_time,
        is_xhr,
    };
    // Functional Utility: Wraps the HTTP request into a generic `NetworkEvent`.
    let net_event = NetworkEvent::HttpRequest(request);

    // Functional Utility: Creates the final `ChromeToDevtoolsControlMsg` to be sent.
    ChromeToDevtoolsControlMsg::NetworkEvent(request_id, net_event)
}

/// Sends a `ChromeToDevtoolsControlMsg` to the DevTools channel.
///
/// This function is a utility to dispatch network-related messages to connected DevTools clients.
fn send_request_to_devtools(
    msg: ChromeToDevtoolsControlMsg,
    devtools_chan: &Sender<DevtoolsControlMsg>,
) {
    // Functional Utility: Sends the prepared message to the DevTools control channel.
    // The `unwrap()` indicates an expectation that the channel send will not fail,
    // which typically implies the channel is always open in a stable state.
    devtools_chan
        .send(DevtoolsControlMsg::FromChrome(msg))
        .unwrap();
}

/// Prepares and sends an HTTP response message to the DevTools channel.
///
/// This function constructs a `DevtoolsHttpResponse` from the provided details
/// and dispatches it as a `NetworkEvent` to connected DevTools clients.
fn send_response_to_devtools(
    devtools_chan: &Sender<DevtoolsControlMsg>,
    request_id: String,
    headers: Option<HeaderMap>,
    status: HttpStatus,
    pipeline_id: PipelineId,
) {
    // Functional Utility: Constructs the `DevtoolsHttpResponse` structure.
    let response = DevtoolsHttpResponse {
        headers,
        status,
        body: None, // Response body is typically handled separately or streamed.
        pipeline_id,
    };
    // Functional Utility: Wraps the HTTP response into a generic `NetworkEvent`.
    let net_event_response = NetworkEvent::HttpResponse(response);

    // Functional Utility: Creates the final `ChromeToDevtoolsControlMsg` to be sent.
    let msg = ChromeToDevtoolsControlMsg::NetworkEvent(request_id, net_event_response);
    // Functional Utility: Sends the message to the DevTools channel, ignoring the result.
    let _ = devtools_chan.send(DevtoolsControlMsg::FromChrome(msg));
}

/// Retrieves basic authentication credentials from the `AuthCache` for a given origin.
///
/// This function looks up the provided `origin` in the authentication cache
/// and, if found, constructs an `Authorization` header with basic credentials.
fn auth_from_cache(
    auth_cache: &RwLock<AuthCache>,
    origin: &ImmutableOrigin,
) -> Option<Authorization<Basic>> {
    // Block Logic: Attempts to read from the authentication cache.
    // Invariant: If an entry exists for the origin, its credentials are used to form an `Authorization` header.
    if let Some(auth_entry) = auth_cache
        .read()
        .unwrap() // Functional Utility: Acquires a read lock on the authentication cache.
        .entries
        .get(&origin.ascii_serialization())
    {
        let user_name = &auth_entry.user_name;
        let password = &auth_entry.password;
        // Functional Utility: Constructs an `Authorization` header with basic authentication.
        Some(Authorization::basic(user_name, password))
    } else {
        // Functional Utility: Returns None if no authentication entry is found for the origin.
        None
    }
}

/// Messages from the IPC route to the fetch worker,
/// used to fill the body with bytes coming-in over IPC.
/// Messages from the IPC route to the fetch worker,
/// used to fill the body with bytes coming-in over IPC.
enum BodyChunk {
    /// A chunk of bytes.
    Chunk(Vec<u8>),
    /// Body is done.
    Done,
}

/// The stream side of the body passed to hyper.
/// The stream side of the body passed to hyper.
enum BodyStream {
    /// A receiver that can be used in Body::wrap_stream,
    /// for streaming the request over the network.
    Chunked(TokioReceiver<Result<Frame<Bytes>, hyper::Error>>),
    /// A body whose bytes are buffered
    /// and sent in one chunk over the network.
    Buffered(UnboundedReceiver<BodyChunk>),
}

/// The sink side of the body passed to hyper,
/// used to enqueue chunks.
/// The sink side of the body passed to hyper,
/// used to enqueue chunks.
enum BodySink {
    /// A Tokio sender used to feed chunks to the network stream.
    Chunked(TokioSender<Result<Frame<Bytes>, hyper::Error>>),
    /// A Crossbeam sender used to send chunks to the fetch worker,
    /// where they will be buffered
    /// in order to ensure they are not streamed them over the network.
    Buffered(UnboundedSender<BodyChunk>),
}

impl BodySink {
    /// Transmits a chunk of bytes through the appropriate sender.
    ///
    /// Depending on whether the body is chunked or buffered, it uses a Tokio
    /// sender (asynchronously) or a Crossbeam sender (synchronously).
    fn transmit_bytes(&self, bytes: Vec<u8>) {
        match self {
            // Block Logic: Handles transmitting bytes for a chunked body.
            // Invariant: Bytes are sent asynchronously to the network stream.
            BodySink::Chunked(ref sender) => {
                let sender = sender.clone();
                // Functional Utility: Spawns an asynchronous task to send the byte chunk.
                HANDLE.lock().unwrap().as_mut().unwrap().spawn(async move {
                    let _ = sender.send(Ok(Frame::data(bytes.into()))).await;
                });
            },
            // Block Logic: Handles transmitting bytes for a buffered body.
            // Invariant: Bytes are sent synchronously to the fetch worker for buffering.
            BodySink::Buffered(ref sender) => {
                // Functional Utility: Sends the byte chunk to the unbounded channel.
                let _ = sender.send(BodyChunk::Chunk(bytes));
            },
        }
    }

    /// Closes the body sink, signaling that no more chunks will be sent.
    ///
    /// For buffered bodies, this sends a `Done` message. For chunked bodies,
    /// no explicit close action is needed on the sender.
    fn close(&self) {
        match self {
            // Functional Utility: No explicit close action for chunked bodies, as the sender's drop handles it.
            BodySink::Chunked(_) => { /* no need to close sender */ },
            // Functional Utility: Sends a `Done` message to indicate the end of the buffered body.
            BodySink::Buffered(ref sender) => {
                let _ = sender.send(BodyChunk::Done);
            },
        }
    }
}

/// Asynchronously obtains an HTTP response using the Hyper client.
///
/// This function constructs and sends an HTTP request, handles streaming of the request body
/// (if present), captures network timing metrics, and integrates with DevTools for network logging.
/// It also manages certificate error overrides.
#[allow(clippy::too_many_arguments)]
async fn obtain_response(
    client: &Client<Connector, crate::connector::BoxedBody>,
    url: &ServoUrl,
    method: &Method,
    request_headers: &mut HeaderMap,
    body: Option<StdArc<Mutex<IpcSender<BodyChunkRequest>>>>,
    source_is_null: bool,
    pipeline_id: &Option<PipelineId>,
    request_id: Option<&str>,
    is_xhr: bool,
    context: &FetchContext,
    fetch_terminated: UnboundedSender<bool>,
) -> Result<(HyperResponse<Decoder>, Option<ChromeToDevtoolsControlMsg>), NetworkError> {
    {
        // Functional Utility: Clones the request headers to be used in the Hyper request.
        let mut headers = request_headers.clone();

        // Functional Utility: Initializes a buffer to store the request body for DevTools.
        let devtools_bytes = StdArc::new(Mutex::new(vec![]));

        // Functional Utility: Percent-encodes the URL according to the URL specification.
        // https://url.spec.whatwg.org/#percent-encoded-bytes
        let encoded_url = url
            .clone()
            .into_url()
            .as_ref()
            .replace('|', "%7C")
            .replace('{', "%7B")
            .replace('}', "%7D");

        // Block Logic: Constructs the Hyper request based on whether a request body is present.
        // Invariant: A valid Hyper request is built, ready to be sent over the network.
        let request = if let Some(chunk_requester) = body {
            // Block Logic: Determines the appropriate body sink and stream based on whether the source is null.
            // Invariant: The body streaming mechanism is correctly configured.
            let (sink, stream) = if source_is_null {
                // Step 4.2 of https://fetch.spec.whatwg.org/#concept-http-network-fetch
                // TODO: this should not be set for HTTP/2(currently not supported?).
                // Functional Utility: Sets the `Transfer-Encoding` header to "chunked" for streaming bodies.
                headers.insert(TRANSFER_ENCODING, HeaderValue::from_static("chunked"));

                // Functional Utility: Creates a Tokio channel for chunked body streaming.
                let (sender, receiver) = channel(1);
                (BodySink::Chunked(sender), BodyStream::Chunked(receiver))
            } else {
                // Note: Hyper seems to already buffer bytes when the request appears not stream-able,
                // see https://github.com/hyperium/hyper/issues/2232#issuecomment-644322104
                //
                // However since this doesn't appear documented, and we're using an ancient version,
                // for now we buffer manually to ensure we don't stream requests
                // to servers that might not know how to handle them.
                // Functional Utility: Creates an unbounded channel for buffered body streaming.
                let (sender, receiver) = unbounded_channel();
                (BodySink::Buffered(sender), BodyStream::Buffered(receiver))
            };

            // Functional Utility: Establishes an IPC channel for body chunk requests/responses.
            let (body_chan, body_port) = ipc::channel().unwrap();

            // Block Logic: If a chunk requester is available, it sends a connection message and requests the first chunk.
            // Invariant: The IPC communication for body streaming is initiated.
            if let Ok(requester) = chunk_requester.lock() {
                // Functional Utility: Sends a connection message to the chunk requester.
                let _ = requester.send(BodyChunkRequest::Connect(body_chan));

                // https://fetch.spec.whatwg.org/#concept-request-transmit-body
                // Request the first chunk, corresponding to Step 3 and 4.
                // Functional Utility: Requests the first chunk of the body.
                let _ = requester.send(BodyChunkRequest::Chunk);
            }

            // Functional Utility: Clones necessary Arcs for use within the IPC router's closure.
            let devtools_bytes = devtools_bytes.clone();
            let chunk_requester2 = chunk_requester.clone();

            // Block Logic: Adds a typed route to the IPC router to handle incoming body chunks.
            // This closure processes received chunks, logs them for DevTools, transmits them, and requests the next chunk.
            ROUTER.add_typed_route(
                body_port,
                Box::new(move |message| {
                    info!("Received message");
                    let bytes: Vec<u8> = match message.unwrap() {
                        BodyChunkResponse::Chunk(bytes) => bytes,
                        BodyChunkResponse::Done => {
                            // Step 3, abort these parallel steps.
                            // Functional Utility: Signals that the fetch is terminated without error.
                            let _ = fetch_terminated.send(false);
                            // Functional Utility: Closes the body sink.
                            sink.close();

                            return;
                        },
                        BodyChunkResponse::Error => {
                            // Step 4 and/or 5.
                            // TODO: differentiate between the two steps,
                            // where step 5 requires setting an `aborted` flag on the fetch.
                            // Functional Utility: Signals that the fetch was terminated due to an error.
                            let _ = fetch_terminated.send(true);
                            // Functional Utility: Closes the body sink.
                            sink.close();

                            return;
                        },
                    };

                    // Functional Utility: Extends the DevTools bytes buffer with the received chunk.
                    devtools_bytes.lock().unwrap().extend_from_slice(&bytes);

                    // Step 5.1.2.2, transmit chunk over the network,
                    // currently implemented by sending the bytes to the fetch worker.
                    // Functional Utility: Transmits the received bytes to the body sink.
                    sink.transmit_bytes(bytes);

                    // Step 5.1.2.3
                    // Request the next chunk.
                    // Functional Utility: Requests the next body chunk.
                    let _ = chunk_requester2
                        .lock()
                        .unwrap()
                        .send(BodyChunkRequest::Chunk);
                }),
            );

            // Block Logic: Creates the request body based on the chosen streaming method (chunked or buffered).
            // Invariant: The Hyper request body is correctly prepared for sending.
            let body = match stream {
                BodyStream::Chunked(receiver) => {
                    // Functional Utility: Wraps the Tokio receiver into a `ReceiverStream` for chunked transfer.
                    let stream = ReceiverStream::new(receiver);
                    BoxBody::new(http_body_util::StreamBody::new(stream))
                },
                BodyStream::Buffered(mut receiver) => {
                    // Accumulate bytes received over IPC into a vector.
                    let mut body = vec![];
                    // Block Logic: Loops to receive all body chunks over IPC and accumulate them.
                    loop {
                        match receiver.recv().await {
                            Some(BodyChunk::Chunk(mut bytes)) => {
                                body.append(&mut bytes);
                            },
                            Some(BodyChunk::Done) => break, // Functional Utility: Breaks loop when `Done` signal is received.
                            None => warn!("Failed to read all chunks from request body."),
                        }
                    }
                    // Functional Utility: Creates a full buffered body from the accumulated bytes.
                    Full::new(body.into()).map_err(|_| unreachable!()).boxed()
                },
            };
            // Functional Utility: Builds a Hyper request with the method, encoded URL, and prepared body.
            HyperRequest::builder()
                .method(method)
                .uri(encoded_url)
                .body(body)
        } else {
            // Functional Utility: Builds a Hyper request with no body if `chunk_requester` is None.
            HyperRequest::builder()
                .method(method)
                .uri(encoded_url)
                .body(
                    http_body_util::Empty::new()
                        .map_err(|_| unreachable!())
                        .boxed(),
                )
        };

        // Functional Utility: Records the start of domain lookup in resource timing.
        context
            .timing
            .lock()
            .unwrap()
            .set_attribute(ResourceAttribute::DomainLookupStart);

        // TODO(#21261) connect_start: set if a persistent connection is *not* used and the last non-redirected
        // fetch passes the timing allow check
        // Functional Utility: Records the start of connection in resource timing.
        let connect_start = CrossProcessInstant::now();
        context
            .timing
            .lock()
            .unwrap()
            .set_attribute(ResourceAttribute::ConnectStart(connect_start));

        // TODO: We currently don't know when the handhhake before the connection is done
        // so our best bet would be to set `secure_connection_start` here when we are currently
        // fetching on a HTTPS url.
        // Block Logic: Records secure connection start if the URL scheme is HTTPS.
        if url.scheme() == "https" {
            context
                .timing
                .lock()
                .unwrap()
                .set_attribute(ResourceAttribute::SecureConnectionStart);
        }

        // Functional Utility: Handles potential errors during request construction.
        let mut request = match request {
            Ok(request) => request,
            Err(e) => return Err(NetworkError::from_http_error(&e)),
        };
        // Functional Utility: Overwrites the request headers with the modified headers.
        *request.headers_mut() = headers.clone();

        // Functional Utility: Records the end of connection in resource timing.
        let connect_end = CrossProcessInstant::now();
        context
            .timing
            .lock()
            .unwrap()
            .set_attribute(ResourceAttribute::ConnectEnd(connect_end));

        // Functional Utility: Clones request ID, pipeline ID, URL, and method for use in async blocks.
        let request_id = request_id.map(|v| v.to_owned());
        let pipeline_id = *pipeline_id;
        let closure_url = url.clone();
        let method = method.clone();
        // Functional Utility: Records the start of sending the request.
        let send_start = CrossProcessInstant::now();

        // Functional Utility: Extracts and clones host, override manager, and headers for use in async blocks.
        let host = request.uri().host().unwrap_or("").to_owned();
        let override_manager = context.state.override_manager.clone();
        let headers = headers.clone();
        // Functional Utility: Determines if the URL scheme is secure.
        let is_secure_scheme = url.is_secure_scheme();

        // Block Logic: Sends the HTTP request using the Hyper client and processes the response.
        // Invariant: The HTTP request is sent, and a raw Hyper response is received or an error is returned.
        client
            .request(request)
            .and_then(move |res| {
                // Functional Utility: Records the end of sending the request.
                let send_end = CrossProcessInstant::now();

                // TODO(#21271) response_start: immediately after receiving first byte of response

                // Block Logic: Prepares a DevTools message with network event details if request_id and pipeline_id are present.
                // Invariant: DevTools is notified about the HTTP request with relevant timing and header information.
                let msg = if let Some(request_id) = request_id {
                    if let Some(pipeline_id) = pipeline_id {
                        Some(prepare_devtools_request(
                            request_id,
                            closure_url,
                            method.clone(),
                            headers,
                            Some(devtools_bytes.lock().unwrap().clone()),
                            pipeline_id,
                            (connect_end - connect_start).unsigned_abs(),
                            (send_end - send_start).unsigned_abs(),
                            is_xhr,
                        ))
                    // TODO: ^This is not right, connect_start is taken before contructing the
                    // request and connect_end at the end of it. send_start is takend before the
                    // connection too. I'm not sure it's currently possible to get the time at the
                    // point between the connection and the start of a request.
                    } else {
                        debug!("Not notifying devtools (no pipeline_id)");
                        None
                    }
                } else {
                    debug!("Not notifying devtools (no request_id)");
                    None
                };

                // Functional Utility: Decodes the Hyper response body using `Decoder::detect`.
                future::ready(Ok((
                    Decoder::detect(res.map(|r| r.boxed()), is_secure_scheme),
                    msg,
                )))
            })
            // Functional Utility: Maps Hyper errors to custom `NetworkError`s, including certificate error removal.
            .map_err(move |error| {
                NetworkError::from_hyper_error(
                    &error,
                    override_manager.remove_certificate_failing_verification(host.as_str()),
                )
            })
            .await
    }
}

/// Implements the main HTTP fetch algorithm as specified by the Fetch standard.
///
/// This function handles the entire fetch lifecycle, including service worker interception,
/// CORS preflight checks, network or cache fetching, and redirect processing.
/// See [HTTP fetch](https://fetch.spec.whatwg.org#http-fetch)
#[async_recursion]
#[allow(clippy::too_many_arguments)]
pub async fn http_fetch(
    fetch_params: &mut FetchParams,
    cache: &mut CorsCache,
    cors_flag: bool,
    cors_preflight_flag: bool,
    authentication_fetch_flag: bool,
    target: Target<'async_recursion>,
    done_chan: &mut DoneChannel,
    context: &FetchContext,
) -> Response {
    // Functional Utility: Resets the done channel at the beginning of a new async fetch.
    *done_chan = None;
    // Functional Utility: References `fetchParams’s request` (Step 1).
    let request = &mut fetch_params.request;

    // Functional Utility: Initializes `response` to null (Step 2).
    let mut response: Option<Response> = None;

    // Block Logic: Handles service worker interception (Step 3).
    // Invariant: Service worker interaction is processed before proceeding with network fetch.
    if request.service_workers_mode == ServiceWorkersMode::All {
        // TODO: Substep 1
        // Set response to the result of invoking handle fetch for request.

        // Substep 2
        if let Some(ref res) = response {
            // Subsubstep 1
            // TODO: transmit body for request

            // Subsubstep 2
            // nothing to do, since actual_response is a function on response

            // Subsubstep 3
            // Functional Utility: Checks conditions that indicate a request failure for opaque responses or redirects.
            if (res.response_type == ResponseType::Opaque && request.mode != RequestMode::NoCors) ||
                (res.response_type == ResponseType::OpaqueRedirect &&
                    request.redirect_mode != RedirectMode::Manual) ||
                (res.url_list.len() > 1 && request.redirect_mode != RedirectMode::Follow) ||
                res.is_network_error()
            {
                // Functional Utility: Returns a network error if the request fails under specific conditions.
                return Response::network_error(NetworkError::Internal("Request failed".into()));
            }

            // Subsubstep 4
            // TODO: set response's CSP list on actual_response
        }
    }

    // Block Logic: If no response has been obtained yet, proceeds with preflight and network/cache fetch (Step 4).
    // Invariant: A response is obtained either from the cache/network or through preflight failures.
    if response.is_none() {
        // Block Logic: Handles CORS preflight checks (Substep 1).
        // Invariant: Preflight checks are performed if `cors_preflight_flag` is true,
        //            ensuring that cross-origin requests are compliant before the main fetch.
        if cors_preflight_flag {
            // Functional Utility: Checks if the request method matches the cache.
            let method_cache_match = cache.match_method(request, request.method.clone());

            // Functional Utility: Determines if the method is mismatched (not safelisted or requires preflight).
            let method_mismatch = !method_cache_match &&
                (!is_cors_safelisted_method(&request.method) || request.use_cors_preflight);
            // Functional Utility: Determines if any headers are mismatched (not safelisted or not in cache).
            let header_mismatch = request.headers.iter().any(|(name, value)| {
                !cache.match_header(request, name) &&
                    !is_cors_safelisted_request_header(&name, &value)
            });

            // Sub-substep 1
            // Block Logic: If there is a method or header mismatch, performs a CORS preflight fetch.
            if method_mismatch || header_mismatch {
                // Functional Utility: Executes the CORS preflight fetch.
                let preflight_result = cors_preflight_fetch(request, cache, context).await;
                // Sub-substep 2
                // Block Logic: If the preflight fetch results in a network error, returns that error.
                if let Some(e) = preflight_result.get_network_error() {
                    return Response::network_error(e.clone());
                }
            }
        }

        // Block Logic: Adjusts service worker mode if redirect mode is "Follow" (Substep 2).
        // Invariant: Service workers are bypassed for requests that follow redirects.
        if request.redirect_mode == RedirectMode::Follow {
            request.service_workers_mode = ServiceWorkersMode::None;
        }

        // Generally, we use a persistent connection, so we will also set other PerformanceResourceTiming
        //   attributes to this as well (domain_lookup_start, domain_lookup_end, connect_start, connect_end,
        //   secure_connection_start)
        // Functional Utility: Records the start of the request for resource timing.
        context
            .timing
            .lock()
            .unwrap()
            .set_attribute(ResourceAttribute::RequestStart);

        // Functional Utility: Executes an HTTP network or cache fetch.
        let mut fetch_result = http_network_or_cache_fetch(
            fetch_params,
            authentication_fetch_flag,
            cors_flag,
            done_chan,
            context,
        )
        .await;

        // Substep 4
        // Block Logic: Performs a CORS check if `cors_flag` is true.
        // Invariant: If CORS check fails, a network error is returned.
        if cors_flag && cors_check(&fetch_params.request, &fetch_result).is_err() {
            return Response::network_error(NetworkError::Internal("CORS check failed".into()));
        }

        // Functional Utility: Marks the response as not being an internal response.
        fetch_result.return_internal = false;
        // Functional Utility: Sets the obtained fetch result as the response.
        response = Some(fetch_result);
    }

    // Functional Utility: References `fetchParams’s request` again.
    let request = &mut fetch_params.request;

    // response is guaranteed to be something by now
    // Functional Utility: Unwraps the response, which is guaranteed to be present at this point.
    let mut response = response.unwrap();

    // TODO: Step 5: cross-origin resource policy check

    // Block Logic: Handles HTTP redirects (Step 6).
    // Invariant: Redirect responses are processed according to the request's redirect mode.
    if response
        .actual_response()
        .status
        .try_code()
        .is_some_and(is_redirect_status)
    {
        // Substep 1.
        // TODO: send RST_STREAM frame
        if response.actual_response().status != StatusCode::SEE_OTHER {
            // TODO: send RST_STREAM frame
        }

        // Substep 2-3.
        // Functional Utility: Extracts and parses the `Location` header to determine the redirect URL.
        let mut location = response
            .actual_response()
            .headers
            .get(header::LOCATION)
            .and_then(|v| {
                HeaderValue::to_str(v)
                    .map(|l| {
                        ServoUrl::parse_with_base(response.actual_response().url(), l)
                            .map_err(|err| err.to_string())
                    })
                    .ok()
            });

        // Substep 4.
        // Block Logic: Appends the fragment from the current URL to the redirect location if missing.
        // Invariant: Fragment identifiers are preserved across redirects.
        if let Some(Ok(ref mut location)) = location {
            if location.fragment().is_none() {
                let current_url = request.current_url();
                location.set_fragment(current_url.fragment());
            }
        }
        // Functional Utility: Updates the actual response's location URL.
        response.actual_response_mut().location_url = location;

        // Substep 5.
        // Block Logic: Processes the redirect based on the request's `RedirectMode`.
        // Invariant: The response is adjusted according to the redirect handling policy.
        response = match request.redirect_mode {
            // Functional Utility: Returns a network error if redirect mode is "Error".
            RedirectMode::Error => {
                Response::network_error(NetworkError::Internal("Redirect mode error".into()))
            },
            // Functional Utility: Filters the response to "OpaqueRedirect" for manual redirects.
            RedirectMode::Manual => response.to_filtered(ResponseType::OpaqueRedirect),
            // Functional Utility: Recursively calls `http_redirect_fetch` for "Follow" redirects.
            RedirectMode::Follow => {
                // set back to default
                response.return_internal = true;
                http_redirect_fetch(
                    fetch_params,
                    cache,
                    response,
                    cors_flag,
                    target,
                    done_chan,
                    context,
                )
                .await
            },
        };
    }

    // set back to default
    // Functional Utility: Ensures `return_internal` is true for the final response.
    response.return_internal = true;
    // Functional Utility: Records the number of redirects for resource timing.
    context
        .timing
        .lock()
        .unwrap()
        .set_attribute(ResourceAttribute::RedirectCount(
            fetch_params.request.redirect_count as u16,
        ));

    // Functional Utility: Assigns the resource timing information to the response.
    response.resource_timing = Arc::clone(&context.timing);

    // Step 6
    // Functional Utility: Returns the final processed response.
    response
}

/// Convenience struct that implements `Drop` to automatically record `redirectEnd` timing.
///
/// This struct is used to ensure that `redirectEnd` is set in `ResourceFetchTiming`
/// when the struct goes out of scope, typically at the end of a redirect chain.
struct RedirectEndTimer(Option<Arc<Mutex<ResourceFetchTiming>>>);

impl RedirectEndTimer {
    /// Prevents the `redirectEnd` timing from being recorded when this timer is dropped.
    ///
    /// This is useful when the timing should be neutered, e.g., if a new timer is
    /// created later in the process.
    fn neuter(&mut self) {
        self.0 = None;
    }
}

impl Drop for RedirectEndTimer {
    /// The `drop` implementation for `RedirectEndTimer`.
    ///
    /// When the `RedirectEndTimer` goes out of scope, this method is called
    /// to set the `RedirectEnd` attribute in the associated `ResourceFetchTiming`.
    fn drop(&mut self) {
        let RedirectEndTimer(resource_fetch_timing_opt) = self;

        // Functional Utility: If `resource_fetch_timing_opt` is Some, it locks the mutex
        // and sets the `RedirectEnd` attribute.
        resource_fetch_timing_opt.as_ref().map_or((), |t| {
            t.lock()
                .unwrap()
                .set_attribute(ResourceAttribute::RedirectEnd(RedirectEndValue::Zero));
        })
    }
}

/// Implements the HTTP redirect fetch algorithm as specified by the Fetch standard.
///
/// This function is responsible for processing a redirect response, updating the request
/// accordingly, and recursively fetching the new URL.
/// See [HTTP redirect fetch](https://fetch.spec.whatwg.org#http-redirect-fetch)
#[async_recursion]
pub async fn http_redirect_fetch(
    fetch_params: &mut FetchParams,
    cache: &mut CorsCache,
    response: Response,
    cors_flag: bool,
    target: Target<'async_recursion>,
    done_chan: &mut DoneChannel,
    context: &FetchContext,
) -> Response {
    // Functional Utility: Initializes a timer to record the `redirectEnd` timing attribute.
    let mut redirect_end_timer = RedirectEndTimer(Some(context.timing.clone()));

    // Functional Utility: References `fetchParams’s request` (Step 1).
    let request = &mut fetch_params.request;

    // Functional Utility: Asserts that the response is internal.
    assert!(response.return_internal);

    // Functional Utility: Clones the `location_url` from the actual response.
    let location_url = response.actual_response().location_url.clone();
    // Block Logic: Processes the `location_url` from the redirect response.
    // Invariant: The redirect URL is validated and parsed correctly, or an error is returned.
    let location_url = match location_url {
        // Functional Utility: Implements Step 2. If no location URL, returns the original response.
        None => return response,
        // Functional Utility: Implements Step 3. If location URL parsing fails, returns a network error.
        Some(Err(err)) => {
            return Response::network_error(NetworkError::Internal(
                "Location URL parse failure: ".to_owned() + &err,
            ));
        },
        // Functional Utility: Implements Step 4. If the scheme is not HTTP or HTTPS, returns a network error.
        Some(Ok(ref url)) if !matches!(url.scheme(), "http" | "https") => {
            return Response::network_error(NetworkError::Internal(
                "Location URL not an HTTP(S) scheme".into(),
            ));
        },
        // Functional Utility: Extracts the valid URL.
        Some(Ok(url)) => url,
    };

    // Step 1 of https://w3c.github.io/resource-timing/#dom-performanceresourcetiming-fetchstart
    // TODO: check origin and timing allow check
    // Functional Utility: Records the `RedirectStart` timing attribute.
    context
        .timing
        .lock()
        .unwrap()
        .set_attribute(ResourceAttribute::RedirectStart(
            RedirectStartValue::FetchStart,
        ));

    // Functional Utility: Records the `FetchStart` timing attribute.
    context
        .timing
        .lock()
        .unwrap()
        .set_attribute(ResourceAttribute::FetchStart);

    // start_time should equal redirect_start if nonzero; else fetch_start
    // Functional Utility: Sets the `StartTime` attribute to `FetchStart`.
    context
        .timing
        .lock()
        .unwrap()
        .set_attribute(ResourceAttribute::StartTime(ResourceTimeValue::FetchStart));

    // Functional Utility: Updates the `StartTime` attribute if `RedirectStart` is non-zero.
    context
        .timing
        .lock()
        .unwrap()
        .set_attribute(ResourceAttribute::StartTime(
            ResourceTimeValue::RedirectStart,
        )); // updates start_time only if redirect_start is nonzero (implying TAO)

    // Block Logic: Implements Step 7. Checks for too many redirects.
    // Invariant: The redirect chain does not exceed the maximum allowed count.
    if request.redirect_count >= 20 {
        return Response::network_error(NetworkError::Internal("Too many redirects".into()));
    }

    // Functional Utility: Implements Step 8. Increments the redirect count.
    request.redirect_count += 1;

    // Step 7
    // Block Logic: Determines if the request is same-origin with the redirect target.
    let same_origin = match request.origin {
        Origin::Origin(ref origin) => *origin == location_url.origin(),
        // Functional Utility: Panics if the origin is `Client` at this stage, indicating an unexpected state.
        Origin::Client => panic!(
            "Request origin should not be client for {}",
            request.current_url()
        ),
    };

    // Functional Utility: Checks if the `location_url` has credentials.
    let has_credentials = has_credentials(&location_url);

    // Block Logic: Implements a CORS credential check.
    // Invariant: Cross-origin requests with credentials are blocked if not same-origin.
    if request.mode == RequestMode::CorsMode && !same_origin && has_credentials {
        return Response::network_error(NetworkError::Internal(
            "Cross-origin credentials check failed".into(),
        ));
    }

    // Block Logic: Implements Step 9. Sets the request origin to opaque if CORS flag is true and origins differ.
    if cors_flag && location_url.origin() != request.current_url().origin() {
        request.origin = Origin::Origin(ImmutableOrigin::new_opaque());
    }

    // Block Logic: Implements Step 10. Blocks cross-origin requests with credentials if CORS flag is true.
    if cors_flag && has_credentials {
        return Response::network_error(NetworkError::Internal("Credentials check failed".into()));
    }

    // Block Logic: Implements Step 11. Checks for specific conditions related to status code and request body.
    if response.actual_response().status != StatusCode::SEE_OTHER &&
        request.body.as_ref().is_some_and(|b| b.source_is_null())
    {
        return Response::network_error(NetworkError::Internal("Request body is not done".into()));
    }

    // Block Logic: Implements Step 12. Adjusts request method and headers for specific redirect status codes.
    if response
        .actual_response()
        .status
        .try_code()
        .is_some_and(|code| {
            ((code == StatusCode::MOVED_PERMANENTLY || code == StatusCode::FOUND) &&
                request.method == Method::POST) ||
                (code == StatusCode::SEE_OTHER &&
                    request.method != Method::HEAD &&
                    request.method != Method::GET)
        })
    {
        // Functional Utility: Implements Step 12.1. Changes the request method to GET.
        request.method = Method::GET;
        // Functional Utility: Clears the request body.
        request.body = None;
        // Functional Utility: Implements Step 12.2. Removes specific content headers.
        for name in &[
            CONTENT_ENCODING,
            CONTENT_LANGUAGE,
            CONTENT_LOCATION,
            CONTENT_TYPE,
        ] {
            request.headers.remove(name);
        }
    }

    // Block Logic: Implements Step 13. Removes CORS non-wildcard request headers if origins differ.
    if location_url.origin() != request.current_url().origin() {
        // This list currently only contains the AUTHORIZATION header
        // https://fetch.spec.whatwg.org/#cors-non-wildcard-request-header-name
        // Functional Utility: Removes the `Authorization` header.
        request.headers.remove(AUTHORIZATION);
    }

    // Block Logic: Implements Step 14. Safely extracts the request body's source if non-null.
    if let Some(body) = request.body.as_mut() {
        body.extract_source();
    }

    // Steps 15-17 relate to timing, which is not implemented 1:1 with the spec.

    // Functional Utility: Implements Step 18. Appends the new location URL to the request's URL list.
    request.url_list.push(location_url);

    // Functional Utility: Implements Step 19. Sets the request's referrer policy on redirect.
    set_requests_referrer_policy_on_redirect(request, response.actual_response());

    // Functional Utility: Implements Step 20 & 21. Determines the recursive flag based on redirect mode.
    let recursive_flag = request.redirect_mode != RedirectMode::Manual;

    // Functional Utility: Implements Step 22. Recursively calls `main_fetch` with updated parameters.
    let fetch_response = main_fetch(
        fetch_params,
        cache,
        recursive_flag,
        target,
        done_chan,
        context,
    )
    .await;

    // TODO: timing allow check
    // Functional Utility: Records the `RedirectEnd` timing attribute.
    context
        .timing
        .lock()
        .unwrap()
        .set_attribute(ResourceAttribute::RedirectEnd(
            RedirectEndValue::ResponseEnd,
        ));
    // Functional Utility: Prevents the `redirectEnd` timer from setting the timing again on drop.
    redirect_end_timer.neuter();

    // Functional Utility: Returns the fetch response.
    fetch_response
}

/// [HTTP network or cache fetch](https://fetch.spec.whatwg.org#http-network-or-cache-fetch)
#[async_recursion]
async fn http_network_or_cache_fetch(
    fetch_params: &mut FetchParams,
    authentication_fetch_flag: bool,
    cors_flag: bool,
    done_chan: &mut DoneChannel,
    context: &FetchContext,
) -> Response {
    // Step 1. Let request be fetchParams’s request.
    let request = &mut fetch_params.request;

    // Step 2. Let httpFetchParams be null.
    let http_fetch_params: &mut FetchParams;
    let mut http_fetch_params_obj: FetchParams;

    // Step 3. Let httpRequest be null. (See step 8 for initialization)

    // Step 4. Let response be null.
    let mut response: Option<Response> = None;

    // Step 7. Let the revalidatingFlag be unset.
    let mut revalidating_flag = false;

    // TODO(#33616): Step 8. Run these steps, but abort when fetchParams is canceled:
    // Step 8.1: If request’s window is "no-window" and request’s redirect mode is "error", then set
    // httpFetchParams to fetchParams and httpRequest to request.
    let request_has_no_window = request.window == RequestWindow::NoWindow;

    let http_request = if request_has_no_window && request.redirect_mode == RedirectMode::Error {
        http_fetch_params = fetch_params;
        &mut http_fetch_params.request
    }
    // Step 8.2 Otherwise:
    else {
        // Step 8.2.1 - 8.2.3: Set httpRequest to a clone of request
        // and Set httpFetchParams to a copy of fetchParams.
        http_fetch_params_obj = fetch_params.clone();

        http_fetch_params = &mut http_fetch_params_obj;
        &mut http_fetch_params.request
    };

    // Step 8.3: Let includeCredentials be true if one of:
    let include_credentials = match http_request.credentials_mode {
        // request’s credentials mode is "include"
        CredentialsMode::Include => true,
        // request’s credentials mode is "same-origin" and request’s response tainting is "basic"
        CredentialsMode::CredentialsSameOrigin
            if http_request.response_tainting == ResponseTainting::Basic =>
        {
            true
        },
        _ => false,
    };

    // Step 8.4: If Cross-Origin-Embedder-Policy allows credentials with request returns false, then
    // set includeCredentials to false.
    // TODO(#33616): Requires request's client object

    // Step 8.5 Let contentLength be httpRequest’s body’s length, if httpRequest’s body is non-null;
    // otherwise null.
    let content_length = http_request
        .body
        .as_ref()
        .and_then(|body| body.len().map(|size| size as u64));

    // Step 8.6 Let contentLengthHeaderValue be null.
    let mut content_length_header_value = None;

    // Step 8.7 If httpRequest’s body is null and httpRequest’s method is `POST` or `PUT`,
    // then set contentLengthHeaderValue to `0`.
    if http_request.body.is_none() && matches!(http_request.method, Method::POST | Method::PUT) {
        content_length_header_value = Some(0);
    }

    // Step 8.8 If contentLength is non-null, then set contentLengthHeaderValue to contentLength,
    // serialized and isomorphic encoded.
    // NOTE: The header will later be serialized using HeaderMap::typed_insert
    if let Some(content_length) = content_length {
        content_length_header_value = Some(content_length);
    };

    // Step 8.9 If contentLengthHeaderValue is non-null, then append (`Content-Length`, contentLengthHeaderValue)
    // to httpRequest’s header list.
    if let Some(content_length_header_value) = content_length_header_value {
        http_request
            .headers
            .typed_insert(ContentLength(content_length_header_value));
    }

    // Step 8.10 If contentLength is non-null and httpRequest’s keepalive is true, then:
    if content_length.is_some() && http_request.keep_alive {
        // TODO(#33616) Keepalive requires request's client object's fetch group
    }

    // Step 8.11: If httpRequest’s referrer is a URL, then:
    match http_request.referrer {
        Referrer::ReferrerUrl(ref http_request_referrer) |
        Referrer::Client(ref http_request_referrer) => {
            // Step 8.11.1: Let referrerValue be httpRequest’s referrer, serialized and isomorphic
            // encoded.
            if let Ok(referer) = http_request_referrer.to_string().parse::<Referer>() {
                // Step 8.11.2: Append (`Referer`, referrerValue) to httpRequest’s header list.
                http_request.headers.typed_insert(referer);
            } else {
                // This error should only happen in cases where hyper and rust-url disagree
                // about how to parse a referer.
                // https://github.com/servo/servo/issues/24175
                error!("Failed to parse {} as referrer", http_request_referrer);
            }
        },
        _ => {},
    };

    // Step 8.12 Append a request `Origin` header for httpRequest.
    append_a_request_origin_header(http_request);

    // Step 8.13 Append the Fetch metadata headers for httpRequest.
    append_the_fetch_metadata_headers(http_request);

    // Step 8.14: If httpRequest’s initiator is "prefetch", then set a structured field value given
    // (`Sec-Purpose`, the token "prefetch") in httpRequest’s header list.
    if http_request.initiator == Initiator::Prefetch {
        if let Ok(value) = HeaderValue::from_str("prefetch") {
            http_request.headers.insert("Sec-Purpose", value);
        }
    }

    // Step 8.15: If httpRequest’s header list does not contain `User-Agent`, then user agents
    // should append (`User-Agent`, default `User-Agent` value) to httpRequest’s header list.
    if !http_request.headers.contains_key(header::USER_AGENT) {
        let user_agent = context.user_agent.clone().into_owned();
        http_request
            .headers
            .typed_insert::<UserAgent>(user_agent.parse().unwrap());
    }

    // Steps 8.16 to 8.18
    match http_request.cache_mode {
        // Step 8.16: If httpRequest’s cache mode is "default" and httpRequest’s header list
        // contains `If-Modified-Since`, `If-None-Match`, `If-Unmodified-Since`, `If-Match`, or
        // `If-Range`, then set httpRequest’s cache mode to "no-store".
        CacheMode::Default if is_no_store_cache(&http_request.headers) => {
            http_request.cache_mode = CacheMode::NoStore;
        },

        // Note that the following steps (8.17 and 8.18) are being considered for removal:
        // https://github.com/whatwg/fetch/issues/722#issuecomment-1420264615

        // Step 8.17: If httpRequest’s cache mode is "no-cache", httpRequest’s prevent no-cache
        // cache-control header modification flag is unset, and httpRequest’s header list does not
        // contain `Cache-Control`, then append (`Cache-Control`, `max-age=0`) to httpRequest’s
        // header list.
        // TODO: Implement request's prevent no-cache cache-control header modification flag
        // https://fetch.spec.whatwg.org/#no-cache-prevent-cache-control
        CacheMode::NoCache if !http_request.headers.contains_key(header::CACHE_CONTROL) => {
            http_request
                .headers
                .typed_insert(CacheControl::new().with_max_age(Duration::from_secs(0)));
        },

        // Step 8.18: If httpRequest’s cache mode is "no-store" or "reload", then:
        CacheMode::Reload | CacheMode::NoStore => {
            // Step 8.18.1: If httpRequest’s header list does not contain `Pragma`, then append
            // (`Pragma`, `no-cache`) to httpRequest’s header list.
            if !http_request.headers.contains_key(header::PRAGMA) {
                http_request.headers.typed_insert(Pragma::no_cache());
            }

            // Step 8.18.2: If httpRequest’s header list does not contain `Cache-Control`, then
            // append (`Cache-Control`, `no-cache`) to httpRequest’s header list.
            if !http_request.headers.contains_key(header::CACHE_CONTROL) {
                http_request
                    .headers
                    .typed_insert(CacheControl::new().with_no_cache());
            }
        },

        _ => {},
    }

    // Step 8.19: If httpRequest’s header list contains `Range`, then append (`Accept-Encoding`,
    // `identity`) to httpRequest’s header list.
    if http_request.headers.contains_key(header::RANGE) {
        if let Ok(value) = HeaderValue::from_str("identity") {
            http_request.headers.insert("Accept-Encoding", value);
        }
    }

    // Step 8.20: Modify httpRequest’s header list per HTTP. Do not append a given header if
    // httpRequest’s header list contains that header’s name.
    // `Accept`, `Accept-Charset`, and `Accept-Language` must not be included at this point.
    http_request.headers.remove(header::HOST);
    // unlike http_loader, we should not set the accept header here
    set_default_accept_encoding(&mut http_request.headers);

    let current_url = http_request.current_url();

    // Step 8.21: If includeCredentials is true, then:
    // TODO some of this step can't be implemented yet
    if include_credentials {
        // Substep 1
        // TODO http://mxr.mozilla.org/servo/source/components/net/http_loader.rs#504
        // XXXManishearth http_loader has block_cookies: support content blocking here too
        set_request_cookies(
            &current_url,
            &mut http_request.headers,
            &context.state.cookie_jar,
        );
        // Substep 2
        if !http_request.headers.contains_key(header::AUTHORIZATION) {
            // Substep 3
            let mut authorization_value = None;

            // Substep 4
            if let Some(basic) = auth_from_cache(&context.state.auth_cache, &current_url.origin()) {
                if !http_request.use_url_credentials || !has_credentials(&current_url) {
                    authorization_value = Some(basic);
                }
            }

            // Substep 5
            if authentication_fetch_flag &&
                authorization_value.is_none() &&
                has_credentials(&current_url)
            {
                authorization_value = Some(Authorization::basic(
                    current_url.username(),
                    current_url.password().unwrap_or(""),
                ));
            }

            // Substep 6
            if let Some(basic) = authorization_value {
                http_request.headers.typed_insert(basic);
            }
        }
    }

    // TODO(#33616) Step 8.22 If there’s a proxy-authentication entry, use it as appropriate.

    // If the cache is not ready to construct a response, wait.
    //
    // The cache is not ready if a previous fetch checked the cache, found nothing,
    // and moved on to a network fetch, and hasn't updated the cache yet with a pending resource.
    //
    // Note that this is a different workflow from the one involving `wait_for_cached_response`.
    // That one happens when a fetch gets a cache hit, and the resource is pending completion from the network.
    {
        let (lock, cvar) = {
            let entry_key = CacheKey::new(http_request);
            let mut state_map = context.state.http_cache_state.lock().unwrap();
            &*state_map
                .entry(entry_key)
                .or_insert_with(|| {
                    Arc::new((
                        Mutex::new(HttpCacheEntryState::ReadyToConstruct),
                        Condvar::new(),
                    ))
                })
                .clone()
        };

        // Start of critical section on http-cache state.
        let mut state = lock.lock().unwrap();
        while let HttpCacheEntryState::PendingStore(_) = *state {
            let (current_state, time_out) = cvar
                .wait_timeout(state, Duration::from_millis(500))
                .unwrap();
            state = current_state;
            if time_out.timed_out() {
                // After a timeout, ignore the pending store.
                break;
            }
        }

        // TODO(#33616): Step 8.23 Set httpCache to the result of determining the
        // HTTP cache partition, given httpRequest.
        if let Ok(http_cache) = context.state.http_cache.read() {
            // Step 8.25.1 Set storedResponse to the result of selecting a response from the httpCache,
            //              possibly needing validation, as per the "Constructing Responses from Caches"
            //              chapter of HTTP Caching, if any.
            let stored_response = http_cache.construct_response(http_request, done_chan);

            // Step 8.25.2 If storedResponse is non-null, then:
            if let Some(response_from_cache) = stored_response {
                let response_headers = response_from_cache.response.headers.clone();
                // Substep 1, 2, 3, 4
                let (cached_response, needs_revalidation) =
                    match (http_request.cache_mode, &http_request.mode) {
                        (CacheMode::ForceCache, _) => (Some(response_from_cache.response), false),
                        (CacheMode::OnlyIfCached, &RequestMode::SameOrigin) => {
                            (Some(response_from_cache.response), false)
                        },
                        (CacheMode::OnlyIfCached, _) |
                        (CacheMode::NoStore, _) |
                        (CacheMode::Reload, _) => (None, false),
                        (_, _) => (
                            Some(response_from_cache.response),
                            response_from_cache.needs_validation,
                        ),
                    };

                if needs_revalidation {
                    revalidating_flag = true;
                    // Substep 5
                    if let Some(http_date) = response_headers.typed_get::<LastModified>() {
                        let http_date: SystemTime = http_date.into();
                        http_request
                            .headers
                            .typed_insert(IfModifiedSince::from(http_date));
                    }
                    if let Some(entity_tag) = response_headers.get(header::ETAG) {
                        http_request
                            .headers
                            .insert(header::IF_NONE_MATCH, entity_tag.clone());
                    }
                } else {
                    // Substep 6
                    response = cached_response;
                }
                if response.is_none() {
                    // Ensure the done chan is not set if we're not using the cached response,
                    // as the cache might have set it to Some if it constructed a pending response.
                    *done_chan = None;

                    // Update the cache state, incrementing the pending store count,
                    // or starting the count.
                    if let HttpCacheEntryState::PendingStore(i) = *state {
                        let new = i + 1;
                        *state = HttpCacheEntryState::PendingStore(new);
                    } else {
                        *state = HttpCacheEntryState::PendingStore(1);
                    }
                }
            }
        }
        // Notify the next thread waiting in line, if there is any.
        if *state == HttpCacheEntryState::ReadyToConstruct {
            cvar.notify_one();
        }
        // End of critical section on http-cache state.
    }

    // Decrement the number of pending stores,
    // and set the state to ready to construct,
    // if no stores are pending.
    fn update_http_cache_state(context: &FetchContext, http_request: &Request) {
        let (lock, cvar) = {
            let entry_key = CacheKey::new(http_request);
            let mut state_map = context.state.http_cache_state.lock().unwrap();
            &*state_map
                .get_mut(&entry_key)
                .expect("Entry in http-cache state to have been previously inserted")
                .clone()
        };
        let mut state = lock.lock().unwrap();
        if let HttpCacheEntryState::PendingStore(i) = *state {
            let new = i - 1;
            if new == 0 {
                *state = HttpCacheEntryState::ReadyToConstruct;
                // Notify the next thread waiting in line, if there is any.
                cvar.notify_one();
            } else {
                *state = HttpCacheEntryState::PendingStore(new);
            }
        }
    }

    async fn wait_for_cached_response(
        done_chan: &mut DoneChannel,
        response: &mut Option<Response>,
    ) {
        if let Some(ref mut ch) = *done_chan {
            // The cache constructed a response with a body of ResponseBody::Receiving.
            // We wait for the response in the cache to "finish",
            // with a body of either Done or Cancelled.
            assert!(response.is_some());

            loop {
                match ch.1.recv().await {
                    Some(Data::Payload(_)) => {},
                    Some(Data::Done) => break, // Return the full response as if it was initially cached as such.
                    Some(Data::Cancelled) => {
                        // The response was cancelled while the fetch was ongoing.
                        // Set response to None, which will trigger a network fetch below.
                        *response = None;
                        break;
                    },
                    _ => panic!("HTTP cache should always send Done or Cancelled"),
                }
            }
        }
        // Set done_chan back to None, it's cache-related usefulness ends here.
        *done_chan = None;
    }

    wait_for_cached_response(done_chan, &mut response).await;

    // TODO(#33616): Step 9. If aborted, then return the appropriate network error for fetchParams.

    // Step 10. If response is null, then:
    if response.is_none() {
        // Step 10.1 If httpRequest’s cache mode is "only-if-cached", then return a network error.
        if http_request.cache_mode == CacheMode::OnlyIfCached {
            // The cache will not be updated,
            // set its state to ready to construct.
            update_http_cache_state(context, http_request);
            return Response::network_error(NetworkError::Internal(
                "Couldn't find response in cache".into(),
            ));
        }

        // Step 10.2 Let forwardResponse be the result of running HTTP-network fetch given httpFetchParams,
        // includeCredentials, and isNewConnectionFetch.
        let forward_response =
            http_network_fetch(http_fetch_params, include_credentials, done_chan, context).await;

        let http_request = &mut http_fetch_params.request;
        // Step 10.3 If httpRequest’s method is unsafe and forwardResponse’s status is in the range 200 to 399,
        // inclusive, invalidate appropriate stored responses in httpCache, as per the
        // "Invalidating Stored Responses" chapter of HTTP Caching, and set storedResponse to null.
        if forward_response.status.in_range(200..=399) && !http_request.method.is_safe() {
            if let Ok(mut http_cache) = context.state.http_cache.write() {
                http_cache.invalidate(http_request, &forward_response);
            }
        }

        // Step 10.4 If the revalidatingFlag is set and forwardResponse’s status is 304, then:
        if revalidating_flag && forward_response.status == StatusCode::NOT_MODIFIED {
            if let Ok(mut http_cache) = context.state.http_cache.write() {
                // Ensure done_chan is None,
                // since the network response will be replaced by the revalidated stored one.
                *done_chan = None;
                response = http_cache.refresh(http_request, forward_response.clone(), done_chan);
            }
            wait_for_cached_response(done_chan, &mut response).await;
        }

        // Step 10.5 If response is null, then:
        if response.is_none() {
            // Step 10.5.1 Set response to forwardResponse.
            let forward_response = response.insert(forward_response);

            // Per https://httpwg.org/specs/rfc9111.html#response.cacheability we must not cache responses
            // if the No-Store directive is present
            if http_request.cache_mode != CacheMode::NoStore {
                // Step 10.5.2 Store httpRequest and forwardResponse in httpCache, as per the
                //             "Storing Responses in Caches" chapter of HTTP Caching.
                if let Ok(mut http_cache) = context.state.http_cache.write() {
                    http_cache.store(http_request, forward_response);
                }
            }
        }
    }

    let http_request = &mut http_fetch_params.request;
    // The cache has been updated, set its state to ready to construct.
    update_http_cache_state(context, http_request);

    let mut response = response.unwrap();

    // FIXME: The spec doesn't tell us to do this *here*, but if we don't do it then
    // tests fail. Where should we do it instead? See also #33615
    if http_request.response_tainting != ResponseTainting::CorsTainting &&
        cross_origin_resource_policy_check(http_request, &response) ==
            CrossOriginResourcePolicy::Blocked
    {
        return Response::network_error(NetworkError::Internal(
            "Cross-origin resource policy check failed".into(),
        ));
    }

    // TODO(#33616): Step 11. Set response’s URL list to a clone of httpRequest’s URL list.
    // TODO(#33616): Step 12. If httpRequest’s header list contains `Range`,
    // then set response’s range-requested flag.
    // TODO(#33616): Step 13 Set response’s request-includes-credentials to includeCredentials.

    // Step 14. If response’s status is 401, httpRequest’s response tainting is not "cors",
    // includeCredentials is true, and request’s window is an environment settings object, then:
    // TODO(#33616): Figure out what to do with request window objects
    if let (Some(StatusCode::UNAUTHORIZED), false, true) =
        (response.status.try_code(), cors_flag, include_credentials)
    {
        // TODO: Step 14.1 Spec says requires testing on multiple WWW-Authenticate headers

        let request = &mut fetch_params.request;

        // Step 14.2 If request’s body is non-null, then:
        if request.body.is_some() {
            // TODO Implement body source
        }

        // Step 14.3 If request’s use-URL-credentials flag is unset or isAuthenticationFetch is true, then:
        if !http_request.use_url_credentials || authentication_fetch_flag {
            let Some(webview_id) = http_request.target_webview_id else {
                return response;
            };
            let Some(credentials) =
                prompt_user_for_credentials(&context.state.embedder_proxy, webview_id)
            else {
                return response;
            };
            let Some(username) = credentials.username else {
                return response;
            };
            let Some(password) = credentials.password else {
                return response;
            };

            if let Err(err) = request.current_url_mut().set_username(&username) {
                error!("error setting username for url: {:?}", err);
                return response;
            };

            if let Err(err) = request.current_url_mut().set_password(Some(&password)) {
                error!("error setting password for url: {:?}", err);
                return response;
            };
        }

        // Make sure this is set to None,
        // since we're about to start a new `http_network_or_cache_fetch`.
        *done_chan = None;

        // Step 14.4 Set response to the result of running HTTP-network-or-cache fetch given fetchParams and true.
        response = http_network_or_cache_fetch(
            fetch_params,
            true, /* authentication flag */
            cors_flag,
            done_chan,
            context,
        )
        .await;
    }

    // Step 15. If response’s status is 407, then:
    if response.status == StatusCode::PROXY_AUTHENTICATION_REQUIRED {
        let request = &mut fetch_params.request;
        // Step 15.1 If request’s window is "no-window", then return a network error.

        if request_has_no_window {
            return Response::network_error(NetworkError::Internal(
                "Can't find Window object".into(),
            ));
        }

        // (Step 15.2 does not exist, requires testing on Proxy-Authenticate headers)

        // TODO(#33616): Step 15.3 If fetchParams is canceled, then return
        // the appropriate network error for fetchParams.

        // Step 15.4 Prompt the end user as appropriate in request’s window
        // window and store the result as a proxy-authentication entry.
        let Some(webview_id) = http_request.target_webview_id else {
            return response;
        };
        let Some(credentials) =
            prompt_user_for_credentials(&context.state.embedder_proxy, webview_id)
        else {
            return response;
        };
        let Some(user_name) = credentials.username else {
            return response;
        };
        let Some(password) = credentials.password else {
            return response;
        };

        // store the credentials as a proxy-authentication entry.
        let entry = AuthCacheEntry {
            user_name,
            password,
        };
        {
            let mut auth_cache = context.state.auth_cache.write().unwrap();
            let key = request.current_url().origin().ascii_serialization();
            auth_cache.entries.insert(key, entry);
        }

        // Make sure this is set to None,
        // since we're about to start a new `http_network_or_cache_fetch`.
        *done_chan = None;

        // Step 15.5 Set response to the result of running HTTP-network-or-cache fetch given fetchParams.
        response = http_network_or_cache_fetch(
            fetch_params,
            false, /* authentication flag */
            cors_flag,
            done_chan,
            context,
        )
        .await;
    }

    // TODO(#33616): Step 16. If all of the following are true:
    // * response’s status is 421
    // * isNewConnectionFetch is false
    // * request’s body is null, or request’s body is non-null and request’s body’s source is non-null
    // then: [..]

    // Step 17. If isAuthenticationFetch is true, then create an authentication entry for request and the given realm.
    if authentication_fetch_flag {
        // TODO(#33616)
    }

    // Step 18. Return response.
    response
}

/// <https://fetch.spec.whatwg.org/#cross-origin-resource-policy-check>
///
/// This is obtained from [cross_origin_resource_policy_check]
#[derive(PartialEq)]
enum CrossOriginResourcePolicy {
    Allowed,
    Blocked,
}

// TODO(#33615): Judging from the name, this appears to be https://fetch.spec.whatwg.org/#cross-origin-resource-policy-check,
//       but the steps aren't even close to the spec. Perhaps this needs to be rewritten?
fn cross_origin_resource_policy_check(
    request: &Request,
    response: &Response,
) -> CrossOriginResourcePolicy {
    // Step 1
    if request.mode != RequestMode::NoCors {
        return CrossOriginResourcePolicy::Allowed;
    }

    // Step 2
    let current_url_origin = request.current_url().origin();
    let same_origin = if let Origin::Origin(ref origin) = request.origin {
        *origin == request.current_url().origin()
    } else {
        false
    };

    if same_origin {
        return CrossOriginResourcePolicy::Allowed;
    }

    // Step 3
    let policy = response
        .headers
        .get(HeaderName::from_static("cross-origin-resource-policy"))
        .map(|h| h.to_str().unwrap_or(""))
        .unwrap_or("");

    // Step 4
    if policy == "same-origin" {
        return CrossOriginResourcePolicy::Blocked;
    }

    // Step 5
    if let Origin::Origin(ref request_origin) = request.origin {
        let schemeless_same_origin = is_schemelessy_same_site(request_origin, &current_url_origin);
        if schemeless_same_origin &&
            (request_origin.scheme() == Some("https") ||
                response.https_state == HttpsState::None)
        {
            return CrossOriginResourcePolicy::Allowed;
        }
    };

    // Step 6
    if policy == "same-site" {
        return CrossOriginResourcePolicy::Blocked;
    }

    CrossOriginResourcePolicy::Allowed
}

// Convenience struct that implements Done, for setting responseEnd on function return
struct ResponseEndTimer(Option<Arc<Mutex<ResourceFetchTiming>>>);

impl ResponseEndTimer {
    fn neuter(&mut self) {
        self.0 = None;
    }
}

impl Drop for ResponseEndTimer {
    fn drop(&mut self) {
        let ResponseEndTimer(resource_fetch_timing_opt) = self;

        resource_fetch_timing_opt.as_ref().map_or((), |t| {
            t.lock()
                .unwrap()
                .set_attribute(ResourceAttribute::ResponseEnd);
        })
    }
}

fn prompt_user_for_credentials(
    embedder_proxy: &Mutex<EmbedderProxy>,
    webview_id: WebViewId,
) -> Option<PromptCredentialsInput> {
    let proxy = embedder_proxy.lock().unwrap();

    let (ipc_sender, ipc_receiver) = ipc::channel().unwrap();

    proxy.send(EmbedderMsg::Prompt(
        webview_id,
        PromptDefinition::Credentials(ipc_sender),
        PromptOrigin::Trusted,
    ));

    let Ok(credentials) = ipc_receiver.recv() else {
        warn!("error getting user credentials");
        return None;
    };

    Some(credentials)
}

/// [HTTP network fetch](https://fetch.spec.whatwg.org/#http-network-fetch)
async fn http_network_fetch(
    fetch_params: &mut FetchParams,
    credentials_flag: bool,
    done_chan: &mut DoneChannel,
    context: &FetchContext,
) -> Response {
    let mut response_end_timer = ResponseEndTimer(Some(context.timing.clone()));

    // Step 1: Let request be fetchParams’s request.
    let request = &mut fetch_params.request;

    // Step 2
    // TODO be able to create connection using current url's origin and credentials

    // Step 3
    // TODO be able to tell if the connection is a failure

    // Step 4
    // TODO: check whether the connection is HTTP/2

    // Step 5
    let url = request.current_url();

    let request_id = context
        .devtools_chan
        .as_ref()
        .map(|_| uuid::Uuid::new_v4().simple().to_string());

    if log_enabled!(log::Level::Info) {
        info!("{:?} request for {}", request.method, url);
        for header in request.headers.iter() {
            debug!(" - {:?}", header);
        }
    }

    // XHR uses the default destination; other kinds of fetches (which haven't been implemented yet)
    // do not. Once we support other kinds of fetches we'll need to be more fine grained here
    // since things like image fetches are classified differently by devtools
    let is_xhr = request.destination == Destination::None;

    // The receiver will receive true if there has been an error streaming the request body.
    let (fetch_terminated_sender, mut fetch_terminated_receiver) = unbounded_channel();

    let body = request.body.as_ref().map(|body| body.take_stream());

    if body.is_none() {
        // There cannot be an error streaming a non-existent body.
        // However in such a case the channel will remain unused
        // and drop inside `obtain_response`.
        // Send the confirmation now, ensuring the receiver will not dis-connect first.
        let _ = fetch_terminated_sender.send(false);
    }

    let response_future = obtain_response(
        &context.state.client,
        &url,
        &request.method,
        &mut request.headers,
        body,
        request
            .body
            .as_ref()
            .map(|body| body.source_is_null())
            .unwrap_or(false),
        &request.pipeline_id,
        request_id.as_deref(),
        is_xhr,
        context,
        fetch_terminated_sender,
    );

    let pipeline_id = request.pipeline_id;
    // This will only get the headers, the body is read later
    let (res, msg) = match response_future.await {
        Ok(wrapped_response) => wrapped_response,
        Err(error) => return Response::network_error(error),
    };

    if log_enabled!(log::Level::Info) {
        debug!("{:?} response for {}", res.version(), url);
        for header in res.headers().iter() {
            debug!(" - {:?}", header);
        }
    }

    // Check if there was an error while streaming the request body.
    //
    match fetch_terminated_receiver.recv().await {
        Some(true) => {
            return Response::network_error(NetworkError::Internal(
                "Request body streaming failed.".into(),
            ));
        },
        Some(false) => {},
        _ => warn!("Failed to receive confirmation request was streamed without error."),
    }

    let header_strings: Vec<&str> = res
        .headers()
        .get_all("Timing-Allow-Origin")
        .iter()
        .map(|header_value| header_value.to_str().unwrap_or(""))
        .collect();
    let wildcard_present = header_strings.iter().any(|header_str| *header_str == "*");
    // The spec: https://www.w3.org/TR/resource-timing-2/#sec-timing-allow-origin
    // says that a header string is either an origin or a wildcard so we can just do a straight
    // check against the document origin
    let req_origin_in_timing_allow = header_strings
        .iter()
        .any(|header_str| match request.origin {
            SpecificOrigin(ref immutable_request_origin) => {
                *header_str == immutable_request_origin.ascii_serialization()
            },
            _ => false,
        });

    let is_same_origin = request.url_list.iter().all(|url| match request.origin {
        SpecificOrigin(ref immutable_request_origin) => url.origin() == *immutable_request_origin,
        _ => false,
    });

    if !(is_same_origin || req_origin_in_timing_allow || wildcard_present) {
        context.timing.lock().unwrap().mark_timing_check_failed();
    }

    let timing = context.timing.lock().unwrap().clone();
    let mut response = Response::new(url.clone(), timing);

    let status_text = res
        .extensions()
        .get::<ReasonPhrase>()
        .map(ReasonPhrase::as_bytes)
        .or_else(|| res.status().canonical_reason().map(str::as_bytes))
        .map(Vec::from)
        .unwrap_or_default();
    response.status = HttpStatus::new(res.status(), status_text);

    info!("got {:?} response for {:?}", res.status(), request.url());
    response.headers = res.headers().clone();
    response.referrer = request.referrer.to_url().cloned();
    response.referrer_policy = request.referrer_policy;

    let res_body = response.body.clone();

    // We're about to spawn a future to be waited on here
    let (done_sender, done_receiver) = unbounded_channel();
    *done_chan = Some((done_sender.clone(), done_receiver));
    let meta = match response
        .metadata()
        .expect("Response metadata should exist at this stage")
    {
        FetchMetadata::Unfiltered(m) => m,
        FetchMetadata::Filtered { unsafe_, .. } => unsafe_,
    };

    let devtools_sender = context.devtools_chan.clone();
    let meta_status = meta.status;
    let meta_headers = meta.headers;
    let cancellation_listener = context.cancellation_listener.clone();
    if cancellation_listener.cancelled() {
        return Response::network_error(NetworkError::Internal("Fetch aborted".into()));
    }

    *res_body.lock().unwrap() = ResponseBody::Receiving(vec![]);
    let res_body2 = res_body.clone();

    if let Some(ref sender) = devtools_sender {
        let sender = sender.lock().unwrap();
        if let Some(m) = msg {
            send_request_to_devtools(m, &sender);
        }

        // --- Tell devtools that we got a response
        // Send an HttpResponse message to devtools with the corresponding request_id
        if let Some(pipeline_id) = pipeline_id {
            send_response_to_devtools(
                &sender,
                request_id.unwrap(),
                meta_headers.map(Serde::into_inner),
                meta_status,
                pipeline_id,
            );
        }
    }

    let done_sender2 = done_sender.clone();
    let done_sender3 = done_sender.clone();
    let timing_ptr2 = context.timing.clone();
    let timing_ptr3 = context.timing.clone();
    let url1 = request.url();
    let url2 = url1.clone();

    HANDLE.lock().unwrap().as_ref().unwrap().spawn(
        res.into_body()
            .map_err(|e| {
                warn!("Error streaming response body: {:?}", e);
            })
            .try_fold(res_body, move |res_body, chunk| {
                if cancellation_listener.cancelled() {
                    *res_body.lock().unwrap() = ResponseBody::Done(vec![]);
                    let _ = done_sender.send(Data::Cancelled);
                    return future::ready(Err(()));
                }
                if let ResponseBody::Receiving(ref mut body) = *res_body.lock().unwrap() {
                    let bytes = chunk;
                    body.extend_from_slice(&bytes);
                    let _ = done_sender.send(Data::Payload(bytes.to_vec()));
                }
                future::ready(Ok(res_body))
            })
            .and_then(move |res_body| {
                debug!("successfully finished response for {:?}", url1);
                let mut body = res_body.lock().unwrap();
                let completed_body = match *body {
                    ResponseBody::Receiving(ref mut body) => std::mem::take(body),
                    _ => vec![],
                };
                *body = ResponseBody::Done(completed_body);
                timing_ptr2
                    .lock()
                    .unwrap()
                    .set_attribute(ResourceAttribute::ResponseEnd);
                let _ = done_sender2.send(Data::Done);
                future::ready(Ok(()))
            })
            .map_err(move |_| {
                debug!("finished response for {:?}", url2);
                let mut body = res_body2.lock().unwrap();
                let completed_body = match *body {
                    ResponseBody::Receiving(ref mut body) => std::mem::take(body),
                    _ => vec![],
                };
                *body = ResponseBody::Done(completed_body);
                timing_ptr3
                    .lock()
                    .unwrap()
                    .set_attribute(ResourceAttribute::ResponseEnd);
                let _ = done_sender3.send(Data::Done);
            }),
    );

    // TODO these substeps aren't possible yet
    // Substep 1

    // Substep 2

    response.https_state = match url.scheme() {
        "https" => HttpsState::Modern,
        _ => HttpsState::None,
    };

    // TODO Read request

    // Step 6-11
    // (needs stream bodies)

    // Step 13
    // TODO this step isn't possible yet (CSP)

    // Step 14, update the cached response, done via the shared response body.

    // TODO this step isn't possible yet
    // Step 15
    if credentials_flag {
        set_cookies_from_headers(&url, &response.headers, &context.state.cookie_jar);
    }
    context
        .state
        .hsts_list
        .write()
        .unwrap()
        .update_hsts_list_from_response(&url, &response.headers);

    // TODO these steps
    // Step 16
    // Substep 1
    // Substep 2
    // Sub-substep 1
    // Sub-substep 2
    // Sub-substep 3
    // Sub-substep 4
    // Substep 3

    // Step 16

    // Ensure we don't override "responseEnd" on successful return of this function
    response_end_timer.neuter();

    response
}

/// [CORS preflight fetch](https://fetch.spec.whatwg.org#cors-preflight-fetch)
async fn cors_preflight_fetch(
    request: &Request,
    cache: &mut CorsCache,
    context: &FetchContext,
) -> Response {
    // Step 1
    let mut preflight = RequestBuilder::new(
        request.target_webview_id,
        request.current_url(),
        request.referrer.clone(),
    )
    .method(Method::OPTIONS)
    .origin(match &request.origin {
        Origin::Client => {
            unreachable!("We shouldn't get Client origin in cors_preflight_fetch.")
        },
        Origin::Origin(origin) => origin.clone(),
    })
    .pipeline_id(request.pipeline_id)
    .initiator(request.initiator)
    .destination(request.destination)
    .referrer_policy(request.referrer_policy)
    .mode(RequestMode::CorsMode)
    .response_tainting(ResponseTainting::CorsTainting)
    .build();

    // Step 2
    preflight
        .headers
        .insert(ACCEPT, HeaderValue::from_static("*/*"));

    // Step 3
    preflight
        .headers
        .typed_insert::<AccessControlRequestMethod>(AccessControlRequestMethod::from(
            request.method.clone(),
        ));

    // Step 4
    let headers = get_cors_unsafe_header_names(&request.headers);

    // Step 5
    if !headers.is_empty() {
        preflight
            .headers
            .typed_insert(AccessControlRequestHeaders::from_iter(headers));
    }

    // Step 6
    let mut fetch_params = FetchParams::new(preflight);
    let response =
        http_network_or_cache_fetch(&mut fetch_params, false, false, &mut None, context).await;
    // Step 7
    if cors_check(request, &response).is_ok() && response.status.code().is_success() {
        // Substep 1
        let mut methods = if response
            .headers
            .contains_key(header::ACCESS_CONTROL_ALLOW_METHODS)
        {
            match response.headers.typed_get::<AccessControlAllowMethods>() {
                Some(methods) => methods.iter().collect(),
                // Substep 3
                None => {
                    return Response::network_error(NetworkError::Internal(
                        "CORS ACAM check failed".into(),
                    ));
                },
            }
        } else {
            vec![]
        };

        // Substep 2
        let header_names = if response
            .headers
            .contains_key(header::ACCESS_CONTROL_ALLOW_HEADERS)
        {
            match response.headers.typed_get::<AccessControlAllowHeaders>() {
                Some(names) => names.iter().collect(),
                // Substep 3
                None => {
                    return Response::network_error(NetworkError::Internal(
                        "CORS ACAH check failed".into(),
                    ));
                },
            }
        } else {
            vec![]
        };

        debug!(
            "CORS check: Allowed methods: {:?}, current method: {:?}",
            methods, request.method
        );

        // Substep 4
        if methods.is_empty() && request.use_cors_preflight {
            methods = vec![request.method.clone()];
        }

        // Substep 5
        if methods
            .iter()
            .all(|m| *m.as_str() != *request.method.as_ref()) &&
            !is_cors_safelisted_method(&request.method) &&
            (request.credentials_mode == CredentialsMode::Include ||
                methods.iter().all(|m| m.as_ref() != "*"))
        {
            return Response::network_error(NetworkError::Internal(
                "CORS method check failed".into(),
            ));
        }

        debug!(
            "CORS check: Allowed headers: {:?}, current headers: {:?}",
            header_names, request.headers
        );

        // Substep 6
        if request.headers.iter().any(|(name, _)| {
            is_cors_non_wildcard_request_header_name(name) &&
                header_names.iter().all(|hn| hn != name)
        }) {
            return Response::network_error(NetworkError::Internal(
                "CORS authorization check failed".into(),
            ));
        }

        // Substep 7
        let unsafe_names = get_cors_unsafe_header_names(&request.headers);
        #[allow(clippy::mutable_key_type)] // We don't mutate the items in the set
        let header_names_set: HashSet<&HeaderName> = HashSet::from_iter(header_names.iter());
        let header_names_contains_star = header_names.iter().any(|hn| hn.as_str() == "*");
        for unsafe_name in unsafe_names.iter() {
            if !header_names_set.contains(unsafe_name) &&
                (request.credentials_mode == CredentialsMode::Include ||
                    !header_names_contains_star)
            {
                return Response::network_error(NetworkError::Internal(
                    "CORS headers check failed".into(),
                ));
            }
        }

        // Substep 8, 9
        let max_age: Duration = response
            .headers
            .typed_get::<AccessControlMaxAge>()
            .map(|acma| acma.into())
            .unwrap_or(Duration::from_secs(5));
        // Substep 10
        // TODO: Need to define what an imposed limit on max-age is

        // Substep 11 ignored, we do have a CORS cache

        // Substep 12, 13
        for method in &methods {
            cache.match_method_and_update(request, method.clone(), max_age);
        }

        // Substep 14, 15
        for header_name in &header_names {
            cache.match_header_and_update(request, header_name, max_age);
        }

        // Substep 16
        return response;
    }

    // Step 8
    Response::network_error(NetworkError::Internal("CORS check failed".into()))
}

/// [CORS check](https://fetch.spec.whatwg.org#concept-cors-check)
fn cors_check(request: &Request, response: &Response) -> Result<(), ()> {
    // Step 1
    let origin = response.headers.typed_get::<AccessControlAllowOrigin>();

    // Step 2
    let origin = origin.ok_or(())?;

    // Step 3
    if request.credentials_mode != CredentialsMode::Include &&
        origin == AccessControlAllowOrigin::ANY
    {
        return Ok(());
    }

    // Step 4
    let origin = match origin.origin() {
        Some(origin) => origin,
        // if it's Any or Null at this point, there's nothing to do but return Err(())
        None => return Err(()),
    };

    match request.origin {
        Origin::Origin(ref o) if o.ascii_serialization() == origin.to_string().trim() => {},
        _ => return Err(()),
    }

    // Step 5
    if request.credentials_mode != CredentialsMode::Include {
        return Ok(());
    }

    // Step 6
    let credentials = response
        .headers
        .typed_get::<AccessControlAllowCredentials>();

    // Step 7
    if credentials.is_some() {
        return Ok(());
    }

    // Step 8
    Err(())
}

fn has_credentials(url: &ServoUrl) -> bool {
    !url.username().is_empty() || url.password().is_some()
}

fn is_no_store_cache(headers: &HeaderMap) -> bool {
    headers.contains_key(header::IF_MODIFIED_SINCE) |
        headers.contains_key(header::IF_NONE_MATCH) |
        headers.contains_key(header::IF_UNMODIFIED_SINCE) |
        headers.contains_key(header::IF_MATCH) |
        headers.contains_key(header::IF_RANGE)
}

/// <https://fetch.spec.whatwg.org/#redirect-status>
fn is_redirect_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::MOVED_PERMANENTLY |
            StatusCode::FOUND |
            StatusCode::SEE_OTHER |
            StatusCode::TEMPORARY_REDIRECT |
            StatusCode::PERMANENT_REDIRECT
    )
}

/// <https://fetch.spec.whatwg.org/#concept-request-tainted-origin>
fn request_has_redirect_tainted_origin(request: &Request) -> bool {
    // Step 1. Assert: request’s origin is not "client".
    let Origin::Origin(request_origin) = &request.origin else {
        panic!("origin cannot be \"client\" at this point in time");
    };

    // Step 2. Let lastURL be null.
    let mut last_url = None;

    // Step 3. For each url of request’s URL list:
    for url in &request.url_list {
        // Step 3.1 If lastURL is null, then set lastURL to url and continue.
        let Some(last_url) = &mut last_url else {
            last_url = Some(url);
            continue;
        };

        // Step 3.2 If url’s origin is not same origin with lastURL’s origin and
        //          request’s origin is not same origin with lastURL’s origin, then return true.
        if url.origin() != last_url.origin() && *request_origin != last_url.origin() {
            return true;
        }

        // Step 3.3 Set lastURL to url.
        *last_url = url;
    }

    // Step 4. Return false.
    false
}

/// <https://fetch.spec.whatwg.org/#serializing-a-request-origin>
fn serialize_request_origin(request: &Request) -> headers::Origin {
    // Step 1. Assert: request’s origin is not "client".
    let Origin::Origin(origin) = &request.origin else {
        panic!("origin cannot be \"client\" at this point in time");
    };

    // Step 2. If request has a redirect-tainted origin, then return "null".
    if request_has_redirect_tainted_origin(request) {
        return headers::Origin::NULL;
    }

    // Step 3. Return request’s origin, serialized.
    serialize_origin(origin)
}

/// Step 3 of <https://fetch.spec.whatwg.org/#serializing-a-request-origin>.
pub fn serialize_origin(origin: &ImmutableOrigin) -> headers::Origin {
    match origin {
        ImmutableOrigin::Opaque(_) => headers::Origin::NULL,
        ImmutableOrigin::Tuple(scheme, host, port) => {
            // Note: This must be kept in sync with `Origin::ascii_serialization()`, which does not
            // use the port number when a default port is used.
            let port = match (scheme.as_ref(), port) {
                ("http" | "ws", 80) | ("https" | "wss", 443) | ("ftp", 21) => None,
                _ => Some(*port),
            };

            // TODO: Ensure that hyper/servo don't disagree about valid origin headers
            headers::Origin::try_from_parts(scheme, &host.to_string(), port)
                .unwrap_or(headers::Origin::NULL)
        },
    }
}

/// <https://fetch.spec.whatwg.org/#append-a-request-origin-header>
fn append_a_request_origin_header(request: &mut Request) {
    // Step 1. Assert: request’s origin is not "client".
    let Origin::Origin(request_origin) = &request.origin else {
        panic!("origin cannot be \"client\" at this point in time");
    };

    // Step 2. Let serializedOrigin be the result of byte-serializing a request origin with request.
    let mut serialized_origin = serialize_request_origin(request);

    // Step 3. If request’s response tainting is "cors" or request’s mode is "websocket",
    //         then append (`Origin`, serializedOrigin) to request’s header list.
    if request.response_tainting == ResponseTainting::CorsTainting ||
        matches!(request.mode, RequestMode::WebSocket { .. })
    {
        request.headers.typed_insert(serialized_origin);
    }
    // Step 4. Otherwise, if request’s method is neither `GET` nor `HEAD`, then:
    else if !matches!(request.method, Method::GET | Method::HEAD) {
        // Step 4.1 If request’s mode is not "cors", then switch on request’s referrer policy:
        if request.mode != RequestMode::CorsMode {
            match request.referrer_policy {
                ReferrerPolicy::NoReferrer => {
                    // Set serializedOrigin to `null`.
                    serialized_origin = headers::Origin::NULL;
                },
                ReferrerPolicy::NoReferrerWhenDowngrade |
                ReferrerPolicy::StrictOrigin |
                ReferrerPolicy::StrictOriginWhenCrossOrigin => {
                    // If request’s origin is a tuple origin, its scheme is "https", and
                    // request’s current URL’s scheme is not "https", then set serializedOrigin to `null`.
                    if let ImmutableOrigin::Tuple(scheme, _, _) = &request_origin {
                        if scheme == "https" && request.current_url().scheme() != "https" {
                            serialized_origin = headers::Origin::NULL;
                        }
                    }
                },
                ReferrerPolicy::SameOrigin => {
                    // If request’s origin is not same origin with request’s current URL’s origin,
                    // then set serializedOrigin to `null`.
                    if *request_origin != request.current_url().origin() {
                        serialized_origin = headers::Origin::NULL;
                    }
                },
                _ => {
                    // Otherwise, do nothing.
                },
            };
        }

        // Step 4.2. Append (`Origin`, serializedOrigin) to request’s header list.
        request.headers.typed_insert(serialized_origin);
    }
}

/// <https://w3c.github.io/webappsec-fetch-metadata/#abstract-opdef-append-the-fetch-metadata-headers-for-a-request>
fn append_the_fetch_metadata_headers(r: &mut Request) {
    // Step 1. If r’s url is not an potentially trustworthy URL, return.
    if !r.url().is_potentially_trustworthy() {
        return;
    }

    // Step 2. Set the Sec-Fetch-Dest header for r.
    set_the_sec_fetch_dest_header(r);

    // Step 3. Set the Sec-Fetch-Mode header for r.
    set_the_sec_fetch_mode_header(r);

    // Step 4. Set the Sec-Fetch-Site header for r.
    set_the_sec_fetch_site_header(r);

    // Step 5. Set the Sec-Fetch-User header for r.
    set_the_sec_fetch_user_header(r);
}

/// <https://w3c.github.io/webappsec-fetch-metadata/#abstract-opdef-set-dest>
fn set_the_sec_fetch_dest_header(r: &mut Request) {
    // Step 1. Assert: r’s url is a potentially trustworthy URL.
    debug_assert!(r.url().is_potentially_trustworthy());

    // Step 2. Let header be a Structured Header whose value is a token.
    // Step 3. If r’s destination is the empty string, set header’s value to the string "empty".
    // Otherwise, set header’s value to r’s destination.
    let header = r.destination;

    // Step 4. Set a structured field value `Sec-Fetch-Dest`/header in r’s header list.
    r.headers.typed_insert(SecFetchDest(header));
}

/// <https://w3c.github.io/webappsec-fetch-metadata/#abstract-opdef-set-mode>
fn set_the_sec_fetch_mode_header(r: &mut Request) {
    // Step 1. Assert: r’s url is a potentially trustworthy URL.
    debug_assert!(r.url().is_potentially_trustworthy());

    // Step 2. Let header be a Structured Header whose value is a token.
    // Step 3. Set header’s value to r’s mode.
    let header = &r.mode;

    // Step 4. Set a structured field value `Sec-Fetch-Mode`/header in r’s header list.
    r.headers.typed_insert(SecFetchMode::from(header));
}

/// <https://w3c.github.io/webappsec-fetch-metadata/#abstract-opdef-set-site>
fn set_the_sec_fetch_site_header(r: &mut Request) {
    // The webappsec spec seems to have a similar issue as
    // https://github.com/whatwg/fetch/issues/1773
    let Origin::Origin(request_origin) = &r.origin else {
        panic!("request origin cannot be \"client\" at this point")
    };

    // Step 1. Assert: r’s url is a potentially trustworthy URL.
    debug_assert!(r.url().is_potentially_trustworthy());

    // Step 2. Let header be a Structured Header whose value is a token.
    // Step 3. Set header’s value to same-origin.
    let mut header = SecFetchSite::SameOrigin;

    // TODO: Step 3. If r is a navigation request that was explicitly caused by a
    // user’s interaction with the user agent, then set header’s value to none.

    // Step 5. If header’s value is not none, then for each url in r’s url list:
    if header != SecFetchSite::None {
        for url in &r.url_list {
            // Step 5.1 If url is same origin with r’s origin, continue.
            if url.origin() == *request_origin {
                continue;
            }

            // Step 5.2 Set header’s value to cross-site.
            header = SecFetchSite::CrossSite;

            // Step 5.3 If r’s origin is not same site with url’s origin, then break.
            if is_same_site(request_origin, &url.origin()) {
                break;
            }

            // Step 5.4 Set header’s value to same-site.
            header = SecFetchSite::SameSite;
        }
    }

    // Step 6. Set a structured field value `Sec-Fetch-Site`/header in r’s header list.
    r.headers.typed_insert(header);
}

/// <https://w3c.github.io/webappsec-fetch-metadata/#abstract-opdef-set-user>
fn set_the_sec_fetch_user_header(r: &mut Request) {
    // Step 1. Assert: r’s url is a potentially trustworthy URL.
    debug_assert!(r.url().is_potentially_trustworthy());

    // Step 2. If r is not a navigation request, or if r’s user-activation is false, return.
    // TODO user activation
    if !r.is_navigation_request() {
        return;
    }

    // Step 3. Let header be a Structured Header whose value is a token.
    // Step 4. Set header’s value to true.
    let header = SecFetchUser;

    // Step 5. Set a structured field value `Sec-Fetch-User`/header in r’s header list.
    r.headers.typed_insert(header);
}

/// <https://w3c.github.io/webappsec-referrer-policy/#set-requests-referrer-policy-on-redirect>
fn set_requests_referrer_policy_on_redirect(request: &mut Request, response: &Response) {
    // Step 1: Let policy be the result of executing § 8.1 Parse a referrer policy from a
    // Referrer-Policy header on actualResponse.
    let referrer_policy: ReferrerPolicy = response
        .headers
        .typed_get::<headers::ReferrerPolicy>()
        .into();

    // Step 2: If policy is not the empty string, then set request’s referrer policy to policy.
    if referrer_policy != ReferrerPolicy::EmptyString {
        request.referrer_policy = referrer_policy;
    }
}
