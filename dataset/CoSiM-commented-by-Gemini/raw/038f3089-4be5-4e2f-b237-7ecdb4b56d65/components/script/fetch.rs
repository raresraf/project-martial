//! This module implements the [WHATWG Fetch specification](https://fetch.spec.whatwg.org/),
//! providing a modern, promise-based API for making network requests, intended as a more
//! powerful and flexible replacement for `XMLHttpRequest`. It serves as the bridge between the
//! JavaScript scripting environment and the browser's networking stack.
//!
//! # Architectural Overview
//!
//! The implementation is centered around the `Fetch` function, which is the main entry point
//! exposed to the JavaScript global scope (e.g., `window.fetch`). This function orchestrates
//! the entire fetch process, from request creation to response handling.
//!
//! Key components and their roles:
//!
//! - **`Fetch` function**: Parses the JavaScript `RequestInfo` and `RequestInit` arguments,
//!   constructs a `Request` object, and initiates the asynchronous fetch operation. It returns
//!   a `Promise` that will eventually resolve with a `Response` or reject with an error.
//!
//! - **`FetchContext`**: A state machine that manages the lifecycle of a single fetch. It holds
//!   the `Promise` to be resolved, the partially constructed `Response` object, and performance
//!   timing data (`ResourceFetchTiming`). It implements the `FetchResponseListener` trait to
//!   react to events from the networking layer (e.g., headers received, data chunks, EOF).
//!
//! - **`FetchCanceller`**: A RAII-based guard that ensures a fetch is cancelled if the corresponding
//!   handle is dropped. This is crucial for resource management, especially when the DOM element
//!   or script environment that initiated the fetch is destroyed.
//!
//! - **Integration with Networking (`net_traits`)**: This module does not perform networking I/O
//!   directly. Instead, it communicates with the core networking thread via an IPC channel
//!   (`CoreResourceThread`). It constructs a `RequestBuilder` object, which is a serializable
//!   representation of the request, and sends it to the networking stack for processing.
//!
//! - **Security**: The module is responsible for enforcing several security policies, including
//!   CORS (Cross-Origin Resource Sharing), Content Security Policy (CSP), and referrer policies.
//!   It collaborates with the `security_manager` and uses `PolicyContainer` to make security
//!   decisions.
//!
//! - **Performance Monitoring**: It integrates with the Resource Timing API by implementing the
//!   `ResourceTimingListener` trait, capturing detailed performance metrics for each fetch
//!   and submitting them to the performance timeline.
//!
//! The overall design is highly asynchronous and event-driven, leveraging Rust's ownership
//! and trait system to ensure safety and modularity while interfacing with both the JavaScript
//! engine (SpiderMonkey, via `jsapi`) and the low-level networking components of the browser engine.

use std::rc::Rc;
use std::sync::{Arc, Mutex};

use base::id::WebViewId;
use content_security_policy as csp;
use ipc_channel::ipc;
use net_traits::policy_container::{PolicyContainer, RequestPolicyContainer};
use net_traits::request::{
    CorsSettings, CredentialsMode, Destination, InsecureRequestsPolicy, Referrer,
    Request as NetTraitsRequest, RequestBuilder, RequestId, RequestMode, ServiceWorkersMode,
};
use net_traits::{
    CoreResourceMsg, CoreResourceThread, FetchChannels, FetchMetadata, FetchResponseListener,
    FetchResponseMsg, FilteredMetadata, Metadata, NetworkError, ResourceFetchTiming,
    ResourceTimingType, cancel_async_fetch,
};
use servo_url::ServoUrl;

use crate::dom::bindings::codegen::Bindings::RequestBinding::{
    RequestInfo, RequestInit, RequestMethods,
};
use crate::dom::bindings::codegen::Bindings::ResponseBinding::Response_Binding::ResponseMethods;
use crate::dom::bindings::codegen::Bindings::ResponseBinding::ResponseType as DOMResponseType;
use crate::dom::bindings::error::Error;
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::{Trusted, TrustedPromise};
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::root::DomRoot;
use crate::dom::bindings::trace::RootedTraceableBox;
use crate::dom::csp::report_csp_violations;
use crate::dom::globalscope::GlobalScope;
use crate::dom::headers::Guard;
use crate::dom::performanceresourcetiming::InitiatorType;
use crate::dom::promise::Promise;
use crate::dom::request::Request;
use crate::dom::response::Response;
use crate::dom::serviceworkerglobalscope::ServiceWorkerGlobalScope;
use crate::network_listener::{self, PreInvoke, ResourceTimingListener, submit_timing_data};
use crate::realms::{InRealm, enter_realm};
use crate::script_runtime::CanGc;

/// `FetchContext` holds the state associated with a single fetch operation. It acts as a
/// state machine that transitions based on events received from the networking layer.
/// This struct implements the `FetchResponseListener` trait, making it the primary
/// consumer of asynchronous fetch data.
struct FetchContext {
    /// The `Promise` that will be resolved or rejected upon completion or failure of the fetch.
    /// This is `Option`-wrapped to allow it to be `take()`-n when the promise is settled,
    /// preventing multiple resolutions or rejections.
    fetch_promise: Option<TrustedPromise>,

    /// The `Response` object that is exposed to JavaScript. It is populated incrementally
    /// as data arrives from the network (e.g., headers, body chunks). It is wrapped
    /// in `Trusted` to ensure it is properly rooted and managed by the garbage collector.
    response_object: Trusted<Response>,

    /// Stores performance metrics for the fetch, such as start time, DNS lookup time,
    /// and time to first byte. This data is used to populate the Performance Timeline API
    /// via the `ResourceTimingListener` trait implementation.
    resource_timing: ResourceFetchTiming,
}

/// `FetchCanceller` is a RAII-based guard that automatically cancels an associated fetch
/// request when it goes out of scope (i.e., is `drop()`-ed).
///
/// This pattern is crucial for preventing resource leaks and unnecessary network activity.
/// For instance, if a component that initiated a fetch is destroyed before the fetch
/// completes, the `FetchCanceller`'s `drop` implementation will be invoked, sending a
/// cancellation message to the networking thread.
#[derive(Default, JSTraceable, MallocSizeOf)]
pub(crate) struct FetchCanceller {
    #[no_trace]
    /// The unique identifier for the network request. This is `None` if the fetch
    /// has already been completed or cancelled, preventing redundant cancellation calls.
    request_id: Option<RequestId>,
}

impl FetchCanceller {
    /// Constructs a new `FetchCanceller` to manage the lifecycle of a request.
    pub(crate) fn new(request_id: RequestId) -> Self {
        Self {
            request_id: Some(request_id),
        }
    }

    /// Explicitly cancels the fetch request if it is still active.
    /// It sends a cancellation message to the networking thread. This is a "best-effort"
    /// operation; the underlying network request may have already completed or failed.
    pub(crate) fn cancel(&mut self) {
        if let Some(request_id) = self.request_id.take() {
            // "You can't always get what you want." - The Rolling Stones.
            // Cancellation is a courtesy call. We don't block or wait for confirmation,
            // as the network process might be busy or the request might already be
            // in a terminal state.
            cancel_async_fetch(vec![request_id]);
        }
    }

    /// Disarms the canceller, preventing it from cancelling the fetch upon being dropped.
    /// This is called when the fetch completes successfully, transferring ownership of
    /// the response stream to the caller.
    pub(crate) fn ignore(&mut self) {
        let _ = self.request_id.take();
    }
}

impl Drop for FetchCanceller {
    /// The `drop` implementation for `FetchCanceller` ensures that the fetch is
    /// cancelled if it is still ongoing when the `FetchCanceller` is dropped.
    fn drop(&mut self) {
        self.cancel()
    }
}

/// Creates a `RequestBuilder` from an existing `NetTraitsRequest`.
/// This utility function is used to clone and re-purpose a request, often for
/// internal processing like redirects or service worker interception, while
/// preserving essential properties of the original request.
fn request_init_from_request(request: NetTraitsRequest) -> RequestBuilder {
    RequestBuilder {
        id: request.id,
        method: request.method.clone(),
        url: request.url(),
        headers: request.headers.clone(),
        unsafe_request: request.unsafe_request,
        body: request.body.clone(),
        service_workers_mode: ServiceWorkersMode::All,
        destination: request.destination,
        synchronous: request.synchronous,
        mode: request.mode.clone(),
        cache_mode: request.cache_mode,
        use_cors_preflight: request.use_cors_preflight,
        credentials_mode: request.credentials_mode,
        use_url_credentials: request.use_url_credentials,
        // The origin is derived from the current global scope, which represents
        // the security context of the code initiating the fetch.
        origin: GlobalScope::current()
            .expect("No current global object")
            .origin()
            .immutable()
            .clone(),
        referrer: request.referrer.clone(),
        referrer_policy: request.referrer_policy,
        pipeline_id: request.pipeline_id,
        target_webview_id: request.target_webview_id,
        redirect_mode: request.redirect_mode,
        integrity_metadata: request.integrity_metadata.clone(),
        cryptographic_nonce_metadata: request.cryptographic_nonce_metadata.clone(),
        url_list: vec![],
        parser_metadata: request.parser_metadata,
        initiator: request.initiator,
        policy_container: request.policy_container,
        insecure_requests_policy: request.insecure_requests_policy,
        has_trustworthy_ancestor_origin: request.has_trustworthy_ancestor_origin,
        https_state: request.https_state,
        response_tainting: request.response_tainting,
        crash: None,
    }
}

/// The `Fetch` method is the entry point for the Fetch API, exposed to JavaScript.
/// It processes the `input` and `init` arguments, constructs a request, and dispatches
/// it to the networking thread. It returns a `Promise` that resolves with the `Response`.
/// This implementation follows the steps outlined in the WHATWG Fetch specification.
/// <https://fetch.spec.whatwg.org/#fetch-method>
#[allow(non_snake_case)]
#[cfg_attr(crown, allow(crown::unrooted_must_root))]
pub(crate) fn Fetch(
    global: &GlobalScope,
    input: RequestInfo,
    init: RootedTraceableBox<RequestInit>,
    comp: InRealm,
    can_gc: CanGc,
) -> Rc<Promise> {
    // Spec Step 1: Let p be a new promise.
    let promise = Promise::new_in_current_realm(comp, can_gc);

    // Spec Step 7: Let responseObject be null.
    // NOTE: We initialize the Response object early to handle potential synchronous errors
    // during request creation, allowing us to associate the error with the response.
    let response = Response::new(global, can_gc);
    // The headers are immutable until the network response provides them.
    response.Headers(can_gc).set_guard(Guard::Immutable);

    // Spec Step 2: Create a new Request object. If this fails (e.g., invalid URL or headers),
    // reject the promise and terminate the algorithm.
    let request = match Request::Constructor(global, None, can_gc, input, init) {
        Err(e) => {
            // Associate the error with the response object and reject the promise.
            response.error_stream(e.clone(), can_gc);
            promise.reject_error(e, can_gc);
            return promise;
        },
        Ok(r) => {
            // Spec Step 3: Let request be requestObject’s request.
            r.get_request()
        },
    };
    let timing_type = request.timing_type();

    // Create a serializable request builder from the internal request object.
    let mut request_init = request_init_from_request(request);
    // The policy container from the global scope provides the security context (e.g., CSP, origin).
    request_init.policy_container =
        RequestPolicyContainer::PolicyContainer(global.policy_container());

    // TODO: Spec Step 4: Handle AbortSignal.

    // Spec Step 5: `globalObject` is the `global` parameter.

    // Spec Step 6: If in a ServiceWorker, disable SW interception for this fetch.
    if global.is::<ServiceWorkerGlobalScope>() {
        request_init.service_workers_mode = ServiceWorkersMode::None;
    }

    // TODO: Spec Steps 8-11: AbortController integration.

    // Spec Step 12: The core logic of dispatching the fetch and processing the response
    // is encapsulated within the `FetchContext` and its listener implementation.
    let fetch_context = Arc::new(Mutex::new(FetchContext {
        fetch_promise: Some(TrustedPromise::new(promise.clone())),
        response_object: Trusted::new(&*response),
        resource_timing: ResourceFetchTiming::new(timing_type),
    }));

    // Dispatch the request to the networking thread for asynchronous processing.
    global.fetch(
        request_init,
        fetch_context,
        global.task_manager().networking_task_source().to_sendable(),
    );

    // Spec Step 13: Return the promise to the caller.
    promise
}

impl PreInvoke for FetchContext {}

impl FetchResponseListener for FetchContext {
    /// Called by the network thread to process the request body. Currently a no-op.
    fn process_request_body(&mut self, _: RequestId) {
        // This would be used for features like streaming request bodies, which are not
        // fully implemented here.
    }

    /// Called by the network thread when the request body has been fully sent. No-op.
    fn process_request_eof(&mut self, _: RequestId) {
        // Marker for the completion of the upload phase.
    }

    /// Handles the initial part of the HTTP response, containing headers and status.
    /// This is a critical step where the promise can be resolved and the response stream begins.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    fn process_response(
        &mut self,
        _: RequestId,
        fetch_metadata: Result<FetchMetadata, NetworkError>,
    ) {
        // The promise must be taken to be resolved. It's an error if it's already gone.
        let promise = self
            .fetch_promise
            .take()
            .expect("fetch promise is missing")
            .root();

        // All JS interactions must happen within the correct realm.
        let _ac = enter_realm(&*promise);
        match fetch_metadata {
            // Spec Step 4.1: If the fetch results in a network error, reject the promise.
            Err(_) => {
                promise.reject_error(
                    Error::Type("Network error occurred".to_string()),
                    CanGc::note(),
                );
                // The promise is now settled, but we keep it to maintain consistent state.
                self.fetch_promise = Some(TrustedPromise::new(promise));
                let response = self.response_object.root();
                // Mark the response as an 'error' type, as per the spec.
                response.set_type(DOMResponseType::Error, CanGc::note());
                response.error_stream(
                    Error::Type("Network error occurred".to_string()),
                    CanGc::note(),
                );
                return;
            },
            // Spec Step 4.2: On a successful response, process the metadata.
            Ok(metadata) => match metadata {
                // An unfiltered response is a standard response.
                FetchMetadata::Unfiltered(m) => {
                    fill_headers_with_metadata(self.response_object.root(), m, CanGc::note());
                    self.response_object
                        .root()
                        .set_type(DOMResponseType::Default, CanGc::note());
                },
                // A filtered response has been modified due to security policies (e.g., CORS).
                FetchMetadata::Filtered { filtered, .. } => match filtered {
                    // A "basic" filtered response exposes a limited set of headers.
                    FilteredMetadata::Basic(m) => {
                        fill_headers_with_metadata(self.response_object.root(), m, CanGc::note());
                        self.response_object
                            .root()
                            .set_type(DOMResponseType::Basic, CanGc::note());
                    },
                    // A "cors" filtered response, from a successful CORS check.
                    FilteredMetadata::Cors(m) => {
                        fill_headers_with_metadata(self.response_object.root(), m, CanGc::note());
                        self.response_object
                            .root()
                            .set_type(DOMResponseType::Cors, CanGc::note());
                    },
                    // An "opaque" response from a no-cors request to a cross-origin resource.
                    // The body is readable, but status and headers are hidden.
                    FilteredMetadata::Opaque => {
                        self.response_object
                            .root()
                            .set_type(DOMResponseType::Opaque, CanGc::note());
                    },
                    // An opaque redirect occurs when a request is redirected to a different origin
                    // under a policy that requires the response to be opaque.
                    FilteredMetadata::OpaqueRedirect(url) => {
                        let r = self.response_object.root();
                        r.set_type(DOMResponseType::Opaqueredirect, CanGc::note());
                        r.set_final_url(url);
                    },
                },
            },
        }

        // Spec Step 4.3: The headers are processed, so we can now resolve the promise
        // with the Response object. The body will stream in separately.
        promise.resolve_native(&self.response_object.root(), CanGc::note());
        self.fetch_promise = Some(TrustedPromise::new(promise));
    }

    /// Receives a chunk of the response body from the network thread and appends it
    /// to the response object's internal stream.
    fn process_response_chunk(&mut self, _: RequestId, chunk: Vec<u8>) {
        let response = self.response_object.root();
        response.stream_chunk(chunk, CanGc::note());
    }

    /// Signifies the end of the response body stream. This finalizes the response object.
    fn process_response_eof(
        &mut self,
        _: RequestId,
        _response: Result<ResourceFetchTiming, NetworkError>,
    ) {
        let response = self.response_object.root();
        let _ac = enter_realm(&*response);
        // Mark the response stream as finished.
        response.finish(CanGc::note());
        // TODO: Handle trailers, which are headers sent after the response body.
        // ... trailerObject is not supported in Servo yet.
    }

    /// Provides mutable access to the resource timing data for this fetch.
    fn resource_timing_mut(&mut self) -> &mut ResourceFetchTiming {
        &mut self.resource_timing
    }

    /// Provides immutable access to the resource timing data.
    fn resource_timing(&self) -> &ResourceFetchTiming {
        &self.resource_timing
    }

    /// Submits the collected resource timing information to the performance timeline,
    /// making it accessible via `performance.getEntriesByType("resource")`.
    fn submit_resource_timing(&mut self) {
        // Navigation timing is handled separately, so we only submit for sub-resources.
        if self.resource_timing.timing_type == ResourceTimingType::Resource {
            network_listener::submit_timing(self, CanGc::note())
        }
    }

    /// Processes Content Security Policy (CSP) violations reported by the network layer
    /// for this fetch.
    fn process_csp_violations(&mut self, _request_id: RequestId, violations: Vec<csp::Violation>) {
        let global = &self.resource_timing_global();
        report_csp_violations(global, violations, None);
    }
}

impl ResourceTimingListener for FetchContext {
    /// Provides metadata required by the Performance API, including the initiator type
    /// (e.g., "fetch") and the final URL of the resource.
    fn resource_timing_information(&self) -> (InitiatorType, ServoUrl) {
        (
            InitiatorType::Fetch,
            self.resource_timing_global().get_url().clone(),
        )
    }

    /// Returns the `GlobalScope` associated with this fetch, which provides the
    /// necessary context for reporting performance data.
    fn resource_timing_global(&self) -> DomRoot<GlobalScope> {
        self.response_object.root().global()
    }
}

/// A utility function to populate the `Response` object's properties from the
/// `Metadata` received from the network layer.
fn fill_headers_with_metadata(r: DomRoot<Response>, m: Metadata, can_gc: CanGc) {
    r.set_headers(m.headers, can_gc);
    r.set_status(&m.status);
    r.set_final_url(m.final_url);
    r.set_redirected(m.redirected);
}

/// A synchronous, blocking convenience function to load an entire resource into memory.
/// This is used for internal operations that require the full resource content upfront,
/// such as loading a script or stylesheet, and is not suitable for large resources.
/// It bypasses the promise-based asynchronous flow of the main Fetch API.
pub(crate) fn load_whole_resource(
    request: RequestBuilder,
    core_resource_thread: &CoreResourceThread,
    global: &GlobalScope,
    can_gc: CanGc,
) -> Result<(Metadata, Vec<u8>), NetworkError> {
    // Ensure the request has the correct HTTPS state from its global context.
    let request = request.https_state(global.get_https_state());
    // Create an IPC channel to receive the response from the networking thread.
    let (action_sender, action_receiver) = ipc::channel().unwrap();
    let url = request.url.clone();
    // Send the fetch request to the core resource thread.
    core_resource_thread
        .send(CoreResourceMsg::Fetch(
            request,
            FetchChannels::ResponseMsg(action_sender),
        ))
        .unwrap();

    let mut buf = vec![];
    let mut metadata = None;
    // Block and loop, processing messages from the network thread until the
    // response is complete (EOF) or an error occurs.
    loop {
        match action_receiver.recv().unwrap() {
            FetchResponseMsg::ProcessRequestBody(..) |
            FetchResponseMsg::ProcessRequestEOF(..) |
            FetchResponseMsg::ProcessCspViolations(..) => {},
            FetchResponseMsg::ProcessResponse(_, Ok(m)) => {
                // Store the metadata when the headers are received.
                metadata = Some(match m {
                    FetchMetadata::Unfiltered(m) => m,
                    FetchMetadata::Filtered { unsafe_, .. } => unsafe_,
                })
            },
            // Append incoming data chunks to the buffer.
            FetchResponseMsg::ProcessResponseChunk(_, data) => buf.extend_from_slice(&data),
            // On end-of-file, submit timing data and return the complete resource.
            FetchResponseMsg::ProcessResponseEOF(_, Ok(_)) => {
                let metadata = metadata.unwrap();
                if let Some(timing) = &metadata.timing {
                    submit_timing_data(global, url, InitiatorType::Other, timing, can_gc);
                }
                return Ok((metadata, buf));
            },
            // Propagate any network errors.
            FetchResponseMsg::ProcessResponse(_, Err(e)) |
            FetchResponseMsg::ProcessResponseEOF(_, Err(e)) => return Err(e),
        }
    }
}

/// Creates a request that may be subject to CORS checks, as defined by the HTML spec.
/// This function configures the request's mode (`cors`, `no-cors`, `same-origin`) and
/// credentials mode based on the context in which a resource is being fetched.
/// <https://html.spec.whatwg.org/multipage/#create-a-potential-cors-request>
#[allow(clippy::too_many_arguments)]
pub(crate) fn create_a_potential_cors_request(
    webview_id: Option<WebViewId>,
    url: ServoUrl,
    destination: Destination,
    cors_setting: Option<CorsSettings>,
    same_origin_fallback: Option<bool>,
    referrer: Referrer,
    insecure_requests_policy: InsecureRequestsPolicy,
    has_trustworthy_ancestor_origin: bool,
    policy_container: PolicyContainer,
) -> RequestBuilder {
    RequestBuilder::new(webview_id, url, referrer)
        // Spec Step 1: Set request's mode.
        .mode(match cors_setting {
            Some(_) => RequestMode::CorsMode,
            None if same_origin_fallback == Some(true) => RequestMode::SameOrigin,
            None => RequestMode::NoCors,
        })
        // Spec Steps 3-4: Set request's credentials mode.
        // For "anonymous", credentials are only sent for same-origin requests.
        // For "use-credentials" or if not specified, they are included.
        .credentials_mode(match cors_setting {
            Some(CorsSettings::Anonymous) => CredentialsMode::CredentialsSameOrigin,
            _ => CredentialsMode::Include,
        })
        // Spec Step 5: Set request's destination.
        .destination(destination)
        .use_url_credentials(true)
        .insecure_requests_policy(insecure_requests_policy)
        .has_trustworthy_ancestor_origin(has_trustworthy_ancestor_origin)
        .policy_container(policy_container)
}
