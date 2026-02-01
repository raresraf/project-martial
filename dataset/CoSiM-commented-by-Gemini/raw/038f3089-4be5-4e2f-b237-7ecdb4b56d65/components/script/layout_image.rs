//! This module provides the infrastructure to initiate network requests for images required by the layout engine.
//! The script thread is responsible for managing these requests to ensure that they can be handled
//! asynchronously, even if the DOM nodes that triggered them are no longer present. This is crucial
//! for maintaining a responsive and non-blocking rendering pipeline.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::sync::Arc;

use content_security_policy as csp;
use net_traits::image_cache::{ImageCache, PendingImageId};
use net_traits::request::{Destination, RequestBuilder as FetchRequestInit, RequestId};
use net_traits::{
    FetchMetadata, FetchResponseListener, FetchResponseMsg, NetworkError, ResourceFetchTiming,
    ResourceTimingType,
};
use servo_url::ServoUrl;

use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::root::DomRoot;
use crate::dom::csp::report_csp_violations;
use crate::dom::document::Document;
use crate::dom::globalscope::GlobalScope;
use crate::dom::node::{Node, NodeTraits};
use crate::dom::performanceresourcetiming::InitiatorType;
use crate::network_listener::{self, PreInvoke, ResourceTimingListener};
use crate::script_runtime::CanGc;

/// Holds the context for an image fetch operation initiated by layout.
/// This includes the pending image ID, a reference to the image cache,
/// resource timing information, and the document that initiated the request.
struct LayoutImageContext {
    /// The unique identifier for the pending image request.
    id: PendingImageId,
    /// A thread-safe reference to the image cache.
    cache: Arc<dyn ImageCache>,
    /// Stores performance timing data for the resource fetch.
    resource_timing: ResourceFetchTiming,
    /// A trusted reference to the document that owns the image.
    doc: Trusted<Document>,
    /// The URL of the image being fetched.
    url: ServoUrl,
}

/// Implements the `FetchResponseListener` trait to handle network events
/// related to the image fetch. It forwards response messages to the image cache.
impl FetchResponseListener for LayoutImageContext {
    fn process_request_body(&mut self, _: RequestId) {}
    fn process_request_eof(&mut self, _: RequestId) {}

    /// Called when the response headers and metadata are received.
    fn process_response(
        &mut self,
        request_id: RequestId,
        metadata: Result<FetchMetadata, NetworkError>,
    ) {
        self.cache.notify_pending_response(
            self.id,
            FetchResponseMsg::ProcessResponse(request_id, metadata),
        );
    }

    /// Called when a chunk of the response body is received.
    fn process_response_chunk(&mut self, request_id: RequestId, payload: Vec<u8>) {
        self.cache.notify_pending_response(
            self.id,
            FetchResponseMsg::ProcessResponseChunk(request_id, payload),
        );
    }

    /// Called when the response is fully received or an error occurs.
    fn process_response_eof(
        &mut self,
        request_id: RequestId,
        response: Result<ResourceFetchTiming, NetworkError>,
    ) {
        self.cache.notify_pending_response(
            self.id,
            FetchResponseMsg::ProcessResponseEOF(request_id, response),
        );
    }

    fn resource_timing_mut(&mut self) -> &mut ResourceFetchTiming {
        &mut self.resource_timing
    }

    fn resource_timing(&self) -> &mut ResourceFetchTiming {
        &mut self.resource_timing
    }

    /// Submits the collected resource timing information to the performance timeline.
    fn submit_resource_timing(&mut self) {
        network_listener::submit_timing(self, CanGc::note())
    }

    /// Processes any CSP violations that occurred during the fetch.
    fn process_csp_violations(&mut self, _request_id: RequestId, violations: Vec<csp::Violation>) {
        let global = &self.resource_timing_global();
        report_csp_violations(global, violations, None);
    }
}

/// Implements the `ResourceTimingListener` trait to provide the necessary
/// information for creating a performance resource timing entry.
impl ResourceTimingListener for LayoutImageContext {
    /// Returns the initiator type and the URL of the resource.
    fn resource_timing_information(&self) -> (InitiatorType, ServoUrl) {
        (InitiatorType::Other, self.url.clone())
    }

    /// Returns the global scope associated with this fetch.
    fn resource_timing_global(&self) -> DomRoot<GlobalScope> {
        self.doc.root().global()
    }
}

/// A marker trait implementation indicating that this listener's methods
/// should always be invoked.
impl PreInvoke for LayoutImageContext {}

/// Initiates a network request to fetch an image required for layout.
///
/// # Arguments
///
/// * `url` - The URL of the image to fetch.
/// * `node` - The DOM node that requires the image.
/// * `id` - The pending image ID assigned by the image cache.
/// * `cache` - A reference to the image cache that will store the image.
pub(crate) fn fetch_image_for_layout(
    url: ServoUrl,
    node: &Node,
    id: PendingImageId,
    cache: Arc<dyn ImageCache>,
) {
    let document = node.owner_document();
    let context = LayoutImageContext {
        id,
        cache,
        resource_timing: ResourceFetchTiming::new(ResourceTimingType::Resource),
        doc: Trusted::new(&document),
        url: url.clone(),
    };

    let request = FetchRequestInit::new(
        Some(document.webview_id()),
        url,
        document.global().get_referrer(),
    )
    .origin(document.origin().immutable().clone())
    .destination(Destination::Image)
    .pipeline_id(Some(document.global().pipeline_id()))
    .insecure_requests_policy(document.insecure_requests_policy())
    .has_trustworthy_ancestor_origin(document.has_trustworthy_ancestor_origin())
    .policy_container(document.policy_container().to_owned());

    // Layout image loads are performed as background fetches so as not to delay the document load event.
    document.fetch_background(request, context);
}
