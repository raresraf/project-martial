//! This module handles the reporting of Content Security Policy (CSP) violations.
//! It defines the logic for creating and dispatching violation reports, both by
//! firing a `SecurityPolicyViolationEvent` and by sending a report to a URI
//! specified in the CSP. This is a core part of the browser's security model,
//! ensuring that policy violations are auditable and can be collected for analysis.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::sync::{Arc, Mutex};

use content_security_policy as csp;
use headers::{ContentType, HeaderMap, HeaderMapExt};
use net_traits::request::{
    CredentialsMode, RequestBody, RequestId, create_request_body_with_content,
};
use net_traits::{
    FetchMetadata, FetchResponseListener, NetworkError, ResourceFetchTiming, ResourceTimingType,
};
use servo_url::ServoUrl;
use stylo_atoms::Atom;

use crate::conversions::Convert;
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::root::DomRoot;
use crate::dom::csp::report_csp_violations;
use crate::dom::csppolicyviolationreport::{
    CSPReportUriViolationReport, SecurityPolicyViolationReport,
};
use crate::dom::event::{Event, EventBubbles, EventCancelable, EventComposed};
use crate::dom::eventtarget::EventTarget;
use crate::dom::performanceresourcetiming::InitiatorType;
use crate::dom::securitypolicyviolationevent::SecurityPolicyViolationEvent;
use crate::dom::types::GlobalScope;
use crate::fetch::create_a_potential_cors_request;
use crate::network_listener::{PreInvoke, ResourceTimingListener, submit_timing};
use crate::script_runtime::CanGc;
use crate::task::TaskOnce;

/// A task that handles the reporting of a single CSP violation.
/// This includes firing a local event and sending a report to a remote endpoint.
pub(crate) struct CSPViolationReportTask {
    /// The global scope in which the violation occurred.
    global: Trusted<GlobalScope>,
    /// The target for the violation event.
    event_target: Trusted<EventTarget>,
    /// The detailed information about the violation.
    violation_report: SecurityPolicyViolationReport,
    /// The policy that was violated.
    violation_policy: csp::Policy,
}

impl CSPViolationReportTask {
    /// Creates a new task to report a CSP violation.
    pub fn new(
        global: Trusted<GlobalScope>,
        event_target: Trusted<EventTarget>,
        violation_report: SecurityPolicyViolationReport,
        violation_policy: csp::Policy,
    ) -> CSPViolationReportTask {
        CSPViolationReportTask {
            global,
            event_target,
            violation_report,
            violation_policy,
        }
    }

    /// Fires a `SecurityPolicyViolationEvent` at the designated event target.
    /// This makes the violation observable to scripts in the page.
    fn fire_violation_event(&self, can_gc: CanGc) {
        let event = SecurityPolicyViolationEvent::new(
            &self.global.root(),
            Atom::from("securitypolicyviolation"),
            EventBubbles::Bubbles,
            EventCancelable::NotCancelable,
            EventComposed::Composed,
            &self.violation_report.clone().convert(),
            can_gc,
        );

        event
            .upcast::<Event>()
            .fire(&self.event_target.root(), can_gc);
    }

    /// Serializes the violation report into a JSON format suitable for sending
    /// to a reporting endpoint, as defined by the CSP specification.
    /// See: <https://www.w3.org/TR/CSP/#deprecated-serialize-violation>
    fn serialize_violation(&self) -> Option<RequestBody> {
        let report_body = CSPReportUriViolationReport {
            // Steps 1-3 from the specification.
            csp_report: self.violation_report.clone().into(),
        };
        // Step 4. Return the result of serializing to JSON.
        Some(create_request_body_with_content(
            &serde_json::to_string(&report_body).unwrap_or("".to_owned()),
        ))
    }

    /// Sends the serialized CSP violation report to the endpoints specified
    /// in the `report-uri` directive of the policy.
    /// See: Step 3.4 of <https://www.w3.org/TR/CSP/#report-violation>
    fn post_csp_violation_to_report_uri(&self, report_uri_directive: &csp::Directive) {
        let global = self.global.root();
        // Step 3.4.1. If a "report-to" directive exists, it takes precedence.
        if self
            .violation_policy
            .contains_a_directive_whose_name_is("report-to")
        {
            return;
        }
        // Step 3.4.2. For each token in the directive's value:
        for token in &report_uri_directive.value {
            // Step 3.4.2.1. Parse the token as a URL.
            let Ok(endpoint) = ServoUrl::parse_with_base(Some(&global.get_url()), token) else {
                // Step 3.4.2.2. If parsing fails, skip this token.
                continue;
            };
            // Step 3.4.2.3. Create a new POST request for the report.
            let mut headers = HeaderMap::with_capacity(1);
            headers.typed_insert(ContentType::from(
                "application/csp-report".parse::<mime::Mime>().unwrap(),
            ));
            let request_body = self.serialize_violation();
            let request = create_a_potential_cors_request(
                None,
                endpoint.clone(),
                csp::Destination::Report,
                None,
                None,
                global.get_referrer(),
                global.insecure_requests_policy(),
                global.has_trustworthy_ancestor_or_current_origin(),
                global.policy_container(),
            )
            .method(http::Method::POST)
            .body(request_body)
            .origin(global.origin().immutable().clone())
            .credentials_mode(CredentialsMode::CredentialsSameOrigin)
            .headers(headers);
            // Step 3.4.2.4. Fetch the request, ignoring the result.
            global.fetch(
                request,
                Arc::new(Mutex::new(CSPReportUriFetchListener {
                    endpoint,
                    global: Trusted::new(&global),
                    resource_timing: ResourceFetchTiming::new(ResourceTimingType::None),
                })),
                global.task_manager().networking_task_source().into(),
            );
        }
    }
}

/// Implements the main logic for reporting a CSP violation, as specified in:
/// <https://w3c.github.io/webappsec-csp/#report-violation>
/// This task is queued to run asynchronously.
impl TaskOnce for CSPViolationReportTask {
    fn run_once(self) {
        // Fire the `securitypolicyviolation` event at the target.
        self.fire_violation_event(CanGc::note());
        // Step 3.4. If a "report-uri" directive exists, send the report.
        if let Some(report_uri_directive) = self
            .violation_policy
            .directive_set
            .iter()
            .find(|directive| directive.name == "report-uri")
        {
            self.post_csp_violation_to_report_uri(report_uri_directive);
        }
    }
}

/// Listener for the network fetch of a CSP violation report.
/// This is primarily used to gather resource timing information for the report itself.
struct CSPReportUriFetchListener {
    /// The endpoint URL to which the report is being sent.
    endpoint: ServoUrl,
    /// Timing data for this resource fetch.
    resource_timing: ResourceFetchTiming,
    /// The global scope from which the report is being sent.
    global: Trusted<GlobalScope>,
}

impl FetchResponseListener for CSPReportUriFetchListener {
    fn process_request_body(&mut self, _: RequestId) {}

    fn process_request_eof(&mut self, _: RequestId) {}

    fn process_response(
        &mut self,
        _: RequestId,
        fetch_metadata: Result<FetchMetadata, NetworkError>,
    ) {
        // The response to a CSP report is typically ignored.
        _ = fetch_metadata;
    }

    fn process_response_chunk(&mut self, _: RequestId, chunk: Vec<u8>) {
        // The response body is ignored.
        _ = chunk;
    }

    fn process_response_eof(
        &mut self,
        _: RequestId,
        response: Result<ResourceFetchTiming, NetworkError>,
    ) {
        // The final timing information is processed, but the result itself is not used further.
        _ = response;
    }

    fn resource_timing_mut(&mut self) -> &mut ResourceFetchTiming {
        &mut self.resource_timing
    }

    fn resource_timing(&self) -> &ResourceFetchTiming {
        &self.resource_timing
    }

    fn submit_resource_timing(&mut self) {
        submit_timing(self, CanGc::note())
    }

    fn process_csp_violations(&mut self, _request_id: RequestId, violations: Vec<csp::Violation>) {
        // Recursively report any CSP violations that occur while sending a CSP report.
        let global = &self.resource_timing_global();
        report_csp_violations(global, violations, None);
    }
}

impl ResourceTimingListener for CSPReportUriFetchListener {
    /// Provides the initiator type and URL for resource timing entries.
    fn resource_timing_information(&self) -> (InitiatorType, ServoUrl) {
        (InitiatorType::Other, self.endpoint.clone())
    }

    /// Returns the global scope associated with this fetch.
    fn resource_timing_global(&self) -> DomRoot<GlobalScope> {
        self.global.root()
    }
}

impl PreInvoke for CSPReportUriFetchListener {
    /// Determines whether the listener's methods should be invoked.
    /// For CSP reports, this is always true.
    fn should_invoke(&self) -> bool {
        true
    }
}
