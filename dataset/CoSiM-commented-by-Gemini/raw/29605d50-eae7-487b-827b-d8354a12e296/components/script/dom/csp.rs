//! This module provides the integration layer between the browser's DOM and the
//! core Content Security Policy (CSP) engine. It implements the algorithms
//! defined in the CSP specification by checking actions against the policies
//! associated with a given global scope (e.g., a Window or Worker).

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::borrow::Cow;

use constellation_traits::{LoadData, LoadOrigin};
use content_security_policy::{
    CheckResult, CspList, Destination, Element as CspElement, Initiator, NavigationCheckType,
    Origin, ParserMetadata, PolicyDisposition, PolicySource, Request, ViolationResource,
};
use http::HeaderMap;
use hyper_serde::Serde;
use js::rust::describe_scripted_caller;

use crate::dom::bindings::codegen::Bindings::WindowBinding::WindowMethods;
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::csppolicyviolationreport::CSPViolationReportBuilder;
use crate::dom::element::Element;
use crate.dom::globalscope::GlobalScope;
use crate::dom::node::{Node, NodeTraits};
use crate::dom::window::Window;
use crate::security_manager::CSPViolationReportTask;

/// Checks if evaluating a JavaScript string (e.g., with `eval` or `setTimeout`)
/// is permitted by the active Content Security Policy.
///
/// <https://www.w3.org/TR/CSP/#can-compile-strings>
pub(crate) fn is_js_evaluation_allowed(global: &GlobalScope, source: &str) -> bool {
    // If there is no CSP, all evaluations are allowed.
    let Some(csp_list) = global.get_csp_list() else {
        return true;
    };

    let (is_js_evaluation_allowed, violations) = csp_list.is_js_evaluation_allowed(source);

    report_csp_violations(global, violations, None);

    is_js_evaluation_allowed == CheckResult::Allowed
}

/// Checks if compiling and instantiating WebAssembly from bytes is permitted
/// by the active Content Security Policy.
///
/// <https://www.w3.org/TR/CSP/#can-compile-wasm-bytes>
pub(crate) fn is_wasm_evaluation_allowed(global: &GlobalScope) -> bool {
    // If there is no CSP, all evaluations are allowed.
    let Some(csp_list) = global.get_csp_list() else {
        return true;
    };

    let (is_wasm_evaluation_allowed, violations) = csp_list.is_wasm_evaluation_allowed();

    report_csp_violations(global, violations, None);

    is_wasm_evaluation_allowed == CheckResult::Allowed
}

/// Determines if a navigation request should be blocked based on the document's CSP.
/// This is used to check navigations initiated by links, scripts, or form submissions.
///
/// <https://www.w3.org/TR/CSP/#should-block-navigation-request>
pub(crate) fn should_navigation_request_be_blocked(
    global: &GlobalScope,
    load_data: &LoadData,
    element: Option<&Element>,
) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        return false;
    };
    // Construct a CSP `Request` object from the browser's `LoadData`.
    let request = Request {
        url: load_data.url.clone().into_url(),
        origin: match &load_data.load_origin {
            LoadOrigin::Script(immutable_origin) => immutable_origin.clone().into_url_origin(),
            _ => Origin::new_opaque(),
        },
        // TODO: populate this field correctly
        redirect_count: 0,
        destination: Destination::None,
        initiator: Initiator::None,
        nonce: "".to_owned(),
        integrity_metadata: "".to_owned(),
        parser_metadata: ParserMetadata::None,
    };
    // TODO: set correct navigation check type for form submission if applicable
    let (result, violations) =
        csp_list.should_navigation_request_be_blocked(&request, NavigationCheckType::Other);

    report_csp_violations(global, violations, element);

    result == CheckResult::Blocked
}

/// Re-export from the core CSP crate for use in inline checks.
pub use content_security_policy::InlineCheckType;

/// Checks if an element's inline behavior (e.g., an inline script or style attribute)
/// should be blocked by the active CSP.
///
/// <https://www.w3.org/TR/CSP/#should-block-inline>
pub(crate) fn should_elements_inline_type_behavior_be_blocked(
    global: &GlobalScope,
    el: &Element,
    type_: InlineCheckType,
    source: &str,
) -> bool {
    let (result, violations) = match global.get_csp_list() {
        None => {
            return false;
        },
        Some(csp_list) => {
            let element = CspElement {
                // The element's nonce is provided to the CSP engine to check against `nonce-` directives.
                nonce: el.nonce_value_if_nonceable().map(Cow::Owned),
            };
            csp_list.should_elements_inline_type_behavior_be_blocked(&element, type_, source)
        },
    };

    report_csp_violations(global, violations, Some(el));

    result == CheckResult::Blocked
}

/// Checks if the creation of a Trusted Types policy is allowed by the `trusted-types`
/// directive in the Content Security Policy.
///
/// <https://w3c.github.io/trusted-types/dist/spec/#should-block-create-policy>
pub(crate) fn is_trusted_type_policy_creation_allowed(
    global: &GlobalScope,
    policy_name: String,
    created_policy_names: Vec<String>,
) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        // If no CSP, creation is not allowed (Trusted Types requires an explicit policy).
        return false;
    };

    let (allowed_by_csp, violations) =
        csp_list.is_trusted_type_policy_creation_allowed(policy_name, created_policy_names);

    report_csp_violations(global, violations, None);

    allowed_by_csp == CheckResult::Allowed
}

/// Checks if a specific DOM sink (like `innerHTML`) requires a Trusted Type object
/// according to the `require-trusted-types-for` directive.
///
/// <https://w3c.github.io/trusted-types/dist/spec/#abstract-opdef-does-sink-type-require-trusted-types>
pub(crate) fn does_sink_type_require_trusted_types(
    global: &GlobalScope,
    sink_group: &str,
    include_report_only_policies: bool,
) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        return false;
    };

    csp_list.does_sink_type_require_trusted_types(sink_group, include_report_only_policies)
}

/// Checks if a Trusted Type violation (i.e., passing a string to a sink that
/// requires a Trusted Type) should be blocked by the CSP.
///
/// <https://w3c.github.io/trusted-types/dist/spec/#should-block-sink-type-mismatch>
pub(crate) fn should_sink_type_mismatch_violation_be_blocked_by_csp(
    global: &GlobalScope,
    sink: &str,
    sink_group: &str,
    source: &str,
) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        return false;
    };

    let (allowed_by_csp, violations) =
        csp_list.should_sink_type_mismatch_violation_be_blocked_by_csp(sink, sink_group, source);

    report_csp_violations(global, violations, None);

    allowed_by_csp == CheckResult::Blocked
}

/// Re-export from the core CSP crate.
pub use content_security_policy::Violation;

/// Reports a set of CSP violations by creating `SecurityPolicyViolationEvent` instances
/// and dispatching them on the appropriate target.
///
/// <https://www.w3.org/TR/CSP/#report-violation>
#[allow(unsafe_code)]
pub(crate) fn report_csp_violations(
    global: &GlobalScope,
    violations: Vec<Violation>,
    element: Option<&Element>,
) {
    // Attempt to get the source file and line number of the script that caused the violation.
    let scripted_caller =
        unsafe { describe_scripted_caller(*GlobalScope::get_cx()) }.unwrap_or_default();
    for violation in violations {
        // Match on the type of violation to extract the resource and a sample.
        let (sample, resource) = match violation.resource {
            ViolationResource::Inline { sample } => (sample, "inline".to_owned()),
            ViolationResource::Url(url) => (None, url.into()),
            ViolationResource::TrustedTypePolicy { sample } => {
                (Some(sample), "trusted-types-policy".to_owned())
            },
            ViolationResource::TrustedTypeSink { sample } => {
                (Some(sample), "trusted-types-sink".to_owned())
            },
            ViolationResource::Eval { sample } => (sample, "eval".to_owned()),
            ViolationResource::WasmEval => (None, "wasm-eval".to_owned()),
        };
        // Construct the violation report object that will be exposed on the event.
        let report = CSPViolationReportBuilder::default()
            .resource(resource)
            .sample(sample)
            .effective_directive(violation.directive.name)
            .original_policy(violation.policy.to_string())
            .report_only(violation.policy.disposition == PolicyDisposition::Report)
            .source_file(scripted_caller.filename.clone())
            .line_number(scripted_caller.line)
            .column_number(scripted_caller.col + 1)
            .build(global);

        // Algorithm to determine the event target, as specified by the spec.
        // Step 1: Let global be violation’s global object.
        // We use `self` as `global`.
        // Step 2: Let target be violation’s element.
        let target = element.and_then(|event_target| {
            // Step 3.1: If target is not null, and global is a Window,
            // and target’s shadow-including root is not global’s associated Document, set target to null.
            if let Some(window) = global.downcast::<Window>() {
                if event_target.upcast::<Node>().owner_document() != window.Document() {
                    return None;
                }
            }
            Some(event_target)
        });
        let target = match target {
            // Step 3.2: If target is null, determine the target based on the global scope.
            None => {
                // Step 3.2.2: If the global is a Window, the target is its associated Document.
                if let Some(window) = global.downcast::<Window>() {
                    Trusted::new(window.Document().upcast())
                } else {
                    // Step 3.2.1: Otherwise, the target is the global object itself (e.g., a Worker).
                    Trusted::new(global.upcast())
                }
            },
            Some(event_target) => Trusted::new(event_target.upcast()),
        };
        // Step 3: Queue a task to create and dispatch the violation event.
        let task =
            CSPViolationReportTask::new(Trusted::new(global), target, report, violation.policy);
        global
            .task_manager()
            .dom_manipulation_task_source()
            .queue(task);
    }
}

/// Parses CSP headers from HTTP metadata to initialize a document's `CspList`.
///
/// <https://www.w3.org/TR/CSP/#initialize-document-csp>
pub(crate) fn parse_csp_list_from_metadata(headers: &Option<Serde<HeaderMap>>) -> Option<CspList> {
    // TODO: Implement step 1 (local scheme special case)
    let headers = headers.as_ref()?;
    let mut csp = headers.get_all("content-security-policy").iter();
    // This silently ignores the CSP if it contains invalid Unicode.
    // We should probably report an error somewhere.
    let c = csp.next().and_then(|c| c.to_str().ok())?;
    // Parse the first `Content-Security-Policy` header.
    let mut csp_list = CspList::parse(c, PolicySource::Header, PolicyDisposition::Enforce);
    // Append any subsequent `Content-Security-Policy` headers.
    for c in csp {
        let c = c.to_str().ok()?;
        csp_list.append(CspList::parse(
            c,
            PolicySource::Header,
            PolicyDisposition::Enforce,
        ));
    }
    let csp_report = headers
        .get_all("content-security-policy-report-only")
        .iter();
    // Append all `Content-Security-Policy-Report-Only` headers.
    // This silently ignores the CSP if it contains invalid Unicode.
    // We should probably report an error somewhere.
    for c in csp_report {
        let c = c.to_str().ok()?;
        csp_list.append(CspList::parse(
            c,
            PolicySource::Header,
            PolicyDisposition::Report,
        ));
    }
    Some(csp_list)
}
