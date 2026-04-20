/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @29605d50-eae7-487b-827b-d8354a12e296/components/script/dom/csp.rs
 * @brief Logic for enforcing and reporting Content Security Policy (CSP) in the script thread.
 * 
 * Functional Intent: Provides the runtime validation layer for CSP directives. It 
 * checks script evaluations, WASM instantiation, navigation requests, and inline 
 * content against the document's active policy list. On failure, it orchestrates 
 * the generation and dispatch of violation reports to the security manager.
 * 
 * Domain: Browser Security, Content Security Policy, Trusted Types.
 */

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
use crate::dom::globalscope::GlobalScope;
use crate::dom::node::{Node, NodeTraits};
use crate::dom::window::Window;
use crate::security_manager::CSPViolationReportTask;

/// is_js_evaluation_allowed - Enforces 'script-src' and 'unsafe-eval' constraints.
/// Corresponds to https://www.w3.org/TR/CSP/#can-compile-strings
pub(crate) fn is_js_evaluation_allowed(global: &GlobalScope, source: &str) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        return true;
    };

    let (is_js_evaluation_allowed, violations) = csp_list.is_js_evaluation_allowed(source);

    // Block Logic: Automated reporting of illegal evaluation attempts.
    report_csp_violations(global, violations, None);

    is_js_evaluation_allowed == CheckResult::Allowed
}

/// is_wasm_evaluation_allowed - Enforces WASM-specific CSP constraints.
/// Corresponds to https://www.w3.org/TR/CSP/#can-compile-wasm-bytes
pub(crate) fn is_wasm_evaluation_allowed(global: &GlobalScope) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        return true;
    };

    let (is_wasm_evaluation_allowed, violations) = csp_list.is_wasm_evaluation_allowed();

    report_csp_violations(global, violations, None);

    is_wasm_evaluation_allowed == CheckResult::Allowed
}

/// should_navigation_request_be_blocked - Validates outgoing navigation via 'navigate-to' or 'frame-src'.
/// Corresponds to https://www.w3.org/TR/CSP/#should-block-navigation-request
pub(crate) fn should_navigation_request_be_blocked(
    global: &GlobalScope,
    load_data: &LoadData,
    element: Option<&Element>,
) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        return false;
    };
    
    // Block Logic: Request context assembly.
    // Logic: Maps engine-internal LoadData to a normalized CSP Request structure.
    let request = Request {
        url: load_data.url.clone().into_url(),
        origin: match &load_data.load_origin {
            LoadOrigin::Script(immutable_origin) => immutable_origin.clone().into_url_origin(),
            _ => Origin::new_opaque(),
        },
        redirect_count: 0,
        destination: Destination::None,
        initiator: Initiator::None,
        nonce: "".to_owned(),
        integrity_metadata: "".to_owned(),
        parser_metadata: ParserMetadata::None,
    };
    
    let (result, violations) =
        csp_list.should_navigation_request_be_blocked(&request, NavigationCheckType::Other);

    report_csp_violations(global, violations, element);

    result == CheckResult::Blocked
}

pub use content_security_policy::InlineCheckType;

/// should_elements_inline_type_behavior_be_blocked - Enforces nonces/hashes for inline content.
/// Corresponds to https://www.w3.org/TR/CSP/#should-block-inline
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
                nonce: el.nonce_value_if_nonceable().map(Cow::Owned),
            };
            csp_list.should_elements_inline_type_behavior_be_blocked(&element, type_, source)
        },
    };

    report_csp_violations(global, violations, Some(el));

    result == CheckResult::Blocked
}

/// is_trusted_type_policy_creation_allowed - Validates Trusted Type policy names against 'trusted-types' directive.
/// Corresponds to https://w3c.github.io/trusted-types/dist/spec/#should-block-create-policy
pub(crate) fn is_trusted_type_policy_creation_allowed(
    global: &GlobalScope,
    policy_name: String,
    created_policy_names: Vec<String>,
) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        return false;
    };

    let (allowed_by_csp, violations) =
        csp_list.is_trusted_type_policy_creation_allowed(policy_name, created_policy_names);

    report_csp_violations(global, violations, None);

    allowed_by_csp == CheckResult::Allowed
}

/// does_sink_type_require_trusted_types - Checks if a specific sink (e.g., innerHTML) is locked down.
/// Corresponds to https://w3c.github.io/trusted-types/dist/spec/#abstract-opdef-does-sink-type-require-trusted-types
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

/// should_sink_type_mismatch_violation_be_blocked_by_csp - Enforces use of Trusted Types for specific sinks.
/// Corresponds to https://w3c.github.io/trusted-types/dist/spec/#should-block-sink-type-mismatch
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

pub use content_security_policy::Violation;

/// report_csp_violations - Orchestrates the asynchronous reporting of policy breaches.
/// Corresponds to https://www.w3.org/TR/CSP/#report-violation
#[allow(unsafe_code)]
pub(crate) fn report_csp_violations(
    global: &GlobalScope,
    violations: Vec<Violation>,
    element: Option<&Element>,
) {
    let scripted_caller =
        unsafe { describe_scripted_caller(*GlobalScope::get_cx()) }.unwrap_or_default();
    
    // Block Logic: Violation iteration and reporting.
    // Invariant: For each violation, a distinct report is generated and queued 
    // on the DOM manipulation task source to avoid blocking the current script.
    for violation in violations {
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

        // Block Logic: Target resolution for event dispatch.
        // Logic: Determines the reporting target (Document vs Global) based on 
        // whether the violating element is connected to the active document.
        let target = element.and_then(|event_target| {
            if let Some(window) = global.downcast::<Window>() {
                if event_target.upcast::<Node>().owner_document() != window.Document() {
                    return None;
                }
            }
            Some(event_target)
        });
        let target = match target {
            None => {
                if let Some(window) = global.downcast::<Window>() {
                    Trusted::new(window.Document().upcast())
                } else {
                    Trusted::new(global.upcast())
                }
            },
            Some(event_target) => Trusted::new(event_target.upcast()),
        };

        // Task Scheduling: Offloads violation processing to the main task manager.
        let task =
            CSPViolationReportTask::new(Trusted::new(global), target, report, violation.policy);
        global
            .task_manager()
            .dom_manipulation_task_source()
            .queue(task);
    }
}

/// parse_csp_list_from_metadata - Ingests CSP directives from raw HTTP headers.
/// Corresponds to https://www.w3.org/TR/CSP/#initialize-document-csp
pub(crate) fn parse_csp_list_from_metadata(headers: &Option<Serde<HeaderMap>>) -> Option<CspList> {
    let headers = headers.as_ref()?;
    
    // Block Logic: Multi-header aggregation.
    // Logic: Collects all 'Content-Security-Policy' and '-Report-Only' headers, 
    // merging them into a single ordered CspList for combined enforcement.
    let mut csp = headers.get_all("content-security-policy").iter();
    let c = csp.next().and_then(|c| c.to_str().ok())?;
    let mut csp_list = CspList::parse(c, PolicySource::Header, PolicyDisposition::Enforce);
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
