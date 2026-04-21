/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file csp.rs
 * @brief Integration layer for Content Security Policy (CSP) enforcement in the DOM.
 * 
 * Functional Intent: Provides high-level interfaces for the DOM to validate 
 * operations (JS/Wasm evaluation, navigation, inline scripts) against active 
 * security policies and orchestrates standardized violation reporting.
 * 
 * Domain: Production Systems, Web Security, Browser Engine (Servo).
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

/// <https://www.w3.org/TR/CSP/#can-compile-strings>
/// @brief Determines if execution of JavaScript from strings (eval) is permitted.
pub(crate) fn is_js_evaluation_allowed(global: &GlobalScope, source: &str) -> bool {
    // Logic: If no CSP is present, default to allowing execution.
    let Some(csp_list) = global.get_csp_list() else {
        return true;
    };

    let (is_js_evaluation_allowed, violations) = csp_list.is_js_evaluation_allowed(source);

    // Side Effect: Dispatches reports for any policy infractions detected during check.
    report_csp_violations(global, violations, None);

    is_js_evaluation_allowed == CheckResult::Allowed
}

/// <https://www.w3.org/TR/CSP/#can-compile-wasm-bytes>
/// @brief Validates if WebAssembly compilation/instantiation is allowed.
pub(crate) fn is_wasm_evaluation_allowed(global: &GlobalScope) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        return true;
    };

    let (is_wasm_evaluation_allowed, violations) = csp_list.is_wasm_evaluation_allowed();

    report_csp_violations(global, violations, None);

    is_wasm_evaluation_allowed == CheckResult::Allowed
}

/// <https://www.w3.org/TR/CSP/#should-block-navigation-request>
/// @brief Intercepts navigation attempts to verify compliance with 'navigate-to' directives.
pub(crate) fn should_navigation_request_be_blocked(
    global: &GlobalScope,
    load_data: &LoadData,
    element: Option<&Element>,
) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        return false;
    };
    
    // Logic: Constructs a standardized CSP Request object from internal LoadData.
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
    
    // Algorithm: Delegates complex URL matching and directive precedence to the CSP library.
    let (result, violations) =
        csp_list.should_navigation_request_be_blocked(&request, NavigationCheckType::Other);

    report_csp_violations(global, violations, element);

    result == CheckResult::Blocked
}

/// Used to determine which inline check to run
pub use content_security_policy::InlineCheckType;

/// <https://www.w3.org/TR/CSP/#should-block-inline>
/// @brief Checks if an inline script or style block violates security constraints.
pub(crate) fn should_elements_inline_type_behavior_be_blocked(
    global: &GlobalScope,
    el: &Element,
    type_: InlineCheckType,
    source: &str,
) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        return false;
    };
    
    // Logic: Extracts cryptographic nonces if available on the triggering element.
    let element = CspElement {
        nonce: el.nonce_value_if_nonceable().map(Cow::Owned),
    };
    let (result, violations) =
        csp_list.should_elements_inline_type_behavior_be_blocked(&element, type_, source);

    report_csp_violations(global, violations, Some(el));

    result == CheckResult::Blocked
}

/// <https://w3c.github.io/trusted-types/dist/spec/#should-block-create-policy>
/// @brief Guards the creation of Trusted Type policies.
pub(crate) fn is_trusted_type_policy_creation_allowed(
    global: &GlobalScope,
    policy_name: String,
    created_policy_names: Vec<String>,
) -> bool {
    let Some(csp_list) = global.get_csp_list() else {
        return true;
    };

    let (allowed_by_csp, violations) =
        csp_list.is_trusted_type_policy_creation_allowed(policy_name, created_policy_names);

    report_csp_violations(global, violations, None);

    allowed_by_csp == CheckResult::Allowed
}

/// <https://w3c.github.io/trusted-types/dist/spec/#abstract-opdef-does-sink-type-require-trusted-types>
/// @brief Checks if a specific sink (e.g., innerHTML) requires Trusted Types.
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

/// <https://w3c.github.io/trusted-types/dist/spec/#should-block-sink-type-mismatch>
/// @brief Validates that data assigned to a secure sink matches the required Trusted Type.
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

/// Used to determine which inline check to run
pub use content_security_policy::Violation;

/// <https://www.w3.org/TR/CSP/#report-violation>
/// @brief Orchestrates the asynchronous reporting of policy violations to the document.
/// 
/// Algorithm: Translates internal Violation structures into DOM-visible security reports.
#[allow(unsafe_code)]
pub(crate) fn report_csp_violations(
    global: &GlobalScope,
    violations: Vec<Violation>,
    element: Option<&Element>,
) {
    // Logic: Captures the JS call stack context for enriched reporting.
    let scripted_caller =
        unsafe { describe_scripted_caller(*GlobalScope::get_cx()) }.unwrap_or_default();
    
    for violation in violations {
        // Block Logic: Resource categorization and sample extraction.
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

        // Block Logic: Target identification based on W3C spec steps.
        let target = element.and_then(|event_target| {
            // Step 3.1: Verify if the triggering element belongs to the active window's document.
            if let Some(window) = global.downcast::<Window>() {
                if event_target.upcast::<Node>().owner_document() != window.Document() {
                    return None;
                }
            }
            Some(event_target)
        });
        
        let target = match target {
            None => {
                // Defaulting logic for orphaned or window-level violations.
                if let Some(window) = global.downcast::<Window>() {
                    Trusted::new(window.Document().upcast())
                } else {
                    Trusted::new(global.upcast())
                }
            },
            Some(event_target) => Trusted::new(event_target.upcast()),
        };
        
        // Block Logic: Task queuing to avoid blocking the current execution context.
        // Synchronization: Queues a task in the DOM manipulation source.
        let task =
            CSPViolationReportTask::new(Trusted::new(global), target, report, violation.policy);
        global
            .task_manager()
            .dom_manipulation_task_source()
            .queue(task);
    }
}

/// <https://www.w3.org/TR/CSP/#initialize-document-csp>
/// @brief Parses raw HTTP headers into a structured list of security policies.
pub(crate) fn parse_csp_list_from_metadata(headers: &Option<Serde<HeaderMap>>) -> Option<CspList> {
    let headers = headers.as_ref()?;
    
    // Logic: Aggregates both 'Content-Security-Policy' and 'Content-Security-Policy-Report-Only'.
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
