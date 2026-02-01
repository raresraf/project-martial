//! This module implements the Content Security Policy (CSP) violation reporting mechanism.
//! It defines structures and logic for detecting and reporting CSP violations,
//! including the creation of `SecurityPolicyViolationEvent`s and the serialization of violation reports.
//!
//! Functional Utility: Enforces web security policies by monitoring and responding to
//! attempts to bypass restrictions on resource loading and script execution.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use js::jsapi::RuntimeCode;
use net_traits::request::Referrer;
use serde::Serialize;
use servo_url::ServoUrl;
use stylo_atoms::Atom;

use crate::conversions::Convert;
use crate::dom::bindings::codegen::Bindings::EventBinding::EventInit;
use crate::dom::bindings::codegen::Bindings::SecurityPolicyViolationEventBinding::{
    SecurityPolicyViolationEventDisposition, SecurityPolicyViolationEventInit,
};
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::event::{Event, EventBubbles, EventCancelable};
use crate::dom::eventtarget::EventTarget;
use crate::dom::securitypolicyviolationevent::SecurityPolicyViolationEvent;
use crate::dom::types::GlobalScope;
use crate::script_runtime::CanGc;
use crate::task::TaskOnce;

pub(crate) struct CSPViolationReporter {
    sample: Option<String>,
    filename: String,
    report_only: bool,
    runtime_code: RuntimeCode,
    line_number: u32,
    column_number: u32,
    target: Trusted<EventTarget>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SecurityPolicyViolationReport {
    sample: Option<String>,
    #[serde(rename = "blockedURL")]
    blocked_url: String,
    referrer: String,
    status_code: u16,
    #[serde(rename = "documentURL")]
    document_url: String,
    source_file: String,
    violated_directive: String,
    effective_directive: String,
    line_number: u32,
    column_number: u32,
    original_policy: String,
    #[serde(serialize_with = "serialize_disposition")]
    disposition: SecurityPolicyViolationEventDisposition,
}

impl CSPViolationReporter {
    pub(crate) fn new(
        global: &GlobalScope,
        sample: Option<String>,
        report_only: bool,
        runtime_code: RuntimeCode,
        filename: String,
        line_number: u32,
        column_number: u32,
    ) -> CSPViolationReporter {
        /// Functional Utility: Constructs a new `CSPViolationReporter` instance.
        /// This is the entry point for creating a reporter that will handle a specific CSP violation.
        CSPViolationReporter {
            sample,
            filename,
            report_only,
            runtime_code,
            line_number,
            column_number,
            target: Trusted::new(global.upcast::<EventTarget>()),
        }
    }

    fn get_report(&self, global: &GlobalScope) -> SecurityPolicyViolationReport {
        /// Functional Utility: Generates a `SecurityPolicyViolationReport` based on the current violation data.
        /// This report contains detailed information about the violation, formatted according to CSP reporting standards.
        SecurityPolicyViolationReport {
            sample: self.sample.clone(),
            // Block Logic: Determine the disposition of the report (report-only or enforced) based on `report_only` flag.
            disposition: match self.report_only {
                true => SecurityPolicyViolationEventDisposition::Report,
                false => SecurityPolicyViolationEventDisposition::Enforce,
            },
            // https://w3c.github.io/webappsec-csp/#violation-resource
            // Block Logic: Determine the `blocked_url` based on the `runtime_code`.
            blocked_url: match self.runtime_code {
                RuntimeCode::JS => "eval".to_owned(),
                RuntimeCode::WASM => "wasm-eval".to_owned(),
            },
            // https://w3c.github.io/webappsec-csp/#violation-referrer
            // Block Logic: Determine the `referrer` URL, stripping sensitive information for reports.
            referrer: match global.get_referrer() {
                Referrer::Client(url) => self.strip_url_for_reports(url),
                Referrer::ReferrerUrl(url) => self.strip_url_for_reports(url),
                _ => "".to_owned(),
            },
            status_code: global.status_code().unwrap_or(200),
            document_url: self.strip_url_for_reports(global.get_url()),
            source_file: self.filename.clone(),
            violated_directive: "script-src".to_owned(),
            effective_directive: "script-src".to_owned(),
            line_number: self.line_number,
            column_number: self.column_number,
            original_policy: String::default(),
        }
    }

    fn fire_violation_event(&self, can_gc: CanGc) {
        /// Functional Utility: Fires a `SecurityPolicyViolationEvent` at the target `EventTarget`.
        /// This event notifies listeners in the DOM about the detected CSP violation.
        let target = self.target.root();
        let global = &target.global();
        let report = self.get_report(global);

        let event = SecurityPolicyViolationEvent::new(
            global,
            Atom::from("securitypolicyviolation"),
            EventBubbles::Bubbles,
            EventCancelable::Cancelable,
            &report.convert(),
            can_gc,
        );

        event.upcast::<Event>().fire(&target, can_gc);
    }

    /// <https://w3c.github.io/webappsec-csp/#strip-url-for-use-in-reports>
    fn strip_url_for_reports(&self, mut url: ServoUrl) -> String {
        /// Functional Utility: Strips sensitive information (fragment, username, password) from a URL
        /// for use in security violation reports, ensuring privacy while providing necessary context.
        ///
        /// <https://w3c.github.io/webappsec-csp/#strip-url-for-use-in-reports>
        let scheme = url.scheme();
        // Block Logic: Check if the URL's scheme is HTTP or HTTPS.
        // Precondition: The URL object is mutable.
        // Invariant: Only HTTP(S) schemes are processed further for sensitive information stripping.
        if scheme != "https" && scheme != "http" {
            return scheme.to_owned();
        }
        // Block Logic: Remove the fragment from the URL.
        url.set_fragment(None);
        // Block Logic: Remove the username from the URL.
        let _ = url.set_username("");
        // Block Logic: Remove the password from the URL.
        let _ = url.set_password(None);
        // Block Logic: Serialize the modified URL back into a string.
        url.into_string()
    }
}

/// Corresponds to the operation in 5.5 Report Violation
/// <https://w3c.github.io/webappsec-csp/#report-violation>
/// > Queue a task to run the following steps:
impl TaskOnce for CSPViolationReporter {
    fn run_once(self) {
        /// Functional Utility: Executes the CSP violation reporting process as a one-time task.
        /// This ensures that the violation event is fired within the appropriate task queue.
        // > If target implements EventTarget, fire an event named securitypolicyviolation
        // > that uses the SecurityPolicyViolationEvent interface
        // > at target with its attributes initialized as follows:
        self.fire_violation_event(CanGc::note());
        // TODO: Support `report-to` directive that corresponds to 5.5.3.5.
    }
}

impl Convert<SecurityPolicyViolationEventInit> for SecurityPolicyViolationReport {
    fn convert(self) -> SecurityPolicyViolationEventInit {
        /// Functional Utility: Converts a `SecurityPolicyViolationReport` into a `SecurityPolicyViolationEventInit` object.
        /// This prepares the violation data for the creation of a DOM `SecurityPolicyViolationEvent`.
        SecurityPolicyViolationEventInit {
            sample: self.sample.unwrap_or_default().into(),
            blockedURI: self.blocked_url.into(),
            referrer: self.referrer.into(),
            statusCode: self.status_code,
            documentURI: self.document_url.into(),
            sourceFile: self.source_file.into(),
            violatedDirective: self.violated_directive.into(),
            effectiveDirective: self.effective_directive.into(),
            lineNumber: self.line_number,
            columnNumber: self.column_number,
            originalPolicy: self.original_policy.into(),
            disposition: self.disposition,
            parent: EventInit::empty(),
        }
    }
}

fn serialize_disposition<S: serde::Serializer>(
    val: &SecurityPolicyViolationEventDisposition,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    /// Functional Utility: Serializes the `SecurityPolicyViolationEventDisposition` enum into a string representation.
    /// This is used for generating the JSON report of a CSP violation.
    match val {
        SecurityPolicyViolationEventDisposition::Report => serializer.serialize_str("report"),
        SecurityPolicyViolationEventDisposition::Enforce => serializer.serialize_str("enforce"),
    }
}
