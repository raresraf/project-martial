/*!
This module is responsible for reporting Content Security Policy (CSP) violations.
It defines the `report_csp_violations` function, which is called when a CSP
violation occurs. This function creates a `CSPViolationReport` and queues a
task to dispatch a `SecurityPolicyViolationEvent` to the appropriate target.

The `report_csp_violations` function follows the steps outlined in the CSP
specification for reporting violations. It determines the target of the event
based on the element that caused the violation and the global object in which
the violation occurred. If the violation occurred in a `Window` global, the
target is the `Document` associated with that `Window`. Otherwise, the target
is the global object itself.
*/

use content_security_policy::{PolicyDisposition, Violation, ViolationResource};
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

/// Reports a set of CSP violations.
/// <https://www.w3.org/TR/CSP/#report-violation>
#[allow(unsafe_code)]
pub(crate) fn report_csp_violations(
    global: &GlobalScope,
    violations: Vec<Violation>,
    element: Option<&Element>,
) {
    let scripted_caller =
        unsafe { describe_scripted_caller(*GlobalScope::get_cx()) }.unwrap_or_default();
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
        // Step 1: Let global be violation’s global object.
        // We use `self` as `global`;
        // Step 2: Let target be violation’s element.
        let target = element.and_then(|event_target| {
            // Step 3.1: If target is not null, and global is a Window,
            // and target’s shadow-including root is not global’s associated Document, set target to null.
            if let Some(window) = global.downcast::<Window>() {
                // If a node is connected, its owner document is always the shadow-including root.
                // If it isn't connected, then it also doesn't have a corresponding document, hence
                // it can't be this document.
                if event_target.upcast::<Node>().owner_document() != window.Document() {
                    return None;
                }
            }
            Some(event_target)
        });
        let target = match target {
            // Step 3.2: If target is null:
            None => {
                // Step 3.2.2: If target is a Window, set target to target’s associated Document.
                if let Some(window) = global.downcast::<Window>() {
                    Trusted::new(window.Document().upcast())
                } else {
                    // Step 3.2.1: Set target to violation’s global object.
                    Trusted::new(global.upcast())
                }
            },
            Some(event_target) => Trusted::new(event_target.upcast()),
        };
        // Step 3: Queue a task to run the following steps:
        let task =
            CSPViolationReportTask::new(Trusted::new(global), target, report, violation.policy);
        global
            .task_manager()
            .dom_manipulation_task_source()
            .queue(task);
    }
}