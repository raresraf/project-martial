/**
 * @3a6e4243-9889-4c55-b8bd-1d5259dfa1aa/components/script/dom/rtcerror.rs
 * @brief DOM binding implementation for the WebRTC RTCError object.
 * 
 * Functional Intent: Provides a specialized DOM exception type for WebRTC-specific 
 * failure modes. It extends the standard DOMException with low-level protocol 
 * metadata, including SDP line numbers, HTTP status codes, and SCTP cause codes, 
 * enabling granular error reporting for peer-to-peer communication.
 * 
 * Domain: WebRTC, DOM Bindings, Error Handling.
 */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use dom_struct::dom_struct;
use js::rust::HandleObject;

use crate::dom::bindings::codegen::Bindings::RTCErrorBinding::{
    RTCErrorDetailType, RTCErrorInit, RTCErrorMethods,
};
use crate::dom::bindings::reflector::{reflect_dom_object_with_proto, DomGlobal};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::bindings::str::DOMString;
use crate::dom::domexception::{DOMErrorName, DOMException};
use crate::dom::globalscope::GlobalScope;
use crate::dom::window::Window;
use crate::script_runtime::CanGc;

/**
 * @brief Represents an RTCError instance, encapsulating exception state and WebRTC metadata.
 * 
 * Logic: Inherits from DOMException and appends protocol-specific diagnostic fields 
 * required by the W3C WebRTC specification.
 */
#[dom_struct]
pub(crate) struct RTCError {
    exception: Dom<DOMException>,
    error_detail: RTCErrorDetailType,
    sdp_line_number: Option<i32>,
    http_request_status_code: Option<i32>,
    sctp_cause_code: Option<i32>,
    received_alert: Option<u32>,
    sent_alert: Option<u32>,
}

impl RTCError {
    /**
     * @brief Internal initializer for the RTCError state machine.
     * Logic: Maps the human-readable message to a standard DOMErrorName and 
     * populates supplementary protocol fields from the RTCErrorInit dictionary.
     */
    fn new_inherited(
        global: &GlobalScope,
        init: &RTCErrorInit,
        message: DOMString,
        can_gc: CanGc,
    ) -> RTCError {
        RTCError {
            exception: Dom::from_ref(&*DOMException::new(
                global,
                DOMErrorName::from(&message).unwrap(),
                can_gc,
            )),
            error_detail: init.errorDetail,
            sdp_line_number: init.sdpLineNumber,
            http_request_status_code: init.httpRequestStatusCode,
            sctp_cause_code: init.sctpCauseCode,
            received_alert: init.receivedAlert,
            sent_alert: init.sentAlert,
        }
    }

    /**
     * @brief Public constructor for script-triggered RTCErrors.
     */
    pub(crate) fn new(
        global: &GlobalScope,
        init: &RTCErrorInit,
        message: DOMString,
        can_gc: CanGc,
    ) -> DomRoot<RTCError> {
        Self::new_with_proto(global, None, init, message, can_gc)
    }

    /**
     * @brief Orchestrates object creation and JS reflection.
     * Logic: Boxes the inherited struct and links it to the JS engine's 
     * prototype chain for the current global context.
     */
    fn new_with_proto(
        global: &GlobalScope,
        proto: Option<HandleObject>,
        init: &RTCErrorInit,
        message: DOMString,
        can_gc: CanGc,
    ) -> DomRoot<RTCError> {
        reflect_dom_object_with_proto(
            Box::new(RTCError::new_inherited(global, init, message, can_gc)),
            global,
            proto,
            can_gc,
        )
    }
}

/**
 * @brief Implementation of the RTCError WebIDL interface methods.
 */
impl RTCErrorMethods<crate::DomTypeHolder> for RTCError {
    /**
     * @brief [Constructor] Implements the RTCError(init, message) script constructor.
     */
    fn Constructor(
        window: &Window,
        proto: Option<HandleObject>,
        can_gc: CanGc,
        init: &RTCErrorInit,
        message: DOMString,
    ) -> DomRoot<RTCError> {
        RTCError::new_with_proto(&window.global(), proto, init, message, can_gc)
    }

    /**
     * @brief Accessor for the high-level error classification (errorDetail).
     */
    fn ErrorDetail(&self) -> RTCErrorDetailType {
        self.error_detail
    }

    /**
     * @brief Accessor for the SDP line number where parsing or validation failed.
     */
    fn GetSdpLineNumber(&self) -> Option<i32> {
        self.sdp_line_number
    }

    /**
     * @brief Accessor for the HTTP status code in case of signaling or fetch failures.
     */
    fn GetHttpRequestStatusCode(&self) -> Option<i32> {
        self.http_request_status_code
    }

    /**
     * @brief Accessor for the SCTP-specific error cause code.
     */
    fn GetSctpCauseCode(&self) -> Option<i32> {
        self.sctp_cause_code
    }

    /**
     * @brief Accessor for the received DTLS/TLS alert value.
     */
    fn GetReceivedAlert(&self) -> Option<u32> {
        self.received_alert
    }

    /**
     * @brief Accessor for the locally sent DTLS/TLS alert value.
     */
    fn GetSentAlert(&self) -> Option<u32> {
        self.sent_alert
    }
}
