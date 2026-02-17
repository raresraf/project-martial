/// @3a6e4243-9889-4c55-b8bd-1d5259dfa1aa/components/script/dom/rtcerror.rs
/// @brief Implements the WebRTC RTCError object for DOM binding.
///
/// This file defines the `RTCError` struct, which represents an error that occurred
/// during an RTC operation, typically in the context of WebRTC. It integrates with
/// the DOM structure and provides methods for accessing error details as per
/// the WebRTC specification.
///
/// The `RTCError` object inherits from `DOMException` and extends it with
/// WebRTC-specific error details such as `errorDetail`, `sdpLineNumber`,
/// `httpRequestStatusCode`, `sctpCauseCode`, `receivedAlert`, and `sentAlert`.
///
/// References:
/// - https://www.w3.org/TR/webrtc/#rtcerror
/// - https://www.w3.org/TR/webrtc/#dom-rtcerror-constructor
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

/// @brief Represents a WebRTC RTCError object, extending `DOMException` with specific error details.
/// This struct holds information relevant to WebRTC errors, such as the type of error detail,
/// SDP line number, HTTP request status code, SCTP cause code, and received/sent alerts.
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

/// @brief Implementation block for the `RTCError` struct.
impl RTCError {
    /// @brief Creates a new `RTCError` instance with inherited properties from `DOMException`.
    /// This is an internal helper function used by `new` and `new_with_proto`.
    /// @param global: The global scope associated with the error.
    /// @param init: Initialization options for the `RTCError`, containing specific error details.
    /// @param message: The error message, used to determine the `DOMErrorName`.
    /// @param can_gc: A flag indicating if the object can be garbage collected.
    /// @return A new `RTCError` instance.
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

    /// @brief Creates a new `RTCError` instance.
    /// This is a public-facing constructor that internally calls `new_with_proto`.
    /// @param global: The global scope associated with the error.
    /// @param init: Initialization options for the `RTCError`.
    /// @param message: The error message.
    /// @param can_gc: A flag indicating if the object can be garbage collected.
    /// @return A `DomRoot` smart pointer owning the newly created `RTCError`.
    pub(crate) fn new(
        global: &GlobalScope,
        init: &RTCErrorInit,
        message: DOMString,
        can_gc: CanGc,
    ) -> DomRoot<RTCError> {
        Self::new_with_proto(global, None, init, message, can_gc)
    }

    /// @brief Creates a new `RTCError` instance with an optional prototype object.
    /// This function reflects the DOM object with a given prototype, linking it
    /// into the DOM's object hierarchy.
    /// @param global: The global scope associated with the error.
    /// @param proto: An optional `HandleObject` representing the prototype for the new object.
    /// @param init: Initialization options for the `RTCError`.
    /// @param message: The error message.
    /// @param can_gc: A flag indicating if the object can be garbage collected.
    /// @return A `DomRoot` smart pointer owning the newly created `RTCError`.
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

/// @brief Implements the `RTCErrorMethods` trait for the `RTCError` struct.
/// This block provides the methods required by the WebRTC specification for `RTCError` objects.
impl RTCErrorMethods<crate::DomTypeHolder> for RTCError {
    /// @brief Constructor for the `RTCError` object as defined in the WebRTC specification.
    /// @param window: The `Window` object providing the global scope.
    /// @param proto: An optional `HandleObject` representing the prototype for the new object.
    /// @param can_gc: A flag indicating if the object can be garbage collected.
    /// @param init: Initialization options for the `RTCError`.
    /// @param message: The error message.
    /// @return A `DomRoot` smart pointer owning the newly created `RTCError`.
    fn Constructor(
        window: &Window,
        proto: Option<HandleObject>,
        can_gc: CanGc,
        init: &RTCErrorInit,
        message: DOMString,
    ) -> DomRoot<RTCError> {
        RTCError::new_with_proto(&window.global(), proto, init, message, can_gc)
    }

    /// @brief Returns the `errorDetail` of the `RTCError`.
    /// Corresponds to https://www.w3.org/TR/webrtc/#dom-rtcerror-errordetail.
    /// @return The `RTCErrorDetailType` enum value indicating the specific error detail.
    fn ErrorDetail(&self) -> RTCErrorDetailType {
        self.error_detail
    }

    /// @brief Returns the `sdpLineNumber` associated with the error, if available.
    /// Corresponds to https://www.w3.org/TR/webrtc/#dom-rtcerror-sdplinenumber.
    /// @return An `Option<i32>` containing the SDP line number or `None`.
    fn GetSdpLineNumber(&self) -> Option<i32> {
        self.sdp_line_number
    }

    /// @brief Returns the `httpRequestStatusCode` associated with the error, if available.
    /// Corresponds to https://www.w3.org/TR/webrtc/#dom-rtcerror.
    /// @return An `Option<i32>` containing the HTTP request status code or `None`.
    fn GetHttpRequestStatusCode(&self) -> Option<i32> {
        self.http_request_status_code
    }

    /// @brief Returns the `sctpCauseCode` associated with the error, if available.
    /// Corresponds to https://www.w3.org/TR/webrtc/#dom-rtcerror-sctpcausecode.
    /// @return An `Option<i32>` containing the SCTP cause code or `None`.
    fn GetSctpCauseCode(&self) -> Option<i32> {
        self.sctp_cause_code
    }

    /// @brief Returns the `receivedAlert` value associated with the error, if available.
    /// Corresponds to https://www.w3.org/TR/webrtc/#dom-rtcerror-receivedalert.
    /// @return An `Option<u32>` containing the received alert value or `None`.
    fn GetReceivedAlert(&self) -> Option<u32> {
        self.received_alert
    }

    /// @brief Returns the `sentAlert` value associated with the error, if available.
    /// Corresponds to https://www.w3.org/TR/webrtc/#dom-rtcerror-sentalert.
    /// @return An `Option<u32>` containing the sent alert value or `None`.
    fn GetSentAlert(&self) -> Option<u32> {
        self.sent_alert
    }
}
