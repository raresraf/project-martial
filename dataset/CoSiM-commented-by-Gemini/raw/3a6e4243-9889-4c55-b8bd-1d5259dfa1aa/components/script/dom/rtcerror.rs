//! This module provides the Rust implementation for the `RTCError` DOM object,
//! as specified by the WebRTC standard. `RTCError` offers more detailed
//! information about a WebRTC-related error than a standard `DOMException`.

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
use crate.dom::domexception::{DOMErrorName, DOMException};
use crate::dom::globalscope::GlobalScope;
use crate::dom::window::Window;
use crate::script_runtime::CanGc;

/// Represents the `RTCError` interface from the WebRTC API, which provides
/// extended error information for WebRTC-related operations.
///
/// This struct wraps a standard `DOMException` and adds additional fields for
/// detailed, RTC-specific context, such as SDP line numbers or SCTP cause codes.
#[dom_struct]
pub(crate) struct RTCError {
    /// The underlying `DOMException` that provides the base error message.
    exception: Dom<DOMException>,
    /// The category of the error, as defined by the WebRTC specification.
    error_detail: RTCErrorDetailType,
    /// If the error relates to SDP, this indicates the line number.
    sdp_line_number: Option<i32>,
    /// If the error is related to an HTTP request, this holds the status code.
    http_request_status_code: Option<i32>,
    /// If the error is SCTP-related, this holds the SCTP cause code.
    sctp_cause_code: Option<i32>,
    /// If the error is a TLS alert, this indicates the alert code received.
    received_alert: Option<u32>,
    /// If the error is a TLS alert, this indicates the alert code sent.
    sent_alert: Option<u32>,
}

impl RTCError {
    /// Creates the internal state of an `RTCError` object.
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

    /// The primary factory function for creating a new `RTCError` DOM object instance.
    /// This is the entry point for creating the object from Rust code.
    pub(crate) fn new(
        global: &GlobalScope,
        init: &RTCErrorInit,
        message: DOMString,
        can_gc: CanGc,
    ) -> DomRoot<RTCError> {
        Self::new_with_proto(global, None, init, message, can_gc)
    }

    /// Creates an `RTCError` instance and associates it with a JavaScript prototype.
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

/// Implements the public, JavaScript-visible API for the `RTCError` object,
/// as defined by the `RTCErrorMethods` trait from the WebIDL bindings.
impl RTCErrorMethods<crate::DomTypeHolder> for RTCError {
    /// Implements the `RTCError` constructor, callable from JavaScript as `new RTCError()`.
    ///
    /// [Specification](https://www.w3.org/TR/webrtc/#dom-rtcerror-constructor)
    fn Constructor(
        window: &Window,
        proto: Option<HandleObject>,
        can_gc: CanGc,
        init: &RTCErrorInit,
        message: DOMString,
    ) -> DomRoot<RTCError> {
        RTCError::new_with_proto(&window.global(), proto, init, message, can_gc)
    }

    /// Implements the `errorDetail` property accessor.
    ///
    /// [Specification](https://www.w3.org/TR/webrtc/#dom-rtcerror-errordetail)
    fn ErrorDetail(&self) -> RTCErrorDetailType {
        self.error_detail
    }

    /// Implements the `sdpLineNumber` property accessor.
    ///
    // [Specification](https://www.w3.org/TR/webrtc/#dom-rtcerror-sdplinenumber)
    fn GetSdpLineNumber(&self) -> Option<i32> {
        self.sdp_line_number
    }

    /// Implements the `httpRequestStatusCode` property accessor.
    ///
    // [Specification](https://www.w3.org/TR/webrtc/#dom-rtcerror)
    fn GetHttpRequestStatusCode(&self) -> Option<i32> {
        self.http_request_status_code
    }

    /// Implements the `sctpCauseCode` property accessor.
    ///
    // [Specification](https://www.w3.org/TR/webrtc/#dom-rtcerror-sctpcausecode)
    fn GetSctpCauseCode(&self) -> Option<i32> {
        self.sctp_cause_code
    }

    /// Implements the `receivedAlert` property accessor.
    ///
    // [Specification](https://www.w3.org/TR/webrtc/#dom-rtcerror-receivedalert)
    fn GetReceivedAlert(&self) -> Option<u32> {
        self.received_alert
    }

    /// Implements the `sentAlert` property accessor.
    ///
    // [Specification](https://www.w3.org/TR/webrtc/#dom-rtcerror-sentalert)
    fn GetSentAlert(&self) -> Option<u32> {
        self.sent_alert
    }
}