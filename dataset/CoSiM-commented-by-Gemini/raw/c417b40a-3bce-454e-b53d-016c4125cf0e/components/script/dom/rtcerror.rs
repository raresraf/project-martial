/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! This module implements the `RTCError` DOM interface, as defined by the WebRTC
//! specification. It provides a specialized error object that extends `DOMException`
//! with additional details pertinent to WebRTC operations, such as SDP and SCTP errors.
//! This acts as the bridge between the browser's internal Rust components and the
//! JavaScript environment.

use dom_struct::dom_struct;
use js::rust::HandleObject;

use crate::dom::bindings::codegen::Bindings::RTCErrorBinding::{
    RTCErrorDetailType, RTCErrorInit, RTCErrorMethods,
};
use crate::dom::bindings::reflector::{reflect_dom_object_with_proto, DomGlobal};
use crate::dom::bindings::root::DomRoot;
use crate::dom::bindings::str::DOMString;
use crate::dom::domexception::DOMException;
use crate::dom::globalscope::GlobalScope;
use crate::dom::window::Window;
use crate::script_runtime::CanGc;

/// The Rust representation of the `RTCError` DOM object.
///
/// This struct composes a standard `DOMException` and enhances it with
/// WebRTC-specific error fields. The `#[dom_struct]` macro automatically
/// generates the necessary boilerplate for integrating this Rust struct
/// with the SpiderMonkey JavaScript engine.
#[dom_struct]
pub(crate) struct RTCError {
    /// The underlying `DOMException` that this `RTCError` is based on.
    exception: DOMException,
    /// The type of error that occurred, e.g., related to network or data format.
    error_detail: RTCErrorDetailType,
    /// The line number in an SDP string where an error was found.
    sdp_line_number: Option<i32>,
    /// For errors returned by a TURN server, the relevant HTTP status code.
    http_request_status_code: Option<i32>,
    /// For SCTP-related errors, the SCTP cause code.
    sctp_cause_code: Option<i32>,
    /// The DTLS alert that was received from the remote peer.
    received_alert: Option<u32>,
    /// The DTLS alert that was sent to the remote peer.
    sent_alert: Option<u32>,
}

/// Contains the internal logic for constructing an `RTCError` object.
impl RTCError {
    /// Creates a new `RTCError` instance, inheriting its base properties from `DOMException`.
    ///
    /// This is the core constructor that initializes the Rust struct from the
    /// `RTCErrorInit` dictionary provided by the JavaScript caller.
    fn new_inherited(init: &RTCErrorInit, message: DOMString) -> RTCError {
        RTCError {
            // Per the spec, RTCError is a type of "OperationError".
            exception: DOMException::new_inherited(message, "OperationError".into()),
            error_detail: init.errorDetail,
            sdp_line_number: init.sdpLineNumber,
            http_request_status_code: init.httpRequestStatusCode,
            sctp_cause_code: init.sctpCauseCode,
            received_alert: init.receivedAlert,
            sent_alert: init.sentAlert,
        }
    }

    /// The primary public entry point for creating a new `RTCError` DOM object.
    pub(crate) fn new(
        global: &GlobalScope,
        init: &RTCErrorInit,
        message: DOMString,
        can_gc: CanGc,
    ) -> DomRoot<RTCError> {
        Self::new_with_proto(global, None, init, message, can_gc)
    }

    /// Creates the `RTCError` Rust struct and "reflects" it into the JavaScript
    /// runtime as a DOM object with the correct prototype chain.
    fn new_with_proto(
        global: &GlobalScope,
        proto: Option<HandleObject>,
        init: &RTCErrorInit,
        message: DOMString,
        can_gc: CanGc,
    ) -> DomRoot<RTCError> {
        reflect_dom_object_with_proto(
            Box::new(RTCError::new_inherited(init, message)),
            global,
            proto,
            can_gc,
        )
    }
}

/// Implements the public API of the `RTCError` object that is exposed to JavaScript,
/// as defined by the `RTCErrorMethods` trait.
impl RTCErrorMethods<crate::DomTypeHolder> for RTCError {
    /// Implements the `new RTCError(init, message)` constructor in JavaScript.
    // https://www.w3.org/TR/webrtc/#dom-rtcerror-constructor
    fn Constructor(
        window: &Window,
        proto: Option<HandleObject>,
        can_gc: CanGc,
        init: &RTCErrorInit,
        message: DOMString,
    ) -> DomRoot<RTCError> {
        RTCError::new_with_proto(&window.global(), proto, init, message, can_gc)
    }

    /// Implements the `errorDetail` property getter.
    // https://www.w3.org/TR/webrtc/#dom-rtcerror-errordetail
    fn ErrorDetail(&self) -> RTCErrorDetailType {
        self.error_detail
    }

    /// Implements the `sdpLineNumber` property getter.
    // https://www.w3.org/TR/webrtc/#dom-rtcerror-sdplinenumber
    fn GetSdpLineNumber(&self) -> Option<i32> {
        self.sdp_line_number
    }

    /// Implements the `httpRequestStatusCode` property getter.
    // https://www.w3.org/TR/webrtc/#dom-rtcerror
    fn GetHttpRequestStatusCode(&self) -> Option<i32> {
        self.http_request_status_code
    }

    /// Implements the `sctpCauseCode` property getter.
    // https://www.w3.org/TR/webrtc/#dom-rtcerror-sctpcausecode
    fn GetSctpCauseCode(&self) -> Option<i32> {
        self.sctp_cause_code
    }

    /// Implements the `receivedAlert` property getter.
    // https://www.w3.org/TR/webrtc/#dom-rtcerror-receivedalert
    fn GetReceivedAlert(&self) -> Option<u32> {
        self.received_alert
    }

    /// Implements the `sentAlert` property getter.
    // https://www.w3.org/TR/webrtc/#dom-rtcerror-sentalert
    fn GetSentAlert(&self) -> Option<u32> {
        self.sent_alert
    }
}