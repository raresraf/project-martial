/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! @file rtcerror.rs
//! @brief Defines the `RTCError` struct and its associated methods for WebRTC-specific error reporting.
//!
//! This module provides the Rust representation and implementation for the WebRTC `RTCError`
//! interface, as specified by the W3C WebRTC API. `RTCError` is a specialized `DOMException`
//! that includes additional fields relevant to WebRTC operations, such as error details,
//! SDP line numbers, HTTP request status codes, SCTP cause codes, and alert information.
//! It facilitates structured reporting of errors encountered during WebRTC communication.

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

/// @struct RTCError
/// @brief Represents a WebRTC-specific error, extending `DOMException` with additional details.
///
/// This struct holds detailed information about errors encountered during WebRTC
/// operations, conforming to the `RTCError` WebIDL interface. It provides
/// specific context such as the type of error detail, SDP line numbers, HTTP status,
/// SCTP cause codes, and received/sent alert numbers.
#[dom_struct]
pub(crate) struct RTCError {
    /// @brief The underlying `DOMException` that this `RTCError` extends.
    /// Functional Utility: Provides common error properties like `name` and `message`.
    exception: DOMException,
    /// @brief Detailed type of the WebRTC error.
    /// Functional Utility: Categorizes the nature of the error (e.g., SDP, ICE, DTLS).
    error_detail: RTCErrorDetailType,
    /// @brief The SDP line number associated with the error, if applicable.
    sdp_line_number: Option<i32>,
    /// @brief The HTTP request status code associated with the error, if applicable.
    http_request_status_code: Option<i32>,
    /// @brief The SCTP cause code associated with the error, if applicable.
    sctp_cause_code: Option<i32>,
    /// @brief The received alert number, if applicable.
    received_alert: Option<u32>,
    /// @brief The sent alert number, if applicable.
    sent_alert: Option<u32>,
}

impl RTCError {
    /// @brief Creates a new `RTCError` instance with inherited properties from `DOMException`.
    ///
    /// Functional Utility: This internal constructor is used to initialize the `RTCError`
    /// struct's fields based on `RTCErrorInit` and a message, while setting up
    /// the base `DOMException` with a default `OperationError` name.
    ///
    /// @param init Initialization dictionary containing WebRTC error details.
    /// @param message The error message.
    /// @returns A new `RTCError` instance.
    fn new_inherited(init: &RTCErrorInit, message: DOMString) -> RTCError {
        RTCError {
            // Block Logic: Initializes the base `DOMException` with the provided message
            // and a standard "OperationError" name, consistent with WebRTC error handling.
            exception: DOMException::new_inherited(message, "OperationError".into()),
            error_detail: init.errorDetail,
            sdp_line_number: init.sdpLineNumber,
            http_request_status_code: init.httpRequestStatusCode,
            sctp_cause_code: init.sctpCauseCode,
            received_alert: init.receivedAlert,
            sent_alert: init.sentAlert,
        }
    }

    /// @brief Creates a new `RTCError` wrapped in a `DomRoot`.
    ///
    /// Functional Utility: This is a public constructor for creating `RTCError`
    /// objects, typically used within the DOM environment. It calls `new_with_proto`
    /// without specifying a custom prototype.
    ///
    /// @param global The `GlobalScope` context for the DOM object.
    /// @param init Initialization dictionary containing WebRTC error details.
    /// @param message The error message.
    /// @param can_gc `CanGc` indicating if the object is garbage collectable.
    /// @returns A `DomRoot` containing the new `RTCError` instance.
    pub(crate) fn new(
        global: &GlobalScope,
        init: &RTCErrorInit,
        message: DOMString,
        can_gc: CanGc,
    ) -> DomRoot<RTCError> {
        Self::new_with_proto(global, None, init, message, can_gc)
    }

    /// @brief Creates a new `RTCError` instance with a specified prototype, wrapped in a `DomRoot`.
    ///
    /// Functional Utility: This constructor reflects the `RTCError` Rust struct
    /// into a JavaScript DOM object, allowing it to be used within the browser's
    /// JavaScript environment. It handles the inheritance and prototype chain setup.
    ///
    /// @param global The `GlobalScope` context for the DOM object.
    /// @param proto An optional `HandleObject` representing the JavaScript prototype.
    /// @param init Initialization dictionary containing WebRTC error details.
    /// @param message The error message.
    /// @param can_gc `CanGc` indicating if the object is garbage collectable.
    /// @returns A `DomRoot` containing the new `RTCError` instance.
    fn new_with_proto(
        global: &GlobalScope,
        proto: Option<HandleObject>,
        init: &RTCErrorInit,
        message: DOMString,
        can_gc: CanGc,
    ) -> DomRoot<RTCError> {
        // Functional Utility: Uses a reflective mechanism to expose the Rust struct
        // as a JavaScript DOM object, linking it to the global scope and prototype chain.
        reflect_dom_object_with_proto(
            Box::new(RTCError::new_inherited(init, message)),
            global,
            proto,
            can_gc,
        )
    }
}

/// @impl RTCErrorMethods
/// @brief Implements the WebIDL methods and attributes for `RTCError`.
///
/// This block provides the getter methods for accessing the various fields
/// of the `RTCError` struct, conforming to the WebRTC specification. It also
/// defines the constructor logic as it would appear in the WebIDL.
impl RTCErrorMethods<crate::DomTypeHolder> for RTCError {
    /// @brief WebIDL Constructor for `RTCError`.
    ///
    /// Functional Utility: This method acts as the entry point for creating
    /// `RTCError` objects from JavaScript, mapping directly to the `RTCError`
    /// constructor defined in the WebRTC IDL.
    ///
    /// @see https://www.w3.org/TR/webrtc/#dom-rtcerror-constructor
    fn Constructor(
        window: &Window,
        proto: Option<HandleObject>,
        can_gc: CanGc,
        init: &RTCErrorInit,
        message: DOMString,
    ) -> DomRoot<RTCError> {
        // Block Logic: Delegates to the internal `new_with_proto` method to construct
        // the `RTCError` instance within the global scope of the window.
        RTCError::new_with_proto(&window.global(), proto, init, message, can_gc)
    }

    /// @brief Getter for the `errorDetail` attribute of `RTCError`.
    ///
    /// Functional Utility: Provides access to the detailed type of the WebRTC error.
    ///
    /// @see https://www.w3.org/TR/webrtc/#dom-rtcerror-errordetail
    fn ErrorDetail(&self) -> RTCErrorDetailType {
        self.error_detail
    }

    /// @brief Getter for the `sdpLineNumber` attribute of `RTCError`.
    ///
    /// Functional Utility: Provides access to the SDP line number associated with the error.
    ///
    /// @see https://www.w3.org/TR/webrtc/#dom-rtcerror-sdplinenumber
    fn GetSdpLineNumber(&self) -> Option<i32> {
        self.sdp_line_number
    }

    /// @brief Getter for the `httpRequestStatusCode` attribute of `RTCError`.
    ///
    /// Functional Utility: Provides access to the HTTP request status code associated with the error.
    ///
    /// @see https://www.w3.org/TR/webrtc/#dom-rtcerror
    fn GetHttpRequestStatusCode(&self) -> Option<i32> {
        self.http_request_status_code
    }

    /// @brief Getter for the `sctpCauseCode` attribute of `RTCError`.
    ///
    /// Functional Utility: Provides access to the SCTP cause code associated with the error.
    ///
    /// @see https://www.w3.org/TR/webrtc/#dom-rtcerror-sctpcausecode
    fn GetSctpCauseCode(&self) -> Option<i32> {
        self.sctp_cause_code
    }

    /// @brief Getter for the `receivedAlert` attribute of `RTCError`.
    ///
    /// Functional Utility: Provides access to the received alert number associated with the error.
    ///
    /// @see https://www.w3.org/TR/webrtc/#dom-rtcerror-receivedalert
    fn GetReceivedAlert(&self) -> Option<u32> {
        self.received_alert
    }

    /// @brief Getter for the `sentAlert` attribute of `RTCError`.
    ///
    /// Functional Utility: Provides access to the sent alert number associated with the error.
    ///
    /// @see https://www.w3.org/TR/webrtc/#dom-rtcerror-sentalert
    fn GetSentAlert(&self) -> Option<u32> {
        self.sent_alert
    }
}