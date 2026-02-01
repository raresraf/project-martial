//! This module implements the Web Streams API's `WritableStream`
//! (<https://streams.spec.whatwg.org/#writablestream-class>), providing the
//! core logic for writing streaming data in a Web-compatible manner. It manages
//! the internal state of the stream (writable, closed, errored), handles
//! backpressure, abort and close operations, and integrates with underlying
//! sinks to process written data.
//!
//! Functional Utility: Enables web applications to programmatically write
//! data to a destination in an asynchronous and non-blocking fashion,
//! supporting various data sources and destinations.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::cell::Cell;
use std::collections::VecDeque;
use std::mem;
use std::ptr::{self};
use std::rc::Rc;

use dom_struct::dom_struct;
use js::jsapi::{Heap, JSObject};
use js::jsval::{JSVal, ObjectValue, UndefinedValue};
use js::rust::{
    HandleObject as SafeHandleObject, HandleValue as SafeHandleValue,
    MutableHandleValue as SafeMutableHandleValue,
};

use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::QueuingStrategyBinding::QueuingStrategy;
use crate::dom::bindings::codegen::Bindings::UnderlyingSinkBinding::UnderlyingSink;
use crate::dom::bindings::codegen::Bindings::WritableStreamBinding::WritableStreamMethods;
use crate::dom::bindings::conversions::ConversionResult;
use crate::dom::bindings::error::Error;
use crate::dom::bindings::import::module::Fallible;
use crate::dom::bindings::reflector::{reflect_dom_object_with_proto, Reflector};
use crate::dom::bindings::root::{Dom, DomRoot, MutNullableDom};
use crate::dom::countqueuingstrategy::{extract_high_water_mark, extract_size_algorithm};
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::promisenativehandler::{Callback, PromiseNativeHandler};
use crate::dom::writablestreamdefaultcontroller::WritableStreamDefaultController;
use crate::dom::writablestreamdefaultwriter::WritableStreamDefaultWriter;
use crate::realms::{enter_realm, InRealm};
use crate::script_runtime::{CanGc, JSContext as SafeJSContext};

impl js::gc::Rootable for AbortAlgorithmFulfillmentHandler {}

/// The fulfillment handler for the abort steps of
/// <https://streams.spec.whatwg.org/#writable-stream-finish-erroring>.
/// Functional Utility: This handler is responsible for resolving the `abort_request_promise`
/// when the underlying sink's abort operation successfully completes, and then
/// rejecting any pending close/closed promises if needed.
#[derive(JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
struct AbortAlgorithmFulfillmentHandler {
    /// A reference to the `WritableStream` being aborted.
    stream: Dom<WritableStream>,
    /// The promise associated with the current abort request.
    #[ignore_malloc_size_of = "Rc is hard"]
    abort_request_promise: Rc<Promise>,
}

impl Callback for AbortAlgorithmFulfillmentHandler {
    /// Callback method invoked when the abort operation's promise is fulfilled.
    /// Functional Utility: Resolves the `abort_request_promise` and then calls
    /// `WritableStreamRejectCloseAndClosedPromiseIfNeeded` to handle stream state.
    /// @param cx The current JavaScript context.
    /// @param _v The value the promise was fulfilled with (unused, as it's typically undefined).
    /// @param _realm The current JavaScript realm.
    /// @param can_gc A `CanGc` token.
    fn callback(&self, cx: SafeJSContext, _v: SafeHandleValue, _realm: InRealm, can_gc: CanGc) {
        // Resolve abortRequest’s promise with undefined.
        self.abort_request_promise.resolve_native(&(), can_gc);

        // Perform ! WritableStreamRejectCloseAndClosedPromiseIfNeeded(stream).
        self.stream
            .as_rooted()
            .reject_close_and_closed_promise_if_needed(cx);
    }
}

impl js::gc::Rootable for AbortAlgorithmRejectionHandler {}

/// The rejection handler for the abort steps of
/// <https://streams.spec.whatwg.org/#writable-stream-finish-erroring>.
/// Functional Utility: This handler is responsible for rejecting the `abort_request_promise`
/// when the underlying sink's abort operation fails, and then rejecting any
/// pending close/closed promises if needed.
#[derive(JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
struct AbortAlgorithmRejectionHandler {
    /// A reference to the `WritableStream` being aborted.
    stream: Dom<WritableStream>,
    /// The promise associated with the current abort request.
    #[ignore_malloc_size_of = "Rc is hard"]
    abort_request_promise: Rc<Promise>,
}

impl Callback for AbortAlgorithmRejectionHandler {
    /// Callback method invoked when the abort operation's promise is rejected.
    /// Functional Utility: Rejects the `abort_request_promise` with the given reason
    /// and then calls `WritableStreamRejectCloseAndClosedPromiseIfNeeded` to handle stream state.
    /// @param cx The current JavaScript context.
    /// @param reason The reason the promise was rejected with.
    /// @param _realm The current JavaScript realm.
    /// @param _can_gc A `CanGc` token.
    fn callback(
        &self,
        cx: SafeJSContext,
        reason: SafeHandleValue,
        _realm: InRealm,
        _can_gc: CanGc,
    ) {
        // Reject abortRequest’s promise with reason.
        self.abort_request_promise.reject_native(&reason);

        // Perform ! WritableStreamRejectCloseAndClosedPromiseIfNeeded(stream).
        self.stream
            .as_rooted()
            .reject_close_and_closed_promise_if_needed(cx);
    }
}
impl js::gc::Rootable for PendingAbortRequest {}

/// Represents a pending abort request for a `WritableStream`.
/// <https://streams.spec.whatwg.org/#pending-abort-request>
/// Functional Utility: Stores the promise, reason, and state related to an
/// ongoing or pending abort operation on a `WritableStream`, ensuring that
/// the abort request can be managed and its outcome propagated.
#[derive(JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
struct PendingAbortRequest {
    /// The promise associated with this abort request.
    /// <https://streams.spec.whatwg.org/#pending-abort-request-promise>
    #[ignore_malloc_size_of = "Rc is hard"]
    promise: Rc<Promise>,

    /// The reason provided for the abort request.
    /// <https://streams.spec.whatwg.org/#pending-abort-request-reason>
    #[ignore_malloc_size_of = "mozjs"]
    reason: Box<Heap<JSVal>>,

    /// A flag indicating whether the stream was already in an "erroring" state when the abort was initiated.
    /// <https://streams.spec.whatwg.org/#pending-abort-request-was-already-erroring>
    was_already_erroring: bool,
}
/// Represents the internal state of a `WritableStream`.
/// <https://streams.spec.whatwg.org/#writablestream-state>
/// Functional Utility: Essential for managing the lifecycle of the stream,
/// dictating which operations are valid at any given time.
#[derive(Clone, Copy, Debug, Default, JSTraceable, MallocSizeOf)]
pub(crate) enum WritableStreamState {
    /// The stream is open and can be written to.
    #[default]
    Writable,
    /// The stream has been successfully closed.
    Closed,
    /// The stream is in the process of erroring.
    Erroring,
    /// The stream has errored and cannot be used further.
    Errored,
}
/// Implements the Web Streams API `WritableStream`.
/// <https://streams.spec.whatwg.org/#ws-class>
/// Functional Utility: Provides an object for writing streamed data to an underlying sink.
/// It manages the stream's state, backpressure, write requests, and error handling
/// according to the Web Streams specification.
#[dom_struct]
pub struct WritableStream {
    /// The `Reflector` provides the bridge between the native Rust object and its JS counterpart.
    reflector_: Reflector,

    /// A flag indicating whether the stream is currently experiencing backpressure.
    /// <https://streams.spec.whatwg.org/#writablestream-backpressure>
    backpressure: Cell<bool>,

    /// A promise that resolves when the stream's close operation is requested.
    /// <https://streams.spec.whatwg.org/#writablestream-closerequest>
    #[ignore_malloc_size_of = "Rc is hard"]
    close_request: DomRefCell<Option<Rc<Promise>>>,

    /// The associated `WritableStreamDefaultController` for managing the stream.
    /// <https://streams.spec.whatwg.org/#writablestream-controller>
    controller: MutNullableDom<WritableStreamDefaultController>,

    /// A flag indicating whether the stream has been detached.
    /// <https://streams.spec.whatwg.org/#writablestream-detached>
    detached: Cell<bool>,

    /// A promise representing a write request that is currently in flight.
    /// <https://streams.spec.whatwg.org/#writablestream-inflightwriterequest>
    #[ignore_malloc_size_of = "Rc is hard"]
    in_flight_write_request: DomRefCell<Option<Rc<Promise>>>,

    /// A promise representing a close request that is currently in flight.
    /// <https://streams.spec.whatwg.org/#writablestream-inflightcloserequest>
    #[ignore_malloc_size_of = "Rc is hard"]
    in_flight_close_request: DomRefCell<Option<Rc<Promise>>>,

    /// Stores details about a pending abort request.
    /// <https://streams.spec.whatwg.org/#writablestream-pendingabortrequest>
    pending_abort_request: DomRefCell<Option<PendingAbortRequest>>,

    /// The current state of the writable stream (e.g., writable, closed, errored).
    /// <https://streams.spec.whatwg.org/#writablestream-state>
    state: Cell<WritableStreamState>,

    /// Stores the JavaScript value representing the error that caused the stream to enter the "errored" state.
    /// <https://streams.spec.whatwg.org/#writablestream-storederror>
    #[ignore_malloc_size_of = "mozjs"]
    stored_error: Heap<JSVal>,

    /// The associated `WritableStreamDefaultWriter` that is currently locked to this stream.
    /// <https://streams.spec.whatwg.org/#writablestream-writer>
    writer: MutNullableDom<WritableStreamDefaultWriter>,

    /// A queue of promises representing pending write requests.
    /// <https://streams.spec.whatwg.org/#writablestream-writerequests>
    #[ignore_malloc_size_of = "Rc is hard"]
    write_requests: DomRefCell<VecDeque<Rc<Promise>>>,
}

impl WritableStream {
    /// Internal constructor for creating an inherited `WritableStream` instance.
    /// Functional Utility: Initializes the internal state variables of the stream to
    /// their default values as specified by the Streams API.
    /// <https://streams.spec.whatwg.org/#initialize-readable-stream>
    /// @return A new `WritableStream` instance with default internal state.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    fn new_inherited() -> WritableStream {
        WritableStream {
            reflector_: Reflector::new(),
            backpressure: Default::default(),
            close_request: Default::default(),
            controller: Default::default(),
            detached: Default::default(),
            in_flight_write_request: Default::default(),
            in_flight_close_request: Default::default(),
            pending_abort_request: Default::default(),
            state: Default::default(),
            stored_error: Default::default(),
            writer: Default::default(),
            write_requests: Default::default(),
        }
    }

    /// Creates a new `DomRoot<WritableStream>` instance, reflecting it into the JavaScript DOM.
    /// Functional Utility: This is the primary entry point for instantiating `WritableStream`
    /// objects that are accessible from JavaScript, ensuring they are properly
    /// rooted and managed by the DOM.
    /// @param global The `GlobalScope` in which the object is created.
    /// @param proto An optional `SafeHandleObject` to use as the JavaScript prototype.
    /// @param can_gc A `CanGc` token.
    /// @return A `DomRoot<WritableStream>` representing the new instance.
    fn new_with_proto(
        global: &GlobalScope,
        proto: Option<SafeHandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<WritableStream> {
        reflect_dom_object_with_proto(
            Box::new(WritableStream::new_inherited()),
            global,
            proto,
            can_gc,
        )
    }

    /// Asserts that the stream currently has no associated controller.
    /// Functional Utility: Used as a precondition check during the setup of a
    /// `WritableStreamDefaultController` to ensure proper initialization order.
    /// <https://streams.spec.whatwg.org/#set-up-writable-stream-default-controller>
    pub(crate) fn assert_no_controller(&self) {
        assert!(self.controller.get().is_none());
    }

    /// Sets the default controller for the stream.
    /// Functional Utility: Associates a `WritableStreamDefaultController` with the
    /// `WritableStream`, delegating control over the underlying sink.
    /// <https://streams.spec.whatwg.org/#set-up-writable-stream-default-controller>
    /// @param controller The `WritableStreamDefaultController` to associate with the stream.
    pub(crate) fn set_default_controller(&self, controller: &WritableStreamDefaultController) {
        self.controller.set(Some(controller));
    }

    /// Checks if the stream's state is "writable".
    /// Functional Utility: Indicates whether the stream is open and ready to accept new writes.
    /// @return `true` if the stream is writable, `false` otherwise.
    pub(crate) fn is_writable(&self) -> bool {
        matches!(self.state.get(), WritableStreamState::Writable)
    }

    /// Checks if the stream's state is "erroring".
    /// Functional Utility: Indicates whether the stream is in the process of erroring.
    /// @return `true` if the stream is erroring, `false` otherwise.
    pub(crate) fn is_erroring(&self) -> bool {
        matches!(self.state.get(), WritableStreamState::Erroring)
    }

    /// Checks if the stream's state is "errored".
    /// Functional Utility: Indicates whether the stream has encountered an unrecoverable error.
    /// @return `true` if the stream is errored, `false` otherwise.
    pub(crate) fn is_errored(&self) -> bool {
        matches!(self.state.get(), WritableStreamState::Errored)
    }

    /// Checks if the stream's state is "closed".
    /// Functional Utility: Indicates whether the stream has been successfully closed.
    /// @return `true` if the stream is closed, `false` otherwise.
    pub(crate) fn is_closed(&self) -> bool {
        matches!(self.state.get(), WritableStreamState::Closed)
    }

    /// Checks if there is a write request currently in flight.
    /// Functional Utility: Indicates whether an asynchronous write operation has been initiated
    /// but has not yet completed or settled.
    /// @return `true` if a write request is in flight, `false` otherwise.
    pub(crate) fn has_in_flight_write_request(&self) -> bool {
        self.in_flight_write_request.borrow().is_some()
    }

    /// Checks if there are any write or close operations currently in flight.
    /// <https://streams.spec.whatwg.org/#writable-stream-has-operation-marked-in-flight>
    /// Functional Utility: Provides an aggregate status of ongoing asynchronous operations
    /// that affect the stream's state transitions (e.g., preventing closure while writes are pending).
    /// @return `true` if any write or close request is in flight, `false` otherwise.
    pub(crate) fn has_operations_marked_inflight(&self) -> bool {
        let in_flight_write_requested = self.in_flight_write_request.borrow().is_some();
        let in_flight_close_requested = self.in_flight_close_request.borrow().is_some();

        in_flight_write_requested || in_flight_close_requested
    }

    /// Retrieves the error stored in the stream.
    /// <https://streams.spec.whatwg.org/#writablestream-storederror>
    /// Functional Utility: Allows inspection of the error that caused the stream to enter
    /// the "errored" state, providing diagnostic information.
    /// @param mut handle_mut A mutable handle to receive the stored error value.
    pub(crate) fn get_stored_error(&self, mut handle_mut: SafeMutableHandleValue) {
        handle_mut.set(self.stored_error.get());
    }

    /// Transitions the `WritableStream` from "erroring" to "errored" state,
    /// rejecting pending write requests and handling abort requests.
    /// <https://streams.spec.whatwg.org/#writable-stream-finish-erroring>
    /// Functional Utility: Centralizes the logic for gracefully transitioning
    /// the stream into a final errored state, cleaning up pending operations,
    /// and ensuring all associated promises are correctly rejected or resolved.
    /// @param cx The JavaScript context.
    /// @param global The `GlobalScope` context.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn finish_erroring(&self, cx: SafeJSContext, global: &GlobalScope, can_gc: CanGc) {
        // Assert: stream.[[state]] is "erroring".
        assert!(self.is_erroring());

        // Assert: ! WritableStreamHasOperationMarkedInFlight(stream) is false.
        assert!(!self.has_operations_marked_inflight());

        // Set stream.[[state]] to "errored".
        self.state.set(WritableStreamState::Errored);

        // Perform ! stream.[[controller]].[[ErrorSteps]]().
        let Some(controller) = self.controller.get() else {
            unreachable!("Stream should have a controller.");
        };
        controller.perform_error_steps();

        // Let storedError be stream.[[storedError]].
        rooted!(in(*cx) let mut stored_error = UndefinedValue());
        self.get_stored_error(stored_error.handle_mut());

        // For each writeRequest of stream.[[writeRequests]]:
        // Block Logic: Rejects all pending write requests with the stored error.
        let write_requests = mem::take(&mut *self.write_requests.borrow_mut());
        for request in write_requests {
            // Reject writeRequest with storedError.
            request.reject(cx, stored_error.handle());
        }

        // Set stream.[[writeRequests]] to an empty list.
        // Done above with `drain`.

        // If stream.[[pendingAbortRequest]] is undefined,
        if self.pending_abort_request.borrow().is_none() {
            // Perform ! WritableStreamRejectCloseAndClosedPromiseIfNeeded(stream).
            self.reject_close_and_closed_promise_if_needed(cx);

            // Return.
            return;
        }

        // Let abortRequest be stream.[[pendingAbortRequest]].
        // Set stream.[[pendingAbortRequest]] to undefined.
        rooted!(in(*cx) let pending_abort_request = self.pending_abort_request.borrow_mut().take());
        if let Some(pending_abort_request) = &*pending_abort_request {
            // If abortRequest’s was already erroring is true,
            if pending_abort_request.was_already_erroring {
                // Reject abortRequest’s promise with storedError.
                pending_abort_request
                    .promise
                    .reject(cx, stored_error.handle());

                // Perform ! WritableStreamRejectCloseAndClosedPromiseIfNeeded(stream).
                self.reject_close_and_closed_promise_if_needed(cx);

                // Return.
                return;
            }

            // Let promise be ! stream.[[controller]].[[AbortSteps]](abortRequest’s reason).
            rooted!(in(*cx) let mut reason = UndefinedValue());
            reason.set(pending_abort_request.reason.get());
            let promise = controller.abort_steps(cx, global, reason.handle(), can_gc);

            // Upon fulfillment of promise,
            rooted!(in(*cx) let mut fulfillment_handler = Some(AbortAlgorithmFulfillmentHandler {
                stream: Dom::from_ref(self),
                abort_request_promise: pending_abort_request.promise.clone(),
            }));

            // Upon rejection of promise with reason r,
            rooted!(in(*cx) let mut rejection_handler = Some(AbortAlgorithmRejectionHandler {
                stream: Dom::from_ref(self),
                abort_request_promise: pending_abort_request.promise.clone(),
            }));

            let handler = PromiseNativeHandler::new(
                global,
                fulfillment_handler.take().map(|h| Box::new(h) as Box<_>),
                rejection_handler.take().map(|h| Box::new(h) as Box<_>),
                can_gc,
            );
            let realm = enter_realm(global);
            let comp = InRealm::Entered(&realm);
            promise.append_native_handler(&handler, comp, can_gc);
        }
    }

    /// Rejects the stream's close request and the writer's closed promise if they exist and the stream is in an errored state.
    /// <https://streams.spec.whatwg.org/#writable-stream-reject-close-and-closed-promise-if-needed>
    /// Functional Utility: Ensures that promises related to closing the stream are appropriately
    /// rejected when the stream has entered an errored state, preventing hanging promises.
    /// @param cx The JavaScript context.
    pub(crate) fn reject_close_and_closed_promise_if_needed(&self, cx: SafeJSContext) {
        // Assert: stream.[[state]] is "errored".
        assert!(self.is_errored());

        rooted!(in(*cx) let mut stored_error = UndefinedValue());
        self.get_stored_error(stored_error.handle_mut());

        // If stream.[[closeRequest]] is not undefined
        // Block Logic: If a close request is pending, it is rejected with the stored error.
        let close_request = self.close_request.borrow_mut().take();
        if let Some(close_request) = close_request {
            // Assert: stream.[[inFlightCloseRequest]] is undefined.
            assert!(self.in_flight_close_request.borrow().is_none());

            // Reject stream.[[closeRequest]] with stream.[[storedError]].
            close_request.reject_native(&stored_error.handle())

            // Set stream.[[closeRequest]] to undefined.
            // Done with `take` above.
        }

        // Let writer be stream.[[writer]].
        // If writer is not undefined,
        // Block Logic: If a writer is associated with the stream, its closed promise is rejected.
        if let Some(writer) = self.writer.get() {
            // Reject writer.[[closedPromise]] with stream.[[storedError]].
            writer.reject_closed_promise_with_stored_error(&stored_error.handle());

            // Set writer.[[closedPromise]].[[PromiseIsHandled]] to true.
            writer.set_close_promise_is_handled();
        }
    }
    /// Checks if a close operation is either queued or currently in flight.
    /// <https://streams.spec.whatwg.org/#writable-stream-close-queued-or-in-flight>
    /// Functional Utility: Used to determine if the stream is already in the process
    /// of being closed, preventing redundant or conflicting close attempts.
    /// @return `true` if a close is queued or in flight, `false` otherwise.
    pub(crate) fn close_queued_or_in_flight(&self) -> bool {
        let close_requested = self.close_request.borrow().is_some();
        let in_flight_close_requested = self.in_flight_close_request.borrow().is_some();

        close_requested || in_flight_close_requested
    }

    /// Resolves the promise associated with the current in-flight write request.
    /// <https://streams.spec.whatwg.org/#writable-stream-finish-in-flight-write>
    /// Functional Utility: Marks the completion of an asynchronous write operation,
    /// resolving the corresponding promise and clearing the in-flight status.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn finish_in_flight_write(&self, can_gc: CanGc) {
        let Some(in_flight_write_request) = self.in_flight_write_request.borrow_mut().take() else {
            // Assert: stream.[[inFlightWriteRequest]] is not undefined.
            unreachable!("Stream should have a write request");
        };

        // Resolve stream.[[inFlightWriteRequest]] with undefined.
        in_flight_write_request.resolve_native(&(), can_gc);

        // Set stream.[[inFlightWriteRequest]] to undefined.
        // Done above with `take`.
    }

    /// Initiates the erroring process for the `WritableStream`.
    /// <https://streams.spec.whatwg.org/#writable-stream-start-erroring>
    /// Functional Utility: Transitions the stream to the "erroring" state, stores the error,
    /// and rejects any ready promises held by the writer. If no operations are in flight,
    /// it proceeds to immediately finish erroring.
    /// @param cx The JavaScript context.
    /// @param global The `GlobalScope` context.
    /// @param error The `SafeHandleValue` representing the error reason.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn start_erroring(
        &self,
        cx: SafeJSContext,
        global: &GlobalScope,
        error: SafeHandleValue,
        can_gc: CanGc,
    ) {
        // Assert: stream.[[storedError]] is undefined.
        assert!(self.stored_error.get().is_undefined());

        // Assert: stream.[[state]] is "writable".
        assert!(self.is_writable());

        // Let controller be stream.[[controller]].
        let Some(controller) = self.controller.get() else {
            unreachable!("Stream should have a controller.");
        };

        // Set stream.[[state]] to "erroring".
        self.state.set(WritableStreamState::Erroring);

        // Set stream.[[storedError]] to reason.
        self.stored_error.set(*error);

        // Let writer be stream.[[writer]].
        if let Some(writer) = self.writer.get() {
            // If writer is not undefined, perform ! WritableStreamDefaultWriterEnsureReadyPromiseRejected
            writer.ensure_ready_promise_rejected(global, error, can_gc);
        }

        // If ! WritableStreamHasOperationMarkedInFlight(stream) is false and controller.[[started]] is true
        // Block Logic: If no operations are in flight and the controller has started, finish erroring immediately.
        if !self.has_operations_marked_inflight() && controller.started() {
            // perform ! WritableStreamFinishErroring
            self.finish_erroring(cx, global, can_gc);
        }
    }

    /// Handles a rejection (error) in the `WritableStream`.
    /// <https://streams.spec.whatwg.org/#writable-stream-deal-with-rejection>
    /// Functional Utility: Orchestrates the error handling logic, transitioning the stream
    /// to an erroring state if currently writable, or completing the erroring process
    /// if it's already in the "erroring" state.
    /// @param cx The JavaScript context.
    /// @param global The `GlobalScope` context.
    /// @param error The `SafeHandleValue` representing the error reason.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn deal_with_rejection(
        &self,
        cx: SafeJSContext,
        global: &GlobalScope,
        error: SafeHandleValue,
        can_gc: CanGc,
    ) {
        // Let state be stream.[[state]].

        // If state is "writable",
        // Block Logic: If the stream is writable, initiate the erroring process.
        if self.is_writable() {
            // Perform ! WritableStreamStartErroring(stream, error).
            self.start_erroring(cx, global, error, can_gc);

            // Return.
            return;
        }

        // Assert: state is "erroring".
        assert!(self.is_erroring());

        // Perform ! WritableStreamFinishErroring(stream).
        // Block Logic: If the stream is already erroring, finalize the erroring process.
        self.finish_erroring(cx, global, can_gc);
    }
    /// Marks the first queued write request as in-flight.
    /// <https://streams.spec.whatwg.org/#writable-stream-mark-first-write-request-in-flight>
    /// Functional Utility: Moves a pending write request from the queue to the in-flight state,
    /// indicating that its associated operation has begun.
    /// Precondition: There must be no write request currently in flight, and the write queue must not be empty.
    /// Postcondition: The first request from `write_requests` is moved to `in_flight_write_request`.
    pub(crate) fn mark_first_write_request_in_flight(&self) {
        let mut in_flight_write_request = self.in_flight_write_request.borrow_mut();
        let mut write_requests = self.write_requests.borrow_mut();

        // Assert: stream.[[inFlightWriteRequest]] is undefined.
        assert!(in_flight_write_request.is_none());

        // Assert: stream.[[writeRequests]] is not empty.
        assert!(!write_requests.is_empty());

        // Let writeRequest be stream.[[writeRequests]][0].
        // Remove writeRequest from stream.[[writeRequests]].
        let write_request = write_requests.pop_front().unwrap();

        // Set stream.[[inFlightWriteRequest]] to writeRequest.
        *in_flight_write_request = Some(write_request);
    }

    /// Marks the queued close request as in-flight.
    /// <https://streams.spec.whatwg.org/#writable-stream-mark-close-request-in-flight>
    /// Functional Utility: Moves a pending close request from the queue to the in-flight state,
    /// indicating that the asynchronous close operation has begun.
    /// Precondition: There must be no close request currently in flight, and a close request must be queued.
    /// Postcondition: The queued close request is moved to `in_flight_close_request`.
    pub(crate) fn mark_close_request_in_flight(&self) {
        let mut in_flight_close_request = self.in_flight_close_request.borrow_mut();
        let mut close_request = self.close_request.borrow_mut();

        // Assert: stream.[[inFlightCloseRequest]] is undefined.
        assert!(in_flight_close_request.is_none());

        // Assert: stream.[[closeRequest]] is not undefined.
        assert!(close_request.is_some());

        // Let closeRequest be stream.[[closeRequest]].
        // Set stream.[[closeRequest]] to undefined.
        let close_request = close_request.take().unwrap();

        // Set stream.[[inFlightCloseRequest]] to closeRequest.
        *in_flight_close_request = Some(close_request);
    }

    /// Completes an in-flight close operation for the `WritableStream`.
    /// <https://streams.spec.whatwg.org/#writable-stream-finish-in-flight-close>
    /// Functional Utility: Resolves the promise associated with the in-flight close request,
    /// transitions the stream to the "closed" state, resolves the writer's closed promise,
    /// and cleans up any related pending abort requests or stored errors.
    /// @param cx The JavaScript context.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn finish_in_flight_close(&self, cx: SafeJSContext, can_gc: CanGc) {
        let Some(in_flight_close_request) = self.in_flight_close_request.borrow_mut().take() else {
            // Assert: stream.[[inFlightCloseRequest]] is not undefined.
            unreachable!("in_flight_close_request must be Some");
        };

        // Resolve stream.[[inFlightCloseRequest]] with undefined.
        in_flight_close_request.resolve_native(&(), can_gc);

        // Set stream.[[inFlightCloseRequest]] to undefined.
        // Done with take above.

        // Assert: stream.[[state]] is "writable" or "erroring".
        assert!(self.is_writable() || self.is_erroring());

        // If state is "erroring",
        if self.is_erroring() {
            // Set stream.[[storedError]] to undefined.
            self.stored_error.set(UndefinedValue());

            // If stream.[[pendingAbortRequest]] is not undefined,
            rooted!(in(*cx) let pending_abort_request = self.pending_abort_request.borrow_mut().take());
            if let Some(pending_abort_request) = &*pending_abort_request {
                // Resolve stream.[[pendingAbortRequest]]'s promise with undefined.
                pending_abort_request.promise.resolve_native(&(), can_gc);

                // Set stream.[[pendingAbortRequest]] to undefined.
                // Done above with `take`.
            }
        }

        // Set stream.[[state]] to "closed".
        self.state.set(WritableStreamState::Closed);

        // Let writer be stream.[[writer]].
        if let Some(writer) = self.writer.get() {
            // If writer is not undefined,
            // resolve writer.[[closedPromise]] with undefined.
            writer.resolve_closed_promise_with_undefined(can_gc);
        }

        // Assert: stream.[[pendingAbortRequest]] is undefined.
        assert!(self.pending_abort_request.borrow().is_none());

        // Assert: stream.[[storedError]] is undefined.
        assert!(self.stored_error.get().is_undefined());
    }
    /// Completes an in-flight close operation with an error for the `WritableStream`.
    /// <https://streams.spec.whatwg.org/#writable-stream-finish-in-flight-close-with-error>
    /// Functional Utility: Rejects the promise associated with the in-flight close request,
    /// and then initiates the general rejection handling for the stream, ensuring
    /// proper error propagation and state transition.
    /// @param cx The JavaScript context.
    /// @param global The `GlobalScope` context.
    /// @param error The `SafeHandleValue` representing the error reason.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn finish_in_flight_close_with_error(
        &self,
        cx: SafeJSContext,
        global: &GlobalScope,
        error: SafeHandleValue,
        can_gc: CanGc,
    ) {
        let Some(in_flight_close_request) = self.in_flight_close_request.borrow_mut().take() else {
            // Assert: stream.[[inFlightCloseRequest]] is not undefined.
            unreachable!("Inflight close request must be defined.");
        };

        // Reject stream.[[inFlightCloseRequest]] with error.
        in_flight_close_request.reject_native(&error);

        // Set stream.[[inFlightCloseRequest]] to undefined.
        // Done above with `take`.

        // Assert: stream.[[state]] is "writable" or "erroring".
        assert!(self.is_erroring() || self.is_writable());

        // If stream.[[pendingAbortRequest]] is not undefined,
        rooted!(in(*cx) let pending_abort_request = self.pending_abort_request.borrow_mut().take());
        if let Some(pending_abort_request) = &*pending_abort_request {
            // Reject stream.[[pendingAbortRequest]]'s promise with error.
            pending_abort_request.promise.reject_native(&error);

            // Set stream.[[pendingAbortRequest]] to undefined.
            // Done above with `take`.
        }

        // Perform ! WritableStreamDealWithRejection(stream, error).
        self.deal_with_rejection(cx, global, error, can_gc);
    }

    /// Completes an in-flight write operation with an error for the `WritableStream`.
    /// <https://streams.spec.whatwg.org/#writable-stream-finish-in-flight-write-with-error>
    /// Functional Utility: Rejects the promise associated with the in-flight write request
    /// and then initiates the general rejection handling for the stream, ensuring
    /// proper error propagation and state transition.
    /// @param cx The JavaScript context.
    /// @param global The `GlobalScope` context.
    /// @param error The `SafeHandleValue` representing the error reason.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn finish_in_flight_write_with_error(
        &self,
        cx: SafeJSContext,
        global: &GlobalScope,
        error: SafeHandleValue,
        can_gc: CanGc,
    ) {
        let Some(in_flight_write_request) = self.in_flight_write_request.borrow_mut().take() else {
            // Assert: stream.[[inFlightWriteRequest]] is not undefined.
            unreachable!("Inflight write request must be defined.");
        };

        // Reject stream.[[inFlightWriteRequest]] with error.
        in_flight_write_request.reject_native(&error);

        // Set stream.[[inFlightWriteRequest]] to undefined.
        // Done above with `take`.

        // Assert: stream.[[state]] is "writable" or "erroring".
        assert!(self.is_errored() || self.is_writable());

        // Perform ! WritableStreamDealWithRejection(stream, error).
        self.deal_with_rejection(cx, global, error, can_gc);
    }

    /// Returns the `WritableStreamDefaultWriter` currently locked to this stream, if any.
    /// Functional Utility: Provides access to the writer object, which can be used
    /// to interact with the stream (e.g., write chunks, close, abort).
    /// @return An `Option<DomRoot<WritableStreamDefaultWriter>>` representing the writer.
    pub(crate) fn get_writer(&self) -> Option<DomRoot<WritableStreamDefaultWriter>> {
        self.writer.get()
    }

    /// Sets the `WritableStreamDefaultWriter` locked to this stream.
    /// Functional Utility: Associates a writer with the stream, marking it as locked.
    /// @param writer The `WritableStreamDefaultWriter` to lock to this stream, or `None` to unlock.
    pub(crate) fn set_writer(&self, writer: Option<&WritableStreamDefaultWriter>) {
        self.writer.set(writer);
    }

    /// Sets the backpressure state of the stream.
    /// Functional Utility: Updates the internal flag indicating whether the stream
    /// is currently experiencing backpressure, affecting subsequent write operations.
    /// @param backpressure A boolean value indicating the new backpressure state.
    pub(crate) fn set_backpressure(&self, backpressure: bool) {
        self.backpressure.set(backpressure);
    }

    /// Returns the current backpressure state of the stream.
    /// Functional Utility: Allows external code to query whether the stream is
    /// currently experiencing backpressure.
    /// @return `true` if the stream is under backpressure, `false` otherwise.
    pub(crate) fn get_backpressure(&self) -> bool {
        self.backpressure.get()
    }

    /// Checks if the stream is currently locked to a writer.
    /// <https://streams.spec.whatwg.org/#is-writable-stream-locked>
    /// Functional Utility: Determines if a `WritableStreamDefaultWriter` has been
    /// acquired for this stream, which prevents other writers from being acquired.
    /// @return `true` if the stream is locked, `false` otherwise.
    pub(crate) fn is_locked(&self) -> bool {
        // If stream.[[writer]] is undefined, return false.
        // Return true.
        self.get_writer().is_some()
    }

    /// Adds a new promise to the queue of pending write requests.
    /// <https://streams.spec.whatwg.org/#writable-stream-add-write-request>
    /// Functional Utility: Enqueues an asynchronous write operation, returning a promise
    /// that will be resolved or rejected based on the outcome of the write.
    /// Precondition: The stream must be locked and in a "writable" state.
    /// Postcondition: A new promise is appended to `write_requests`.
    /// @param global The `GlobalScope` context.
    /// @param can_gc A `CanGc` token.
    /// @return A new `Rc<Promise>` representing the write request.
    pub(crate) fn add_write_request(&self, global: &GlobalScope, can_gc: CanGc) -> Rc<Promise> {
        // Assert: ! IsWritableStreamLocked(stream) is true.
        assert!(self.is_locked());

        // Assert: stream.[[state]] is "writable".
        assert!(self.is_writable());

        // Let promise be a new promise.
        let promise = Promise::new(global, can_gc);

        // Append promise to stream.[[writeRequests]].
        self.write_requests.borrow_mut().push_back(promise.clone());

        // Return promise.
        promise
    }
    // Returns the rooted controller of the stream, if any.
    pub(crate) fn get_controller(&self) -> Option<DomRoot<WritableStreamDefaultController>> {
        self.controller.get()
    }

    /// <https://streams.spec.whatwg.org/#writable-stream-abort>
    pub(crate) fn abort(
        &self,
        cx: SafeJSContext,
        global: &GlobalScope,
        provided_reason: SafeHandleValue,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        // If stream.[[state]] is "closed" or "errored",
        if self.is_closed() || self.is_errored() {
            // return a promise resolved with undefined.
            return Promise::new_resolved(global, cx, (), can_gc);
        }

        // TODO: Signal abort on stream.[[controller]].[[abortController]] with reason.

        // TODO: If state is "closed" or "errored", return a promise resolved with undefined.
        // Note: state may have changed because of signal above.

        // If stream.[[pendingAbortRequest]] is not undefined,
        if self.pending_abort_request.borrow().is_some() {
            // return stream.[[pendingAbortRequest]]'s promise.
            return self
                .pending_abort_request
                .borrow()
                .as_ref()
                .expect("Pending abort request must be Some.")
                .promise
                .clone();
        }

        // Assert: state is "writable" or "erroring".
        assert!(self.is_writable() || self.is_erroring());

        // Let wasAlreadyErroring be false.
        let mut was_already_erroring = false;
        rooted!(in(*cx) let undefined_reason = UndefinedValue());

        // If state is "erroring",
        let reason = if self.is_erroring() {
            // Set wasAlreadyErroring to true.
            was_already_erroring = true;

            // Set reason to undefined.
            undefined_reason.handle()
        } else {
            // Use the provided reason.
            provided_reason
        };

        // Let promise be a new promise.
        let promise = Promise::new(global, can_gc);

        // Set stream.[[pendingAbortRequest]] to a new pending abort request
        // whose promise is promise,
        // reason is reason,
        // and was already erroring is wasAlreadyErroring.
        *self.pending_abort_request.borrow_mut() = Some(PendingAbortRequest {
            promise: promise.clone(),
            reason: Heap::boxed(reason.get()),
            was_already_erroring,
        });

        // If wasAlreadyErroring is false,
        if !was_already_erroring {
            // perform ! WritableStreamStartErroring(stream, reason)
            self.start_erroring(cx, global, reason, can_gc);
        }

        // Return promise.
        promise
    }

    /// <https://streams.spec.whatwg.org/#writable-stream-close>
    pub(crate) fn close(
        &self,
        cx: SafeJSContext,
        global: &GlobalScope,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        // Let state be stream.[[state]].
        // If state is "closed" or "errored",
        if self.is_closed() || self.is_errored() {
            // return a promise rejected with a TypeError exception.
            let promise = Promise::new(global, can_gc);
            promise.reject_error(Error::Type("Stream is closed or errored.".to_string()));
            return promise;
        }

        // Assert: state is "writable" or "erroring".
        assert!(self.is_writable() || self.is_erroring());

        // Assert: ! WritableStreamCloseQueuedOrInFlight(stream) is false.
        assert!(!self.close_queued_or_in_flight());

        // Let promise be a new promise.
        let promise = Promise::new(global, can_gc);

        // Set stream.[[closeRequest]] to promise.
        *self.close_request.borrow_mut() = Some(promise.clone());

        // Let writer be stream.[[writer]].
        // If writer is not undefined,
        if let Some(writer) = self.writer.get() {
            // and stream.[[backpressure]] is true,
            // and state is "writable",
            if self.get_backpressure() && self.is_writable() {
                // resolve writer.[[readyPromise]] with undefined.
                writer.resolve_ready_promise_with_undefined(can_gc);
            }
        }

        // Perform ! WritableStreamDefaultControllerClose(stream.[[controller]]).
        let Some(controller) = self.controller.get() else {
            unreachable!("Stream must have a controller.");
        };
        controller.close(cx, global, can_gc);

        // Return promise.
        promise
    }

    /// <https://streams.spec.whatwg.org/#writable-stream-default-writer-get-desired-size>
    /// Note: implement as a stream method, as opposed to a writer one, for convenience.
    pub(crate) fn get_desired_size(&self) -> Option<f64> {
        // Let stream be writer.[[stream]].
        // Stream is `self`.

        // Let state be stream.[[state]].
        // If state is "errored" or "erroring", return null.
        if self.is_errored() || self.is_erroring() {
            return None;
        }

        // If state is "closed", return 0.
        if self.is_closed() {
            return Some(0.);
        }

        let Some(controller) = self.controller.get() else {
            unreachable!("Stream must have a controller.");
        };
        Some(controller.get_desired_size())
    }

    /// <https://streams.spec.whatwg.org/#acquire-writable-stream-default-writer>
    pub(crate) fn aquire_default_writer(
        &self,
        cx: SafeJSContext,
        global: &GlobalScope,
        can_gc: CanGc,
    ) -> Result<DomRoot<WritableStreamDefaultWriter>, Error> {
        // Let writer be a new WritableStreamDefaultWriter object.
        let writer = WritableStreamDefaultWriter::new(global, None, can_gc);

        // Perform ? SetUpWritableStreamDefaultWriter(writer, stream).
        writer.setup(cx, self, can_gc)?;

        // Return writer.
        Ok(writer)
    }

    /// <https://streams.spec.whatwg.org/#writable-stream-update-backpressure>
    pub(crate) fn update_backpressure(
        &self,
        backpressure: bool,
        global: &GlobalScope,
        can_gc: CanGc,
    ) {
        // Assert: stream.[[state]] is "writable".
        self.is_writable();

        // Assert: ! WritableStreamCloseQueuedOrInFlight(stream) is false.
        assert!(!self.close_queued_or_in_flight());

        // Let writer be stream.[[writer]].
        let writer = self.get_writer();
        if writer.is_some() && backpressure != self.get_backpressure() {
            // If writer is not undefined
            let writer = writer.expect("Writer is some, as per the above check.");
            // and backpressure is not stream.[[backpressure]],
            if backpressure {
                // If backpressure is true, set writer.[[readyPromise]] to a new promise.
                let promise = Promise::new(global, can_gc);
                writer.set_ready_promise(promise);
            } else {
                // Otherwise,
                // Assert: backpressure is false.
                assert!(!backpressure);
                // Resolve writer.[[readyPromise]] with undefined.
                writer.resolve_ready_promise_with_undefined(can_gc);
            }
        };

        // Set stream.[[backpressure]] to backpressure.
        self.set_backpressure(backpressure);
    }
}

impl WritableStreamMethods<crate::DomTypeHolder> for WritableStream {
    /// <https://streams.spec.whatwg.org/#ws-constructor>
    fn Constructor(
        cx: SafeJSContext,
        global: &GlobalScope,
        proto: Option<SafeHandleObject>,
        can_gc: CanGc,
        underlying_sink: Option<*mut JSObject>,
        strategy: &QueuingStrategy,
    ) -> Fallible<DomRoot<WritableStream>> {
        // If underlyingSink is missing, set it to null.
        rooted!(in(*cx) let underlying_sink_obj = underlying_sink.unwrap_or(ptr::null_mut()));

        // Let underlyingSinkDict be underlyingSink,
        // converted to an IDL value of type UnderlyingSink.
        let underlying_sink_dict = if !underlying_sink_obj.is_null() {
            rooted!(in(*cx) let obj_val = ObjectValue(underlying_sink_obj.get()));
            match UnderlyingSink::new(cx, obj_val.handle()) {
                Ok(ConversionResult::Success(val)) => val,
                Ok(ConversionResult::Failure(error)) => return Err(Error::Type(error.to_string())),
                _ => {
                    return Err(Error::JSFailed);
                },
            }
        } else {
            UnderlyingSink::empty()
        };

        if !underlying_sink_dict.type_.handle().is_undefined() {
            // If underlyingSinkDict["type"] exists, throw a RangeError exception.
            return Err(Error::Range("type is set".to_string()));
        }

        // Perform ! InitializeWritableStream(this).
        let stream = WritableStream::new_with_proto(global, proto, can_gc);

        // Let sizeAlgorithm be ! ExtractSizeAlgorithm(strategy).
        let size_algorithm = extract_size_algorithm(strategy, can_gc);

        // Let highWaterMark be ? ExtractHighWaterMark(strategy, 1).
        let high_water_mark = extract_high_water_mark(strategy, 1.0)?;

        // Perform ? SetUpWritableStreamDefaultControllerFromUnderlyingSink
        let controller = WritableStreamDefaultController::new(
            global,
            &underlying_sink_dict,
            high_water_mark,
            size_algorithm,
            can_gc,
        );

        // Note: this must be done before `setup`,
        // otherwise `thisOb` is null in the start callback.
        controller.set_underlying_sink_this_object(underlying_sink_obj.handle());

        // Perform ? SetUpWritableStreamDefaultController
        controller.setup(cx, global, &stream, &underlying_sink_dict.start, can_gc)?;

        Ok(stream)
    }

    /// <https://streams.spec.whatwg.org/#ws-locked>
    fn Locked(&self) -> bool {
        // Return ! IsWritableStreamLocked(this).
        self.is_locked()
    }

    /// <https://streams.spec.whatwg.org/#ws-abort>
    fn Abort(
        &self,
        cx: SafeJSContext,
        reason: SafeHandleValue,
        realm: InRealm,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        let global = GlobalScope::from_safe_context(cx, realm);

        // If ! IsWritableStreamLocked(this) is true,
        if self.is_locked() {
            // return a promise rejected with a TypeError exception.
            let promise = Promise::new(&global, can_gc);
            promise.reject_error(Error::Type("Stream is locked.".to_string()));
            return promise;
        }

        // Return ! WritableStreamAbort(this, reason).
        self.abort(cx, &global, reason, can_gc)
    }

    /// <https://streams.spec.whatwg.org/#ws-close>
    fn Close(&self, realm: InRealm, can_gc: CanGc) -> Rc<Promise> {
        let cx = GlobalScope::get_cx();
        let global = GlobalScope::from_safe_context(cx, realm);

        // If ! IsWritableStreamLocked(this) is true,
        if self.is_locked() {
            // return a promise rejected with a TypeError exception.
            let promise = Promise::new(&global, can_gc);
            promise.reject_error(Error::Type("Stream is locked.".to_string()));
            return promise;
        }

        // If ! WritableStreamCloseQueuedOrInFlight(this) is true
        if self.close_queued_or_in_flight() {
            // return a promise rejected with a TypeError exception.
            let promise = Promise::new(&global, can_gc);
            promise.reject_error(Error::Type(
                "Stream has closed queued or in-flight".to_string(),
            ));
            return promise;
        }

        // Return ! WritableStreamClose(this).
        self.close(cx, &global, can_gc)
    }

    /// <https://streams.spec.whatwg.org/#ws-get-writer>
    fn GetWriter(
        &self,
        realm: InRealm,
        can_gc: CanGc,
    ) -> Result<DomRoot<WritableStreamDefaultWriter>, Error> {
        let cx = GlobalScope::get_cx();
        let global = GlobalScope::from_safe_context(cx, realm);

        // Return ? AcquireWritableStreamDefaultWriter(this).
        self.aquire_default_writer(cx, &global, can_gc)
    }
}
