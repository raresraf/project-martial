//! This module implements the Fetch API's concept of a "body",
//! which represents the content of a request or response. It handles
//! the extraction of various data types into a `ReadableStream`,
//! the transmission of body chunks over IPC, and the consumption
//! of body data by script-side logic.
//!
//! Functional Utility: Centralizes the complex logic of managing
//! request/response payloads, ensuring consistent behavior across
//! different body initializers and facilitating efficient data transfer.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::rc::Rc;
use std::{ptr, slice, str};

use encoding_rs::{Encoding, UTF_8};
use ipc_channel::ipc::{self, IpcReceiver, IpcSender};
use ipc_channel::router::ROUTER;
use js::jsapi::{Heap, JSObject, JS_ClearPendingException, Value as JSValue};
use js::jsval::{JSVal, UndefinedValue};
use js::rust::wrappers::{JS_GetPendingException, JS_ParseJSON};
use js::rust::HandleValue;
use js::typedarray::{ArrayBufferU8, Uint8};
use mime::{self, Mime};
use net_traits::request::{
    BodyChunkRequest, BodyChunkResponse, BodySource as NetBodySource, RequestBody,
};
use script_traits::serializable::BlobImpl;
use url::form_urlencoded;

use crate::dom::bindings::buffer_source::create_buffer_source;
use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::BlobBinding::Blob_Binding::BlobMethods;
use crate::dom::bindings::codegen::Bindings::FormDataBinding::FormDataMethods;
use crate::dom::bindings::codegen::Bindings::XMLHttpRequestBinding::BodyInit;
use crate::dom::bindings::error::{Error, Fallible};
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::{DomGlobal, DomObject};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::bindings::str::{DOMString, USVString};
use crate::dom::bindings::trace::RootedTraceableBox;
use crate::dom::blob::{normalize_type_string, Blob};
use crate::dom::formdata::FormData;
use crate::dom::globalscope::GlobalScope;
use crate::dom::htmlformelement::{encode_multipart_form_data, generate_boundary};
use crate::dom::promise::Promise;
use crate::dom::promisenativehandler::{Callback, PromiseNativeHandler};
use crate::dom::readablestream::{get_read_promise_bytes, get_read_promise_done, ReadableStream};
use crate::dom::urlsearchparams::URLSearchParams;
use crate::realms::{enter_realm, AlreadyInRealm, InRealm};
use crate::script_runtime::{CanGc, JSContext};
use crate::task_source::SendableTaskSource;

/// The Dom object, or ReadableStream, that is the source of a body.
/// <https://fetch.spec.whatwg.org/#concept-body-source>
#[derive(Clone, PartialEq)]
pub(crate) enum BodySource {
    /// Indicates that the body originates from a `ReadableStream`.
    /// Functional Utility: Allows for streaming data directly from a JavaScript `ReadableStream`.
    Null,
    /// Indicates that the body originates from another DOM object (e.g., `Blob`, `FormData`, `ArrayBuffer`).
    /// Functional Utility: Represents bodies that are fully available in memory or can be re-extracted.
    /// TODO: store the actual object
    /// and re-extract a stream on re-direct.
    Object,
}

/// The reason to stop reading from the body.
enum StopReading {
    /// Indicates that the stream has encountered an error and reading should cease.
    Error,
    /// Indicates that the stream has reached its end and reading should cease.
    Done,
}

/// The IPC route handler for <https://fetch.spec.whatwg.org/#concept-request-transmit-body>.
/// This handler runs in the script process and queues tasks to perform operations
/// on the stream, transmitting body chunks over IPC to the network process.
///
/// Functional Utility: Bridges the gap between script-side body management
/// (e.g., `ReadableStream`) and network-side transmission, enabling efficient
/// and asynchronous data transfer.
#[derive(Clone)]
struct TransmitBodyConnectHandler {
    /// The `ReadableStream` from which the body data is read.
    stream: Trusted<ReadableStream>,
    /// The task source for queuing asynchronous operations.
    task_source: SendableTaskSource,
    /// An IPC sender to send `BodyChunkResponse` (data chunks) back to the network process.
    bytes_sender: Option<IpcSender<BodyChunkResponse>>,
    /// An IPC sender to send `BodyChunkRequest` (control messages) to the network process.
    control_sender: IpcSender<BodyChunkRequest>,
    /// Optional in-memory storage for the body bytes if available.
    in_memory: Option<Vec<u8>>,
    /// Flag indicating if the in-memory body has already been transmitted.
    in_memory_done: bool,
    /// The `BodySource` type, indicating the origin of the body.
    source: BodySource,
}

impl TransmitBodyConnectHandler {
    /// Constructs a new `TransmitBodyConnectHandler`.
    ///
    /// @param stream The `ReadableStream` providing the body data.
    /// @param task_source The task source for scheduling operations.
    /// @param control_sender The IPC sender for control requests.
    /// @param in_memory Optional pre-loaded body bytes.
    /// @param source The type of `BodySource`.
    pub(crate) fn new(
        stream: Trusted<ReadableStream>,
        task_source: SendableTaskSource,
        control_sender: IpcSender<BodyChunkRequest>,
        in_memory: Option<Vec<u8>>,
        source: BodySource,
    ) -> TransmitBodyConnectHandler {
        TransmitBodyConnectHandler {
            stream,
            task_source,
            bytes_sender: None,
            control_sender,
            in_memory,
            in_memory_done: false,
            source,
        }
    }

    /// Resets the `in_memory_done` flag.
    /// Functional Utility: Called when a stream is re-extracted from the source
    /// to support re-directs, allowing the in-memory data to be resent.
    pub(crate) fn reset_in_memory_done(&mut self) {
        self.in_memory_done = false;
    }

    /// Re-extracts the source to support streaming it again for a re-direct.
    /// Functional Utility: Sets up a new IPC route for re-transmitting the body
    /// upon a network redirect.
    ///
    /// @param chunk_request_receiver The IPC receiver for body chunk requests.
    ///
    /// TODO: actually re-extract the source, instead of just cloning data, to support Blob.
    fn re_extract(&mut self, chunk_request_receiver: IpcReceiver<BodyChunkRequest>) {
        let mut body_handler = self.clone();
        body_handler.reset_in_memory_done();

        ROUTER.add_typed_route(
            chunk_request_receiver,
            Box::new(move |message| {
                // Block Logic: Handles various `BodyChunkRequest` messages to manage body transmission.
                // Conditional logic to process different types of body chunk requests.
                let request = message.unwrap();
                match request {
                    BodyChunkRequest::Connect(sender) => {
                        body_handler.start_reading(sender);
                    },
                    BodyChunkRequest::Extract(receiver) => {
                        body_handler.re_extract(receiver);
                    },
                    BodyChunkRequest::Chunk => body_handler.transmit_source(),
                    // Note: this is actually sent from this process
                    // by the TransmitBodyPromiseHandler when reading stops.
                    BodyChunkRequest::Done => {
                        body_handler.stop_reading(StopReading::Done);
                    },
                    // Note: this is actually sent from this process
                    // by the TransmitBodyPromiseHandler when the stream errors.
                    BodyChunkRequest::Error => {
                        body_handler.stop_reading(StopReading::Error);
                    },
                }
            }),
        );
    }

    /// In case of re-direct, and of a source available in memory,
    /// send it all in one chunk.
    ///
    /// Functional Utility: Optimizes re-transmission for in-memory bodies
    /// by sending the entire content in a single `BodyChunkResponse::Chunk`.
    ///
    /// Precondition: `bytes_sender` must be `Some(IpcSender)`.
    ///
    /// TODO: this method should be deprecated
    /// in favor of making `re_extract` actually re-extract a stream from the source.
    /// See #26686
    fn transmit_source(&mut self) {
        // Precondition: `in_memory_done` flag indicates if the body has already been transmitted.
        // Block Logic: If in-memory transmission is complete, stop reading.
        if self.in_memory_done {
            // Step 5.1.3
            self.stop_reading(StopReading::Done);
            return;
        }

        // Invariant: `ReadableStream(Null)` sources should not be re-directed.
        // Block Logic: Ensures that readable stream sources are not re-directed in an unsupported manner.
        if let BodySource::Null = self.source {
            panic!("ReadableStream(Null) sources should not re-redirect.");
        }

        // Block Logic: If in-memory bytes exist, clone and send them, then mark as done.
        // Precondition: `self.in_memory` contains the byte vector to transmit.
        if let Some(bytes) = self.in_memory.clone() {
            // The memoized bytes are sent so we mark it as done again
            self.in_memory_done = true;
            let _ = self
                .bytes_sender
                .as_ref()
                .expect("No bytes sender to transmit source.")
                .send(BodyChunkResponse::Chunk(bytes.clone()));
            return;
        }
        warn!("Re-directs for file-based Blobs not supported yet.");
    }

    /// Takes the IPC sender sent by `net`, allowing this handler to send body chunks with it.
    /// This method also serves as the entry point to <https://fetch.spec.whatwg.org/#concept-request-transmit-body>.
    ///
    /// Functional Utility: Establishes the communication channel for transmitting body data
    /// from the script process to the network process.
    ///
    /// Precondition: `sender` is a valid `IpcSender<BodyChunkResponse>`.
    fn start_reading(&mut self, sender: IpcSender<BodyChunkResponse>) {
        self.bytes_sender = Some(sender);

        // Block Logic: If the `BodySource` is `Null` (indicating a `ReadableStream`),
        //              acquire a reader for it asynchronously.
        // Precondition: `self.source` indicates the origin of the body.
        if self.source == BodySource::Null {
            let stream = self.stream.clone();
            self.task_source
                .queue(task!(start_reading_request_body_stream: move || {
                    // Step 1, Let body be request’s body.
                    let rooted_stream = stream.root();

                    // TODO: Step 2, If body is null.

                    // Step 3, get a reader for stream.
                    rooted_stream.acquire_default_reader(CanGc::note())
                        .expect("Couldn't acquire a reader for the body stream.");

                    // Note: this algorithm continues when the first chunk is requested by `net`.
                }));
        }
    }

    /// Drops the IPC sender previously sent by `net`.
    ///
    /// Functional Utility: Terminates the body transmission process,
    /// sending a final `Error` or `Done` message to the network process
    /// based on the `reason`.
    ///
    /// Precondition: `bytes_sender` must be `Some(IpcSender)` when this method is called.
    fn stop_reading(&mut self, reason: StopReading) {
        let bytes_sender = self
            .bytes_sender
            .take()
            .expect("Stop reading called multiple times on TransmitBodyConnectHandler.");
        // Block Logic: Sends an `Error` or `Done` message based on the `reason`.
        // Invariant: Ensures that the network process is notified of the body transmission's final state.
        match reason {
            StopReading::Error => {
                let _ = bytes_sender.send(BodyChunkResponse::Error);
            },
            StopReading::Done => {
                let _ = bytes_sender.send(BodyChunkResponse::Done);
            },
        }
    }

    /// Implements Step 4 and following of <https://fetch.spec.whatwg.org/#concept-request-transmit-body>.
    ///
    /// Functional Utility: Transmits a single chunk of the body, either directly from
    /// in-memory data or by asynchronously reading from a `ReadableStream` via promises.
    ///
    /// Precondition: `bytes_sender` must be `Some(IpcSender)`.
    fn transmit_body_chunk(&mut self) {
        // Precondition: `in_memory_done` flag indicates if the body has already been transmitted.
        // Block Logic: If in-memory transmission is complete, stop reading.
        if self.in_memory_done {
            // Step 5.1.3
            self.stop_reading(StopReading::Done);
            return;
        }

        let stream = self.stream.clone();
        let control_sender = self.control_sender.clone();
        let bytes_sender = self
            .bytes_sender
            .clone()
            .expect("No bytes sender to transmit chunk.");

        // Block Logic: If the data is available in memory, send it all at once, bypassing SpiderMonkey for transmission.
        // Invariant: Optimized path for in-memory data to avoid overhead of stream processing.
        if let Some(bytes) = self.in_memory.clone() {
            let _ = bytes_sender.send(BodyChunkResponse::Chunk(bytes));
            // Mark this body as `done` so that we can stop reading in the next tick,
            // matching the behavior of the promise-based flow
            self.in_memory_done = true;
            return;
        }

        // Block Logic: For `ReadableStream` sources, queue a task to set up a promise handler
        //              for reading the next chunk.
        // Precondition: `self.stream` is a valid `ReadableStream`.
        self.task_source.queue(
            task!(setup_native_body_promise_handler: move || {
                let rooted_stream = stream.root();
                let global = rooted_stream.global();

                // Step 4, the result of reading a chunk from body’s stream with reader.
                let promise = rooted_stream.read_a_chunk(CanGc::note());

                // Step 5, the parallel steps waiting for and handling the result of the read promise,
                // are a combination of the promise native handler here,
                // and the corresponding IPC route in `component::net::http_loader`.
                let promise_handler = Box::new(TransmitBodyPromiseHandler {
                    bytes_sender: bytes_sender.clone(),
                    stream: Dom::from_ref(&rooted_stream.clone()),
                    control_sender: control_sender.clone(),
                });

                let rejection_handler = Box::new(TransmitBodyPromiseRejectionHandler {
                    bytes_sender,
                    stream: rooted_stream,
                    control_sender,
                });

                let handler =
                    PromiseNativeHandler::new(&global, Some(promise_handler), Some(rejection_handler), CanGc::note());

                let realm = enter_realm(&*global);
                let comp = InRealm::Entered(&realm);
                promise.append_native_handler(&handler, comp, CanGc::note());
            })
        );
    }
}

/// The handler for successful read promises of body streams, used in
/// <https://fetch.spec.whatwg.org/#concept-request-transmit-body>.
///
/// Functional Utility: Processes successfully read chunks from a `ReadableStream`,
/// sending them via IPC to the network process, and managing the stream's state.
#[derive(Clone, JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
struct TransmitBodyPromiseHandler {
    /// The IPC sender used to transmit body chunks to the network process.
    #[ignore_malloc_size_of = "Channels are hard"]
    #[no_trace]
    bytes_sender: IpcSender<BodyChunkResponse>,
    /// The `ReadableStream` from which the body data is being read.
    stream: Dom<ReadableStream>,
    /// The IPC sender for control requests to the network process.
    #[ignore_malloc_size_of = "Channels are hard"]
    #[no_trace]
    control_sender: IpcSender<BodyChunkRequest>,
}

impl Callback for TransmitBodyPromiseHandler {
    /// Implements Step 5 of <https://fetch.spec.whatwg.org/#concept-request-transmit-body>.
    /// Functional Utility: This callback is invoked when a `ReadableStream`'s `read()`
    /// promise resolves, processing the received data or `done` signal.
    fn callback(&self, cx: JSContext, v: HandleValue, _realm: InRealm, _can_gc: CanGc) {
        // Precondition: `v` contains the result of the `ReadableStream` read operation.
        let is_done = match get_read_promise_done(cx, &v) {
            Ok(is_done) => is_done,
            Err(_) => {
                // Step 5.5, the "otherwise" steps.
                // Block Logic: If there's an error getting the 'done' status, send a Done signal (as per spec)
                // and stop reading the stream.
                // Invariant: An error in determining 'done' status implies the stream is effectively done from a transmission perspective.
                // TODO: terminate fetch.
                let _ = self.control_sender.send(BodyChunkRequest::Done);
                return self.stream.stop_reading();
            },
        };

        // Conditional logic based on whether the stream has indicated completion.
        if is_done {
            // Step 5.3, the "done" steps.
            // Block Logic: If the stream is done, send a Done signal and stop reading.
            // Invariant: A 'done' signal indicates the successful end of the stream.
            // TODO: queue a fetch task on request to process request end-of-body.
            let _ = self.control_sender.send(BodyChunkRequest::Done);
            return self.stream.stop_reading();
        }

        // Block Logic: Extracts the chunk of bytes from the read result.
        let chunk = match get_read_promise_bytes(cx, &v) {
            Ok(chunk) => chunk,
            Err(_) => {
                // Step 5.5, the "otherwise" steps.
                // Block Logic: If there's an error getting the bytes, send an Error signal and stop reading.
                // Invariant: An error during byte retrieval signals a failure in transmission.
                let _ = self.control_sender.send(BodyChunkRequest::Error);
                return self.stream.stop_reading();
            },
        };

        // Step 5.1 and 5.2, transmit chunk.
        // Block Logic: Sends the successfully read data chunk to the network process.
        // Precondition: `self.bytes_sender` is established to transmit `BodyChunkResponse::Chunk`.
        // TODO: queue a fetch task on request to process request body for request.
        let _ = self.bytes_sender.send(BodyChunkResponse::Chunk(chunk));
    }
}

/// The handler of read promise rejections of body streams, used in
/// <https://fetch.spec.whatwg.org/#concept-request-transmit-body>.
///
/// Functional Utility: Catches and propagates rejections from `ReadableStream`
/// read operations to the network process via IPC, signaling an error in body transmission.
#[derive(Clone, JSTraceable, MallocSizeOf)]
struct TransmitBodyPromiseRejectionHandler {
    /// The IPC sender used to transmit body chunks to the network process.
    #[ignore_malloc_size_of = "Channels are hard"]
    #[no_trace]
    bytes_sender: IpcSender<BodyChunkResponse>,
    /// The `ReadableStream` from which the body data is being read.
    stream: DomRoot<ReadableStream>,
    /// The IPC sender for control requests to the network process.
    #[ignore_malloc_size_of = "Channels are hard"]
    #[no_trace]
    control_sender: IpcSender<BodyChunkRequest>,
}

impl Callback for TransmitBodyPromiseRejectionHandler {
    /// Implements the rejection steps (Step 5.4) of <https://fetch.spec.whatwg.org/#concept-request-transmit-body>.
    /// Functional Utility: Ensures that if a read operation on the `ReadableStream` fails,
    /// an error message is sent over the control channel and the stream is stopped.
    fn callback(&self, _cx: JSContext, _v: HandleValue, _realm: InRealm, _can_gc: CanGc) {
        // Step 5.4, the "rejection" steps.
        // Block Logic: Sends an error message via the control sender and stops reading the stream.
        // Invariant: A rejected promise indicates a failure in reading the stream.
        let _ = self.control_sender.send(BodyChunkRequest::Error);
        self.stream.stop_reading();
    }
}

/// The result of <https://fetch.spec.whatwg.org/#concept-bodyinit-extract>.
/// Functional Utility: Encapsulates all the necessary information about a body
/// once it has been extracted from a `BodyInit` type, preparing it for either
/// network transmission or script-side consumption.
pub(crate) struct ExtractedBody {
    /// The `ReadableStream` representing the body's content.
    pub(crate) stream: DomRoot<ReadableStream>,
    /// The source type of the body.
    pub(crate) source: BodySource,
    /// The total number of bytes in the body, if known.
    pub(crate) total_bytes: Option<usize>,
    /// The content type of the body.
    pub(crate) content_type: Option<DOMString>,
}

impl ExtractedBody {
    /// Builds a request body from the extracted body, to be sent over IPC to `net`
    /// for use with `concept-request-transmit-body` (<https://fetch.spec.whatwg.org/#concept-request-transmit-body>).
    ///
    /// Also returns the corresponding `ReadableStream`, to be stored on the request
    /// in script, and potentially used as part of `consume_body` (<https://fetch.spec.whatwg.org/#concept-body-consume-body>).
    ///
    /// Transmitting a body over fetch and consuming it in script are mutually exclusive
    /// operations, since each will lock the stream to a reader.
    ///
    /// Functional Utility: Prepares the extracted body for cross-process transmission
    /// while also returning a reference to the stream for local script processing.
    pub(crate) fn into_net_request_body(self) -> (RequestBody, DomRoot<ReadableStream>) {
        let ExtractedBody {
            stream,
            total_bytes,
            content_type: _,
            source,
        } = self;

        // First, setup some infra to be used to transmit body
        //  from `components::script` to `components::net`.
        let (chunk_request_sender, chunk_request_receiver) = ipc::channel().unwrap();

        let trusted_stream = Trusted::new(&*stream);

        let global = stream.global();
        let task_source = global.task_manager().networking_task_source();

        // Block Logic: If data is already in memory, use it directly, bypassing SpiderMonkey for transmission.
        // Invariant: Pre-loaded in-memory data allows for a more efficient transmission path.
        let in_memory = stream.get_in_memory_bytes();

        let net_source = match source {
            BodySource::Null => NetBodySource::Null,
            _ => NetBodySource::Object,
        };

        let mut body_handler = TransmitBodyConnectHandler::new(
            trusted_stream,
            task_source.into(),
            chunk_request_sender.clone(),
            in_memory,
            source,
        );

        ROUTER.add_typed_route(
            chunk_request_receiver,
            Box::new(move |message| {
                // Block Logic: Handles various `BodyChunkRequest` messages from the network process.
                // Conditional logic to process different types of body chunk requests, managing the body transmission.
                match message.unwrap() {
                    BodyChunkRequest::Connect(sender) => {
                        body_handler.start_reading(sender);
                    },
                    BodyChunkRequest::Extract(receiver) => {
                        body_handler.re_extract(receiver);
                    },
                    BodyChunkRequest::Chunk => body_handler.transmit_body_chunk(),
                    // Note: this is actually sent from this process
                    // by the TransmitBodyPromiseHandler when reading stops.
                    BodyChunkRequest::Done => {
                        body_handler.stop_reading(StopReading::Done);
                    },
                    // Note: this is actually sent from this process
                    // by the TransmitBodyPromiseHandler when the stream errors.
                    BodyChunkRequest::Error => {
                        body_handler.stop_reading(StopReading::Error);
                    },
                }
            }),
        );

        // Return `components::net` view into this request body,
        // which can be used by `net` to transmit it over the network.
        let request_body = RequestBody::new(chunk_request_sender, net_source, total_bytes);

        // Also return the stream for this body, which can be used by script to consume it.
        (request_body, stream)
    }

    /// Checks if the data of the stream of this extracted body is available in memory.
    /// Functional Utility: Provides a quick way to determine if the body content
    /// is readily available without requiring stream operations.
    pub(crate) fn in_memory(&self) -> bool {
        self.stream.in_memory()
    }
}

/// A trait for types that can be "extracted" into an `ExtractedBody`.
/// <https://fetch.spec.whatwg.org/#concept-bodyinit-extract>
pub(crate) trait Extractable {
    /// Extracts the body content into an `ExtractedBody` structure.
    ///
    /// @param global The `GlobalScope` context.
    /// @param can_gc A `CanGc` token for garbage collection awareness.
    /// @return A `Fallible<ExtractedBody>` containing the extracted body or an `Error`.
    fn extract(&self, global: &GlobalScope, can_gc: CanGc) -> Fallible<ExtractedBody>;
}

impl Extractable for BodyInit {
    /// Implements the `extract` method for `BodyInit` enum.
    /// <https://fetch.spec.whatwg.org/#concept-bodyinit-extract>
    /// Functional Utility: Provides a unified way to handle different `BodyInit` types,
    /// converting them into a `ReadableStream` and `ExtractedBody`.
    fn extract(&self, global: &GlobalScope, can_gc: CanGc) -> Fallible<ExtractedBody> {
        match self {
            BodyInit::String(ref s) => s.extract(global, can_gc),
            BodyInit::URLSearchParams(ref usp) => usp.extract(global, can_gc),
            BodyInit::Blob(ref b) => b.extract(global, can_gc),
            BodyInit::FormData(ref formdata) => formdata.extract(global, can_gc),
            BodyInit::ArrayBuffer(ref typedarray) => {
                let bytes = typedarray.to_vec();
                let total_bytes = bytes.len();
                let stream = ReadableStream::new_from_bytes(global, bytes, can_gc)?;
                Ok(ExtractedBody {
                    stream,
                    total_bytes: Some(total_bytes),
                    content_type: None,
                    source: BodySource::Object,
                })
            },
            BodyInit::ArrayBufferView(ref typedarray) => {
                let bytes = typedarray.to_vec();
                let total_bytes = bytes.len();
                let stream = ReadableStream::new_from_bytes(global, bytes, can_gc)?;
                Ok(ExtractedBody {
                    stream,
                    total_bytes: Some(total_bytes),
                    content_type: None,
                    source: BodySource::Object,
                })
            },
            BodyInit::ReadableStream(stream) => {
                // TODO:
                // 1. If the keepalive flag is set, then throw a TypeError.

                // Block Logic: Checks if the stream is already locked or disturbed, which would prevent reading.
                // Precondition: The stream must not be locked or disturbed to be extracted.
                if stream.is_locked() || stream.is_disturbed() {
                    return Err(Error::Type(
                        "The body's stream is disturbed or locked".to_string(),
                    ));
                }

                Ok(ExtractedBody {
                    stream: stream.clone(),
                    total_bytes: None,
                    content_type: None,
                    source: BodySource::Null,
                })
            },
        }
    }
}

impl Extractable for Vec<u8> {
    /// Extracts a `Vec<u8>` into an `ExtractedBody`.
    /// Functional Utility: Treats a raw byte vector as a body source,
    /// creating a `ReadableStream` from it.
    fn extract(&self, global: &GlobalScope, can_gc: CanGc) -> Fallible<ExtractedBody> {
        let bytes = self.clone();
        let total_bytes = self.len();
        let stream = ReadableStream::new_from_bytes(global, bytes, can_gc)?;
        Ok(ExtractedBody {
            stream,
            total_bytes: Some(total_bytes),
            content_type: None,
            // A vec is used only in `submit_entity_body`.
            source: BodySource::Object,
        })
    }
}

impl Extractable for Blob {
    /// Extracts a `Blob` into an `ExtractedBody`.
    /// Functional Utility: Converts a `Blob` object into a `ReadableStream`
    /// for body processing, preserving its content type and size.
    fn extract(&self, _global: &GlobalScope, can_gc: CanGc) -> Fallible<ExtractedBody> {
        let blob_type = self.Type();
        let content_type = if blob_type.as_ref().is_empty() {
            None
        } else {
            Some(blob_type)
        };
        let total_bytes = self.Size() as usize;
        let stream = self.get_stream(can_gc)?;
        Ok(ExtractedBody {
            stream,
            total_bytes: Some(total_bytes),
            content_type,
            source: BodySource::Object,
        })
    }
}

impl Extractable for DOMString {
    /// Extracts a `DOMString` into an `ExtractedBody`.
    /// Functional Utility: Converts a JavaScript string into a UTF-8 byte array
    /// and then into a `ReadableStream`, setting the appropriate `text/plain`
    /// content type.
    fn extract(&self, global: &GlobalScope, can_gc: CanGc) -> Fallible<ExtractedBody> {
        let bytes = self.as_bytes().to_owned();
        let total_bytes = bytes.len();
        let content_type = Some(DOMString::from("text/plain;charset=UTF-8"));
        let stream = ReadableStream::new_from_bytes(global, bytes, can_gc)?;
        Ok(ExtractedBody {
            stream,
            total_bytes: Some(total_bytes),
            content_type,
            source: BodySource::Object,
        })
    }
}

impl Extractable for FormData {
    /// Extracts `FormData` into an `ExtractedBody`.
    /// Functional Utility: Encodes `FormData` into a `multipart/form-data`
    /// byte array, generating a unique boundary, and then creates a `ReadableStream`.
    fn extract(&self, global: &GlobalScope, can_gc: CanGc) -> Fallible<ExtractedBody> {
        let boundary = generate_boundary();
        let bytes = encode_multipart_form_data(&mut self.datums(), boundary.clone(), UTF_8);
        let total_bytes = bytes.len();
        let content_type = Some(DOMString::from(format!(
            "multipart/form-data;boundary={}",
            boundary
        )));
        let stream = ReadableStream::new_from_bytes(global, bytes, can_gc)?;
        Ok(ExtractedBody {
            stream,
            total_bytes: Some(total_bytes),
            content_type,
            source: BodySource::Object,
        })
    }
}

impl Extractable for URLSearchParams {
    /// Extracts `URLSearchParams` into an `ExtractedBody`.
    /// Functional Utility: Serializes URL search parameters into an
    /// `application/x-www-form-urlencoded` byte array and creates a `ReadableStream`.
    fn extract(&self, global: &GlobalScope, can_gc: CanGc) -> Fallible<ExtractedBody> {
        let bytes = self.serialize_utf8().into_bytes();
        let total_bytes = bytes.len();
        let content_type = Some(DOMString::from(
            "application/x-www-form-urlencoded;charset=UTF-8",
        ));
        let stream = ReadableStream::new_from_bytes(global, bytes, can_gc)?;
        Ok(ExtractedBody {
            stream,
            total_bytes: Some(total_bytes),
            content_type,
            source: BodySource::Object,
        })
    }
}

/// Enumerates the different types of body data that can be consumed.
/// Functional Utility: Provides a clear classification for how fetched data
/// should be interpreted and processed after retrieval.
#[derive(Clone, Copy, JSTraceable, MallocSizeOf)]
pub(crate) enum BodyType {
    /// The body content is a `Blob`.
    Blob,
    /// The body content is raw bytes.
    Bytes,
    /// The body content is `FormData`.
    FormData,
    /// The body content is JSON.
    Json,
    /// The body content is plain text.
    Text,
    /// The body content is an `ArrayBuffer`.
    ArrayBuffer,
}

/// Represents the various forms that fetched data can take after processing.
/// Functional Utility: Unifies the return types for different data packaging
/// algorithms, allowing for flexible handling of diverse body content.
pub(crate) enum FetchedData {
    /// Fetched data as a `String`.
    Text(String),
    /// Fetched data as a JavaScript JSON object.
    Json(RootedTraceableBox<Heap<JSValue>>),
    /// Fetched data as a `Blob` DOM object.
    BlobData(DomRoot<Blob>),
    /// Fetched data as raw bytes (represented as a JavaScript `Uint8Array`).
    Bytes(RootedTraceableBox<Heap<*mut JSObject>>),
    /// Fetched data as `FormData` DOM object.
    FormData(DomRoot<FormData>),
    /// Fetched data as a JavaScript `ArrayBuffer`.
    ArrayBuffer(RootedTraceableBox<Heap<*mut JSObject>>),
    /// Indicates that a JavaScript exception occurred during data processing.
    JSException(RootedTraceableBox<Heap<JSVal>>),
}

/// Handles the rejection of promises during the body consumption process.
/// Functional Utility: Ensures that errors encountered while reading from a
/// `ReadableStream` are correctly propagated as rejections of the main
/// `result_promise`.
#[derive(Clone, JSTraceable, MallocSizeOf)]
struct ConsumeBodyPromiseRejectionHandler {
    /// The promise that represents the overall result of consuming the body.
    #[ignore_malloc_size_of = "Rc are hard"]
    result_promise: Rc<Promise>,
}

impl Callback for ConsumeBodyPromiseRejectionHandler {
    /// Continues Step 4 of <https://fetch.spec.whatwg.org/#concept-body-consume-body>,
    /// specifically Step 3 of <https://fetch.spec.whatwg.org/#concept-read-all-bytes-from-readablestream>,
    /// handling the rejection steps.
    /// Functional Utility: When a promise related to reading a chunk rejects,
    /// this callback rejects the main `result_promise` with the encountered error.
    fn callback(&self, cx: JSContext, v: HandleValue, _realm: InRealm, _can_gc: CanGc) {
        self.result_promise.reject(cx, v);
    }
}

impl js::gc::Rootable for ConsumeBodyPromiseHandler {}

#[derive(Clone, JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
/// The promise handler used to consume the body, implementing steps described in
/// <https://fetch.spec.whatwg.org/#concept-body-consume-body>.
/// Functional Utility: Manages the asynchronous process of reading all chunks from a `ReadableStream`,
/// accumulating them, and then packaging the final data according to the specified `BodyType`.
struct ConsumeBodyPromiseHandler {
    /// The promise that represents the overall result of consuming the body.
    #[ignore_malloc_size_of = "Rc are hard"]
    result_promise: Rc<Promise>,
    /// The `ReadableStream` being consumed. Optional because it's moved when done.
    stream: Option<Dom<ReadableStream>>,
    /// The type of body data expected (e.g., `Json`, `Text`, `Blob`).
    body_type: DomRefCell<Option<BodyType>>,
    /// The MIME type of the body data.
    mime_type: DomRefCell<Option<Vec<u8>>>,
    /// Accumulated bytes read from the stream.
    bytes: DomRefCell<Option<Vec<u8>>>,
}

impl ConsumeBodyPromiseHandler {
    /// Implements Step 5 of <https://fetch.spec.whatwg.org/#concept-body-consume-body>.
    /// Functional Utility: Once all bytes have been read and accumulated, this method
    /// performs the final data packaging based on `body_type` and resolves the
    /// `result_promise` with the appropriate `FetchedData`.
    fn resolve_result_promise(&self, cx: JSContext, can_gc: CanGc) {
        let body_type = self.body_type.borrow_mut().take().unwrap();
        let mime_type = self.mime_type.borrow_mut().take().unwrap();
        let body = self.bytes.borrow_mut().take().unwrap();

        let pkg_data_results = run_package_data_algorithm(cx, body, body_type, mime_type, can_gc);

        match pkg_data_results {
            Ok(results) => {
                match results {
                    FetchedData::Text(s) => {
                        self.result_promise.resolve_native(&USVString(s), can_gc)
                    },
                    FetchedData::Json(j) => self.result_promise.resolve_native(&j, can_gc),
                    FetchedData::BlobData(b) => self.result_promise.resolve_native(&b, can_gc),
                    FetchedData::FormData(f) => self.result_promise.resolve_native(&f, can_gc),
                    FetchedData::Bytes(b) => self.result_promise.resolve_native(&b, can_gc),
                    FetchedData::ArrayBuffer(a) => self.result_promise.resolve_native(&a, can_gc),
                    FetchedData::JSException(e) => self.result_promise.reject_native(&e.handle()),
                };
            },
            Err(err) => self.result_promise.reject_error(err),
        }
    }
}

impl Callback for ConsumeBodyPromiseHandler {
    /// Continues Step 4 of <https://fetch.spec.whatwg.org/#concept-body-consume-body>,
    /// specifically Step 3 of <https://fetch.spec.whatwg.org/#concept-read-all-bytes-from-readablestream>.
    ///
    /// Functional Utility: This callback is invoked each time a chunk is read from the stream.
    /// It either accumulates the chunk or, if the stream is done, resolves the final promise.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    fn callback(&self, cx: JSContext, v: HandleValue, _realm: InRealm, can_gc: CanGc) {
        let stream = self
            .stream
            .as_ref()
            .expect("ConsumeBodyPromiseHandler has no stream in callback.");

        let is_done = match get_read_promise_done(cx, &v) {
            Ok(is_done) => is_done,
            Err(err) => {
                stream.stop_reading();
                // When read is fulfilled with a value that doesn't matches with neither of the above patterns.
                return self.result_promise.reject_error(err);
            },
        };

        if is_done {
            // Block Logic: If the stream indicates completion, resolve the accumulated body.
            // When read is fulfilled with an object whose done property is true.
            self.resolve_result_promise(cx, can_gc);
        } else {
            let chunk = match get_read_promise_bytes(cx, &v) {
                Ok(chunk) => chunk,
                Err(err) => {
                    stream.stop_reading();
                    // When read is fulfilled with a value that matches with neither of the above patterns
                    return self.result_promise.reject_error(err);
                },
            };

            let mut bytes = self
                .bytes
                .borrow_mut()
                .take()
                .expect("No bytes for ConsumeBodyPromiseHandler.");

            // Append the value property to bytes.
            bytes.extend_from_slice(&chunk);

            let global = stream.global();

            // Run the above step again.
            let read_promise = stream.read_a_chunk(can_gc);

            let promise_handler = Box::new(ConsumeBodyPromiseHandler {
                result_promise: self.result_promise.clone(),
                stream: self.stream.clone(),
                body_type: DomRefCell::new(self.body_type.borrow_mut().take()),
                mime_type: DomRefCell::new(self.mime_type.borrow_mut().take()),
                bytes: DomRefCell::new(Some(bytes)),
            });

            let rejection_handler = Box::new(ConsumeBodyPromiseRejectionHandler {
                result_promise: self.result_promise.clone(),
            });

            let handler = PromiseNativeHandler::new(
                &global,
                Some(promise_handler),
                Some(rejection_handler),
                can_gc,
            );

            let realm = enter_realm(&*global);
            let comp = InRealm::Entered(&realm);
            read_promise.append_native_handler(&handler, comp, can_gc);
        }
    }
}

/// Consumes a body from an object implementing `BodyMixin`.
/// <https://fetch.spec.whatwg.org/#concept-body-consume-body>
///
/// Functional Utility: Initiates the process of reading an entire body from a `ReadableStream`
/// associated with a `BodyMixin` object, returning a `Promise` that resolves with the
/// packaged data.
///
/// @param object The object whose body is to be consumed.
/// @param body_type The expected type of the body data.
/// @param can_gc A `CanGc` token for garbage collection awareness.
/// @return A `Rc<Promise>` that will resolve with the consumed and packaged body data.
#[cfg_attr(crown, allow(crown::unrooted_must_root))]
pub(crate) fn consume_body<T: BodyMixin + DomObject>(
    object: &T,
    body_type: BodyType,
    can_gc: CanGc,
) -> Rc<Promise> {
    let in_realm_proof = AlreadyInRealm::assert();
    let promise = Promise::new_in_current_realm(InRealm::Already(&in_realm_proof), can_gc);

    // Step 1: Check if the body's stream is disturbed or locked.
    // Invariant: A disturbed or locked stream cannot be consumed.
    if object.is_disturbed() || object.is_locked() {
        promise.reject_error(Error::Type(
            "The body's stream is disturbed or locked".to_string(),
        ));
        return promise;
    }

    consume_body_with_promise(
        object,
        body_type,
        promise.clone(),
        InRealm::Already(&in_realm_proof),
        can_gc,
    );

    promise
}

/// Helper function to consume a body with a given promise.
/// <https://fetch.spec.whatwg.org/#concept-body-consume-body>
/// Functional Utility: Encapsulates the core logic for setting up the
/// `ReadableStream` reader and initiating the asynchronous byte-reading process.
fn consume_body_with_promise<T: BodyMixin + DomObject>(
    object: &T,
    body_type: BodyType,
    promise: Rc<Promise>,
    comp: InRealm,
    can_gc: CanGc,
) {
    let global = object.global();

    // Step 2: Obtain the ReadableStream from the object's body.
    let stream = match object.body() {
        Some(stream) => stream,
        None => ReadableStream::new_from_bytes(&global, Vec::with_capacity(0), can_gc)
            .expect("ReadableStream::new_from_bytes should not fail with an empty Vec<u8>"),
    };

    // Step 3: Acquire a default reader for the stream.
    // Invariant: The stream must be acquirable for reading.
    if stream.acquire_default_reader(can_gc).is_err() {
        return promise.reject_error(Error::Type(
            "The response's stream is disturbed or locked".to_string(),
        ));
    }

    // Step 4: Read all the bytes.
    // This starts the asynchronous process which continues in `ConsumeBodyPromiseHandler`.

    // Step 1 of
    // https://fetch.spec.whatwg.org/#concept-read-all-bytes-from-readablestream
    let read_promise = stream.read_a_chunk(can_gc);

    let cx = GlobalScope::get_cx();
    rooted!(in(*cx) let mut promise_handler = Some(ConsumeBodyPromiseHandler {
        result_promise: promise.clone(),
        stream: Some(Dom::from_ref(&stream)),
        body_type: DomRefCell::new(Some(body_type)),
        mime_type: DomRefCell::new(Some(object.get_mime_type(can_gc))),
        // Step 2: Initialize an empty byte vector to accumulate chunks.
        bytes: DomRefCell::new(Some(vec![])),
    }));

    let rejection_handler = Box::new(ConsumeBodyPromiseRejectionHandler {
        result_promise: promise,
    });

    let handler = PromiseNativeHandler::new(
        &object.global(),
        promise_handler.take().map(|h| Box::new(h) as Box<_>),
        Some(rejection_handler),
        can_gc,
    );
    // We are already in a realm and a script.
    read_promise.append_native_handler(&handler, comp, can_gc);
}

/// Implements the "package data" algorithm from <https://fetch.spec.whatwg.org/#concept-body-package-data>.
/// Functional Utility: Takes raw bytes and converts them into a `FetchedData` variant
/// based on the `body_type` and `mime_type`, handling various data formats like Text, JSON, Blob, etc.
///
/// @param cx The JavaScript context.
/// @param bytes The raw byte vector to package.
/// @param body_type The `BodyType` indicating how to interpret the bytes.
/// @param mime_type The MIME type of the data.
/// @param can_gc A `CanGc` token for garbage collection awareness.
/// @return A `Fallible<FetchedData>` containing the packaged data or an `Error`.
fn run_package_data_algorithm(
    cx: JSContext,
    bytes: Vec<u8>,
    body_type: BodyType,
    mime_type: Vec<u8>,
    can_gc: CanGc,
) -> Fallible<FetchedData> {
    let mime = &*mime_type;
    let in_realm_proof = AlreadyInRealm::assert_for_cx(cx);
    let global = GlobalScope::from_safe_context(cx, InRealm::Already(&in_realm_proof));
    match body_type {
        BodyType::Text => run_text_data_algorithm(bytes),
        BodyType::Json => run_json_data_algorithm(cx, bytes),
        BodyType::Blob => run_blob_data_algorithm(&global, bytes, mime, can_gc),
        BodyType::FormData => run_form_data_algorithm(&global, bytes, mime, can_gc),
        BodyType::ArrayBuffer => run_array_buffer_data_algorithm(cx, bytes, can_gc),
        BodyType::Bytes => run_bytes_data_algorithm(cx, bytes, can_gc),
    }
}

/// Implements the "text data" algorithm.
/// Functional Utility: Converts a byte vector into a `String`, handling invalid UTF-8
/// sequences by replacing them with replacement characters.
///
/// @param bytes The byte vector to convert to text.
/// @return A `Fallible<FetchedData>` containing the text or an `Error`.
fn run_text_data_algorithm(bytes: Vec<u8>) -> Fallible<FetchedData> {
    Ok(FetchedData::Text(
        String::from_utf8_lossy(&bytes).into_owned(),
    ))
}

/// Implements the "JSON data" algorithm.
/// Functional Utility: Parses a byte vector as JSON, stripping any UTF-8 BOM,
/// and converts it into a JavaScript `JSValue`.
///
/// @param cx The JavaScript context.
/// @param bytes The byte vector containing JSON data.
/// @return A `Fallible<FetchedData>` containing the JavaScript JSON object or a `JSException`.
#[allow(unsafe_code)]
fn run_json_data_algorithm(cx: JSContext, bytes: Vec<u8>) -> Fallible<FetchedData> {
    // The JSON spec allows implementations to either ignore UTF-8 BOM or treat it as an error.
    // `JS_ParseJSON` treats this as an error, so it is necessary for us to strip it if present.
    //
    // https://datatracker.ietf.org/doc/html/rfc8259#section-8.1
    let json_text = decode_to_utf16_with_bom_removal(&bytes, UTF_8);
    rooted!(in(*cx) let mut rval = UndefinedValue());
    // Block Logic: Calls SpiderMonkey's `JS_ParseJSON` to parse the UTF-16 JSON string.
    unsafe {
        if !JS_ParseJSON(
            *cx,
            json_text.as_ptr(),
            json_text.len() as u32,
            rval.handle_mut(),
        ) {
            rooted!(in(*cx) let mut exception = UndefinedValue());
            // Invariant: If parsing fails, a pending JavaScript exception must exist.
            assert!(JS_GetPendingException(*cx, exception.handle_mut()));
            JS_ClearPendingException(*cx); // Clear the pending exception after capturing.
            return Ok(FetchedData::JSException(RootedTraceableBox::from_box(
                Heap::boxed(exception.get()),
            )));
        }
        let rooted_heap = RootedTraceableBox::from_box(Heap::boxed(rval.get()));
        Ok(FetchedData::Json(rooted_heap))
    }
}

/// Implements the "Blob data" algorithm.
/// Functional Utility: Creates a new `Blob` DOM object from a byte vector and MIME type.
///
/// @param root The `GlobalScope` context.
/// @param bytes The raw byte vector for the `Blob`.
/// @param mime The MIME type of the `Blob`.
/// @param can_gc A `CanGc` token.
/// @return A `Fallible<FetchedData>` containing the `Blob` object.
fn run_blob_data_algorithm(
    root: &GlobalScope,
    bytes: Vec<u8>,
    mime: &[u8],
    can_gc: CanGc,
) -> Fallible<FetchedData> {
    let mime_string = if let Ok(s) = String::from_utf8(mime.to_vec()) {
        s
    } else {
        "".to_string()
    };
    let blob = Blob::new(
        root,
        BlobImpl::new_from_bytes(bytes, normalize_type_string(&mime_string)),
        can_gc,
    );
    Ok(FetchedData::BlobData(blob))
}

/// Implements the "FormData data" algorithm.
/// Functional Utility: Parses a byte vector as form data (specifically
/// `application/x-www-form-urlencoded`) and converts it into a `FormData` DOM object.
///
/// @param root The `GlobalScope` context.
/// @param bytes The byte vector containing form data.
/// @param mime The MIME type of the data.
/// @param can_gc A `CanGc` token.
/// @return A `Fallible<FetchedData>` containing the `FormData` object or an `Error` for unsupported MIME types.
fn run_form_data_algorithm(
    root: &GlobalScope,
    bytes: Vec<u8>,
    mime: &[u8],
    can_gc: CanGc,
) -> Fallible<FetchedData> {
    let mime_str = str::from_utf8(mime).unwrap_or_default();
    let mime: Mime = mime_str
        .parse()
        .map_err(|_| Error::Type("Inappropriate MIME-type for Body".to_string()))?;

    // TODO
    // ... Parser for Mime(TopLevel::Multipart, SubLevel::FormData, _)
    // ... is not fully determined yet.
    // Block Logic: Currently only supports `application/x-www-form-urlencoded` for `FormData`.
    if mime.type_() == mime::APPLICATION && mime.subtype() == mime::WWW_FORM_URLENCODED {
        let entries = form_urlencoded::parse(&bytes);
        let formdata = FormData::new(None, root, can_gc);
        for (k, e) in entries {
            formdata.Append(USVString(k.into_owned()), USVString(e.into_owned()));
        }
        return Ok(FetchedData::FormData(formdata));
    }

    Err(Error::Type("Inappropriate MIME-type for Body".to_string()))
}

/// Implements the "bytes data" algorithm.
/// Functional Utility: Creates a JavaScript `Uint8Array` (`Heap<*mut JSObject>`)
/// from a byte vector.
///
/// @param cx The JavaScript context.
/// @param bytes The byte vector.
/// @param can_gc A `CanGc` token.
/// @return A `Fallible<FetchedData>` containing the JavaScript `Uint8Array`.
fn run_bytes_data_algorithm(cx: JSContext, bytes: Vec<u8>, can_gc: CanGc) -> Fallible<FetchedData> {
    rooted!(in(*cx) let mut array_buffer_ptr = ptr::null_mut::<JSObject>());

    create_buffer_source::<Uint8>(cx, &bytes, array_buffer_ptr.handle_mut(), can_gc)
        .map_err(|_| Error::JSFailed)?;

    let rooted_heap = RootedTraceableBox::from_box(Heap::boxed(array_buffer_ptr.get()));
    Ok(FetchedData::Bytes(rooted_heap))
}

/// Implements the "ArrayBuffer data" algorithm.
/// Functional Utility: Creates a JavaScript `ArrayBuffer` (`Heap<*mut JSObject>`)
/// from a byte vector.
///
/// @param cx The JavaScript context.
/// @param bytes The byte vector.
/// @param can_gc A `CanGc` token.
/// @return A `Fallible<FetchedData>` containing the JavaScript `ArrayBuffer`.
pub(crate) fn run_array_buffer_data_algorithm(
    cx: JSContext,
    bytes: Vec<u8>,
    can_gc: CanGc,
) -> Fallible<FetchedData> {
    rooted!(in(*cx) let mut array_buffer_ptr = ptr::null_mut::<JSObject>());

    create_buffer_source::<ArrayBufferU8>(cx, &bytes, array_buffer_ptr.handle_mut(), can_gc)
        .map_err(|_| Error::JSFailed)?;

    let rooted_heap = RootedTraceableBox::from_box(Heap::boxed(array_buffer_ptr.get()));
    Ok(FetchedData::ArrayBuffer(rooted_heap))
}

/// Decodes a byte slice to UTF-16, optionally removing a Byte Order Mark (BOM).
///
/// Functional Utility: Provides a robust decoding mechanism for JSON-like data,
/// ensuring compatibility with JSON specifications that allow or forbid BOMs.
///
/// @param bytes The input byte slice.
/// @param encoding The `Encoding` to use for decoding (e.g., `UTF_8`).
/// @return A `Vec<u16>` containing the decoded UTF-16 characters.
#[allow(unsafe_code)]
pub(crate) fn decode_to_utf16_with_bom_removal(
    bytes: &[u8],
    encoding: &'static Encoding,
) -> Vec<u16> {
    let mut decoder = encoding.new_decoder_with_bom_removal();
    let capacity = decoder
        .max_utf16_buffer_length(bytes.len())
        .expect("Overflow");
    let mut utf16 = Vec::with_capacity(capacity);
    // Block Logic: Creates an unsafe mutable slice from the vector's raw parts to enable direct decoding into it.
    let extra = unsafe { slice::from_raw_parts_mut(utf16.as_mut_ptr(), capacity) };
    let (_, read, written, _) = decoder.decode_to_utf16(bytes, extra, true);
    assert_eq!(read, bytes.len());
    // Block Logic: Unsafely sets the vector's length to reflect the number of bytes actually written by the decoder.
    unsafe { utf16.set_len(written) }
    utf16
}

/// A trait representing the "Body" concept from the Fetch API.
/// <https://fetch.spec.whatwg.org/#body>
/// Functional Utility: Defines the common interface for objects that can
/// serve as the body of a request or response, providing methods to
/// check its state (disturbed, locked) and access its content (`ReadableStream`).
pub(crate) trait BodyMixin {
    /// Checks if the body's stream is disturbed.
    /// <https://fetch.spec.whatwg.org/#concept-body-disturbed>
    fn is_disturbed(&self) -> bool;
    /// Returns the `ReadableStream` associated with the body.
    /// <https://fetch.spec.whatwg.org/#dom-body-body>
    fn body(&self) -> Option<DomRoot<ReadableStream>>;
    /// Checks if the body's stream is locked.
    /// <https://fetch.spec.whatwg.org/#concept-body-locked>
    fn is_locked(&self) -> bool;
    /// Returns the MIME type of the body.
    /// <https://fetch.spec.whatwg.org/#concept-body-mime-type>
    fn get_mime_type(&self, can_gc: CanGc) -> Vec<u8>;
}
