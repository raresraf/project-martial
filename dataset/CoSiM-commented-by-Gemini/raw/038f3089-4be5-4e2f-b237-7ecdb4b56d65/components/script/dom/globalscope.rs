/*!
This module defines the `GlobalScope` struct, which is the base class for
all global scopes in Servo, including `Window`, `WorkerGlobalScope`, and
`WorkletGlobalScope`. It provides the common functionality that is shared
between all global scopes, such as:

- Event handling
- Timers
- Btoa and atob
- Structured cloning
- Fetching resources
- CSP
- And more.

The `GlobalScope` is also responsible for managing the lifetime of the
JavaScript runtime and the garbage collector.
*/

use std::cell::{Cell, OnceCell, Ref};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Index;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{mem, ptr};

use base::id::{
    BlobId, BroadcastChannelRouterId, MessagePortId, MessagePortRouterId, PipelineId,
    ServiceWorkerId, ServiceWorkerRegistrationId, WebViewId,
};
use constellation_traits::{
    BlobData, BlobImpl, BroadcastMsg, FileBlob, LoadData, LoadOrigin, MessagePortImpl,
    MessagePortMsg, PortMessageTask, ScriptToConstellationChan, ScriptToConstellationMessage,
};
use content_security_policy::{
    CheckResult, CspList, Destination, Initiator, NavigationCheckType, ParserMetadata,
    PolicyDisposition, PolicySource, Request,
};
use crossbeam_channel::Sender;
use devtools_traits::{PageError, ScriptToDevtoolsControlMsg};
use dom_struct::dom_struct;
use embedder_traits::EmbedderMsg;
use http::HeaderMap;
use hyper_serde::Serde;
use ipc_channel::ipc::{self, IpcSender};
use ipc_channel::router::ROUTER;
use js::glue::{IsWrapper, UnwrapObjectDynamic};
use js::jsapi::{
    Compile1, CurrentGlobalOrNull, GetNonCCWObjectGlobal, HandleObject, Heap,
    InstantiateGlobalStencil, InstantiateOptions, JSContext, JSObject, JSScript, SetScriptPrivate,
};
use js::jsval::{PrivateValue, UndefinedValue};
use js::panic::maybe_resume_unwind;
use js::rust::wrappers::{JS_ExecuteScript, JS_GetScriptPrivate};
use js::rust::{
    CompileOptionsWrapper, CustomAutoRooter, CustomAutoRooterGuard, HandleValue,
    MutableHandleValue, ParentRuntime, Runtime, get_object_class, transform_str_to_source_text,
};
use js::{JSCLASS_IS_DOMJSCLASS, JSCLASS_IS_GLOBAL};
use net_traits::blob_url_store::{BlobBuf, get_blob_origin};
use net_traits::filemanager_thread::{
    FileManagerResult, FileManagerThreadMsg, ReadFileProgress, RelativePos,
};
use net_traits::image_cache::ImageCache;
use net_traits::policy_container::PolicyContainer;
use net_traits::request::{InsecureRequestsPolicy, Referrer, RequestBuilder};
use net_traits::response::HttpsState;
use net_traits::{
    CoreResourceMsg, CoreResourceThread, FetchResponseListener, IpcSend, ReferrerPolicy,
    ResourceThreads, fetch_async,
};
use profile_traits::{ipc as profile_ipc, mem as profile_mem, time as profile_time};
use script_bindings::interfaces::GlobalScopeHelpers;
use servo_url::{ImmutableOrigin, MutableOrigin, ServoUrl};
use timers::{TimerEventRequest, TimerId};
use url::Origin;
use uuid::Uuid;
#[cfg(feature = "webgpu")]
use webgpu_traits::{DeviceLostReason, WebGPUDevice};

use super::bindings::codegen::Bindings::MessagePortBinding::StructuredSerializeOptions;
#[cfg(feature = "webgpu")]
use super::bindings::codegen::Bindings::WebGPUBinding::GPUDeviceLostReason;
use super::bindings::error::Fallible;
use super::bindings::trace::{HashMapTracedValues, RootedTraceableBox};
use super::serviceworkerglobalscope::ServiceWorkerGlobalScope;
use super::transformstream::CrossRealmTransform;
use crate::dom::bindings::cell::{DomRefCell, RefMut};
use crate::dom::bindings::codegen::Bindings::BroadcastChannelBinding::BroadcastChannelMethods;
use crate::dom::bindings::codegen::Bindings::EventSourceBinding::EventSource_Binding::EventSourceMethods;
use crate::dom::bindings::codegen::Bindings::FunctionBinding::Function;
use crate::dom::bindings::codegen::Bindings::NotificationBinding::NotificationPermissionCallback;
use crate::dom::bindings::codegen::Bindings::PermissionStatusBinding::{
    PermissionName, PermissionState,
};
use crate::dom::bindings::codegen::Bindings::VoidFunctionBinding::VoidFunction;
use crate::dom::bindings::codegen::Bindings::WindowBinding::WindowMethods;
use crate::dom::bindings::codegen::Bindings::WorkerGlobalScopeBinding::WorkerGlobalScopeMethods;
use crate::dom::bindings::conversions::{root_from_object, root_from_object_static};
use crate::dom::bindings::error::{Error, ErrorInfo, report_pending_exception};
use crate::dom::bindings::frozenarray::CachedFrozenArray;
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::{Trusted, TrustedPromise};
use crate::dom::bindings::reflector::{DomGlobal, DomObject};
use crate::dom::bindings::root::{Dom, DomRoot, MutNullableDom};
use crate::dom::bindings::settings_stack::{AutoEntryScript, entry_global, incumbent_global};
use crate::dom::bindings::str::DOMString;
use crate::dom::bindings::structuredclone;
use crate::dom::bindings::trace::CustomTraceable;
use crate::dom::bindings::weakref::{DOMTracker, WeakRef};
use crate::dom::blob::Blob;
use crate::dom::broadcastchannel::BroadcastChannel;
use crate::dom::crypto::Crypto;
use crate::dom::csp::report_csp_violations;
use crate::dom::dedicatedworkerglobalscope::{
    DedicatedWorkerControlMsg, DedicatedWorkerGlobalScope,
};
use crate::dom::element::Element;
use crate::dom::errorevent::ErrorEvent;
use crate::dom::event::{Event, EventBubbles, EventCancelable, EventStatus};
use crate::dom::eventsource::EventSource;
use crate::dom::eventtarget::EventTarget;
use crate::dom::file::File;
use crate::dom::htmlscriptelement::{ScriptId, SourceCode};
use crate::dom::messageport::MessagePort;
use crate::dom::paintworkletglobalscope::PaintWorkletGlobalScope;
use crate::dom::performance::Performance;
use crate::dom::performanceobserver::VALID_ENTRY_TYPES;
use crate::dom::promise::Promise;
use crate::dom::readablestream::{CrossRealmTransformReadable, ReadableStream};
use crate::dom::serviceworker::ServiceWorker;
use crate::dom::serviceworkerregistration::ServiceWorkerRegistration;
use crate::dom::trustedtypepolicyfactory::TrustedTypePolicyFactory;
use crate::dom::types::MessageEvent;
use crate::dom::underlyingsourcecontainer::UnderlyingSourceType;
#[cfg(feature = "webgpu")]
use crate::dom::webgpu::gpudevice::GPUDevice;
#[cfg(feature = "webgpu")]
use crate::dom::webgpu::identityhub::IdentityHub;
use crate::dom::window::Window;
use crate::dom::workerglobalscope::WorkerGlobalScope;
use crate::dom::workletglobalscope::WorkletGlobalScope;
use crate::dom::writablestream::CrossRealmTransformWritable;
use crate::messaging::{CommonScriptMsg, ScriptEventLoopReceiver, ScriptEventLoopSender};
use crate::microtask::{Microtask, MicrotaskQueue, UserMicrotask};
use crate::network_listener::{NetworkListener, PreInvoke};
use crate::realms::{InRealm, enter_realm};
use crate::script_module::{
    DynamicModuleList, ImportMap, ModuleScript, ModuleTree, ResolvedModule, ScriptFetchOptions,
};
use crate::script_runtime::{CanGc, JSContext as SafeJSContext, ThreadSafeJSContext};
use crate::script_thread::{ScriptThread, with_script_thread};
use crate::task_manager::TaskManager;
use crate::task_source::SendableTaskSource;
use crate::timers::{
    IsInterval, OneshotTimerCallback, OneshotTimerHandle, OneshotTimers, TimerCallback,
    TimerEventId, TimerSource,
};
use crate::unminify::unminified_path;

/// A struct that automatically closes a worker when it is dropped.
#[derive(JSTraceable)]
pub(crate) struct AutoCloseWorker {
    /// <https://html.spec.whatwg.org/multipage/#dom-workerglobalscope-closing>
    closing: Arc<AtomicBool>,
    /// A handle to join on the worker thread.
    join_handle: Option<JoinHandle<()>>,
    /// A sender of control messages,
    /// currently only used to signal shutdown.
    #[no_trace]
    control_sender: Sender<DedicatedWorkerControlMsg>,
    /// The context to request an interrupt on the worker thread.
    #[no_trace]
    context: ThreadSafeJSContext,
}

impl Drop for AutoCloseWorker {
    /// <https://html.spec.whatwg.org/multipage/#terminate-a-worker>
    fn drop(&mut self) {
        // Step 1.
        self.closing.store(true, Ordering::SeqCst);

        if self
            .control_sender
            .send(DedicatedWorkerControlMsg::Exit)
            .is_err()
        {
            warn!("Couldn't send an exit message to a dedicated worker.");
        }

        self.context.request_interrupt_callback();

        // TODO: step 2 and 3.
        // Step 4 is unnecessary since we don't use actual ports for dedicated workers.
        if self
            .join_handle
            .take()
            .expect("No handle to join on worker.")
            .join()
            .is_err()
        {
            warn!("Failed to join on dedicated worker thread.");
        }
    }
}

/// The `GlobalScope` struct.
#[dom_struct]
pub(crate) struct GlobalScope {
    eventtarget: EventTarget,
    crypto: MutNullableDom<Crypto>,

    /// A [`TaskManager`] for this [`GlobalScope`].
    task_manager: OnceCell<TaskManager>,

    /// The message-port router id for this global, if it is managing ports.
    message_port_state: DomRefCell<MessagePortState>,

    /// The broadcast channels state this global, if it is managing any.
    broadcast_channel_state: DomRefCell<BroadcastChannelState>,

    /// The blobs managed by this global, if any.
    blob_state: DomRefCell<HashMapTracedValues<BlobId, BlobInfo>>,

    /// <https://w3c.github.io/ServiceWorker/#environment-settings-object-service-worker-registration-object-map>
    registration_map: DomRefCell<
        HashMapTracedValues<ServiceWorkerRegistrationId, Dom<ServiceWorkerRegistration>>,
    >,

    /// <https://w3c.github.io/ServiceWorker/#environment-settings-object-service-worker-object-map>
    worker_map: DomRefCell<HashMapTracedValues<ServiceWorkerId, Dom<ServiceWorker>>>,

    /// Pipeline id associated with this global.
    #[no_trace]
    pipeline_id: PipelineId,

    /// A flag to indicate whether the developer tools has requested
    /// live updates from the worker.
    devtools_wants_updates: Cell<bool>,

    /// Timers (milliseconds) used by the Console API.
    console_timers: DomRefCell<HashMap<DOMString, Instant>>,

    /// module map is used when importing JavaScript modules
    /// <https://html.spec.whatwg.org/multipage/#concept-settings-object-module-map>
    #[ignore_malloc_size_of = "mozjs"]
    module_map: DomRefCell<HashMapTracedValues<ServoUrl, Rc<ModuleTree>>>,

    #[ignore_malloc_size_of = "mozjs"]
    inline_module_map: DomRefCell<HashMap<ScriptId, Rc<ModuleTree>>>,

    /// For providing instructions to an optional devtools server.
    #[no_trace]
    devtools_chan: Option<IpcSender<ScriptToDevtoolsControlMsg>>,

    /// For sending messages to the memory profiler.
    #[ignore_malloc_size_of = "channels are hard"]
    #[no_trace]
    mem_profiler_chan: profile_mem::ProfilerChan,

    /// For sending messages to the time profiler.
    #[ignore_malloc_size_of = "channels are hard"]
    #[no_trace]
    time_profiler_chan: profile_time::ProfilerChan,

    /// A handle for communicating messages to the constellation thread.
    #[ignore_malloc_size_of = "channels are hard"]
    #[no_trace]
    script_to_constellation_chan: ScriptToConstellationChan,

    /// <https://html.spec.whatwg.org/multipage/#in-error-reporting-mode>
    in_error_reporting_mode: Cell<bool>,

    /// Associated resource threads for use by DOM objects like XMLHttpRequest,
    /// including resource_thread, filemanager_thread and storage_thread
    #[no_trace]
    resource_threads: ResourceThreads,

    /// The mechanism by which time-outs and intervals are scheduled.
    /// <https://html.spec.whatwg.org/multipage/#timers>
    timers: OnceCell<OneshotTimers>,

    /// The origin of the globalscope
    #[no_trace]
    origin: MutableOrigin,

    /// <https://html.spec.whatwg.org/multipage/#concept-environment-creation-url>
    #[no_trace]
    creation_url: ServoUrl,

    /// <https://html.spec.whatwg.org/multipage/#concept-environment-top-level-creation-url>
    #[no_trace]
    top_level_creation_url: Option<ServoUrl>,

    /// A map for storing the previous permission state read results.
    permission_state_invocation_results: DomRefCell<HashMap<PermissionName, PermissionState>>,

    /// The microtask queue associated with this global.
    ///
    /// It is refcounted because windows in the same script thread share the
    /// same microtask queue.
    ///
    /// <https://html.spec.whatwg.org/multipage/#microtask-queue>
    #[ignore_malloc_size_of = "Rc<T> is hard"]
    microtask_queue: Rc<MicrotaskQueue>,

    /// Vector storing closing references of all workers
    #[ignore_malloc_size_of = "Arc"]
    list_auto_close_worker: DomRefCell<Vec<AutoCloseWorker>>,

    /// Vector storing references of all eventsources.
    event_source_tracker: DOMTracker<EventSource>,

    /// Storage for watching rejected promises waiting for some client to
    /// consume their rejection.
    /// Promises in this list have been rejected in the last turn of the
    /// event loop without the rejection being handled.
    /// Note that this can contain nullptrs in place of promises removed because
    /// they're consumed before it'd be reported.
    ///
    /// <https://html.spec.whatwg.org/multipage/#about-to-be-notified-rejected-promises-list>
    #[ignore_malloc_size_of = "mozjs"]
    // `Heap` values must stay boxed, as they need semantics like `Pin`
    // (that is, they cannot be moved).
    #[allow(clippy::vec_box)]
    uncaught_rejections: DomRefCell<Vec<Box<Heap<*mut JSObject>>>>,

    /// Promises in this list have previously been reported as rejected
    /// (because they were in the above list), but the rejection was handled
    /// in the last turn of the event loop.
    ///
    /// <https://html.spec.whatwg.org/multipage/#outstanding-rejected-promises-weak-set>
    #[ignore_malloc_size_of = "mozjs"]
    // `Heap` values must stay boxed, as they need semantics like `Pin`
    // (that is, they cannot be moved).
    #[allow(clippy::vec_box)]
    consumed_rejections: DomRefCell<Vec<Box<Heap<*mut JSObject>>>>,

    /// Identity Manager for WebGPU resources
    #[ignore_malloc_size_of = "defined in wgpu"]
    #[no_trace]
    #[cfg(feature = "webgpu")]
    gpu_id_hub: Arc<IdentityHub>,

    /// WebGPU devices
    #[cfg(feature = "webgpu")]
    gpu_devices: DomRefCell<HashMapTracedValues<WebGPUDevice, WeakRef<GPUDevice>>>,

    // https://w3c.github.io/performance-timeline/#supportedentrytypes-attribute
    #[ignore_malloc_size_of = "mozjs"]
    frozen_supported_performance_entry_types: CachedFrozenArray,

    /// currect https state (from previous request)
    #[no_trace]
    https_state: Cell<HttpsState>,

    /// The stack of active group labels for the Console APIs.
    console_group_stack: DomRefCell<Vec<DOMString>>,

    /// The count map for the Console APIs.
    ///
    /// <https://console.spec.whatwg.org/#count>
    console_count_map: DomRefCell<HashMap<DOMString, usize>>,

    /// List of ongoing dynamic module imports.
    dynamic_modules: DomRefCell<DynamicModuleList>,

    /// Is considered in a secure context
    inherited_secure_context: Option<bool>,

    /// Directory to store unminified scripts for this window if unminify-js
    /// opt is enabled.
    unminified_js_dir: Option<String>,

    /// The byte length queuing strategy size function that will be initialized once
    /// `size` getter of `ByteLengthQueuingStrategy` is called.
    ///
    /// <https://streams.spec.whatwg.org/#byte-length-queuing-strategy-size-function>
    #[ignore_malloc_size_of = "Rc<T> is hard"]
    byte_length_queuing_strategy_size_function: OnceCell<Rc<Function>>,

    /// The count queuing strategy size function that will be initialized once
    /// `size` getter of `CountQueuingStrategy` is called.
    ///
    /// <https://streams.spec.whatwg.org/#count-queuing-strategy-size-function>
    #[ignore_malloc_size_of = "Rc<T> is hard"]
    count_queuing_strategy_size_function: OnceCell<Rc<Function>>,

    #[ignore_malloc_size_of = "Rc<T> is hard"]
    notification_permission_request_callback_map:
        DomRefCell<HashMap<String, Rc<NotificationPermissionCallback>>>,

    /// An import map allows control over module specifier resolution.
    /// For now, only Window global objects have their import map modified from the initial empty one.
    ///
    /// <https://html.spec.whatwg.org/multipage/#import-maps>
    import_map: DomRefCell<ImportMap>,

    /// <https://html.spec.whatwg.org/multipage/#resolved-module-set>
    resolved_module_set: DomRefCell<HashSet<ResolvedModule>>,
}

/// A wrapper for glue-code between the ipc router and the event-loop.
struct MessageListener {
    task_source: SendableTaskSource,
    context: Trusted<GlobalScope>,
}

/// A wrapper for broadcasts coming in over IPC, and the event-loop.
struct BroadcastListener {
    task_source: SendableTaskSource,
    context: Trusted<GlobalScope>,
}

type FileListenerCallback = Box<dyn Fn(Rc<Promise>, Fallible<Vec<u8>>) + Send>;

/// A wrapper for the handling of file data received by the ipc router
struct FileListener {
    /// State should progress as either of:
    /// - Some(Empty) => Some(Receiving) => None
    /// - Some(Empty) => None
    state: Option<FileListenerState>,
    task_source: SendableTaskSource,
}

enum FileListenerTarget {
    Promise(TrustedPromise, FileListenerCallback),
    Stream(Trusted<ReadableStream>),
}

enum FileListenerState {
    Empty(FileListenerTarget),
    Receiving(Vec<u8>, FileListenerTarget),
}

#[derive(JSTraceable, MallocSizeOf)]
/// A holder of a weak reference for a DOM blob or file.
pub(crate) enum BlobTracker {
    /// A weak ref to a DOM file.
    File(WeakRef<File>),
    /// A weak ref to a DOM blob.
    Blob(WeakRef<Blob>),
}

#[derive(JSTraceable, MallocSizeOf)]
/// The info pertaining to a blob managed by this global.
pub(crate) struct BlobInfo {
    /// The weak ref to the corresponding DOM object.
    tracker: BlobTracker,
    /// The data and logic backing the DOM object.
    #[no_trace]
    blob_impl: BlobImpl,
    /// Whether this blob has an outstanding URL,
    /// <https://w3c.github.io/FileAPI/#url>.
    has_url: bool,
}

/// The result of looking-up the data for a Blob,
/// containing either the in-memory bytes,
/// or the file-id.
enum BlobResult {
    Bytes(Vec<u8>),
    File(Uuid, usize),
}

/// Data representing a message-port managed by this global.
#[derive(JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
pub(crate) struct ManagedMessagePort {
    /// The DOM port.
    dom_port: Dom<MessagePort>,
    /// The logic and data backing the DOM port.
    /// The option is needed to take out the port-impl
    /// as part of its transferring steps,
    /// without having to worry about rooting the dom-port.
    #[no_trace]
    port_impl: Option<MessagePortImpl>,
    /// We keep ports pending when they are first transfer-received,
    /// and only add them, and ask the constellation to complete the transfer,
    /// in a subsequent task if the port hasn't been re-transfered.
    pending: bool,
    /// Whether the port has been closed by script in this global,
    /// so it can be removed.
    explicitly_closed: bool,
    /// The handler for `message` or `messageerror` used in the cross realm transform,
    /// if any was setup with this port.
    cross_realm_transform: Option<CrossRealmTransform>,
}

/// State representing whether this global is currently managing broadcast channels.
#[derive(JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
pub(crate) enum BroadcastChannelState {
    /// The broadcast-channel router id for this global, and a queue of managed channels.
    /// Step 9, "sort destinations"
    /// of <https://html.spec.whatwg.org/multipage/#dom-broadcastchannel-postmessage>
    /// requires keeping track of creation order, hence the queue.
    Managed(
        #[no_trace] BroadcastChannelRouterId,
        /// The map of channel-name to queue of channels, in order of creation.
        HashMap<DOMString, VecDeque<Dom<BroadcastChannel>>>,
    ),
    /// This global is not managing any broadcast channels at this time.
    UnManaged,
}

/// State representing whether this global is currently managing messageports.
#[derive(JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
pub(crate) enum MessagePortState {
    /// The message-port router id for this global, and a map of managed ports.
    Managed(
        #[no_trace] MessagePortRouterId,
        HashMapTracedValues<MessagePortId, ManagedMessagePort>,
    ),
    /// This global is not managing any ports at this time.
    UnManaged,
}

impl BroadcastListener {
    /// Handle a broadcast coming in over IPC,
    /// by queueing the appropriate task on the relevant event-loop.
    fn handle(&self, event: BroadcastMsg) {
        let context = self.context.clone();

        // Note: strictly speaking we should just queue the message event tasks,
        // not queue a task that then queues more tasks.
        // This however seems to be hard to avoid in the light of the IPC.
        // One can imagine queueing tasks directly,
        // for channels that would be in the same script-thread.
        self.task_source
            .queue(task!(broadcast_message_event: move || {
                let global = context.root();
                // Step 10 of https://html.spec.whatwg.org/multipage/#dom-broadcastchannel-postmessage,
                // For each BroadcastChannel object destination in destinations, queue a task.
                global.broadcast_message_event(event, None);
            }));
    }
}

impl MessageListener {
    /// A new message came in, handle it via a task enqueued on the event-loop.
    /// A task is required, since we are using a trusted globalscope,
    /// and we can only access the root from the event-loop.
    fn notify(&self, msg: MessagePortMsg) {
        match msg {
            MessagePortMsg::CompleteTransfer(ports) => {
                let context = self.context.clone();
                self.task_source.queue(
                    task!(process_complete_transfer: move || {
                        let global = context.root();

                        let router_id = match global.port_router_id() {
                            Some(router_id) => router_id,
                            None => {
                                // If not managing any ports, no transfer can succeed,
                                // so just send back everything.
                                let _ = global.script_to_constellation_chan().send(
                                    ScriptToConstellationMessage::MessagePortTransferResult(None, vec![], ports),
                                );
                                return;
                            }
                        };

                        let mut succeeded = vec![];
                        let mut failed = HashMap::new();

                        for (id, info) in ports.into_iter() {
                            if global.is_managing_port(&id) {
                                succeeded.push(id);
                                global.complete_port_transfer(
                                    id,
                                    info.port_message_queue,
                                    info.disentangled,
                                    CanGc::note()
                                );
                            } else {
                                failed.insert(id, info);
                            }
                        }
                        let _ = global.script_to_constellation_chan().send(
                            ScriptToConstellationMessage::MessagePortTransferResult(Some(router_id), succeeded, failed),
                        );
                    })
                );
            },
            MessagePortMsg::CompletePendingTransfer(port_id, info) => {
                let context = self.context.clone();
                self.task_source.queue(task!(complete_pending: move || {
                    let global = context.root();
                    global.complete_port_transfer(port_id, info.port_message_queue, info.disentangled, CanGc::note());
                }));
            },
            MessagePortMsg::CompleteDisentanglement(port_id) => {
                let context = self.context.clone();
                self.task_source
                    .queue(task!(try_complete_disentanglement: move || {
                        let global = context.root();
                        global.try_complete_disentanglement(port_id, CanGc::note());
                    }));
            },
            MessagePortMsg::NewTask(port_id, task) => {
                let context = self.context.clone();
                self.task_source.queue(task!(process_new_task: move || {
                    let global = context.root();
                    global.route_task_to_port(port_id, task, CanGc::note());
                }));
            },
        }
    }
}

/// Callback used to enqueue file chunks to streams as part of FileListener.
fn stream_handle_incoming(stream: &ReadableStream, bytes: Fallible<Vec<u8>>, can_gc: CanGc) {
    match bytes {
        Ok(b) => {
            stream.enqueue_native(b, can_gc);
        },
        Err(e) => {
            stream.error_native(e, can_gc);
        },
    }
}

/// Callback used to close streams as part of FileListener.
fn stream_handle_eof(stream: &ReadableStream, can_gc: CanGc) {
    stream.controller_close_native(can_gc);
}

impl FileListener {
    fn handle(&mut self, msg: FileManagerResult<ReadFileProgress>) {
        match msg {
            Ok(ReadFileProgress::Meta(blob_buf)) => match self.state.take() {
                Some(FileListenerState::Empty(target)) => {
                    let bytes = if let FileListenerTarget::Stream(ref trusted_stream) = target {
                        let trusted = trusted_stream.clone();

                        let task = task!(enqueue_stream_chunk: move || {
                            let stream = trusted.root();
                            stream_handle_incoming(&stream, Ok(blob_buf.bytes), CanGc::note());
                        });
                        self.task_source.queue(task);

                        Vec::with_capacity(0)
                    } else {
                        blob_buf.bytes
                    };

                    self.state = Some(FileListenerState::Receiving(bytes, target));
                },
                _ => panic!(
                    "Unexpected FileListenerState when receiving ReadFileProgress::Meta msg."
                ),
            },
            Ok(ReadFileProgress::Partial(mut bytes_in)) => match self.state.take() {
                Some(FileListenerState::Receiving(mut bytes, target)) => {
                    if let FileListenerTarget::Stream(ref trusted_stream) = target {
                        let trusted = trusted_stream.clone();

                        let task = task!(enqueue_stream_chunk: move || {
                            let stream = trusted.root();
                            stream_handle_incoming(&stream, Ok(bytes_in), CanGc::note());
                        });

                        self.task_source.queue(task);
                    } else {
                        bytes.append(&mut bytes_in);
                    };

                    self.state = Some(FileListenerState::Receiving(bytes, target));
                },
                _ => panic!(
                    "Unexpected FileListenerState when receiving ReadFileProgress::Partial msg."
                ),
            },
            Ok(ReadFileProgress::EOF) => match self.state.take() {
                Some(FileListenerState::Receiving(bytes, target)) => match target {
                    FileListenerTarget::Promise(trusted_promise, callback) => {
                        let task = task!(resolve_promise: move || {
                            let promise = trusted_promise.root();
                            let _ac = enter_realm(&*promise.global());
                            callback(promise, Ok(bytes));
                        });

                        self.task_source.queue(task);
                    },
                    FileListenerTarget::Stream(trusted_stream) => {
                        let trusted = trusted_stream.clone();

                        let task = task!(enqueue_stream_chunk: move || {
                            let stream = trusted.root();
                            stream_handle_eof(&stream, CanGc::note());
                        });

                        self.task_source.queue(task);
                    },
                },
                _ => {
                    panic!("Unexpected FileListenerState when receiving ReadFileProgress::EOF msg.")
                },
            },
            Err(_) => match self.state.take() {
                Some(FileListenerState::Receiving(_, target)) |
                Some(FileListenerState::Empty(target)) => {
                    let error = Err(Error::Network);

                    match target {
                        FileListenerTarget::Promise(trusted_promise, callback) => {
                            self.task_source.queue(task!(reject_promise: move || {
                                let promise = trusted_promise.root();
                                let _ac = enter_realm(&*promise.global());
                                callback(promise, error);
                            }));
                        },
                        FileListenerTarget::Stream(trusted_stream) => {
                            self.task_source.queue(task!(error_stream: move || {
                                let stream = trusted_stream.root();
                                stream_handle_incoming(&stream, error, CanGc::note());
                            }));
                        },
                    }
                },
                _ => panic!("Unexpected FileListenerState when receiving Err msg."),
            },
        }
    }
}

impl GlobalScope {
    /// A sender to the event loop of this global scope. This either sends to the Worker event loop
    /// or the ScriptThread event loop in the case of a `Window`. This can be `None` for dedicated
    /// workers that are not currently handling a message.
    pub(crate) fn webview_id(&self) -> Option<WebViewId> {
        if let Some(window) = self.downcast::<Window>() {
            Some(window.webview_id())
        } else if let Some(dedicated) = self.downcast::<DedicatedWorkerGlobalScope>() {
            dedicated.webview_id()
        } else {
            // ServiceWorkerGlobalScope, PaintWorklet, or DissimilarOriginWindow
            None
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_inherited(
        pipeline_id: PipelineId,
        devtools_chan: Option<IpcSender<ScriptToDevtoolsControlMsg>>,
        mem_profiler_chan: profile_mem::ProfilerChan,
        time_profiler_chan: profile_time::ProfilerChan,
        script_to_constellation_chan: ScriptToConstellationChan,
        resource_threads: ResourceThreads,
        origin: MutableOrigin,
        creation_url: ServoUrl,
        top_level_creation_url: Option<ServoUrl>,
        microtask_queue: Rc<MicrotaskQueue>,
        #[cfg(feature = "webgpu")] gpu_id_hub: Arc<IdentityHub>,
        inherited_secure_context: Option<bool>,
        unminify_js: bool,
    ) -> Self {
        Self {
            task_manager: Default::default(),
            message_port_state: DomRefCell::new(MessagePortState::UnManaged),
            broadcast_channel_state: DomRefCell::new(BroadcastChannelState::UnManaged),
            blob_state: Default::default(),
            eventtarget: EventTarget::new_inherited(),
            crypto: Default::default(),
            registration_map: DomRefCell::new(HashMapTracedValues::new()),
            worker_map: DomRefCell::new(HashMapTracedValues::new()),
            pipeline_id,
            devtools_wants_updates: Default::default(),
            console_timers: DomRefCell::new(Default::default()),
            module_map: DomRefCell::new(Default::default()),
            inline_module_map: DomRefCell::new(Default::default()),
            devtools_chan,
            mem_profiler_chan,
            time_profiler_chan,
            script_to_constellation_chan,
            in_error_reporting_mode: Default::default(),
            resource_threads,
            timers: OnceCell::default(),
            origin,
            creation_url,
            top_level_creation_url,
            permission_state_invocation_results: Default::default(),
            microtask_queue,
            list_auto_close_worker: Default::default(),
            event_source_tracker: DOMTracker::new(),
            uncaught_rejections: Default::default(),
            consumed_rejections: Default::default(),
            #[cfg(feature = "webgpu")]
            gpu_id_hub,
            #[cfg(feature = "webgpu")]
            gpu_devices: DomRefCell::new(HashMapTracedValues::new()),
            frozen_supported_performance_entry_types: CachedFrozenArray::new(),
            https_state: Cell::new(HttpsState::None),
            console_group_stack: DomRefCell::new(Vec::new()),
            console_count_map: Default::default(),
            dynamic_modules: DomRefCell::new(DynamicModuleList::new()),
            inherited_secure_context,
            unminified_js_dir: unminify_js.then(|| unminified_path("unminified-js")),
            byte_length_queuing_strategy_size_function: OnceCell::new(),
            count_queuing_strategy_size_function: OnceCell::new(),
            notification_permission_request_callback_map: Default::default(),
            import_map: Default::default(),
            resolved_module_set: Default::default(),
        }
    }

    /// The message-port router Id of the global, if any
    fn port_router_id(&self) -> Option<MessagePortRouterId> {
        if let MessagePortState::Managed(id, _message_ports) = &*self.message_port_state.borrow() {
            Some(*id)
        } else {
            None
        }
    }

    /// Is this global managing a given port?
    fn is_managing_port(&self, port_id: &MessagePortId) -> bool {
        if let MessagePortState::Managed(_router_id, message_ports) =
            &*self.message_port_state.borrow()
        {
            return message_ports.contains_key(port_id);
        }
        false
    }

    fn timers(&self) -> &OneshotTimers {
        self.timers.get_or_init(|| OneshotTimers::new(self))
    }

    /// <https://w3c.github.io/ServiceWorker/#get-the-service-worker-registration-object>
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_serviceworker_registration(
        &self,
        script_url: &ServoUrl,
        scope: &ServoUrl,
        registration_id: ServiceWorkerRegistrationId,
        installing_worker: Option<ServiceWorkerId>,
        _waiting_worker: Option<ServiceWorkerId>,
        _active_worker: Option<ServiceWorkerId>,
        can_gc: CanGc,
    ) -> DomRoot<ServiceWorkerRegistration> {
        // Step 1
        let mut registrations = self.registration_map.borrow_mut();

        if let Some(registration) = registrations.get(&registration_id) {
            // Step 3
            return DomRoot::from_ref(&**registration);
        }

        // Step 2.1 -> 2.5
        let new_registration =
            ServiceWorkerRegistration::new(self, scope.clone(), registration_id, can_gc);

        // Step 2.6
        if let Some(worker_id) = installing_worker {
            let worker = self.get_serviceworker(script_url, scope, worker_id, can_gc);
            new_registration.set_installing(&worker);
        }

        // TODO: 2.7 (waiting worker)

        // TODO: 2.8 (active worker)

        // Step 2.9
        registrations.insert(registration_id, Dom::from_ref(&*new_registration));

        // Step 3
        new_registration
    }

    /// <https://w3c.github.io/ServiceWorker/#get-the-service-worker-object>
    pub(crate) fn get_serviceworker(
        &self,
        script_url: &ServoUrl,
        scope: &ServoUrl,
        worker_id: ServiceWorkerId,
        can_gc: CanGc,
    ) -> DomRoot<ServiceWorker> {
        // Step 1
        let mut workers = self.worker_map.borrow_mut();

        if let Some(worker) = workers.get(&worker_id) {
            // Step 3
            DomRoot::from_ref(&**worker)
        } else {
            // Step 2.1
            // TODO: step 2.2, worker state.
            let new_worker =
                ServiceWorker::new(self, script_url.clone(), scope.clone(), worker_id, can_gc);

            // Step 2.3
            workers.insert(worker_id, Dom::from_ref(&*new_worker));

            // Step 3
            new_worker
        }
    }

    /// Complete the transfer of a message-port.
    fn complete_port_transfer(
        &self,
        port_id: MessagePortId,
        tasks: VecDeque<PortMessageTask>,
        disentangled: bool,
        can_gc: CanGc,
    ) {
        let should_start = if let MessagePortState::Managed(_id, message_ports) =
            &mut *self.message_port_state.borrow_mut()
        {
            match message_ports.get_mut(&port_id) {
                None => {
                    panic!("complete_port_transfer called for an unknown port.");
                },
                Some(managed_port) => {
                    if managed_port.pending {
                        panic!("CompleteTransfer msg received for a pending port.");
                    }
                    if let Some(port_impl) = managed_port.port_impl.as_mut() {
                        port_impl.complete_transfer(tasks);
                        if disentangled {
                            port_impl.disentangle();
                            managed_port.dom_port.disentangle();
                        }
                        port_impl.enabled()
                    } else {
                        panic!("managed-port has no port-impl.");
                    }
                },
            }
        } else {
            panic!("complete_port_transfer called for an unknown port.");
        };
        if should_start {
            self.start_message_port(&port_id, can_gc);
        }
    }

    /// The closing of `otherPort`, if it is in a different global.
    /// <https://html.spec.whatwg.org/multipage/#disentangle>
    fn try_complete_disentanglement(&self, port_id: MessagePortId, can_gc: CanGc) {
        let dom_port = if let MessagePortState::Managed(_id, message_ports) =
            &mut *self.message_port_state.borrow_mut()
        {
            let mut dom_port = None;
            for port_id in &[initiator_port, &other_port] {
                match message_ports.get_mut(port_id) {
                    None => {
                        continue;
                    },
                    Some(managed_port) => {
                        let port_impl = managed_port
                            .port_impl
                            .as_mut()
                            .expect("managed-port has no port-impl.");
                        managed_port.dom_port.disentangle();
                        port_impl.disentangle();

                        if **port_id == other_port {
                            dom_port = Some(managed_port.dom_port.as_rooted())
                        }
                    },
                }
            }
            dom_port
        } else {
            panic!("disentangle_port called on a global not managing any ports.");
        };

        // Fire an event named close at `otherPort`.
        // Note: done here if the port is managed by the same global as `initialPort`.
        if let Some(dom_port) = dom_port {
            dom_port.upcast().fire_event(atom!("close"), can_gc);
        }

        let chan = self.script_to_constellation_chan().clone();
        let initiator_port = *initiator_port;
        self.task_manager()
            .port_message_queue()
            .queue(task!(post_message: move || {
                // Note: we do this in a task to ensure it doesn't affect messages that are still to be routed,
                // see the task queueing in `post_messageport_msg`.
                let res = chan.send(ScriptToConstellationMessage::DisentanglePorts(initiator_port, Some(other_port)));
                if res.is_err() {
                    warn!("Sending DisentanglePorts failed");
                }
            }));
    }

    /// <https://html.spec.whatwg.org/multipage/#entangle>
    pub(crate) fn entangle_ports(&self, port1: MessagePortId, port2: MessagePortId) {
        if let MessagePortState::Managed(_id, message_ports) =
            &mut *self.message_port_state.borrow_mut()
        {
            for (port_id, entangled_id) in &[(port1, port2), (port2, port1)] {
                match message_ports.get_mut(port_id) {
                    None => {
                        return warn!("entangled_ports called on a global not managing the port.");
                    },
                    Some(managed_port) => {
                        if let Some(port_impl) = managed_port.port_impl.as_mut() {
                            managed_port.dom_port.entangle(*entangled_id);
                            port_impl.entangle(*entangled_id);
                        } else {
                            panic!("managed-port has no port-impl.");
                        }
                    },
                }
            }
        } else {
            panic!("entangled_ports called on a global not managing any ports.");
        }

        let _ = self
            .script_to_constellation_chan()
            .send(ScriptToConstellationMessage::EntanglePorts(port1, port2));
    }
}