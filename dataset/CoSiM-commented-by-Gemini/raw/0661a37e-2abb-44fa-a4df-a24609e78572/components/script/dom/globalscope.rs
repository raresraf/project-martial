
/**
 * @file globalscope.rs
 * @brief Implementation of the `GlobalScope` interface, the common base for `Window` and `WorkerGlobalScope`.
 *
 * This module defines the `GlobalScope` struct, which implements the shared functionality of all
 * global scopes in the browser, such as `Window`, `WorkerGlobalScope`, `ServiceWorkerGlobalScope`,
 * etc. It provides a common foundation for features that are accessible in all these contexts,
 * including event handling, timers, and access to the origin and other security-related properties.
 *
 * ## Core Functionality:
 *
 * - **Event Handling:** Implements the `EventTarget` interface, allowing global scopes to dispatch
 *   and listen for events.
 * - **Timers:** Provides the implementation for `setTimeout()`, `setInterval()`, `clearTimeout()`,
 *   and `clearInterval()`.
 * - **Origin and Security:** Manages the global scope's origin, which is crucial for the same-origin
 *   policy and other security checks.
 * - **Task Management:** Includes a `TaskManager` for queueing and executing tasks on the global
 *   scope's event loop.
 * - **Web Messaging:** Manages `MessagePort` and `BroadcastChannel` communication.
 * - **Blob and File Handling:** Provides infrastructure for creating and managing `Blob` and `File`
 *   objects.
 * - **JavaScript Execution:** Contains methods for evaluating JavaScript code within the global
 *   scope's context.
 *
 * This module is a central piece of the DOM implementation, ensuring that different types of
 * global scopes have a consistent set of capabilities as defined by various web standards.
 *
 * @see https://html.spec.whatwg.org/multipage/webappapis.html#global-objects
 */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::cell::{Cell, OnceCell};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
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
    BlobData, BlobImpl, BroadcastMsg, FileBlob, MessagePortImpl, MessagePortMsg, PortMessageTask,
    ScriptToConstellationChan, ScriptToConstellationMessage,
};
use content_security_policy::{CheckResult, CspList, PolicyDisposition};
use crossbeam_channel::Sender;
use devtools_traits::{PageError, ScriptToDevtoolsControlMsg};
use dom_struct::dom_struct;
use embedder_traits::{
    EmbedderMsg, GamepadEvent, GamepadSupportedHapticEffects, GamepadUpdateType,
};
use ipc_channel::ipc::{self, IpcSender};
use ipc_channel::router::ROUTER;
use js::glue::{IsWrapper, UnwrapObjectDynamic};
use js::jsapi::{
    Compile1, CurrentGlobalOrNull, GetNonCCWObjectGlobal, HandleObject, Heap,
    InstantiateGlobalStencil, InstantiateOptions, JSContext, JSObject, JSScript, RuntimeCode,
    SetScriptPrivate,
};
use js::jsval::{PrivateValue, UndefinedValue};
use js::panic::maybe_resume_unwind;
use js::rust::wrappers::{JS_ExecuteScript, JS_GetScriptPrivate};
use js::rust::{
    CompileOptionsWrapper, CustomAutoRooter, CustomAutoRooterGuard, HandleValue,
    MutableHandleValue, ParentRuntime, Runtime, describe_scripted_caller, get_object_class,
    transform_str_to_source_text,
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
use timers::{TimerEventId, TimerEventRequest, TimerSource};
use uuid::Uuid;
#[cfg(feature = "webgpu")]
use webgpu_traits::{DeviceLostReason, WebGPUDevice};

use super::bindings::codegen::Bindings::MessagePortBinding::StructuredSerializeOptions;
#[cfg(feature = "webgpu")]
use super::bindings::codegen::Bindings::WebGPUBinding::GPUDeviceLostReason;
use super::bindings::error::Fallible;
use super::bindings::trace::{HashMapTracedValues, RootedTraceableBox};
use super::serviceworkerglobalscope::ServiceWorkerGlobalScope;
use crate::dom::bindings::cell::{DomRefCell, RefMut};
use crate::dom::bindings::codegen::Bindings::BroadcastChannelBinding::BroadcastChannelMethods;
use crate::dom::bindings::codegen::Bindings::EventSourceBinding::EventSource_Binding::EventSourceMethods;
use crate::dom::bindings::codegen::Bindings::FunctionBinding::Function;
use crate::dom::bindings::codegen::Bindings::ImageBitmapBinding::{
    ImageBitmapOptions, ImageBitmapSource,
};
use crate::dom::bindings::codegen::Bindings::NavigatorBinding::NavigatorMethods;
use crate::dom::bindings::codegen::Bindings::NotificationBinding::NotificationPermissionCallback;
use crate::dom::bindings::codegen::Bindings::PerformanceBinding::Performance_Binding::PerformanceMethods;
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
use crate::dom::dedicatedworkerglobalscope::{
    DedicatedWorkerControlMsg, DedicatedWorkerGlobalScope,
};
use crate::dom::errorevent::ErrorEvent;
use crate::dom::event::{Event, EventBubbles, EventCancelable, EventStatus};
use crate::dom::eventsource::EventSource;
use crate::dom::eventtarget::EventTarget;
use crate::dom::file::File;
use crate::dom::gamepad::{Gamepad, contains_user_gesture};
use crate::dom::gamepadevent::GamepadEventType;
use crate::dom::htmlscriptelement::{ScriptId, SourceCode};
use crate::dom::imagebitmap::ImageBitmap;
use crate::dom::messageevent::MessageEvent;
use crate::dom::messageport::MessagePort;
use crate::dom::paintworkletglobalscope::PaintWorkletGlobalScope;
use crate::dom::performance::Performance;
use crate::dom::performanceobserver::VALID_ENTRY_TYPES;
use crate::dom::promise::Promise;
use crate::dom::readablestream::ReadableStream;
use crate::dom::serviceworker::ServiceWorker;
use crate::dom::serviceworkerregistration::ServiceWorkerRegistration;
use crate::dom::trustedtypepolicyfactory::TrustedTypePolicyFactory;
use crate::dom::underlyingsourcecontainer::UnderlyingSourceType;
#[cfg(feature = "webgpu")]
use crate::dom::webgpu::gpudevice::GPUDevice;
#[cfg(feature = "webgpu")]
use crate::dom::webgpu::identityhub::IdentityHub;
use crate::dom::window::Window;
use crate::dom::workerglobalscope::WorkerGlobalScope;
use crate::dom::workletglobalscope::WorkletGlobalScope;
use crate::messaging::{CommonScriptMsg, ScriptEventLoopReceiver, ScriptEventLoopSender};
use crate::microtask::{Microtask, MicrotaskQueue, UserMicrotask};
use crate::network_listener::{NetworkListener, PreInvoke};
use crate::realms::{AlreadyInRealm, InRealm, enter_realm};
use crate::script_module::{DynamicModuleList, ModuleScript, ModuleTree, ScriptFetchOptions};
use crate::script_runtime::{CanGc, JSContext as SafeJSContext, ThreadSafeJSContext};
use crate::script_thread::{ScriptThread, with_script_thread};
use crate::security_manager::CSPViolationReporter;
use crate::task_manager::TaskManager;
use crate::task_source::SendableTaskSource;
use crate::timers::{
    IsInterval, OneshotTimerCallback, OneshotTimerHandle, OneshotTimers, TimerCallback,
};
use crate::unminify::unminified_path;

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

/**
 * @brief The common implementation for all global scopes (e.g., `Window`, `WorkerGlobalScope`).
 *
 * This struct holds the state and implements the functionality that is shared across all types of
 * global scopes in the DOM.
 */
#[dom_struct]
pub(crate) struct GlobalScope {
    /// The `EventTarget` implementation for this global scope.
    eventtarget: EventTarget,
    /// The `Crypto` object associated with this global scope.
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
    creation_url: Option<ServoUrl>,

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
}
//...
