/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! The script thread is the thread that owns the DOM in memory, runs JavaScript, and triggers
//! layout. It's in charge of processing events for all same-origin pages in a frame
//! tree, and manages the entire lifetime of pages in the frame tree from initial request to
//! teardown.
//!
//! Page loads follow a two-step process. When a request for a new page load is received, the
//! network request is initiated and the relevant data pertaining to the new page is stashed.
//! While the non-blocking request is ongoing, the script thread is free to process further events,
//! noting when they pertain to ongoing loads (such as resizes/viewport adjustments). When the
//! initial response is received for an ongoing load, the second phase starts - the frame tree
//! entry is created, along with the Window and Document objects, and the appropriate parser
//! takes over the response body. Once parsing is complete, the document lifecycle for loading
//! a page runs its course and the script thread returns to processing events in the main event
//! loop.

// Block Logic: Standard library imports for common data structures, concurrency, and time management.
// Functional Utility: Provides fundamental building blocks for thread synchronization, atomic operations,
//                     collections (HashMap, HashSet), and time-related functionalities crucial for
//                     event loop management and performance monitoring.
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::option::Option;
use std::rc::Rc;
use std::result::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

// Block Logic: Imports for core browser functionalities and inter-process communication (IPC).
// Functional Utility: Provides definitions for background hang monitoring, browsing context
//                     and pipeline identifiers, WebGL pipelines, time management (chrono),
//                     cross-thread channels (crossbeam_channel), DevTools communication,
//                     embedder-specific messages (InputEvent, Theme), Euclidean geometry (euclid),
//                     font services, HTTP headers, HTML5 parsing primitives, IPC channels and router,
//                     JavaScript runtime bindings (js), media context, performance metrics,
//                     image caching, network request/response handling, memory/time profiling,
//                     layout interface, WebDriver commands, and various script thread behaviors.
//                     Also includes WebGPU and Webrender APIs.
use background_hang_monitor_api::{
    BackgroundHangMonitor, BackgroundHangMonitorExitSignal, HangAnnotation, MonitoredComponentId,
    MonitoredComponentType,
};
use base::cross_process_instant::CrossProcessInstant;
use base::id::{
    BrowsingContextId, HistoryStateId, PipelineId, PipelineNamespace, TopLevelBrowsingContextId,
};
use base::Epoch;
use canvas_traits::webgl::WebGLPipeline;
use chrono::{DateTime, Local};
use crossbeam_channel::unbounded;
use devtools_traits::{
    CSSError, DevtoolScriptControlMsg, DevtoolsPageInfo, NavigationState,
    ScriptToDevtoolsControlMsg, WorkerId,
};
use embedder_traits::{EmbedderMsg, InputEvent, MediaSessionActionType, Theme};
use euclid::default::Rect;
use fonts::{FontContext, SystemFontServiceProxy};
use headers::{HeaderMapExt, LastModified, ReferrerPolicy as ReferrerPolicyHeader};
use html5ever::{local_name, namespace_url, ns};
use hyper_serde::Serde;
use ipc_channel::ipc;
use ipc_channel::router::ROUTER;
use js::glue::GetWindowProxyClass;
use js::jsapi::{
    JSContext as UnsafeJSContext, JSTracer, JS_AddInterruptCallback, SetWindowProxyClass,
};
use js::jsval::UndefinedValue;
use js::rust::ParentRuntime;
use media::WindowGLContext;
use metrics::{PaintTimeMetrics, MAX_TASK_NS};
use mime::{self, Mime};
use net_traits::image_cache::{ImageCache, PendingImageResponse};
use net_traits::request::{RequestId, Referrer};
use net_traits::response::ResponseInit;
use net_traits::storage_thread::StorageType;
use net_traits::{
    FetchMetadata, FetchResponseListener, FetchResponseMsg, Metadata, NetworkError,
    ResourceFetchTiming, ResourceThreads, ResourceTimingType,
};
use percent_encoding::percent_decode;
use profile_traits::mem::ReportsChan;
use profile_traits::time::ProfilerCategory;
use profile_traits::time_profile;
use script_layout_interface::{
    node_id_from_scroll_id, LayoutConfig, LayoutFactory, ReflowGoal, ScriptThreadFactory,
};
use script_traits::webdriver_msg::WebDriverScriptCommand;
use script_traits::{
    ConstellationInputEvent, DiscardBrowsingContext, DocumentActivity, EventResult,
    InitialScriptState, JsEvalResult, LoadData, LoadOrigin, NavigationHistoryBehavior,
    NewLayoutInfo, Painter, ProgressiveWebMetricType, ScriptMsg, ScriptThreadMessage,
    ScriptToConstellationChan, ScrollState, StructuredSerializedData, UpdatePipelineIdReason,
    WindowSizeData, WindowSizeType,
};
use servo_atoms::Atom;
use servo_config::opts;
use servo_url::{ImmutableOrigin, MutableOrigin, ServoUrl};
use style::dom::OpaqueNode;
use style::thread_state::{self, ThreadState};
use timers::{TimerEventRequest, TimerScheduler};
use url::Position;
#[cfg(feature = "webgpu")]
use webgpu::{WebGPUDevice, WebGPUMsg};
use webrender_api::DocumentId;
use webrender_traits::{CompositorHitTestResult, CrossProcessCompositorApi};

// Block Logic: Internal crate imports for DOM manipulation, scripting, and browser engine components.
// Functional Utility: Provides access to core DOM structures (`Document`, `Element`, `Window`),
//                     JavaScript bindings (`DomRefCell`, `JSTraceable`), custom element handling,
//                     mutation observers, HTML elements (`HTMLAnchorElement`, `HTMLIFrameElement`),
//                     performance monitoring, parsing, WebGPU integration, worklet management,
//                     fetch cancellation, inter-thread messaging, microtask queuing, navigation,
//                     script module options, and JavaScript runtime context.
use crate::document_collection::DocumentCollection;
use crate::document_loader::DocumentLoader;
use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::DocumentBinding::{
    DocumentMethods, DocumentReadyState,
};
use crate::dom::bindings::codegen::Bindings::NavigatorBinding::NavigatorMethods;
use crate::dom::bindings::codegen::Bindings::WindowBinding::WindowMethods;
use crate::dom::bindings::conversions::{
    ConversionResult, FromJSValConvertible, StringificationBehavior,
};
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::root::{
    Dom, DomRoot, MutNullableDom, RootCollection, ThreadLocalStackRoots,
};
use crate::dom::bindings::settings_stack::AutoEntryScript;
use crate::dom::bindings::str::DOMString;
use crate::dom::bindings::trace::{HashMapTracedValues, JSTraceable};
use crate::dom::customelementregistry::{
    CallbackReaction, CustomElementDefinition, CustomElementReactionStack,
};
use crate::dom::document::{
    Document, DocumentSource, FocusType, HasBrowsingContext, IsHTMLDocument, TouchEventResult,
};
use crate::dom::element::Element;
use crate::dom::globalscope::GlobalScope;
use crate::dom::htmlanchorelement::HTMLAnchorElement;
use crate::dom::htmliframeelement::HTMLIFrameElement;
use crate::dom::htmlslotelement::HTMLSlotElement;
use crate::dom::mutationobserver::MutationObserver;
use crate::dom::node::{Node, NodeTraits, ShadowIncluding};
use crate::dom::performanceentry::PerformanceEntry;
use crate::dom::performancepainttiming::PerformancePaintTiming;
use crate::dom::servoparser::{ParserContext, ServoParser};
#[cfg(feature = "webgpu")]
use crate::dom::webgpu::identityhub::IdentityHub;
use crate::dom::window::Window;
use crate::dom::windowproxy::{CreatorBrowsingContextInfo, WindowProxy};
use crate::dom::worklet::WorkletThreadPool;
use crate::dom::workletglobalscope::WorkletGlobalScopeInit;
use crate::fetch::FetchCanceller;
use crate::messaging::{
    CommonScriptMsg, MainThreadScriptMsg, MixedMessage, ScriptEventLoopSender,
    ScriptThreadReceivers, ScriptThreadSenders,
};
use crate::microtask::{Microtask, MicrotaskQueue};
use crate::navigation::{InProgressLoad, NavigationListener};
use crate::realms::enter_realm;
use crate::script_module::ScriptFetchOptions;
use crate::script_runtime::{
    CanGc, JSContext, Runtime, ScriptThreadEventCategory, ThreadSafeJSContext,
};
use crate::task_queue::TaskQueue;
use crate::task_source::{SendableTaskSource, TaskSourceName};
use crate::{devtools, webdriver_handlers};

// Block Logic: Thread-local storage for the current `ScriptThread` instance.
// Functional Utility: Provides a mechanism to access the `ScriptThread` instance associated
//                     with the current thread, essential for thread-specific operations
//                     and ensuring correct context.
thread_local!(static SCRIPT_THREAD_ROOT: Cell<Option<*const ScriptThread>> = const { Cell::new(None) });

// Functional Utility: Safely retrieves an optional reference to the current `ScriptThread` instance
//                     from thread-local storage, allowing operations that may or may not require
//                     the script thread context.
fn with_optional_script_thread<R>(f: impl FnOnce(Option<&ScriptThread>) -> R) -> R {
    SCRIPT_THREAD_ROOT.with(|root| {
        f(root
            .get()
            .and_then(|script_thread| unsafe { script_thread.as_ref() }))
    })
}

// Functional Utility: Retrieves a mandatory reference to the current `ScriptThread` instance
//                     from thread-local storage, providing a default value if not found.
pub(crate) fn with_script_thread<R: Default>(f: impl FnOnce(&ScriptThread) -> R) -> R {
    with_optional_script_thread(|script_thread| script_thread.map(f).unwrap_or_default())
}

/// # Safety
///
/// The `JSTracer` argument must point to a valid `JSTracer` in memory. In addition,
/// implementors of this method must ensure that all active objects are properly traced
/// or else the garbage collector may end up collecting objects that are still reachable.
// Block Logic: Implements garbage collector tracing for the `ScriptThread`.
// Functional Utility: Ensures that all reachable objects within the `ScriptThread`'s memory
//                     are correctly identified by the JavaScript garbage collector, preventing
//                     premature deallocation and memory corruption.
//
// Pre-condition: `JSTracer` points to a valid tracer; implementors must ensure proper tracing of active objects.
pub(crate) unsafe fn trace_thread(tr: *mut JSTracer) {
    with_script_thread(|script_thread| {
        trace!("tracing fields of ScriptThread");
        script_thread.trace(tr);
    })
}

// We borrow the incomplete parser contexts mutably during parsing,
// which is fine except that parsing can trigger evaluation,
// which can trigger GC, and so we can end up tracing the script
// thread during parsing. For this reason, we don't trace the
// incomplete parser contexts during GC.
/// `IncompleteParserContexts` wraps a `RefCell` containing a vector of `(PipelineId, ParserContext)`.
/// This is used to hold parser contexts that are not yet fully processed.
///
/// Pre-condition: Parser contexts are associated with a valid `PipelineId`.
/// Invariant: Not traced during garbage collection to prevent issues with mutable borrowing during evaluation.
pub(crate) struct IncompleteParserContexts(RefCell<Vec<(PipelineId, ParserContext)>>);

unsafe_no_jsmanaged_fields!(TaskQueue<MainThreadScriptMsg>);

/// The `ScriptThread` is a core component of Servo responsible for:
/// - Owning the DOM in memory.
/// - Running JavaScript.
/// - Triggering layout.
/// - Processing events for all same-origin pages in a frame tree.
/// - Managing the lifecycle of pages from request to teardown.
#[derive(JSTraced)]
// ScriptThread instances are rooted on creation, so this is okay
#[cfg_attr(crown, allow(crown::unrooted_must_root))]
pub struct ScriptThread {
    /// <https://html.spec.whatwg.org/multipage/#last-render-opportunity-time>
    last_render_opportunity_time: DomRefCell<Option<Instant>>,
    /// The documents for pipelines managed by this thread.
    documents: DomRefCell<DocumentCollection>,
    /// The window proxies known by this thread.
    /// TODO: this map grows, but never shrinks. Issue #15258.
    window_proxies: DomRefCell<HashMapTracedValues<BrowsingContextId, Dom<WindowProxy>>>,
    /// A list of data pertaining to loads that have not yet received a network response.
    incomplete_loads: DomRefCell<Vec<InProgressLoad>>,
    /// A vector containing parser contexts which have not yet been fully processed.
    incomplete_parser_contexts: IncompleteParserContexts,
    /// Image cache for this script thread.
    #[no_trace]
    image_cache: Arc<dyn ImageCache>,

    /// A [`ScriptThreadReceivers`] holding all of the incoming `Receiver`s for messages
    /// to this [`ScriptThread`].
    receivers: ScriptThreadReceivers,

    /// A [`ScriptThreadSenders`] that holds all outgoing sending channels necessary to communicate
    /// to other parts of Servo.
    senders: ScriptThreadSenders,

    /// A handle to the resource thread. This is an `Arc` to avoid running out of file descriptors if
    /// there are many iframes.
    #[no_trace]
    resource_threads: ResourceThreads,

    /// A queue of tasks to be executed in this script-thread.
    task_queue: TaskQueue<MainThreadScriptMsg>,

    /// The dedicated means of communication with the background-hang-monitor for this script-thread.
    #[no_trace]
    background_hang_monitor: Box<dyn BackgroundHangMonitor>,
    /// A flag set to `true` by the BHM on exit, and checked from within the interrupt handler.
    closing: Arc<AtomicBool>,

    /// A [`TimerScheduler`] used to schedule timers for this [`ScriptThread`]. Timers are handled
    /// in the [`ScriptThread`] event loop.
    #[no_trace]
    timer_scheduler: RefCell<TimerScheduler>,

    /// A proxy to the `SystemFontService` to use for accessing system font lists.
    #[no_trace]
    system_font_service: Arc<SystemFontServiceProxy>,

    /// The JavaScript runtime.
    js_runtime: Rc<Runtime>,

    /// The topmost element over the mouse.
    topmost_mouse_over_target: MutNullableDom<Element>,

    /// List of pipelines that have been owned and closed by this script thread.
    #[no_trace]
    closed_pipelines: DomRefCell<HashSet<PipelineId>>,

    /// <https://html.spec.whatwg.org/multipage/#microtask-queue>
    microtask_queue: Rc<MicrotaskQueue>,

    /// Microtask Queue for adding support for mutation observer microtasks
    mutation_observer_microtask_queued: Cell<bool>,

    /// The unit of related similar-origin browsing contexts' list of MutationObserver objects
    mutation_observers: DomRefCell<Vec<Dom<MutationObserver>>>,

    /// <https://dom.spec.whatwg.org/#signal-slot-list>
    signal_slots: DomRefCell<Vec<Dom<HTMLSlotElement>>>,

    /// A handle to the WebGL thread
    #[no_trace]
    webgl_chan: Option<WebGLPipeline>,

    /// The WebXR device registry
    #[no_trace]
    #[cfg(feature = "webxr")]
    webxr_registry: Option<webxr_api::Registry>,

    /// The worklet thread pool
    worklet_thread_pool: DomRefCell<Option<Rc<WorkletThreadPool>>>,

    /// A list of pipelines containing documents that finished loading all their blocking
    /// resources during a turn of the event loop.
    docs_with_no_blocking_loads: DomRefCell<HashSet<Dom<Document>>>,

    /// <https://html.spec.whatwg.org/multipage/#custom-element-reactions-stack>
    custom_element_reaction_stack: CustomElementReactionStack,

    /// The Webrender Document ID associated with this thread.
    #[no_trace]
    webrender_document: DocumentId,

    /// Cross-process access to the compositor's API.
    #[no_trace]
    compositor_api: CrossProcessCompositorApi,

    /// Periodically print out on which events script threads spend their processing time.
    profile_script_events: bool,

    /// Print Progressive Web Metrics to console.
    print_pwm: bool,

    /// Emits notifications when there is a relayout.
    relayout_event: bool,

    /// Unminify Javascript.
    unminify_js: bool,

    /// Directory with stored unminified scripts
    local_script_source: Option<String>,

    /// Unminify Css.
    unminify_css: bool,

    /// Where to load userscripts from, if any. An empty string will load from
    /// the resources/user-agent-js directory, and if the option isn't passed userscripts
    /// won't be loaded
    userscripts_path: Option<String>,

    /// An optional string allowing the user agent to be set for testing.
    user_agent: Cow<'static, str>,

    /// Application window's GL Context for Media player
    #[no_trace]
    player_context: WindowGLContext,

    /// A set of all nodes ever created in this script thread
    node_ids: DomRefCell<HashSet<String>>,

    /// Code is running as a consequence of a user interaction
    is_user_interacting: Cell<bool>,

    /// Identity manager for WebGPU resources
    #[no_trace]
    #[cfg(feature = "webgpu")]
    gpu_id_hub: Arc<IdentityHub>,

    // Secure context
    inherited_secure_context: Option<bool>,

    /// A factory for making new layouts. This allows layout to depend on script.
    #[no_trace]
    layout_factory: Arc<dyn LayoutFactory>,
}

// Block Logic: Represents an exit signal for the Background Hang Monitor (BHM).
// Functional Utility: Allows the BHM to signal the script thread to initiate an exit process,
//                     setting a closing flag and requesting a JavaScript interrupt callback.
/// `BHMExitSignal` is an implementation of `BackgroundHangMonitorExitSignal` that
/// allows the Background Hang Monitor to signal a `ScriptThread` to gracefully
/// initiate its shutdown process.
struct BHMExitSignal {
    /// A shared atomic boolean flag that, when set to `true`, indicates the script
    /// thread should begin closing.
    closing: Arc<AtomicBool>,
    /// A thread-safe handle to the JavaScript context, used to request an interrupt
    /// callback, thereby waking up a potentially blocked JavaScript engine.
    js_context: ThreadSafeJSContext,
}

impl BackgroundHangMonitorExitSignal for BHMExitSignal {
    /// Signals the script thread to exit by setting the `closing` flag and requesting
    /// a JavaScript interrupt callback.
    ///
    /// Pre-condition: The `js_context` is valid and interruptible.
    /// Post-condition: The `closing` flag is set to `true`, and an interrupt is requested
    /// from the JavaScript engine, allowing the script thread to gracefully shut down.
    fn signal_to_exit(&self) {
        self.closing.store(true, Ordering::SeqCst);
        self.js_context.request_interrupt_callback();
    }
}

#[allow(unsafe_code)]
// Block Logic: Interrupt callback for the JavaScript engine.
// Functional Utility: Provides a mechanism to interrupt ongoing JavaScript execution,
//                     allowing the script thread to respond to external signals like
//                     shutdown requests.
//
// Pre-condition: JavaScript context (`_cx`) is valid.
// Invariant: Returns `true` if JavaScript execution should continue, `false` otherwise,
//            triggering `prepare_for_shutdown` if execution is to be halted.
unsafe extern "C" fn interrupt_callback(_cx: *mut UnsafeJSContext) -> bool {
    let res = ScriptThread::can_continue_running();
    if !res {
        ScriptThread::prepare_for_shutdown();
    }
    res
}

/// In the event of thread panic, all data on the stack runs its destructor. However, there
/// are no reachable, owning pointers to the DOM memory, so it never gets freed by default
/// when the script thread fails. The ScriptMemoryFailsafe uses the destructor bomb pattern
/// to forcibly tear down the JS realms for pages associated with the failing ScriptThread.
///
/// `ScriptMemoryFailsafe` ensures that if a `ScriptThread` panics, the associated
/// JavaScript realms are properly deallocated, preventing memory leaks. It uses the
/// "destructor bomb" pattern.
struct ScriptMemoryFailsafe<'a> {
    /// An optional reference to the `ScriptThread` this failsafe is associated with.
    /// `None` if the failsafe has been "neuter"-ed.
    owner: Option<&'a ScriptThread>,
}

impl<'a> ScriptMemoryFailsafe<'a> {
    /// Disarms the failsafe, preventing any cleanup actions from being performed when
    /// the failsafe is dropped.
    ///
    /// Post-condition: `self.owner` is set to `None`.
    fn neuter(&mut self) {
        self.owner = None;
    }

    /// Creates a new `ScriptMemoryFailsafe` instance, associating it with the given
    /// `ScriptThread`.
    ///
    /// # Arguments
    /// * `owner` - A reference to the `ScriptThread` that this failsafe will protect.
    ///
    /// Post-condition: A new `ScriptMemoryFailsafe` is returned, ready to clean up
    /// JavaScript realms if the `owner` thread panics.
    fn new(owner: &'a ScriptThread) -> ScriptMemoryFailsafe<'a> {
        ScriptMemoryFailsafe { owner: Some(owner) }
    }
}

impl Drop for ScriptMemoryFailsafe<'_> {
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    /// Implements the drop logic for `ScriptMemoryFailsafe`. If the failsafe is still
    /// armed (i.e., `owner` is `Some`), it iterates through all documents owned by
    /// the `ScriptThread` and clears their JavaScript runtime to facilitate proper
    /// deallocation of DOM memory.
    ///
    /// Pre-condition: Called when the `ScriptMemoryFailsafe` goes out of scope.
    /// Post-condition: If the owner is present, JavaScript runtimes for owned documents are cleared.
    fn drop(&mut self) {
        if let Some(owner) = self.owner {
            for (_, document) in owner.documents.borrow().iter() {
                document.window().clear_js_runtime_for_script_deallocation();
            }
        }
    }
}

impl ScriptThreadFactory for ScriptThread {
    // Block Logic: Creates and initializes a new script thread.
    // Functional Utility: Spawns a new thread, sets up its environment (thread state, namespaces, roots),
    //                     constructs a `ScriptThread` instance, performs initial page loading,
    //                     and starts the script thread's event loop with memory reporting.
    //
    // Pre-condition: Valid `InitialScriptState`, `LayoutFactory`, `SystemFontServiceProxy`,
    //                `LoadData`, and `user_agent` are provided.
    // Invariant: A new script thread is successfully created and initialized, or a panic occurs.
    /// Creates and initializes a new script thread.
    ///
    /// # Arguments
    /// * `state` - The initial state for the script thread.
    /// * `layout_factory` - A factory for creating layout instances.
    /// * `system_font_service` - A proxy to the system font service.
    /// * `load_data` - The initial load data for the script thread.
    /// * `user_agent` - The user agent string for the script thread.
    ///
    /// Pre-condition: All input parameters are valid.
    /// Post-condition: A new thread is spawned, initialized as a `ScriptThread`, and its event
    /// loop is started, handling initial page load and memory reporting.
    fn create(
        state: InitialScriptState,
        layout_factory: Arc<dyn LayoutFactory>,
        system_font_service: Arc<SystemFontServiceProxy>,
        load_data: LoadData,
        user_agent: Cow<'static, str>,
    ) {
        thread::Builder::new()
            .name(format!("Script{:?}", state.id))
            .spawn(move || {
                thread_state::initialize(ThreadState::SCRIPT | ThreadState::LAYOUT);
                PipelineNamespace::install(state.pipeline_namespace_id);
                TopLevelBrowsingContextId::install(state.top_level_browsing_context_id);
                let roots = RootCollection::new();
                let _stack_roots = ThreadLocalStackRoots::new(&roots);
                let id = state.id;
                let browsing_context_id = state.browsing_context_id;
                let top_level_browsing_context_id = state.top_level_browsing_context_id;
                let parent_info = state.parent_info;
                let opener = state.opener;
                let memory_profiler_sender = state.memory_profiler_sender.clone();
                let window_size = state.window_size;

                let script_thread =
                    ScriptThread::new(state, layout_factory, system_font_service, user_agent);

                SCRIPT_THREAD_ROOT.with(|root| {
                    root.set(Some(&script_thread as *const _));
                });

                let mut failsafe = ScriptMemoryFailsafe::new(&script_thread);

                let origin = MutableOrigin::new(load_data.url.origin());
                script_thread.pre_page_load(InProgressLoad::new(
                    id,
                    browsing_context_id,
                    top_level_browsing_context_id,
                    parent_info,
                    opener,
                    window_size,
                    origin,
                    load_data,
                ));

                let reporter_name = format!("script-reporter-{:?}", id);
                memory_profiler_sender.run_with_memory_reporting(
                    || {
                        script_thread.start(CanGc::note());
                        let _ = script_thread
                            .senders
                            .content_process_shutdown_sender
                            .send(());
                    },
                    reporter_name,
                    ScriptEventLoopSender::MainThread(script_thread.senders.self_sender.clone()),
                    CommonScriptMsg::CollectReports,
                );

                // This must always be the very last operation performed before the thread completes
                failsafe.neuter();
            })
            .expect("Thread spawning failed");
    }
}

impl ScriptThread {
    /// Returns a handle to the parent JavaScript runtime.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: A `ParentRuntime` handle is returned, allowing creation of child runtimes.
    pub(crate) fn runtime_handle() -> ParentRuntime {
        with_optional_script_thread(|script_thread| {
            script_thread.unwrap().js_runtime.prepare_for_new_child()
        })
    }

    /// Checks if the script thread is in a state to continue running, primarily by checking
    /// the `closing` flag set by the Background Hang Monitor.
    ///
    /// Post-condition: Returns `true` if the thread can continue, `false` if it's signaled to close.
    pub(crate) fn can_continue_running() -> bool {
        with_script_thread(|script_thread| script_thread.can_continue_running_inner())
    }

    /// Initiates the shutdown process for the script thread, preparing it for termination.
    /// This involves canceling all tasks for owned documents.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: All tasks for documents owned by this thread are canceled.
    pub(crate) fn prepare_for_shutdown() {
        with_script_thread(|script_thread| {
            script_thread.prepare_for_shutdown_inner();
        })
    }

    /// Sets the flag indicating whether a mutation observer microtask has been queued for processing.
    ///
    /// # Arguments
    /// * `value` - `true` if a microtask is queued, `false` otherwise.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The `mutation_observer_microtask_queued` flag is updated.
    pub(crate) fn set_mutation_observer_microtask_queued(value: bool) {
        with_script_thread(|script_thread| {
            script_thread
                .mutation_observer_microtask_queued
                .set(value);
        })
    }

    /// Returns `true` if a mutation observer microtask is currently queued, `false` otherwise.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The current state of `mutation_observer_microtask_queued` is returned.
    pub(crate) fn is_mutation_observer_microtask_queued() -> bool {
        with_script_thread(|script_thread| script_thread.mutation_observer_microtask_queued.get())
    }

    /// Adds a `MutationObserver` to the list of observers managed by this script thread.
    ///
    /// # Arguments
    /// * `observer` - A reference to the `MutationObserver` to add.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The `observer` is added to `script_thread.mutation_observers`.
    pub(crate) fn add_mutation_observer(observer: &MutationObserver) {
        with_script_thread(|script_thread| {
            script_thread
                .mutation_observers
                .borrow_mut()
                .push(Dom::from_ref(observer));
        })
    }

    /// Retrieves a vector of all `MutationObserver` objects currently managed by this script thread.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: Returns a `Vec` of `DomRoot<MutationObserver>` representing all active observers.
    pub(crate) fn get_mutation_observers() -> Vec<DomRoot<MutationObserver>> {
        with_script_thread(|script_thread| {
            script_thread
                .mutation_observers
                .borrow()
                .iter()
                .map(|o| DomRoot::from_ref(&**o))
                .collect()
        })
    }

    /// Adds an `HTMLSlotElement` to the list of signal slots managed by this script thread.
    ///
    /// # Arguments
    /// * `observer` - A reference to the `HTMLSlotElement` to add.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The `observer` is added to `script_thread.signal_slots`.
    pub(crate) fn add_signal_slot(observer: &HTMLSlotElement) {
        with_script_thread(|script_thread| {
            script_thread
                .signal_slots
                .borrow_mut()
                .push(Dom::from_ref(observer));
        })
    }

    /// Takes ownership of and returns all `HTMLSlotElement` objects from the signal slots,
    /// also removing them from the internal list.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: All `HTMLSlotElement`s are returned, and the internal list is cleared.
    pub(crate) fn take_signal_slots() -> Vec<DomRoot<HTMLSlotElement>> {
        with_script_thread(|script_thread| {
            script_thread
                .signal_slots
                .take()
                .into_iter()
                .inspect(|slot| {
                    slot.remove_from_signal_slots();
                })
                .map(|slot| slot.as_rooted())
                .collect()
        })
    }

    /// Marks a document as having no blocked loads, indicating it has finished loading
    /// all its blocking resources.
    ///
    /// # Arguments
    /// * `doc` - A reference to the `Document` to mark.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The `doc` is added to `script_thread.docs_with_no_blocking_loads`.
    pub(crate) fn mark_document_with_no_blocked_loads(doc: &Document) {
        with_script_thread(|script_thread| {
            script_thread
                .docs_with_no_blocking_loads
                .borrow_mut()
                .insert(Dom::from_ref(doc));
        })
    }

    /// Notifies the script thread that page headers are available, potentially returning a
    /// `ServoParser` for further processing if an incomplete load matches the `pipeline_id`.
    ///
    /// # Arguments
    /// * `id` - The `PipelineId` of the page for which headers are available.
    /// * `metadata` - Optional metadata associated with the page.
    /// * `can_gc` - A `CanGc` token indicating if garbage collection is permitted.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: If an incomplete load matches, a `ServoParser` is returned.
    pub(crate) fn page_headers_available(
        id: &PipelineId,
        metadata: Option<Metadata>,
        can_gc: CanGc,
    ) -> Option<DomRoot<ServoParser>> {
        with_script_thread(|script_thread| {
            script_thread.handle_page_headers_available(id, metadata, can_gc)
        })
    }

    /// Processes a single `CommonScriptMsg` event as if it were the next event
    /// in the queue for this window's event loop.
    ///
    /// # Arguments
    /// * `msg` - The `CommonScriptMsg` to process.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The message is handled, and `true` is returned if further events
    /// should be processed, `false` otherwise.
    pub(crate) fn process_event(msg: CommonScriptMsg) -> bool {
        with_script_thread(|script_thread| {
            if !script_thread.can_continue_running_inner() {
                return false;
            }
            script_thread.handle_msg_from_script(MainThreadScriptMsg::Common(msg));
            true
        })
    }

    /// Schedules a [`TimerEventRequest`] on this [`ScriptThread`]'s [`TimerScheduler`].
    ///
    /// # Arguments
    /// * `request` - The `TimerEventRequest` to schedule.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The timer request is added to the `timer_scheduler`.
    pub(crate) fn schedule_timer(&self, request: TimerEventRequest) {
        self.timer_scheduler.borrow_mut().schedule_timer(request);
    }

    // https://html.spec.whatwg.org/multipage/#await-a-stable-state
    /// Enqueues a microtask to await a stable state, typically used before performing
    /// operations that require the DOM to be settled.
    ///
    /// # Arguments
    /// * `task` - The `Microtask` to enqueue.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The `task` is added to the `microtask_queue`.
    pub(crate) fn await_stable_state(task: Microtask) {
        with_script_thread(|script_thread| {
            script_thread
                .microtask_queue
                .enqueue(task, script_thread.get_cx());
        });
    }

    /// Checks if two origins are "similar enough" for a load operation, primarily
    /// to prevent cross-origin JavaScript URL evaluation.
    ///
    /// # Arguments
    /// * `source` - The `LoadOrigin` of the resource.
    /// * `target` - The `ImmutableOrigin` of the target.
    ///
    /// Pre-condition: None.
    /// Post-condition: Returns `true` if the origins are considered safe for interaction, `false` otherwise.
    /// See <https://github.com/whatwg/html/issues/2591> for more context.
    pub(crate) fn check_load_origin(source: &LoadOrigin, target: &ImmutableOrigin) -> bool {
        match (source, target) {
            (LoadOrigin::Constellation, _) | (LoadOrigin::WebDriver, _) => {
                // Always allow loads initiated by the constellation or webdriver.
                true
            },
            (_, ImmutableOrigin::Opaque(_)) => {
                // If the target is opaque, allow.
                // This covers newly created about:blank auxiliaries, and iframe with no src.
                // TODO: https://github.com/servo/servo/issues/22879
                true
            },
            (LoadOrigin::Script(source_origin), _) => source_origin == target,
        }
    }

    /// Initiates navigation to a new URL within a browsing context.
    /// This corresponds to Step 13 of <https://html.spec.whatwg.org/multipage/#navigate>.
    ///
    /// # Arguments
    /// * `browsing_context` - The `BrowsingContextId` where navigation should occur.
    /// * `pipeline_id` - The `PipelineId` associated with the current document.
    /// * `load_data` - The `LoadData` specifying the target URL and other load parameters.
    /// * `history_handling` - Defines how this navigation should affect the session history.
    ///
    /// Pre-condition: `browsing_context` and `pipeline_id` are valid.
    /// Post-condition: A navigation task is queued; for JavaScript URLs, the script is evaluated,
    /// otherwise a `LoadUrl` message is sent to the Constellation.
    pub(crate) fn navigate(
        browsing_context: BrowsingContextId,
        pipeline_id: PipelineId,
        mut load_data: LoadData,
        history_handling: NavigationHistoryBehavior,
    ) {
        with_script_thread(|script_thread| {
            let is_javascript = load_data.url.scheme() == "javascript";
            // If resource is a request whose url's scheme is "javascript"
            // https://html.spec.whatwg.org/multipage/#javascript-protocol
            if is_javascript {
                let window = match script_thread.documents.borrow().find_window(pipeline_id) {
                    None => return,
                    Some(window) => window,
                };
                let global = window.as_global_scope();
                let trusted_global = Trusted::new(global);
                let sender = script_thread
                    .senders
                    .pipeline_to_constellation_sender
                    .clone();
                let task = task!(navigate_javascript: move || {
                    // Important re security. See https://github.com/servo/servo/issues/23373
                    // TODO: check according to https://w3c.github.io/webappsec-csp/#should-block-navigation-request
                    if let Some(window) = trusted_global.root().downcast::<Window>() {
                        if ScriptThread::check_load_origin(&load_data.load_origin, &window.get_url().origin()) {
                            ScriptThread::eval_js_url(&trusted_global.root(), &mut load_data, CanGc::note());
                            sender
                                .send((pipeline_id, ScriptMsg::LoadUrl(load_data, history_handling)))
                                .unwrap();
                        }
                    }
                });
                global
                    .task_manager()
                    .dom_manipulation_task_source()
                    .queue(task);
            } else {
                if let Some(ref sender) = script_thread.senders.devtools_server_sender {
                    let _ = sender.send(ScriptToDevtoolsControlMsg::Navigate(
                        browsing_context,
                        NavigationState::Start(load_data.url.clone()),
                    ));
                }

                script_thread
                    .senders
                    .pipeline_to_constellation_sender
                    .send((pipeline_id, ScriptMsg::LoadUrl(load_data, history_handling)))
                    .expect("Sending a LoadUrl message to the constellation failed");
            }
        });
    }

    /// Processes an `AttachLayout` event, which involves attaching a new layout to a pipeline.
    ///
    /// # Arguments
    /// * `new_layout_info` - Information about the new layout to attach.
    /// * `origin` - The mutable origin associated with the layout.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: A new layout is handled for the specified pipeline, and the event is profiled.
    pub(crate) fn process_attach_layout(new_layout_info: NewLayoutInfo, origin: MutableOrigin) {
        with_script_thread(|script_thread| {
            let pipeline_id = Some(new_layout_info.new_pipeline_id);
            script_thread.profile_event(
                ScriptThreadEventCategory::AttachLayout,
                pipeline_id,
                || {
                    script_thread.handle_new_layout(new_layout_info, origin);
                },
            )
        });
    }

    /// Retrieves the top-level browsing context ID for a given `BrowsingContextId`.
    ///
    /// # Arguments
    /// * `sender_pipeline` - The `PipelineId` of the sender.
    /// * `browsing_context_id` - The `BrowsingContextId` for which to find the top-level ID.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: Returns an `Option<TopLevelBrowsingContextId>` from the Constellation.
    pub(crate) fn get_top_level_for_browsing_context(
        sender_pipeline: PipelineId,
        browsing_context_id: BrowsingContextId,
    ) -> Option<TopLevelBrowsingContextId> {
        with_script_thread(|script_thread| {
            script_thread.ask_constellation_for_top_level_info(sender_pipeline, browsing_context_id)
        })
    }

    /// Finds a `Document` associated with a given `PipelineId`.
    ///
    /// # Arguments
    /// * `id` - The `PipelineId` of the document to find.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: Returns an `Option<DomRoot<Document>>` if the document is found.
    pub(crate) fn find_document(id: PipelineId) -> Option<DomRoot<Document>> {
        with_script_thread(|script_thread| script_thread.documents.borrow().find_document(id))
    }

    /// Sets whether the user is currently interacting with the page.
    ///
    /// # Arguments
    /// * `interacting` - `true` if the user is interacting, `false` otherwise.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The `is_user_interacting` flag is updated.
    pub(crate) fn set_user_interacting(interacting: bool) {
        with_script_thread(|script_thread| {
            script_thread.is_user_interacting.set(interacting);
        });
    }

    /// Returns `true` if the user is currently interacting with the page, `false` otherwise.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The current state of the `is_user_interacting` flag is returned.
    pub(crate) fn is_user_interacting() -> bool {
        with_script_thread(|script_thread| script_thread.is_user_interacting.get())
    }

    /// Retrieves a set of `PipelineId`s for all fully active documents.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: Returns a `HashSet<PipelineId>` containing IDs of fully active documents.
    pub(crate) fn get_fully_active_document_ids() -> HashSet<PipelineId> {
        with_script_thread(|script_thread| {
            script_thread
                .documents
                .borrow()
                .iter()
                .filter_map(|(id, document)| {
                    if document.is_fully_active() {
                        Some(id)
                    } else {
                        None
                    }
                })
                .fold(HashSet::new(), |mut set, id| {
                    let _ = set.insert(id);
                    set
                })
        })
    }

    /// Finds a `WindowProxy` associated with a given `BrowsingContextId`.
    ///
    /// # Arguments
    /// * `id` - The `BrowsingContextId` of the window proxy to find.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: Returns an `Option<DomRoot<WindowProxy>>` if the window proxy is found.
    pub(crate) fn find_window_proxy(id: BrowsingContextId) -> Option<DomRoot<WindowProxy>> {
        with_script_thread(|script_thread| {
            script_thread
                .window_proxies
                .borrow()
                .get(&id)
                .map(|context| DomRoot::from_ref(&**context))
        })
    }

    /// Finds a `WindowProxy` by its name.
    ///
    /// # Arguments
    /// * `name` - The `DOMString` representing the name of the window proxy to find.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: Returns an `Option<DomRoot<WindowProxy>>` if a matching window proxy is found.
    pub(crate) fn find_window_proxy_by_name(name: &DOMString) -> Option<DomRoot<WindowProxy>> {
        with_script_thread(|script_thread| {
            for (_, proxy) in script_thread.window_proxies.borrow().iter() {
                if proxy.get_name() == *name {
                    return Some(DomRoot::from_ref(&**proxy));
                }
            }
            None
        })
    }

    /// Returns the `WorkletThreadPool` associated with this script thread,
    /// creating it if it doesn't already exist.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: A reference-counted `WorkletThreadPool` is returned.
    pub(crate) fn worklet_thread_pool() -> Rc<WorkletThreadPool> {
        with_optional_script_thread(|script_thread| {
            let script_thread = script_thread.unwrap();
            script_thread
                .worklet_thread_pool
                .borrow_mut()
                .get_or_insert_with(|| {
                    let init = WorkletGlobalScopeInit {
                        to_script_thread_sender: script_thread.senders.self_sender.clone(),
                        resource_threads: script_thread.resource_threads.clone(),
                        mem_profiler_chan: script_thread.senders.memory_profiler_sender.clone(),
                        time_profiler_chan: script_thread.senders.time_profiler_sender.clone(),
                        devtools_chan: script_thread.senders.devtools_server_sender.clone(),
                        to_constellation_sender: script_thread
                            .senders
                            .pipeline_to_constellation_sender
                            .clone(),
                        image_cache: script_thread.image_cache.clone(),
                        user_agent: script_thread.user_agent.clone(),
                        #[cfg(feature = "webgpu")]
                        gpu_id_hub: script_thread.gpu_id_hub.clone(),
                        inherited_secure_context: script_thread.inherited_secure_context,
                    };
                    Rc::new(WorkletThreadPool::spawn(init))
                })
                .clone()
        })
    }

    /// Handles the registration of a paint worklet.
    ///
    /// # Arguments
    /// * `pipeline_id` - The `PipelineId` of the document registering the worklet.
    /// * `name` - The name of the paint worklet.
    /// * `properties` - A vector of property atoms associated with the worklet.
    /// * `painter` - A boxed `Painter` trait object for rendering.
    ///
    /// Pre-condition: `pipeline_id` corresponds to an active window.
    /// Post-condition: The paint worklet is registered with the document's layout.
    fn handle_register_paint_worklet(
        &self,
        pipeline_id: PipelineId,
        name: Atom,
        properties: Vec<Atom>,
        painter: Box<dyn Painter>,
    ) {
        let Some(window) = self.documents.borrow().find_window(pipeline_id) else {
            warn!("Paint worklet registered after pipeline {pipeline_id} closed.");
            return;
        };

        window
            .layout_mut()
            .register_paint_worklet_modules(name, properties, painter);
    }

    /// Pushes a new element queue onto the custom element reaction stack.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: A new empty queue is added to `custom_element_reaction_stack`.
    pub(crate) fn push_new_element_queue() {
        with_script_thread(|script_thread| {
            script_thread
                .custom_element_reaction_stack
                .push_new_element_queue();
        })
    }

    /// Pops the current element queue from the custom element reaction stack.
    ///
    /// # Arguments
    /// * `can_gc` - A `CanGc` token indicating if garbage collection is permitted.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS and the stack is not empty.
    /// Post-condition: The top queue is removed and processed.
    pub(crate) fn pop_current_element_queue(can_gc: CanGc) {
        with_script_thread(|script_thread| {
            script_thread
                .custom_element_reaction_stack
                .pop_current_element_queue(can_gc);
        })
    }

    /// Enqueues a callback reaction for a custom element.
    ///
    /// # Arguments
    /// * `element` - The `Element` for which the reaction is enqueued.
    /// * `reaction` - The `CallbackReaction` to enqueue.
    /// * `definition` - Optional `CustomElementDefinition` associated with the element.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The reaction is added to the current element queue on the stack.
    pub(crate) fn enqueue_callback_reaction(
        element: &Element,
        reaction: CallbackReaction,
        definition: Option<Rc<CustomElementDefinition>>,
    ) {
        with_script_thread(|script_thread| {
            script_thread
                .custom_element_reaction_stack
                .enqueue_callback_reaction(element, reaction, definition);
        })
    }

    /// Enqueues an upgrade reaction for a custom element.
    ///
    /// # Arguments
    /// * `element` - The `Element` for which the reaction is enqueued.
    /// * `definition` - The `CustomElementDefinition` for the element.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The upgrade reaction is added to the current element queue on the stack.
    pub(crate) fn enqueue_upgrade_reaction(
        element: &Element,
        definition: Rc<CustomElementDefinition>,
    ) {
        with_script_thread(|script_thread| {
            script_thread
                .custom_element_reaction_stack
                .enqueue_upgrade_reaction(element, definition);
        })
    }

    /// Invokes the backup element queue on the custom element reaction stack.
    ///
    /// # Arguments
    /// * `can_gc` - A `CanGc` token indicating if garbage collection is permitted.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The backup element queue is processed.
    pub(crate) fn invoke_backup_element_queue(can_gc: CanGc) {
        with_script_thread(|script_thread| {
            script_thread
                .custom_element_reaction_stack
                .invoke_backup_element_queue(can_gc);
        })
    }

    /// Saves a node ID, ensuring its uniqueness within the script thread.
    ///
    /// # Arguments
    /// * `node_id` - The `String` representation of the node ID to save.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: The `node_id` is inserted into `script_thread.node_ids`.
    pub(crate) fn save_node_id(node_id: String) {
        with_script_thread(|script_thread| {
            script_thread.node_ids.borrow_mut().insert(node_id);
        })
    }

    /// Checks if a node ID already exists within the script thread.
    ///
    /// # Arguments
    /// * `node_id` - The string representation of the node ID to check.
    ///
    /// Pre-condition: A `ScriptThread` instance exists in the current thread's TLS.
    /// Post-condition: Returns `true` if the `node_id` exists, `false` otherwise.
    pub(crate) fn has_node_id(node_id: &str) -> bool {
        with_script_thread(|script_thread| script_thread.node_ids.borrow().contains(node_id))
    }

    /// Creates a new `ScriptThread` instance.
    ///
    /// # Arguments
    /// * `state` - The initial state for the script thread.
    /// * `layout_factory` - A factory for creating layout instances.
    /// * `system_font_service` - A proxy to the system font service.
    /// * `user_agent` - The user agent string for the script thread.
    ///
    /// Pre-condition: All input parameters are valid.
    /// Post-condition: A new `ScriptThread` instance is returned, initialized with its
    /// runtime, receivers, senders, and other components.
    pub(crate) fn new(
        state: InitialScriptState,
        layout_factory: Arc<dyn LayoutFactory>,
        system_font_service: Arc<SystemFontServiceProxy>,
        user_agent: Cow<'static, str>,
    ) -> ScriptThread {
        let (self_sender, self_receiver) = unbounded();
        let runtime = Runtime::new(Some(SendableTaskSource {
            sender: ScriptEventLoopSender::MainThread(self_sender.clone()),
            pipeline_id: state.id,
            name: TaskSourceName::Networking,
            canceller: Default::default(),
        }));
        let cx = runtime.cx();

        unsafe {
            SetWindowProxyClass(cx, GetWindowProxyClass());
            JS_AddInterruptCallback(cx, Some(interrupt_callback));
        }

        // Ask the router to proxy IPC messages from the control port to us.
        let constellation_receiver =
            ROUTER.route_ipc_receiver_to_new_crossbeam_receiver(state.constellation_receiver);

        // Ask the router to proxy IPC messages from the devtools to us.
        let devtools_server_sender = state.devtools_server_sender;
        let (ipc_devtools_sender, ipc_devtools_receiver) = ipc::channel().unwrap();
        let devtools_server_receiver = devtools_server_sender
            .as_ref()
            .map(|_| ROUTER.route_ipc_receiver_to_new_crossbeam_receiver(ipc_devtools_receiver))
            .unwrap_or_else(crossbeam_channel::never);

        let task_queue = TaskQueue::new(self_receiver, self_sender.clone());

        let closing = Arc::new(AtomicBool::new(false));
        let background_hang_monitor_exit_signal = BHMExitSignal {
            closing: closing.clone(),
            js_context: runtime.thread_safe_js_context(),
        };

        let background_hang_monitor = state.background_hang_monitor_register.register_component(
            MonitoredComponentId(state.id, MonitoredComponentType::Script),
            Duration::from_millis(1000),
            Duration::from_millis(5000),
            Some(Box::new(background_hang_monitor_exit_signal)),
        );

        let (image_cache_sender, image_cache_receiver) = unbounded();
        let (ipc_image_cache_sender, ipc_image_cache_receiver) = ipc::channel().unwrap();
        ROUTER.add_typed_route(
            ipc_image_cache_receiver,
            Box::new(move |message| {
                let _ = image_cache_sender.send(message.unwrap());
            }),
        );

        let receivers = ScriptThreadReceivers {
            constellation_receiver,
            image_cache_receiver,
            devtools_server_receiver,
            // Initialized to `never` until WebGPU is initialized.
            #[cfg(feature = "webgpu")]
            webgpu_receiver: RefCell::new(crossbeam_channel::never()),
        };

        let opts = opts::get();
        let senders = ScriptThreadSenders {
            self_sender,
            #[cfg(feature = "bluetooth")]
            bluetooth_sender: state.bluetooth_sender,
            constellation_sender: state.constellation_sender,
            pipeline_to_constellation_sender: state.pipeline_to_constellation_sender.sender.clone(),
            layout_to_constellation_ipc_sender: state.layout_to_constellation_ipc_sender,
            image_cache_sender: ipc_image_cache_sender,
            time_profiler_sender: state.time_profiler_sender,
            memory_profiler_sender: state.memory_profiler_sender,
            devtools_server_sender,
            devtools_client_to_script_thread_sender: ipc_devtools_sender,
            content_process_shutdown_sender: state.content_process_shutdown_sender,
        };

        ScriptThread {
            documents: DomRefCell::new(DocumentCollection::default()),
            last_render_opportunity_time: Default::default(),
            window_proxies: DomRefCell::new(HashMapTracedValues::new()),
            incomplete_loads: DomRefCell::new(vec![]),
            incomplete_parser_contexts: IncompleteParserContexts(RefCell::new(vec![])),
            senders,
            receivers,
            image_cache: state.image_cache.clone(),
            resource_threads: state.resource_threads,
            task_queue,
            background_hang_monitor,
            closing,
            timer_scheduler: Default::default(),
            microtask_queue: runtime.microtask_queue.clone(),
            js_runtime: Rc::new(runtime),
            topmost_mouse_over_target: MutNullableDom::new(Default::default()),
            closed_pipelines: DomRefCell::new(HashSet::new()),
            mutation_observer_microtask_queued: Default::default(),
            mutation_observers: Default::default(),
            signal_slots: Default::default(),
            system_font_service,
            webgl_chan: state.webgl_chan,
            #[cfg(feature = "webxr")]
            webxr_registry: state.webxr_registry,
            worklet_thread_pool: Default::default(),
            docs_with_no_blocking_loads: Default::default(),
            custom_element_reaction_stack: CustomElementReactionStack::new(),
            webrender_document: state.webrender_document,
            compositor_api: state.compositor_api,
            profile_script_events: opts.debug.profile_script_events,
            print_pwm: opts.print_pwm,
            relayout_event: opts.debug.relayout_event,
            unminify_js: opts.unminify_js,
            local_script_source: opts.local_script_source.clone(),
            unminify_css: opts.unminify_css,
            userscripts_path: opts.userscripts.clone(),
            user_agent,
            player_context: state.player_context,
            node_ids: Default::default(),
            is_user_interacting: Cell::new(false),
            #[cfg(feature = "webgpu")]
            gpu_id_hub: Arc::new(IdentityHub::default()),
            inherited_secure_context: state.inherited_secure_context,
            layout_factory,
        }
    }

    #[allow(unsafe_code)]
    /// Returns a `JSContext` representing the current JavaScript execution context.
    ///
    /// Pre-condition: The JavaScript runtime is initialized.
    /// Post-condition: A `JSContext` wrapper around the raw `*mut UnsafeJSContext` is returned.
    pub(crate) fn get_cx(&self) -> JSContext {
        unsafe { JSContext::from_ptr(self.js_runtime.cx()) }
    }

    /// Internal method to check if the script thread is in a state to continue running.
    ///
    /// Post-condition: Returns `true` if the `closing` flag is `false`, `false` otherwise.
    fn can_continue_running_inner(&self) -> bool {
        if self.closing.load(Ordering::SeqCst) {
            return false;
        }
        true
    }

    /// Internal method to prepare the script thread for shutdown. This involves
    /// canceling all pending tasks for documents owned by this thread.
    ///
    /// Post-condition: All tasks for owned documents are canceled.
    fn prepare_for_shutdown_inner(&self) {
        let docs = self.documents.borrow();
        for (_, document) in docs.iter() {
            document
                .owner_global()
                .task_manager()
                .cancel_all_tasks_and_ignore_future_tasks();
        }
    }

    /// Starts the script thread's main event loop. The thread will continuously
    /// process messages until it receives an exit signal.
    ///
    /// # Arguments
    /// * `can_gc` - A `CanGc` token indicating if garbage collection is permitted.
    ///
    /// Pre-condition: The `ScriptThread` is initialized.
    /// Post-condition: The thread enters a loop, processing messages and tasks until shutdown.
    pub(crate) fn start(&self, can_gc: CanGc) {
        debug!("Starting script thread.");
        while self.handle_msgs(can_gc) {
            // Go on...
            debug!("Running script thread.");
        }
        debug!("Stopped script thread.");
    }

    /// Processes a compositor mouse move event, handling hit testing and updating the
    /// topmost mouse-over target.
    ///
    /// # Arguments
    /// * `document` - A reference to the `Document` where the event occurred.
    /// * `hit_test_result` - Optional `CompositorHitTestResult` from the compositor.
    /// * `pressed_mouse_buttons` - A bitmask indicating currently pressed mouse buttons.
    /// * `can_gc` - A `CanGc` token.
    ///
    /// Pre-condition: `document` is valid.
    /// Post-condition: Mouse move event is processed, `topmost_mouse_over_target` is updated,
    /// and `EmbedderMsg::Status` is potentially sent to the embedder.
    fn process_mouse_move_event(
        &self,
        document: &Document,
        hit_test_result: Option<CompositorHitTestResult>,
        pressed_mouse_buttons: u16,
        can_gc: CanGc,
    ) {
        // Get the previous target temporarily
        let prev_mouse_over_target = self.topmost_mouse_over_target.get();

        unsafe {
            document.handle_mouse_move_event(
                hit_test_result,
                pressed_mouse_buttons,
                &self.topmost_mouse_over_target,
                can_gc,
            )
        }

        // Short-circuit if nothing changed
        if self.topmost_mouse_over_target.get() == prev_mouse_over_target {
            return;
        }

        let mut state_already_changed = false;

        // Notify Constellation about the topmost anchor mouse over target.
        let window = document.window();
        if let Some(target) = self.topmost_mouse_over_target.get() {
            if let Some(anchor) = target
                .upcast::<Node>()
                .inclusive_ancestors(ShadowIncluding::No)
                .filter_map(DomRoot::downcast::<HTMLAnchorElement>)
                .next()
            {
                let status = anchor
                    .upcast::<Element>()
                    .get_attribute(&ns!(), &local_name!("href"))
                    .and_then(|href| {
                        let value = href.value();
                        let url = document.url();
                        url.join(&value).map(|url| url.to_string()).ok()
                    });
                let event = EmbedderMsg::Status(document.webview_id(), status);
                window.send_to_embedder(event);

                state_already_changed = true;
            }
        }

        // We might have to reset the anchor state
        if !state_already_changed {
            if let Some(target) = prev_mouse_over_target {
                if target
                    .upcast::<Node>()
                    .inclusive_ancestors(ShadowIncluding::No)
                    .filter_map(DomRoot::downcast::<HTMLAnchorElement>)
                    .next()
                    .is_some()
                {
                    let event = EmbedderMsg::Status(window.webview_id(), None);
                    window.send_to_embedder(event);
                }
            }
        }
    }

    /// Processes pending input events received from the compositor.
    ///
    /// # Arguments
    /// * `pipeline_id` - The `PipelineId` of the document to process events for.
    /// * `can_gc` - A `CanGc` token.
    ///
    /// Pre-condition: `pipeline_id` corresponds to an active and non-closed document.
    /// Post-condition: All pending input events for the document are dispatched, and
    /// the user interaction state is updated.
    fn proces_pending_input_events(&self, pipeline_id: PipelineId, can_gc: CanGc) {
        let Some(document) = self.documents.borrow().find_document(pipeline_id) else {
            warn!("Processing pending compositor events for closed pipeline {pipeline_id}.");
            return;
        };
        // Do not handle events if the BC has been, or is being, discarded
        if document.window().Closed() {
            warn!("Compositor event sent to a pipeline with a closed window {pipeline_id}.");
            return;
        };
        ScriptThread::set_user_interacting(true);

        let window = document.window();
        let _realm = enter_realm(document.window());
        for event in document.take_pending_input_events().into_iter() {
            match event.event {
                InputEvent::MouseButton(mouse_button_event) => {
                    document.handle_mouse_button_event(
                        mouse_button_event,
                        event.hit_test_result,
                        event.pressed_mouse_buttons,
                        can_gc,
                    );
                },
                InputEvent::MouseMove(_) => {
                    // The event itself is unnecessary here, because the point in the viewport is in the hit test.
                    self.process_mouse_move_event(
                        &document,
                        event.hit_test_result,
                        event.pressed_mouse_buttons,
                        can_gc,
                    );
                },
                InputEvent::Touch(touch_event) => {
                    let touch_result =
                        document.handle_touch_event(touch_event, event.hit_test_result, can_gc);
                    match touch_result {
                        TouchEventResult::Processed(handled) => {
                            let result = if handled {
                                EventResult::DefaultAllowed(touch_event.action)
                            } else {
                                EventResult::DefaultPrevented(touch_event.event_type)
                            };
                            let message = ScriptMsg::TouchEventProcessed(result);
                            self.senders
                                .pipeline_to_constellation_sender
                                .send((pipeline_id, message))
                                .unwrap();
                        },
                        _ => {
                            // TODO: Calling preventDefault on a touchup event should prevent clicks.
                        },
                    }
                },
                InputEvent::Wheel(wheel_event) => {
                    document.handle_wheel_event(wheel_event, event.hit_test_result, can_gc);
                },
                InputEvent::Keyboard(keyboard_event) => {
                    document.dispatch_key_event(keyboard_event, can_gc);
                },
                InputEvent::Ime(ime_event) => {
                    document.dispatch_ime_event(ime_event, can_gc);
                },
                InputEvent::Gamepad(gamepad_event) => {
                    window.as_global_scope().handle_gamepad_event(gamepad_event);
                },
                InputEvent::EditingAction(editing_action_event) => {
                    document.handle_editing_action(editing_action_event, can_gc);
                },
            }
        }
        ScriptThread::set_user_interacting(false);
    }

    /// Updates the rendering based on the HTML specification's "update the rendering" steps.
    /// This function attempts to update the rendering and performs a microtask checkpoint
    /// if rendering was actually updated.
    ///
    /// # Arguments
    /// * `requested_by_compositor` - `true` if this update was explicitly requested by the compositor.
    /// * `can_gc` - A `CanGc` token.
    ///
    /// Pre-condition: The script thread is running and not in a closing state.
    /// Post-condition: The rendering is updated for all active documents, animation callbacks are
    /// processed, and a microtask checkpoint is performed.
    /// See <https://html.spec.whatwg.org/multipage/#update-the-rendering>.
    pub(crate) fn update_the_rendering(&self, requested_by_compositor: bool, can_gc: CanGc) {
        *self.last_render_opportunity_time.borrow_mut() = Some(Instant::now());

        if !self.can_continue_running_inner() {
            return;
        }

        // Run rafs for all pipeline, if a raf tick was received for any.
        // This ensures relative ordering of rafs between parent doc and iframes.
        let should_run_rafs = self
            .documents
            .borrow()
            .iter()
            .any(|(_, doc)| doc.is_fully_active() && doc.has_received_raf_tick());

        let any_animations_running = self.documents.borrow().iter().any(|(_, document)| {
            document.is_fully_active() && document.animations().running_animation_count() != 0
        });

        // TODO: The specification says to filter out non-renderable documents,
        // as well as those for which a rendering update would be unnecessary,
        // but this isn't happening here.

        // If we aren't explicitly running rAFs, this update wasn't requested by the compositor,
        // and we are running animations, then wait until the compositor tells us it is time to
        // update the rendering via a TickAllAnimations message.
        if !requested_by_compositor && any_animations_running {
            return;
        }

        // > 2. Let docs be all fully active Document objects whose relevant agent's event loop
        // > is eventLoop, sorted arbitrarily except that the following conditions must be
        // > met:
        //
        // > Any Document B whose container document is A must be listed after A in the
        // > list.
        //
        // > If there are two documents A and B that both have the same non-null container
        // > document C, then the order of A and B in the list must match the
        // > shadow-including tree order of their respective navigable containers in C's
        // > node tree.
        //
        // > In the steps below that iterate over docs, each Document must be processed in
        // > the order it is found in the list.
        let documents_in_order = self.documents.borrow().documents_in_order();

        // TODO: The specification reads: "for doc in docs" at each step whereas this runs all
        // steps per doc in docs. Currently `<iframe>` resizing depends on a parent being able to
        // queue resize events on a child and have those run in the same call to this method, so
        // that needs to be sorted out to fix this.
        for pipeline_id in documents_in_order.iter() {
            let document = self
                .documents
                .borrow()
                .find_document(*pipeline_id)
                .expect("Got pipeline for Document not managed by this ScriptThread.");

            if !document.is_fully_active() {
                continue;
            }

            // TODO(#31581): The steps in the "Revealing the document" section need to be implemente
            // `proces_pending_input_events` handles the focusing steps as well as other events
            // from the compositor.

            // TODO: Should this be broken and to match the specification more closely? For instance see
            // https://html.spec.whatwg.org/multipage/#flush-autofocus-candidates.
            self.proces_pending_input_events(*pipeline_id, can_gc);

            // TODO(#31665): Implement the "run the scroll steps" from
            // https://drafts.csswg.org/cssom-view/#document-run-the-scroll-steps.

            // > 8. For each doc of docs, run the resize steps for doc. [CSSOMVIEW]
            if document.window().run_the_resize_steps(can_gc) {
                // Evaluate media queries and report changes.
                document
                    .window()
                    .evaluate_media_queries_and_report_changes(can_gc);

                // https://html.spec.whatwg.org/multipage/#img-environment-changes
                // As per the spec, this can be run at any time.
                document.react_to_environment_changes()
            }

            // > 11. For each doc of docs, update animations and send events for doc, passing
            // > in relative high resolution time given frameTimestamp and doc's relevant
            // > global object as the timestamp [WEBANIMATIONS]
            document.update_animations_and_send_events(can_gc);

            // TODO(#31866): Implement "run the fullscreen steps" from
            // https://fullscreen.spec.whatwg.org/multipage/#run-the-fullscreen-steps.

            // TODO(#31868): Implement the "context lost steps" from
            // https://html.spec.whatwg.org/multipage/#context-lost-steps.

            // > 14. For each doc of docs, run the animation frame callbacks for doc, passing
            // > in the relative high resolution time given frameTimestamp and doc's
            // > relevant global object as the timestamp.
            if should_run_rafs {
                document.run_the_animation_frame_callbacks();
            }

            // Run the resize observer steps.
            let _realm = enter_realm(&*document);
            let mut depth = Default::default();
            while document.gather_active_resize_observations_at_depth(&depth, can_gc) {
                // Note: this will reflow the doc.
                depth = document.broadcast_active_resize_observations(can_gc);
            }

            if document.has_skipped_resize_observations() {
                document.deliver_resize_loop_error_notification(can_gc);
            }

            // TODO(#31870): Implement step 17: if the focused area of doc is not a focusable area,
            // then run the focusing steps for document's viewport.

            // TODO: Perform pending transition operations from
            // https://drafts.csswg.org/css-view-transitions/#perform-pending-transition-operations.

            // TODO(#31021): Run the update intersection observations steps from
            // https://w3c.github.io/IntersectionObserver/#run-the-update-intersection-observations-steps

            // TODO: Mark paint timing from https://w3c.github.io/paint-timing.

            #[cfg(feature = "webgpu")]
            document.update_rendering_of_webgpu_canvases();

            // > Step 22: For each doc of docs, update the rendering or user interface of
            // > doc and its node navigable to reflect the current state.
            let window = document.window();
            if document.is_fully_active() {
                window.reflow(ReflowGoal::UpdateTheRendering, can_gc);
            }

            // TODO: Process top layer removals according to
            // https://drafts.csswg.org/css-position-4/#process-top-layer-removals.
        }

        // Perform a microtask checkpoint as the specifications says that *update the rendering*
        // should be run in a task and a microtask checkpoint is always done when running tasks.
        self.perform_a_microtask_checkpoint(can_gc);

        // If there are pending reflows, they were probably caused by the execution of
        // the microtask checkpoint above and we should spin the event loop one more
        // time to resolve them.
        self.schedule_rendering_opportunity_if_necessary();
    }

    /// Schedules the next rendering opportunity if there are pending reflows and
    /// rendering is not already driven by the compositor.
    ///
    /// Pre-condition: `ScriptThread` is initialized.
    /// Post-condition: A task to update rendering is queued if necessary.
    fn schedule_rendering_opportunity_if_necessary(&self) {
        // If any Document has active animations of rAFs, then we should be receiving
        // regular rendering opportunities from the compositor (or fake animation frame
        // ticks). In this case, don't schedule an opportunity, just wait for the next
        // one.
        if self.documents.borrow().iter().any(|(_, document)| {
            document.is_fully_active() &&
                (document.animations().running_animation_count() != 0 ||
                    document.has_active_request_animation_frame_callbacks())
        }) {
            return;
        }

        let Some((_, document)) = self.documents.borrow().iter().find(|(_, document)| {
            document.is_fully_active() &&
                !document.window().layout_blocked() &&
                document.needs_reflow().is_some()
        }) else {
            return;
        };

        // Queues a task to update the rendering.
        // <https://html.spec.whatwg.org/multipage/#event-loop-processing-model:queue-a-global-task>
        //
        // Note: The specification says to queue a task using the navigable's active
        // window, but then updates the rendering for all documents.
        //
        // This task is empty because any new IPC messages in the ScriptThread trigger a
        // rendering update when animations are not running.
        let _realm = enter_realm(&*document);
        document
            .owner_global()
            .task_manager()
            .rendering_task_source()
            .queue_unconditionally(task!(update_the_rendering: move || { }));
    }

    /// Handles incoming messages from other tasks and the task queue, processing them
    /// in the script thread's event loop.
    ///
    /// # Arguments
    /// * `can_gc` - A `CanGc` token.
    ///
    /// Pre-condition: The script thread is running.
    /// Post-condition: Messages are processed, and the function returns `true` to continue
    /// the event loop, or `false` if an exit message was handled.
    fn handle_msgs(&self, can_gc: CanGc) -> bool {
        // Proritize rendering tasks and others, and gather all other events as `sequential`.
        let mut sequential = vec![];

        // Notify the background-hang-monitor we are waiting for an event.
        self.background_hang_monitor.notify_wait();

        // Receive at least one message so we don't spinloop.
        debug!("Waiting for event.");
        let mut event = self
            .receivers
            .recv(&self.task_queue, &self.timer_scheduler.borrow());

        let mut compositor_requested_update_the_rendering = false;
        loop {
            debug!("Handling event: {event:?}");

            // Dispatch any completed timers, so that their tasks can be run below.
            self.timer_scheduler
                .borrow_mut()
                .dispatch_completed_timers();

            let _realm = event.pipeline_id().map(|id| {
                let global = self.documents.borrow().find_global(id);
                global.map(|global| enter_realm(&*global))
            });

            // https://html.spec.whatwg.org/multipage/#event-loop-processing-model step 7
            match event {
                // This has to be handled before the ResizeMsg below,
                // otherwise the page may not have been added to the
                // child list yet, causing the find() to fail.
                MixedMessage::FromConstellation(ScriptThreadMessage::AttachLayout(
                    new_layout_info,
                )) => {
                    let pipeline_id = new_layout_info.new_pipeline_id;
                    self.profile_event(
                        ScriptThreadEventCategory::AttachLayout,
                        Some(pipeline_id),
                        || {
                            // If this is an about:blank or about:srcdoc load, it must share the
                            // creator's origin. This must match the logic in the constellation
                            // when creating a new pipeline
                            let not_an_about_blank_and_about_srcdoc_load =
                                new_layout_info.load_data.url.as_str() != "about:blank" &&
                                    new_layout_info.load_data.url.as_str() != "about:srcdoc";
                            let origin = if not_an_about_blank_and_about_srcdoc_load {
                                MutableOrigin::new(new_layout_info.load_data.url.origin())
                            } else if let Some(parent) =
                                new_layout_info.parent_info.and_then(|pipeline_id| {
                                    self.documents.borrow().find_document(pipeline_id)
                                })
                            {
                                parent.origin().clone()
                            } else if let Some(creator) = new_layout_info
                                .load_data
                                .creator_pipeline_id
                                .and_then(|pipeline_id| {
                                    self.documents.borrow().find_document(pipeline_id)
                                })
                            {
                                creator.origin().clone()
                            } else {
                                MutableOrigin::new(ImmutableOrigin::new_opaque())
                            };

                            self.handle_new_layout(new_layout_info, origin);
                        },
                    )
                },
                MixedMessage::FromConstellation(ScriptThreadMessage::Resize(
                    id,
                    size,
                    size_type,
                )) => {
                    self.handle_resize_message(id, size, size_type);
                },
                MixedMessage::FromConstellation(ScriptThreadMessage::Viewport(id, rect)) => self
                    .profile_event(ScriptThreadEventCategory::SetViewport, Some(id), || {
                        self.handle_viewport(id, rect);
                    }),
                MixedMessage::FromConstellation(ScriptThreadMessage::TickAllAnimations(
                    pipeline_id,
                    tick_type,
                )) => {
                    if let Some(document) = self.documents.borrow().find_document(pipeline_id) {
                        document.note_pending_animation_tick(tick_type);
                        compositor_requested_update_the_rendering = true;
                    } else {
                        warn!(
                            "Trying to note pending animation tick for closed pipeline {}.",
                            pipeline_id
                        )
                    }
                },
                MixedMessage::FromConstellation(ScriptThreadMessage::SendInputEvent(id, event)) => {
                    self.handle_input_event(id, event)
                },
                MixedMessage::FromScript(MainThreadScriptMsg::Common(CommonScriptMsg::Task(
                    _,
                    _,
                    _,
                    TaskSourceName::Rendering,
                ))) => {
                    // Instead of interleaving any number of update the rendering tasks with other
                    // message handling, we run those steps only once at the end of each call of
                    // this function.
                },
                MixedMessage::FromScript(MainThreadScriptMsg::Inactive) => {
                    // An event came-in from a document that is not fully-active, it has been stored by the task-queue.
                    // Continue without adding it to "sequential".
                },
                MixedMessage::FromConstellation(ScriptThreadMessage::ExitFullScreen(id)) => self
                    .profile_event(ScriptThreadEventCategory::ExitFullscreen, Some(id), || {
                        self.handle_exit_fullscreen(id, can_gc);
                    }),
                _ => {
                    sequential.push(event);
                },
            }

            // If any of our input sources has an event pending, we'll perform another iteration
            // and check for more resize events. If there are no events pending, we'll move
            // on and execute the sequential non-resize events we've seen.
            match self.receivers.try_recv(&self.task_queue) {
                Some(new_event) => event = new_event,
                None => break,
            }
        }

        // Process the gathered events.
        debug!("Processing events.");
        for msg in sequential {
            debug!("Processing event {:?}.", msg);
            let category = self.categorize_msg(&msg);
            let pipeline_id = msg.pipeline_id();
            let _realm = pipeline_id.and_then(|id| {
                let global = self.documents.borrow().find_global(id);
                global.map(|global| enter_realm(&*global))
            });

            if self.closing.load(Ordering::SeqCst) {
                // If we've received the closed signal from the BHM, only handle exit messages.
                match msg {
                    MixedMessage::FromConstellation(ScriptThreadMessage::ExitScriptThread) => {
                        self.handle_exit_script_thread_msg(can_gc);
                        return false;
                    },
                    MixedMessage::FromConstellation(ScriptThreadMessage::ExitPipeline(
                        pipeline_id,
                        discard_browsing_context,
                    )) => {
                        self.handle_exit_pipeline_msg(
                            pipeline_id,
                            discard_browsing_context,
                            can_gc,
                        );
                    },
                    _ => {},
                }
                continue;
            }

            let exiting = self.profile_event(category, pipeline_id, move || {
                match msg {
                    MixedMessage::FromConstellation(ScriptThreadMessage::ExitScriptThread) => {
                        self.handle_exit_script_thread_msg(can_gc);
                        return true;
                    },
                    MixedMessage::FromConstellation(inner_msg) => {
                        self.handle_msg_from_constellation(inner_msg, can_gc)
                    },
                    MixedMessage::FromScript(inner_msg) => self.handle_msg_from_script(inner_msg),
                    MixedMessage::FromDevtools(inner_msg) => {
                        self.handle_msg_from_devtools(inner_msg, can_gc)
                    },
                    MixedMessage::FromImageCache(inner_msg) => {
                        self.handle_msg_from_image_cache(inner_msg)
                    },
                    #[cfg(feature = "webgpu")]
                    MixedMessage::FromWebGPUServer(inner_msg) => {
                        self.handle_msg_from_webgpu_server(inner_msg, can_gc)
                    },
                    MixedMessage::TimerFired => {},
                }

                false
            });

            // If an `ExitScriptThread` message was handled above, bail out now.
            if exiting {
                return false;
            }

            // https://html.spec.whatwg.org/multipage/#event-loop-processing-model step 6
            // TODO(#32003): A microtask checkpoint is only supposed to be performed after running a task.
            self.perform_a_microtask_checkpoint(can_gc);
        }

        {
            // https://html.spec.whatwg.org/multipage/#the-end step 6
            let mut docs = self.docs_with_no_blocking_loads.borrow_mut();
            for document in docs.iter() {
                let _realm = enter_realm(&**document);
                document.maybe_queue_document_completion();
            }
            docs.clear();
        }

        // Update the rendering whenever we receive an IPC message. This may not actually do anything if
        // we are running animations and the compositor hasn't requested a new frame yet via a TickAllAnimatons
        // message.
        self.update_the_rendering(compositor_requested_update_the_rendering, can_gc);

        true
    }

    /// Categorizes an incoming `MixedMessage` into a `ScriptThreadEventCategory` for profiling.
    ///
    /// # Arguments
    /// * `msg` - The `MixedMessage` to categorize.
    ///
    /// Pre-condition: `msg` is a valid message type.
    /// Post-condition: Returns the corresponding `ScriptThreadEventCategory`.
    fn categorize_msg(&self, msg: &MixedMessage) -> ScriptThreadEventCategory {
        match *msg {
            MixedMessage::FromConstellation(ref inner_msg) => match *inner_msg {
                ScriptThreadMessage::SendInputEvent(_, _) => ScriptThreadEventCategory::InputEvent,
                _ => ScriptThreadEventCategory::ConstellationMsg,
            },
            // TODO https://github.com/servo/servo/issues/18998
            MixedMessage::FromDevtools(_) => ScriptThreadEventCategory::DevtoolsMsg,
            MixedMessage::FromImageCache(_) => ScriptThreadEventCategory::ImageCacheMsg,
            MixedMessage::FromScript(ref inner_msg) => match *inner_msg {
                MainThreadScriptMsg::Common(CommonScriptMsg::Task(category, ..)) => category,
                MainThreadScriptMsg::RegisterPaintWorklet { .. } => {
                    ScriptThreadEventCategory::WorkletEvent
                },
                _ => ScriptThreadEventCategory::ScriptEvent,
            },
            #[cfg(feature = "webgpu")]
            MixedMessage::FromWebGPUServer(_) => ScriptThreadEventCategory::WebGPUMsg,
            MixedMessage::TimerFired => ScriptThreadEventCategory::TimerEvent,
        }
    }

    /// Profiles an event by recording its execution duration and handling any associated
    /// background hang monitor notifications.
    ///
    /// # Arguments
    /// * `category` - The `ScriptThreadEventCategory` of the event.
    /// * `pipeline_id` - Optional `PipelineId` associated with the event.
    /// * `f` - A closure containing the code to be profiled.
    ///
    /// Pre-condition: `background_hang_monitor` is initialized.
    /// Post-condition: The closure `f` is executed, its duration is measured, and profiling
    /// data is potentially sent to the time profiler. If `profile_script_events` is `true`,
    /// the event is profiled based on its category.
    fn profile_event<F, R>(
        &self,
        category: ScriptThreadEventCategory,
        pipeline_id: Option<PipelineId>,
        f: F,
    ) -> R
    where
        F: FnOnce() -> R,
    {
        self.background_hang_monitor
            .notify_activity(HangAnnotation::Script(category.into()));
        let start = Instant::now();
        let value = if self.profile_script_events {
            let profiler_chan = self.senders.time_profiler_sender.clone();
            match category {
                ScriptThreadEventCategory::AttachLayout => {
                    time_profile!(ProfilerCategory::ScriptAttachLayout, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::ConstellationMsg => time_profile!(
                    ProfilerCategory::ScriptConstellationMsg,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::DevtoolsMsg => {
                    time_profile!(ProfilerCategory::ScriptDevtoolsMsg, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::DocumentEvent => time_profile!(
                    ProfilerCategory::ScriptDocumentEvent,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::InputEvent => {
                    time_profile!(ProfilerCategory::ScriptInputEvent, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::FileRead => {
                    time_profile!(ProfilerCategory::ScriptFileRead, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::FontLoading => {
                    time_profile!(ProfilerCategory::ScriptFontLoading, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::FormPlannedNavigation => time_profile!(
                    ProfilerCategory::ScriptPlannedNavigation,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::HistoryEvent => {
                    time_profile!(ProfilerCategory::ScriptHistoryEvent, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::ImageCacheMsg => time_profile!(
                    ProfilerCategory::ScriptImageCacheMsg,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::NetworkEvent => {
                    time_profile!(ProfilerCategory::ScriptNetworkEvent, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::PortMessage => {
                    time_profile!(ProfilerCategory::ScriptPortMessage, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::Resize => {
                    time_profile!(ProfilerCategory::ScriptResize, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::ScriptEvent => {
                    time_profile!(ProfilerCategory::ScriptEvent, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::SetScrollState => time_profile!(
                    ProfilerCategory::ScriptSetScrollState,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::UpdateReplacedElement => time_profile!(
                    ProfilerCategory::ScriptUpdateReplacedElement,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::StylesheetLoad => time_profile!(
                    ProfilerCategory::ScriptStylesheetLoad,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::SetViewport => {
                    time_profile!(ProfilerCategory::ScriptSetViewport, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::TimerEvent => {
                    time_profile!(ProfilerCategory::ScriptTimerEvent, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::WebSocketEvent => time_profile!(
                    ProfilerCategory::ScriptWebSocketEvent,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::WorkerEvent => {
                    time_profile!(ProfilerCategory::ScriptWorkerEvent, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::WorkletEvent => {
                    time_profile!(ProfilerCategory::ScriptWorkletEvent, None, profiler_chan, f)
                },
                ScriptThreadEventCategory::ServiceWorkerEvent => time_profile!(
                    ProfilerCategory::ScriptServiceWorkerEvent,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::EnterFullscreen => time_profile!(
                    ProfilerCategory::ScriptEnterFullscreen,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::ExitFullscreen => time_profile!(
                    ProfilerCategory::ScriptExitFullscreen,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::PerformanceTimelineTask => time_profile!(
                    ProfilerCategory::ScriptPerformanceEvent,
                    None,
                    profiler_chan,
                    f
                ),
                ScriptThreadEventCategory::Rendering => {
                    time_profile!(ProfilerCategory::ScriptRendering, None, profiler_chan, f)
                },
                #[cfg(feature = "webgpu")]
                ScriptThreadEventCategory::WebGPUMsg => {
                    time_profile!(ProfilerCategory::ScriptWebGPUMsg, None, profiler_chan, f)
                },
            }
        } else {
            f()
        };
        let task_duration = start.elapsed();
        for (doc_id, doc) in self.documents.borrow().iter() {
            if let Some(pipeline_id) = pipeline_id {
                if pipeline_id == doc_id && task_duration.as_nanos() > MAX_TASK_NS.into() {
                    if self.print_pwm {
                        println!(
                            "Task took longer than max allowed ({:?}) {:?}",
                            category,
                            task_duration.as_nanos()
                        );
                    }
                    doc.start_tti();
                }
            }
            doc.record_tti_if_necessary();
        }
        value
    }

    /// Handles an incoming `ScriptThreadMessage` from the Constellation.
    ///
    /// # Arguments
    /// * `msg` - The `ScriptThreadMessage` to handle.
    /// * `can_gc` - A `CanGc` token.
    ///
    /// Pre-condition: The script thread is running.
    /// Post-condition: The message is processed, leading to state changes or actions within the script thread.
    fn handle_msg_from_constellation(&self, msg: ScriptThreadMessage, can_gc: CanGc) {
        match msg {
            ScriptThreadMessage::StopDelayingLoadEventsMode(pipeline_id) => {
                self.handle_stop_delaying_load_events_mode(pipeline_id)
            },
            ScriptThreadMessage::NavigateIframe(
                parent_pipeline_id,
                browsing_context_id,
                load_data,
                history_handling,
            ) => self.handle_navigate_iframe(
                parent_pipeline_id,
                browsing_context_id,
                load_data,
                history_handling,
                can_gc,
            ),
            ScriptThreadMessage::UnloadDocument(pipeline_id) => {
                self.handle_unload_document(pipeline_id, can_gc)
            },
            ScriptThreadMessage::ResizeInactive(id, new_size) => {
                self.handle_resize_inactive_msg(id, new_size)
            },
            ScriptThreadMessage::ThemeChange(_, theme) => {
                self.handle_theme_change_msg(theme);
            },
            ScriptThreadMessage::GetTitle(pipeline_id) => self.handle_get_title_msg(pipeline_id),
            ScriptThreadMessage::SetDocumentActivity(pipeline_id, activity) => {
                self.handle_set_document_activity_msg(pipeline_id, activity)
            },
            ScriptThreadMessage::SetThrottled(pipeline_id, throttled) => {
                self.handle_set_throttled_msg(pipeline_id, throttled)
            },
            ScriptThreadMessage::SetThrottledInContainingIframe(
                parent_pipeline_id,
                browsing_context_id,
                throttled,
            ) => self.handle_set_throttled_in_containing_iframe_msg(
                parent_pipeline_id,
                browsing_context_id,
                throttled,
            ),
            ScriptThreadMessage::PostMessage {
                target: target_pipeline_id,
                source: source_pipeline_id,
                source_browsing_context,
                target_origin: origin,
                source_origin,
                data,
            } => self.handle_post_message_msg(
                target_pipeline_id,
                source_pipeline_id,
                source_browsing_context,
                origin,
                source_origin,
                data,
            ),
            ScriptThreadMessage::UpdatePipelineId(
                parent_pipeline_id,
                browsing_context_id,
                top_level_browsing_context_id,
                new_pipeline_id,
                reason,
            ) => self.handle_update_pipeline_id(
                parent_pipeline_id,
                browsing_context_id,
                top_level_browsing_context_id,
                new_pipeline_id,
                reason,
                can_gc,
            ),
            ScriptThreadMessage::UpdateHistoryState(pipeline_id, history_state_id, url) => {
                self.handle_update_history_state_msg(pipeline_id, history_state_id, url, can_gc)
            },
            ScriptThreadMessage::RemoveHistoryStates(pipeline_id, history_states) => {
                self.handle_remove_history_states(pipeline_id, history_states)
            },
            ScriptThreadMessage::FocusIFrame(parent_pipeline_id, frame_id) => {
                self.handle_focus_iframe_msg(parent_pipeline_id, frame_id, can_gc)
            },
            ScriptThreadMessage::WebDriverScriptCommand(pipeline_id, msg) => {
                self.handle_webdriver_msg(pipeline_id, msg, can_gc)
            },
            ScriptThreadMessage::WebFontLoaded(pipeline_id, success) => {
                self.handle_web_font_loaded(pipeline_id, success)
            },
            ScriptThreadMessage::DispatchIFrameLoadEvent {
                target: browsing_context_id,
                parent: parent_id,
                child: child_id,
            } => self.handle_iframe_load_event(parent_id, browsing_context_id, child_id, can_gc),
            ScriptThreadMessage::DispatchStorageEvent(
                pipeline_id,
                storage,
                url,
                key,
                old_value,
                new_value,
            ) => self.handle_storage_event(pipeline_id, storage, url, key, old_value, new_value),
            ScriptThreadMessage::ReportCSSError(pipeline_id, filename, line, column, msg) => {
                self.handle_css_error_reporting(pipeline_id, filename, line, column, msg)
            },
            ScriptThreadMessage::Reload(pipeline_id) => self.handle_reload(pipeline_id, can_gc),
            ScriptThreadMessage::ExitPipeline(pipeline_id, discard_browsing_context) => {
                self.handle_exit_pipeline_msg(pipeline_id, discard_browsing_context, can_gc)
            },
            ScriptThreadMessage::PaintMetric(pipeline_id, metric_type, metric_value) => {
                self.handle_paint_metric(pipeline_id, metric_type, metric_value, can_gc)
            },
            ScriptThreadMessage::MediaSessionAction(pipeline_id, action) => {
                self.handle_media_session_action(pipeline_id, action, can_gc)
            },
            #[cfg(feature = "webgpu")]
            ScriptThreadMessage::SetWebGPUPort(port) => {
                *self.receivers.webgpu_receiver.borrow_mut() =
                    ROUTER.route_ipc_receiver_to_new_crossbeam_receiver(port);
            },
            msg @ ScriptThreadMessage::AttachLayout(..) |
            msg @ ScriptThreadMessage::Viewport(..) |
            msg @ ScriptThreadMessage::Resize(..) |
            msg @ ScriptThreadMessage::ExitFullScreen(..) |
            msg @ ScriptThreadMessage::SendInputEvent(..) |
            msg @ ScriptThreadMessage::TickAllAnimations(..) |
            msg @ ScriptThreadMessage::ExitScriptThread => {
                panic!("should have handled {:?} already", msg)
            },
            ScriptThreadMessage::SetScrollStates(pipeline_id, scroll_states) => {
                self.handle_set_scroll_states_msg(pipeline_id, scroll_states)
            },
            ScriptThreadMessage::SetEpochPaintTime(pipeline_id, epoch, time) => {
                self.handle_set_epoch_paint_time(pipeline_id, epoch, time)
            },
        }
    }

    /// Handles `ScriptThreadMessage::SetScrollStates`, updating the scroll state of a window.
    ///
    /// # Arguments
    /// * `pipeline_id` - The `PipelineId` of the window to update.
    /// * `scroll_states` - A vector of `ScrollState` objects to apply.
    ///
    /// Pre-condition: `pipeline_id` corresponds to an active window.
    /// Post-condition: The window's layout scroll states are updated, and viewport/node scroll offsets are adjusted.
    fn handle_set_scroll_states_msg(
        &self,
        pipeline_id: PipelineId,
        scroll_states: Vec<ScrollState>,
    ) {
        let Some(window) = self.documents.borrow().find_window(pipeline_id) else {
            warn!("Received scroll states for closed pipeline {pipeline_id}");
            return;
        };

        self.profile_event(
            ScriptThreadEventCategory::SetScrollState,
            Some(pipeline_id),
            || {
                window.layout_mut().set_scroll_states(&scroll_states);

                let mut scroll_offsets = HashMap::new();
                for scroll_state in scroll_states.into_iter() {
                    let scroll_offset = scroll_state.scroll_offset;
                    if scroll_state.scroll_id.is_root() {
                        window.update_viewport_for_scroll(-scroll_offset.x, -scroll_offset.y);
                    } else if let Some(node_id) =
                        node_id_from_scroll_id(scroll_state.scroll_id.0 as usize)
                    {
                        scroll_offsets.insert(OpaqueNode(node_id), -scroll_offset);
                    }
                }
                window.set_scroll_offsets(scroll_offsets)
            },
        )
    }

    /// Handles `ScriptThreadMessage::SetEpochPaintTime`, updating the paint time for a given epoch.
    ///
    /// # Arguments
    /// * `pipeline_id` - The `PipelineId` of the window.
    /// * `epoch` - The `Epoch` for which to set the paint time.
    /// * `time` - The `CrossProcessInstant` representing the paint time.
    ///
    /// Pre-condition: `pipeline_id` corresponds to an active window.
    /// Post-condition: The layout's epoch paint time is updated.
    fn handle_set_epoch_paint_time(
        &self,
        pipeline_id: PipelineId,
        epoch: Epoch,
        time: CrossProcessInstant,
    ) {
        let Some(window) = self.documents.borrow().find_window(pipeline_id) else {
            warn!("Received set epoch paint time message for closed pipeline {pipeline_id}.");
            return;
        };
        window.layout_mut().set_epoch_paint_time(epoch, time);
    }

    #[cfg(feature = "webgpu")]
    fn handle_msg_from_webgpu_server(&self, msg: WebGPUMsg, can_gc: CanGc) {
        match msg {
            _ => {},
        }
    }
}
