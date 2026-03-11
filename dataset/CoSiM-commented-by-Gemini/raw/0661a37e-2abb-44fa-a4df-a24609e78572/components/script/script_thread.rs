
/**
 * @file script_thread.rs
 * @brief Core of the browser's script thread, responsible for managing the DOM, running JavaScript, and handling page loads.
 *
 * This module defines the `ScriptThread` struct and its associated logic, which represents the main
 * event loop for a set of same-origin browsing contexts. It is the central component that orchestrates
 * the entire lifecycle of a web page, from the initial network request to the final teardown.
 *
 * ## Core Responsibilities:
 *
 * - **DOM Management:** The script thread owns the Document Object Model (DOM) for all the pages it
 *   manages. All DOM mutations and accesses must happen on this thread.
 * - **JavaScript Execution:** It hosts the JavaScript runtime (SpiderMonkey) and is responsible for
 *   executing all JavaScript code, including user scripts, event handlers, and timers.
 * - **Event Loop:** It runs a message-based event loop, processing events from various sources like
 *   user input (forwarded from the compositor), network responses, and timers.
 * - **Page Lifecycle:** It manages the entire page load process. This includes initiating network
 *   requests for new pages, parsing HTML/CSS, constructing the DOM tree, and running the document
 *   lifecycle events (e.g., `DOMContentLoaded`, `load`).
 * - **Layout and Rendering:** It triggers layout (reflow) operations when the DOM changes and
 *   coordinates with the layout and compositor threads to update the rendering of the page.
 * - **Communication:** It communicates with other threads and processes, such as the constellation
 *   (the main browser process), the compositor, and network threads, through a system of channels.
 *
 * ## Architectural Overview:
 *
 * The `ScriptThread` is designed to be single-threaded to avoid the complexities of concurrent DOM
 * access. It interacts with other parts of the browser engine asynchronously. For example, network
 * requests are non-blocking; the script thread initiates a request and then continues processing
 * other events. When the response arrives, a message is sent back to the script thread's event
 * loop for processing.
 *
 * The main struct, `ScriptThread`, holds the state for all the documents it manages, including
 * the DOM trees, window proxies, and ongoing loads. The event loop is implemented in the
 * `handle_msgs` method, which continuously receives and dispatches messages from various channels.
 *
 * ## Key Data Structures:
 *
 * - `ScriptThread`: The main struct representing the script thread and its state.
 * - `DocumentCollection`: A collection of all `Document` objects managed by this thread.
 * - `InProgressLoad`: Represents a page load that is currently in progress.
 * - `TaskQueue`: A queue for tasks to be executed on the script thread.
 * - `MicrotaskQueue`: Manages the JavaScript microtask queue (e.g., for Promises).
 *
 * @see https://html.spec.whatwg.org/multipage/webappapis.html#event-loops
 */
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

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::option::Option;
use std::rc::Rc;
use std::result::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use background_hang_monitor_api::{
    BackgroundHangMonitor, BackgroundHangMonitorExitSignal, HangAnnotation, MonitoredComponentId,
    MonitoredComponentType,
};
use base::cross_process_instant::CrossProcessInstant;
use base::id::{BrowsingContextId, HistoryStateId, PipelineId, PipelineNamespace, WebViewId};
use canvas_traits::webgl::WebGLPipeline;
use chrono::{DateTime, Local};
use compositing_traits::CrossProcessCompositorApi;
use constellation_traits::{
    JsEvalResult, LoadData, LoadOrigin, NavigationHistoryBehavior, ScriptToConstellationChan,
    ScriptToConstellationMessage, ScrollState, StructuredSerializedData, WindowSizeType,
};
use crossbeam_channel::unbounded;
use devtools_traits::{
    CSSError, DevtoolScriptControlMsg, DevtoolsPageInfo, NavigationState,
    ScriptToDevtoolsControlMsg, WorkerId,
};
use embedder_traits::user_content_manager::UserContentManager;
use embedder_traits::{
    CompositorHitTestResult, EmbedderMsg, InputEvent, MediaSessionActionType, Theme,
    ViewportDetails, WebDriverScriptCommand,
};
use euclid::default::Rect;
use fonts::{FontContext, SystemFontServiceProxy};
use headers::{HeaderMapExt, LastModified, ReferrerPolicy as ReferrerPolicyHeader};
use html5ever::{local_name, namespace_url, ns};
use http::header::REFRESH;
use hyper_serde::Serde;
use ipc_channel::ipc;
use ipc_channel::router::ROUTER;
use js::glue::GetWindowProxyClass;
use js::jsapi::{
    JS_AddInterruptCallback, JSContext as UnsafeJSContext, JSTracer, SetWindowProxyClass,
};
use js::jsval::UndefinedValue;
use js::rust::ParentRuntime;
use media::WindowGLContext;
use metrics::MAX_TASK_NS;
use mime::{self, Mime};
use net_traits::image_cache::{ImageCache, PendingImageResponse};
use net_traits::request::{Referrer, RequestId};
use net_traits::response::ResponseInit;
use net_traits::storage_thread::StorageType;
use net_traits::{
    FetchMetadata, FetchResponseListener, FetchResponseMsg, Metadata, NetworkError,
    ResourceFetchTiming, ResourceThreads, ResourceTimingType,
};
use percent_encoding::percent_decode;
use profile_traits::mem::{ProcessReports, ReportsChan};
use profile_traits::time::ProfilerCategory;
use profile_traits::time_profile;
use script_layout_interface::{
    LayoutConfig, LayoutFactory, ReflowGoal, ScriptThreadFactory, node_id_from_scroll_id,
};
use script_traits::{
    ConstellationInputEvent, DiscardBrowsingContext, DocumentActivity, InitialScriptState,
    NewLayoutInfo, Painter, ProgressiveWebMetricType, ScriptThreadMessage, UpdatePipelineIdReason,
};
use servo_config::opts;
use servo_url::{ImmutableOrigin, MutableOrigin, ServoUrl};
use style::dom::OpaqueNode;
use style::thread_state::{self, ThreadState};
use stylo_atoms::Atom;
use timers::{TimerEventRequest, TimerScheduler};
use url::Position;
#[cfg(feature = "webgpu")]
use webgpu_traits::{WebGPUDevice, WebGPUMsg};
use webrender_api::DocumentId;

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
    CanGc, JSContext, JSContextHelper, Runtime, ScriptThreadEventCategory, ThreadSafeJSContext,
};
use crate::task_queue::TaskQueue;
use crate::task_source::{SendableTaskSource, TaskSourceName};
use crate::{devtools, webdriver_handlers};

/// `thread_local!` macro storing a raw pointer to the `ScriptThread` instance.
/// This allows other modules to get access to the current script thread without passing it around.
thread_local!(static SCRIPT_THREAD_ROOT: Cell<Option<*const ScriptThread>> = const { Cell::new(None) });

/// Provides safe access to the current `ScriptThread` instance if it exists.
fn with_optional_script_thread<R>(f: impl FnOnce(Option<&ScriptThread>) -> R) -> R {
    SCRIPT_THREAD_ROOT.with(|root| {
        f(root
            .get()
            .and_then(|script_thread| unsafe { script_thread.as_ref() }))
    })
}

/**
 * Executes a closure with a reference to the current `ScriptThread`.
 *
 * This function provides a safe way to access the thread-local `ScriptThread` instance.
 * It panics if the script thread is not available.
 *
 * @param f A closure that takes a `&ScriptThread` and returns a value.
 * @return The value returned by the closure, or a default value if the script thread is not set.
 */
pub(crate) fn with_script_thread<R: Default>(f: impl FnOnce(&ScriptThread) -> R) -> R {
    with_optional_script_thread(|script_thread| script_thread.map(f).unwrap_or_default())
}

/**
 * Traces all JSTraceable objects within the script thread for garbage collection.
 *
 * This function is called by the JavaScript engine's garbage collector. It is crucial
 * for preventing memory leaks by ensuring that all reachable JavaScript objects are
 * marked as live.
 *
 * # Safety
 *
 * The caller must ensure that the `JSTracer` pointer is valid. The implementation
 * must trace all `JSTraceable` fields within the `ScriptThread`.
 *
 * @param tr A raw pointer to the `JSTracer` used by the garbage collector.
 */
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
pub(crate) struct IncompleteParserContexts(RefCell<Vec<(PipelineId, ParserContext)>>);

unsafe_no_jsmanaged_fields!(TaskQueue<MainThreadScriptMsg>);

/**
 * The `ScriptThread` is the heart of the browser's content process. It is responsible for
 * executing JavaScript, managing the DOM, and handling events for a set of related browsing
 * contexts.
 *
 * This struct encapsulates all the state necessary for a script thread to function, including
 * communication channels, the JavaScript runtime, and the collection of documents it manages.
 *
 * @see https://html.spec.whatwg.org/multipage/webappapis.html#event-loops
 */
#[derive(JSTraceable)]
// ScriptThread instances are rooted on creation, so this is okay
#[cfg_attr(crown, allow(crown::unrooted_must_root))]
pub struct ScriptThread {
    /**
     * @brief The time of the last rendering opportunity.
     * @see https://html.spec.whatwg.org/multipage/#last-render-opportunity-time
     */
    last_render_opportunity_time: DomRefCell<Option<Instant>>,
    /**
     * @brief Collection of all documents managed by this script thread.
     *
     * This is the central repository for all `Document` objects within the script thread's
     * purview.
     */
    documents: DomRefCell<DocumentCollection>,
    /**
     * @brief A map of browsing context IDs to their corresponding `WindowProxy` objects.
     *
     * This allows the script thread to manage and interact with different windows and frames.
     */
    window_proxies: DomRefCell<HashMapTracedValues<BrowsingContextId, Dom<WindowProxy>>>,
    /**
     * @brief A list of ongoing page loads that have not yet received a network response.
     *
     * This is used to track the state of pending navigations.
     */
    incomplete_loads: DomRefCell<Vec<InProgressLoad>>,
    /**
     * @brief A vector containing parser contexts which have not yet been fully processed.
     *
     * This is used to manage the parsing of incoming HTML or XML data.
     */
    incomplete_parser_contexts: IncompleteParserContexts,
    /**
     * @brief A shared handle to the image cache.
     *
     * This allows the script thread to request and manage cached images.
     */
    #[no_trace]
    image_cache: Arc<dyn ImageCache>,

    /**
     * @brief Holds all the incoming `Receiver`s for messages to this `ScriptThread`.
     *
     * This is the entry point for all asynchronous communication to the script thread.
     */
    receivers: ScriptThreadReceivers,

    /**
     * @brief Holds all outgoing sending channels necessary to communicate to other parts of Servo.
     *
     * This is the exit point for all asynchronous communication from the script thread.
     */
    senders: ScriptThreadSenders,

    /**
     * @brief A handle to the resource loading threads.
     *
     * This is used to initiate network requests for resources like images, scripts, and stylesheets.
     */
    #[no_trace]
    resource_threads: ResourceThreads,

    /**
     * @brief The main task queue for the script thread's event loop.
     *
     * This queue holds tasks that need to be executed on the script thread, such as event handlers
     * and timer callbacks.
     */
    task_queue: TaskQueue<MainThreadScriptMsg>,

    /**
     * @brief A handle to the background hang monitor.
     *
     * This is used to detect and report hangs in the script thread.
     */
    #[no_trace]
    background_hang_monitor: Box<dyn BackgroundHangMonitor>,
    /**
     * @brief A flag that is set to `true` when the script thread is requested to shut down.
     *
     * This is used to gracefully terminate the event loop.
     */
    closing: Arc<AtomicBool>,

    /**
     * @brief The timer scheduler for this script thread.
     *
     * This is used to manage `setTimeout` and `setInterval` timers.
     */
    #[no_trace]
    timer_scheduler: RefCell<TimerScheduler>,

    /**
     * @brief A proxy to the system font service.
     *
     * This is used to access system fonts for text rendering.
     */
    #[no_trace]
    system_font_service: Arc<SystemFontServiceProxy>,

    /**
     * @brief The JavaScript runtime for this script thread.
     *
     * This holds the SpiderMonkey engine instance.
     */
    js_runtime: Rc<Runtime>,

    /**
     * @brief The topmost element that the mouse is currently over.
     *
     * This is used for hit testing and dispatching mouse events.
     */
    topmost_mouse_over_target: MutNullableDom<Element>,

    /**
     * @brief A set of pipeline IDs that have been closed.
     *
     * This is used to track the lifecycle of pipelines and prevent use-after-free errors.
     */
    #[no_trace]
    closed_pipelines: DomRefCell<HashSet<PipelineId>>,

    /**
     * @brief The microtask queue for this script thread.
     * @see https://html.spec.whatwg.org/multipage/#microtask-queue
     */
    microtask_queue: Rc<MicrotaskQueue>,

    /**
     * @brief A flag indicating whether a mutation observer microtask has been queued.
     *
     * This is an optimization to avoid unnecessary checks of the mutation observer queue.
     */
    mutation_observer_microtask_queued: Cell<bool>,

    /**
     * @brief The list of `MutationObserver` objects for this script thread.
     */
    mutation_observers: DomRefCell<Vec<Dom<MutationObserver>>>,

    /**
     * @brief A list of `<slot>` elements that need to have their `signal` dispatched.
     * @see https://dom.spec.whatwg.org/#signal-slot-list
     */
    signal_slots: DomRefCell<Vec<Dom<HTMLSlotElement>>>,

    /**
     * @brief A handle to the WebGL thread.
     *
     * This is used to offload WebGL rendering from the script thread.
     */
    #[no_trace]
    webgl_chan: Option<WebGLPipeline>,

    /// The WebXR device registry
    #[no_trace]
    #[cfg(feature = "webxr")]
    webxr_registry: Option<webxr_api::Registry>,

    /**
     * @brief The thread pool for worklets.
     *
     * This is used to execute `PaintWorklet`, `AudioWorklet`, etc.
     */
    worklet_thread_pool: DomRefCell<Option<Rc<WorkletThreadPool>>>,

    /**
     * @brief A list of pipelines containing documents that finished loading all their blocking
     * resources during a turn of the event loop.
     */
    docs_with_no_blocking_loads: DomRefCell<HashSet<Dom<Document>>>,

    /**
     * @brief The stack for custom element reactions.
     * @see https://html.spec.whatwg.org/multipage/#custom-element-reactions-stack
     */
    custom_element_reaction_stack: CustomElementReactionStack,

    /**
     * @brief The Webrender document ID associated with this thread.
     *
     * This is used to identify the document in the Webrender scene.
     */
    #[no_trace]
    webrender_document: DocumentId,

    /**
     * @brief A cross-process API for interacting with the compositor.
     */
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

    /**
     * @brief Manages user-provided content, such as user stylesheets and scripts.
     */
    #[no_trace]
    user_content_manager: UserContentManager,

    /**
     * @brief The application window's GL context for media players.
     */
    #[no_trace]
    player_context: WindowGLContext,

    /**
     * @brief A set of all node IDs ever created in this script thread.
     *
     * This is used for debugging and ensuring node ID uniqueness.
     */
    node_ids: DomRefCell<HashSet<String>>,

    /**
     * @brief A flag indicating whether the code is currently running as a result of a user interaction.
     *
     * This is used to determine whether certain powerful APIs are allowed to be used.
     */
    is_user_interacting: Cell<bool>,

    /**
     * @brief The identity manager for WebGPU resources.
     *
     * This is used to track and manage WebGPU objects.
     */
    #[no_trace]
    #[cfg(feature = "webgpu")]
    gpu_id_hub: Arc<IdentityHub>,

    // Secure context
    inherited_secure_context: Option<bool>,

    /**
     * @brief A factory for creating new layout instances.
     *
     * This allows layout to depend on script, enabling things like custom layout APIs.
     */
    #[no_trace]
    layout_factory: Arc<dyn LayoutFactory>,
}
// ...
