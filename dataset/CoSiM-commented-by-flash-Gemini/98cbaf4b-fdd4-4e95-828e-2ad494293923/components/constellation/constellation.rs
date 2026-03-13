//! The `Constellation`, Servo's Grand Central Station
//!
//! The constellation tracks all information kept globally by the
//! browser engine, which includes:
//!
//! * The set of all `EventLoop` objects. Each event loop is
//!   the constellation's view of a script thread. The constellation
//!   interacts with a script thread by message-passing.
//!
//! * The set of all `Pipeline` objects. Each pipeline gives the
//!   constellation's view of a `Window`, with its script thread and
//!   layout.  Pipelines may share script threads.
//!
//! * The set of all `BrowsingContext` objects. Each browsing context
//!   gives the constellation's view of a `WindowProxy`.
//!   Each browsing context stores an independent
//!   session history, created by navigation. The session
//!   history can be traversed, for example by the back and forwards UI,
//!   so each session history maintains a list of past and future pipelines,
//!   as well as the current active pipeline.
//!
//! There are two kinds of browsing context: top-level ones (for
//! example tabs in a browser UI), and nested ones (typically caused
//! by `iframe` elements). Browsing contexts have a hierarchy
//! (typically caused by `iframe`s containing `iframe`s), giving rise
//! to a forest whose roots are top-level browsing context.  The logical
//! relationship between these types is:
//!
//! ```text
//! +------------+                      +------------+                 +---------+
//! |  Browsing  | ------parent?------> |  Pipeline  | --event_loop--> |  Event  |
//! |  Context   | ------current------> |            |                 |  Loop   |
//! |            | ------prev*--------> |            | <---pipeline*-- |         |
//! |            | ------next*--------> |            |                 +---------+
//! |            |                      |            |
//! |            | <-top_level--------- |            |
//! |            | <-browsing_context-- |            |
//! +------------+                      +------------+
//! ```
//
//! The constellation also maintains channels to threads, including:
//!
//! * The script thread.
//! * The graphics compositor.
//! * The font cache, image cache, and resource manager, which load
//!   and cache shared fonts, images, or other resources.
//! * The service worker manager.
//! * The devtools and webdriver servers.
//!
//! The constellation passes messages between the threads, and updates its state
//! to track the evolving state of the browsing context tree.
//!
//! The constellation acts as a logger, tracking any `warn!` messages from threads,
//! and converting any `error!` or `panic!` into a crash report.
//!
//! Since there is only one constellation, and its responsibilities include crash reporting,
//! it is very important that it does not panic.
//!
//! It's also important that the constellation not deadlock. In particular, we need
//! to be careful that we don't introduce any cycles in the can-block-on relation.
//! Blocking is typically introduced by `receiver.recv()`, which blocks waiting for the
//! sender to send some data. Servo tries to achieve deadlock-freedom by using the following
//! can-block-on relation:
//!
//! * Constellation can block on compositor
//! * Constellation can block on embedder
//! * Script can block on anything (other than script)
//! * Blocking is transitive (if T1 can block on T2 and T2 can block on T3 then T1 can block on T3)
//! * Nothing can block on itself!
//!
//! There is a complexity intoduced by IPC channels, since they do not support
//! non-blocking send. This means that as well as `receiver.recv()` blocking,
//! `sender.send(data)` can also block when the IPC buffer is full. For this reason it is
//! very important that all IPC receivers where we depend on non-blocking send
//! use a router to route IPC messages to an mpsc channel. The reason why that solves
//! the problem is that under the hood, the router uses a dedicated thread to forward
//! messages, and:
//!
//! * Anything (other than a routing thread) can block on a routing thread
//!
//! See <https://github.com/servo/servo/issues/14704>

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::borrow::{Cow, ToOwned};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::mem::replace;
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};
use std::{process, thread};

use background_hang_monitor::HangMonitorRegister;
use background_hang_monitor_api::{
    BackgroundHangMonitorControlMsg, BackgroundHangMonitorRegister, HangMonitorAlert,
};
use base::id::{
    BroadcastChannelRouterId, BrowsingContextGroupId, BrowsingContextId, HistoryStateId,
    MessagePortId, MessagePortRouterId, PipelineId, PipelineNamespace, PipelineNamespaceId,
    PipelineNamespaceRequest, TopLevelBrowsingContextId, WebViewId,
};
use base::Epoch;
#[cfg(feature = "bluetooth")]
use bluetooth_traits::BluetoothRequest;
use canvas_traits::canvas::{CanvasId, CanvasMsg};
use canvas_traits::webgl::WebGLThreads;
use canvas_traits::ConstellationCanvasMsg;
use compositing_traits::{
    CompositorMsg, CompositorProxy, ConstellationMsg as FromCompositorMsg, SendableFrameTree,
};
use crossbeam_channel::{select, unbounded, Receiver, Sender};
use devtools_traits::{
    ChromeToDevtoolsControlMsg, DevtoolsControlMsg, DevtoolsPageInfo, NavigationState,
    ScriptToDevtoolsControlMsg,
};
use embedder_traits::input_events::MouseButtonAction;
use embedder_traits::resources::{self, Resource};
use embedder_traits::{
    Cursor, EmbedderMsg, EmbedderProxy, ImeEvent, InputEvent, MediaSessionActionType,
    MediaSessionEvent, MediaSessionPlaybackState, MouseButton, MouseButtonEvent, Theme,
    TraversalDirection,
};
use euclid::default::Size2D as UntypedSize2D;
use euclid::Size2D;
use fonts::SystemFontServiceProxy;
use ipc_channel::ipc::{self, IpcReceiver, IpcSender};
use ipc_channel::router::ROUTER;
use ipc_channel::Error as IpcError;
use keyboard_types::webdriver::Event as WebDriverInputEvent;
use log::{debug, error, info, trace, warn};
use media::WindowGLContext;
use net_traits::pub_domains::reg_host;
use net_traits::request::Referrer;
use net_traits::storage_thread::{StorageThreadMsg, StorageType};
use net_traits::{self, IpcSend, ReferrerPolicy, ResourceThreads};
use profile_traits::{mem, time};
use script_layout_interface::{LayoutFactory, ScriptThreadFactory};
use script_traits::{
    webdriver_msg, AnimationState, AnimationTickType, AuxiliaryBrowsingContextLoadInfo,
    BroadcastMsg, ConstellationInputEvent, DiscardBrowsingContext, DocumentActivity, DocumentState,
    IFrameLoadInfo, IFrameLoadInfoWithData, IFrameSandboxState, IFrameSizeMsg, Job,
    LayoutMsg as FromLayoutMsg, LoadData, LoadOrigin, LogEntry, MessagePortMsg,
    NavigationHistoryBehavior, PortMessageTask, SWManagerMsg, SWManagerSenders,
    ScriptMsg as FromScriptMsg, ScriptThreadMessage, ScriptToConstellationChan,
    ServiceWorkerManagerFactory, ServiceWorkerMsg, StructuredSerializedData,
    UpdatePipelineIdReason, WebDriverCommandMsg, WindowSizeData, WindowSizeType,
};
use serde::{Deserialize, Serialize};
use servo_config::{opts, pref};
use servo_rand::{random, Rng, ServoRng, SliceRandom};
use servo_url::{Host, ImmutableOrigin, ServoUrl};
use style_traits::CSSPixel;
#[cfg(feature = "webgpu")]
use webgpu::swapchain::WGPUImageMap;
#[cfg(feature = "webgpu")]
use webgpu::{self, WebGPU, WebGPURequest, WebGPUResponse};
#[cfg(feature = "webgpu")]
use webrender::RenderApi;
use webrender::RenderApiSender;
use webrender_api::DocumentId;
use webrender_traits::{CompositorHitTestResult, WebrenderExternalImageRegistry};

use crate::browsingcontext::{
    AllBrowsingContextsIterator, BrowsingContext, FullyActiveBrowsingContextsIterator,
    NewBrowsingContextInfo,
};
use crate::event_loop::EventLoop;
use crate::pipeline::{InitialPipelineState, Pipeline};
use crate::serviceworker::ServiceWorkerUnprivilegedContent;
use crate::session_history::{
    JointSessionHistory, NeedsToReload, SessionHistoryChange, SessionHistoryDiff,
};
use crate::webview::WebViewManager;

/// Represents a map of pipeline IDs to their associated load data and navigation history behavior,
/// used for navigations awaiting embedder approval.
type PendingApprovalNavigations = HashMap<PipelineId, (LoadData, NavigationHistoryBehavior)>;

/// The `TransferState` enum represents the various states a `MessagePort` can be in
/// within the Constellation.
#[derive(Debug)]
enum TransferState {
    /// The port is currently managed by a global (script thread),
    /// identified by its router ID.
    Managed(MessagePortRouterId),
    /// The port is currently in-transfer to another global.
    /// Incoming tasks are buffered until the transfer is complete.
    TransferInProgress(VecDeque<PortMessageTask>),
    /// A global has requested the transfer to be completed, and the Constellation
    /// is awaiting confirmation of success or failure.
    CompletionInProgress(MessagePortRouterId),
    /// A transfer completion failed, usually because the port was shipped
    /// while completion was in progress. Incoming messages are buffered,
    /// and the Constellation awaits the return of the previous buffer.
    CompletionFailed(VecDeque<PortMessageTask>),
    /// A transfer completion failed, and another global has requested a transfer.
    /// Messages are still buffered, awaiting the return of the buffer from the
    /// global that failed to complete the transfer.
    CompletionRequested(MessagePortRouterId, VecDeque<PortMessageTask>),
    /// The entangled port (the other end of the message channel) has been removed.
    /// This port should also be removed once it becomes managed again.
    EntangledRemoved,
}

/// `MessagePortInfo` stores information about a message port tracked by the Constellation.
struct MessagePortInfo {
    /// The current operational state of the message port.
    state: TransferState,

    /// The ID of the entangled `MessagePort`, if one exists, forming a two-way communication channel.
    entangled_with: Option<MessagePortId>,
}

#[cfg(feature = "webgpu")]
/// `WebrenderWGPU` encapsulates WebRender-related objects specifically required by WebGPU threads.
struct WebrenderWGPU {
    /// The WebRender API instance, used for sending rendering commands.
    webrender_api: RenderApi,

    /// A shared, thread-safe registry for WebRender external images,
    /// allowing WebGPU to integrate its rendered textures.
    webrender_external_images: Arc<Mutex<WebrenderExternalImageRegistry>>,

    /// A map used to store WebGPU textures that are to be rendered by WebRender.
    wgpu_image_map: WGPUImageMap,
}

/// `WebView` stores bookkeeping data for a single top-level browsing context (or webview)
/// managed by the Constellation.
struct WebView {
    /// The `BrowsingContextId` of the currently focused browsing context within this webview,
    /// specifically for handling key events.
    focused_browsing_context_id: BrowsingContextId,

    /// The `JointSessionHistory` for this webview, managing navigation history.
    session_history: JointSessionHistory,
}

/// `BrowsingContextGroup` represents a group of browsing contexts as defined by the HTML specification.
///
/// <https://html.spec.whatwg.org/multipage/#browsing-context-group>
#[derive(Clone, Default)]
struct BrowsingContextGroup {
    /// A set of `TopLevelBrowsingContextId`s that belong to this group.
    top_level_browsing_context_set: HashSet<TopLevelBrowsingContextId>,

    /// A map of `Host` to `EventLoop` references. Event loops are shared between scripts
    /// with the same eTLD+1 within the same browsing context group to support `document.domain`
    /// and shared DOM objects.
    event_loops: HashMap<Host, Weak<EventLoop>>,

    /// A map of `Host` to `WebGPU` instances within this browsing context group,
    /// enabled when the "webgpu" feature is active.
    #[cfg(feature = "webgpu")]
    webgpus: HashMap<Host, WebGPU>,
}

/// The `Constellation` itself. In the servo browser, there is one
/// constellation, which maintains all of the browser global data.
/// In embedded applications, there may be more than one constellation,
/// which are independent of each other.
///
/// The constellation may be in a different process from the pipelines,
/// and communicates using IPC.
///
/// It is parameterized over a `LayoutThreadFactory` and a
/// `ScriptThreadFactory` (which in practice are implemented by
/// `LayoutThread` in the `layout` crate, and `ScriptThread` in
/// the `script` crate). Script and layout communicate using a `Message`
/// type.
pub struct Constellation<STF, SWF> {
    /// An ipc-sender/threaded-receiver pair
    /// to facilitate installing pipeline namespaces in threads
    /// via a per-process installer.
    namespace_receiver: Receiver<Result<PipelineNamespaceRequest, IpcError>>,
    namespace_ipc_sender: IpcSender<PipelineNamespaceRequest>,

    /// An IPC channel for script threads to send messages to the constellation.
    /// This is the script threads' view of `script_receiver`.
    script_sender: IpcSender<(PipelineId, FromScriptMsg)>,

    /// A channel for the constellation to receive messages from script threads.
    /// This is the constellation's view of `script_sender`.
    script_receiver: Receiver<Result<(PipelineId, FromScriptMsg), IpcError>>,

    /// A handle to register components for hang monitoring.
    /// `None` when in multiprocess mode, as the hang monitor is then per-process.
    background_monitor_register: Option<Box<dyn BackgroundHangMonitorRegister>>,

    /// Channels to control all background-hang monitors.
    /// TODO: store them on the relevant BrowsingContextGroup,
    /// so that they could be controlled on a "per-tab/event-loop" basis.
    background_monitor_control_senders: Vec<IpcSender<BackgroundHangMonitorControlMsg>>,

    /// A channel for the background hang monitor to send messages
    /// to the constellation.
    background_hang_monitor_sender: IpcSender<HangMonitorAlert>,

    /// A channel for the constellation to receiver messages
    /// from the background hang monitor.
    background_hang_monitor_receiver: Receiver<Result<HangMonitorAlert, IpcError>>,

    /// A factory for creating layouts. This allows customizing the kind
    /// of layout created for a [`Constellation`] and prevents a circular crate
    /// dependency between script and layout.
    layout_factory: Arc<dyn LayoutFactory>,

    /// An IPC channel for layout to send messages to the constellation.
    /// This is the layout's view of `layout_receiver`.
    layout_sender: IpcSender<FromLayoutMsg>,

    /// A channel for the constellation to receive messages from layout.
    /// This is the constellation's view of `layout_sender`.
    layout_receiver: Receiver<Result<FromLayoutMsg, IpcError>>,

    /// A channel for the constellation to receive messages from the compositor thread.
    compositor_receiver: Receiver<FromCompositorMsg>,

    /// A channel through which messages can be sent to the embedder.
    embedder_proxy: EmbedderProxy,

    /// A channel (the implementation of which is port-specific) for the
    /// constellation to send messages to the compositor thread.
    compositor_proxy: CompositorProxy,

    /// Bookkeeping data for all webviews in the constellation.
    webviews: WebViewManager<WebView>,

    /// Channels for the constellation to send messages to the public
    /// resource-related threads. There are two groups of resource threads: one
    /// for public browsing, and one for private browsing.
    public_resource_threads: ResourceThreads,

    /// Channels for the constellation to send messages to the private
    /// resource-related threads.  There are two groups of resource
    /// threads: one for public browsing, and one for private
    /// browsing.
    private_resource_threads: ResourceThreads,

    /// A channel for the constellation to send messages to the font
    /// cache thread.
    system_font_service: Arc<SystemFontServiceProxy>,

    /// A channel for the constellation to send messages to the
    /// devtools thread.
    devtools_sender: Option<Sender<DevtoolsControlMsg>>,

    /// An IPC channel for the constellation to send messages to the
    /// bluetooth thread.
    #[cfg(feature = "bluetooth")]
    bluetooth_ipc_sender: IpcSender<BluetoothRequest>,

    /// A map of origin to sender to a Service worker manager.
    sw_managers: HashMap<ImmutableOrigin, IpcSender<ServiceWorkerMsg>>,

    /// An IPC channel for Service Worker Manager threads to send
    /// messages to the constellation.  This is the SW Manager thread's
    /// view of `swmanager_receiver`.
    swmanager_ipc_sender: IpcSender<SWManagerMsg>,

    /// A channel for the constellation to receive messages from the
    /// Service Worker Manager thread. This is the constellation's view of
    /// `swmanager_sender`.
    swmanager_receiver: Receiver<Result<SWManagerMsg, IpcError>>,

    /// A channel for the constellation to send messages to the
    /// time profiler thread.
    time_profiler_chan: time::ProfilerChan,

    /// A channel for the constellation to send messages to the
    /// memory profiler thread.
    mem_profiler_chan: mem::ProfilerChan,

    /// A single WebRender document the constellation operates on.
    webrender_document: DocumentId,

    /// Webrender related objects required by WebGPU threads
    #[cfg(feature = "webgpu")]
    webrender_wgpu: WebrenderWGPU,

    /// A map of message-port Id to info.
    message_ports: HashMap<MessagePortId, MessagePortInfo>,

    /// A map of router-id to ipc-sender, to route messages to ports.
    message_port_routers: HashMap<MessagePortRouterId, IpcSender<MessagePortMsg>>,

    /// A map of broadcast routers to their IPC sender.
    broadcast_routers: HashMap<BroadcastChannelRouterId, IpcSender<BroadcastMsg>>,

    /// A map of origin to a map of channel-name to a list of relevant routers.
    broadcast_channels: HashMap<ImmutableOrigin, HashMap<String, Vec<BroadcastChannelRouterId>>>,

    /// The set of all the pipelines in the browser.  (See the `pipeline` module
    /// for more details.)
    pipelines: HashMap<PipelineId, Pipeline>,

    /// The set of all the browsing contexts in the browser.
    browsing_contexts: HashMap<BrowsingContextId, BrowsingContext>,

    /// A user agent holds a a set of browsing context groups.
    ///
    /// <https://html.spec.whatwg.org/multipage/#browsing-context-group-set>
    browsing_context_group_set: HashMap<BrowsingContextGroupId, BrowsingContextGroup>,

    /// The Id counter for BrowsingContextGroup.
    browsing_context_group_next_id: u32,

    /// When a navigation is performed, we do not immediately update
    /// the session history, instead we ask the event loop to begin loading
    /// the new document, and do not update the browsing context until the
    /// document is active. Between starting the load and it activating,
    /// we store a `SessionHistoryChange` object for the navigation in progress.
    pending_changes: Vec<SessionHistoryChange>,

    /// Pipeline IDs are namespaced in order to avoid name collisions,
    /// and the namespaces are allocated by the constellation.
    next_pipeline_namespace_id: PipelineNamespaceId,

    /// The size of the top-level window.
    window_size: WindowSizeData,

    /// Bits of state used to interact with the webdriver implementation
    webdriver: WebDriverData,

    /// Document states for loaded pipelines (used only when writing screenshots).
    document_states: HashMap<PipelineId, DocumentState>,

    /// Are we shutting down?
    shutting_down: bool,

    /// Have we seen any warnings? Hopefully always empty!
    /// The buffer contains `(thread_name, reason)` entries.
    handled_warnings: VecDeque<(Option<String>, String)>,

    /// The random number generator and probability for closing pipelines.
    /// This is for testing the hardening of the constellation.
    random_pipeline_closure: Option<(ServoRng, f32)>,

    /// Phantom data that keeps the Rust type system happy.
    phantom: PhantomData<(STF, SWF)>,

    /// Entry point to create and get channels to a WebGLThread.
    webgl_threads: Option<WebGLThreads>,

    /// The XR device registry
    webxr_registry: Option<webxr_api::Registry>,

    /// A channel through which messages can be sent to the canvas paint thread.
    canvas_sender: Sender<ConstellationCanvasMsg>,

    /// An IPC sender for canvas-related messages.
    canvas_ipc_sender: IpcSender<CanvasMsg>,

    /// Navigation requests from script awaiting approval from the embedder.
    pending_approval_navigations: PendingApprovalNavigations,

    /// Bitmask which indicates which combination of mouse buttons are
    /// currently being pressed.
    pressed_mouse_buttons: u16,

    /// If True, exits on thread failure instead of displaying about:failure
    hard_fail: bool,

    /// Pipeline ID of the active media session.
    active_media_session: Option<PipelineId>,

    /// User agent string to report in network requests.
    user_agent: Cow<'static, str>,

    /// The image bytes associated with the RippyPNG embedder resource.
    /// Read during startup and provided to image caches that are created
    /// on an as-needed basis, rather than retrieving it every time.
    rippy_data: Vec<u8>,
}

/// `InitialConstellationState` holds all the initial state and channels
/// required to construct a `Constellation` instance.
pub struct InitialConstellationState {
    /// A channel through which messages can be sent to the embedder.
    pub embedder_proxy: EmbedderProxy,

    /// A channel through which messages can be sent to the compositor in-process.
    pub compositor_proxy: CompositorProxy,

    /// A channel to the developer tools, if applicable.
    pub devtools_sender: Option<Sender<DevtoolsControlMsg>>,

    /// A channel to the bluetooth thread.
    #[cfg(feature = "bluetooth")]
    pub bluetooth_thread: IpcSender<BluetoothRequest>,

    /// A proxy to the `SystemFontService` which manages the list of system fonts.
    pub system_font_service: Arc<SystemFontServiceProxy>,

    /// A channel to the resource thread for public browsing.
    pub public_resource_threads: ResourceThreads,

    /// A channel to the resource thread for private browsing.
    pub private_resource_threads: ResourceThreads,

    /// A channel to the time profiler thread.
    pub time_profiler_chan: time::ProfilerChan,

    /// A channel to the memory profiler thread.
    pub mem_profiler_chan: mem::ProfilerChan,

    /// Webrender document ID.
    pub webrender_document: DocumentId,

    /// Webrender API sender.
    pub webrender_api_sender: RenderApiSender,

    /// Webrender external images registry.
    pub webrender_external_images: Arc<Mutex<WebrenderExternalImageRegistry>>,

    /// Entry point to create and get channels to a WebGLThread.
    pub webgl_threads: Option<WebGLThreads>,

    /// The XR device registry
    pub webxr_registry: Option<webxr_api::Registry>,

    /// User agent string to report in network requests.
    pub user_agent: Cow<'static, str>,

    #[cfg(feature = "webgpu")]
    pub wgpu_image_map: WGPUImageMap,
}

/// `WebDriverData` holds state specific to the WebDriver implementation.
struct WebDriverData {
    /// A channel to send load status updates for a specific pipeline.
    load_channel: Option<(PipelineId, IpcSender<webdriver_msg::LoadStatus>)>,
    /// A channel to send window resize updates.
    resize_channel: Option<IpcSender<WindowSizeData>>,
}

impl WebDriverData {
    /// Creates a new `WebDriverData` instance with default (empty) values.
    fn new() -> WebDriverData {
        WebDriverData {
            load_channel: None,
            resize_channel: None,
        }
    }
}

/// `ReadyToSave` enumerates the possible states when preparing to save an image,
/// particularly in the context of reftests.
#[derive(Debug, PartialEq)]
enum ReadyToSave {
    /// No top-level browsing context is available.
    NoTopLevelBrowsingContext,
    /// There are pending session history changes.
    PendingChanges,
    /// The document is currently loading.
    DocumentLoading,
    /// A mismatch in WebRender epochs.
    EpochMismatch,
    /// The pipeline is unknown or invalid.
    PipelineUnknown,
    /// The image is ready to be saved.
    Ready,
}

/// `ExitPipelineMode` defines how a pipeline should exit.
#[derive(Clone, Copy, Debug)]
enum ExitPipelineMode {
    /// Normal exit, which involves updating the compositor and notifying script threads.
    Normal,
    /// Forced exit, where the compositor is not notified, and script involvement is bypassed.
    Force,
}

/// The number of warnings to include in each crash report.
const WARNINGS_BUFFER_SIZE: usize = 32;

/// Routes messages from an `IpcReceiver` to a new `crossbeam_channel::Receiver`,
/// preserving any IPC errors.
///
/// # Arguments
/// * `ipc_receiver` - The `IpcReceiver` to route messages from.
///
/// Pre-condition: `ipc_receiver` is a valid IPC channel.
/// Post-condition: Returns a `Receiver` that yields `Result<T, IpcError>` values.
fn route_ipc_receiver_to_new_crossbeam_receiver_preserving_errors<T>(
    ipc_receiver: IpcReceiver<T>,
) -> Receiver<Result<T, IpcError>>
where
    T: for<'de> Deserialize<'de> + Serialize + Send + 'static,
{
    let (crossbeam_sender, crossbeam_receiver) = unbounded();
    ROUTER.add_typed_route(
        ipc_receiver,
        Box::new(move |message| {
            let _ = crossbeam_sender.send(message);
        }),
    );
    crossbeam_receiver
}

impl<STF, SWF> Constellation<STF, SWF>
where
    STF: ScriptThreadFactory,
    SWF: ServiceWorkerManagerFactory,
{
    /// Creates and starts a new `Constellation` thread.
    ///
    /// # Arguments
    /// * `state` - Initial state and channels for the Constellation.
    /// * `layout_factory` - A factory for creating layout threads.
    /// * `initial_window_size` - The initial size of the top-level window.
    /// * `random_pipeline_closure_probability` - Optional probability for randomly closing pipelines (for testing).
    /// * `random_pipeline_closure_seed` - Optional seed for the RNG used for random pipeline closure.
    /// * `hard_fail` - If true, the Constellation exits on thread failure.
    /// * `canvas_create_sender` - Sender for canvas creation messages.
    /// * `canvas_ipc_sender` - IPC sender for canvas messages.
    ///
    /// Pre-condition: All provided channels and factories are valid.
    /// Post-condition: A new `Constellation` thread is spawned and its `CompositorMsg` sender is returned.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            skip(state, layout_factory),
            fields(servo_profiling = true),
            level = "trace",
        )
    )]
    pub fn start(
        state: InitialConstellationState,
        layout_factory: Arc<dyn LayoutFactory>,
        initial_window_size: WindowSizeData,
        random_pipeline_closure_probability: Option<f32>,
        random_pipeline_closure_seed: Option<usize>,
        hard_fail: bool,
        canvas_create_sender: Sender<ConstellationCanvasMsg>,
        canvas_ipc_sender: IpcSender<CanvasMsg>,
    ) -> Sender<FromCompositorMsg> {
        let (compositor_sender, compositor_receiver) = unbounded();

        // service worker manager to communicate with constellation
        let (swmanager_ipc_sender, swmanager_ipc_receiver) =
            ipc::channel().expect("ipc channel failure");

        thread::Builder::new()
            .name("Constellation".to_owned())
            .spawn(move || {
                let (script_ipc_sender, script_ipc_receiver) =
                    ipc::channel().expect("ipc channel failure");
                let script_receiver =
                    route_ipc_receiver_to_new_crossbeam_receiver_preserving_errors(
                        script_ipc_receiver,
                    );

                let (namespace_ipc_sender, namespace_ipc_receiver) =
                    ipc::channel().expect("ipc channel failure");
                let namespace_receiver =
                    route_ipc_receiver_to_new_crossbeam_receiver_preserving_errors(
                        namespace_ipc_receiver,
                    );

                let (background_hang_monitor_ipc_sender, background_hang_monitor_ipc_receiver) =
                    ipc::channel().expect("ipc channel failure");
                let background_hang_monitor_receiver =
                    route_ipc_receiver_to_new_crossbeam_receiver_preserving_errors(
                        background_hang_monitor_ipc_receiver,
                    );

                // If we are in multiprocess mode,
                // a dedicated per-process hang monitor will be initialized later inside the content process.
                // See run_content_process in servo/lib.rs
                let (background_monitor_register, background_hang_monitor_control_ipc_senders) =
                    if opts::get().multiprocess {
                        (None, vec![])
                    } else {
                        let (
                            background_hang_monitor_control_ipc_sender,
                            background_hang_monitor_control_ipc_receiver,
                        ) = ipc::channel().expect("ipc channel failure");
                        (
                            Some(HangMonitorRegister::init(
                                background_hang_monitor_ipc_sender.clone(),
                                background_hang_monitor_control_ipc_receiver,
                                opts::get().background_hang_monitor,
                            )),
                            vec![background_hang_monitor_control_ipc_sender],
                        )
                    };

                let (layout_ipc_sender, layout_ipc_receiver) =
                    ipc::channel().expect("ipc channel failure");
                let layout_receiver =
                    route_ipc_receiver_to_new_crossbeam_receiver_preserving_errors(
                        layout_ipc_receiver,
                    );

                let swmanager_receiver =
                    route_ipc_receiver_to_new_crossbeam_receiver_preserving_errors(
                        swmanager_ipc_receiver,
                    );

                // Zero is reserved for the embedder.
                PipelineNamespace::install(PipelineNamespaceId(1));

                #[cfg(feature = "webgpu")]
                let webrender_wgpu = WebrenderWGPU {
                    webrender_api: state.webrender_api_sender.create_api(),
                    webrender_external_images: state.webrender_external_images,
                    wgpu_image_map: state.wgpu_image_map,
                };

                let rippy_data = resources::read_bytes(Resource::RippyPNG);

                let mut constellation: Constellation<STF, SWF> = Constellation {
                    namespace_receiver,
                    namespace_ipc_sender,
                    script_sender: script_ipc_sender,
                    background_hang_monitor_sender: background_hang_monitor_ipc_sender,
                    background_hang_monitor_receiver,
                    background_monitor_register,
                    background_monitor_control_senders: background_hang_monitor_control_ipc_senders,
                    layout_sender: layout_ipc_sender,
                    script_receiver,
                    compositor_receiver,
                    layout_factory,
                    layout_receiver,
                    embedder_proxy: state.embedder_proxy,
                    compositor_proxy: state.compositor_proxy,
                    webviews: WebViewManager::default(),
                    devtools_sender: state.devtools_sender,
                    #[cfg(feature = "bluetooth")]
                    bluetooth_ipc_sender: state.bluetooth_thread,
                    public_resource_threads: state.public_resource_threads,
                    private_resource_threads: state.private_resource_threads,
                    system_font_service: state.system_font_service,
                    sw_managers: Default::default(),
                    swmanager_receiver,
                    swmanager_ipc_sender,
                    browsing_context_group_set: Default::default(),
                    browsing_context_group_next_id: Default::default(),
                    message_ports: HashMap::new(),
                    message_port_routers: HashMap::new(),
                    broadcast_routers: HashMap::new(),
                    broadcast_channels: HashMap::new(),
                    pipelines: HashMap::new(),
                    browsing_contexts: HashMap::new(),
                    pending_changes: vec![],
                    // We initialize the namespace at 2, since we reserved
                    // namespace 0 for the embedder, and 0 for the constellation
                    next_pipeline_namespace_id: PipelineNamespaceId(2),
                    time_profiler_chan: state.time_profiler_chan,
                    mem_profiler_chan: state.mem_profiler_chan,
                    window_size: initial_window_size,
                    phantom: PhantomData,
                    webdriver: WebDriverData::new(),
                    document_states: HashMap::new(),
                    webrender_document: state.webrender_document,
                    #[cfg(feature = "webgpu")]
                    webrender_wgpu,
                    shutting_down: false,
                    handled_warnings: VecDeque::new(),
                    random_pipeline_closure: random_pipeline_closure_probability.map(|prob| {
                        let seed = random_pipeline_closure_seed.unwrap_or_else(random);
                        let rng = ServoRng::new_manually_reseeded(seed as u64);
                        warn!("Randomly closing pipelines.");
                        info!("Using seed {} for random pipeline closure.", seed);
                        (rng, prob)
                    }),
                    webgl_threads: state.webgl_threads,
                    webxr_registry: state.webxr_registry,
                    canvas_sender: canvas_create_sender,
                    canvas_ipc_sender,
                    pending_approval_navigations: HashMap::new(),
                    pressed_mouse_buttons: 0,
                    hard_fail,
                    active_media_session: None,
                    user_agent: state.user_agent,
                    rippy_data,
                };

                constellation.run();
            })
            .expect("Thread spawning failed");

        compositor_sender
    }

    /// The main event loop for the constellation.
    /// This function continuously processes incoming requests from various threads
    /// until the constellation is shutting down and all pipelines are closed.
    ///
    /// Pre-condition: The Constellation has been initialized.
    /// Post-condition: The Constellation processes all pending requests and gracefully shuts down.
    fn run(&mut self) {
        while !self.shutting_down || !self.pipelines.is_empty() {
            // Randomly close a pipeline if --random-pipeline-closure-probability is set
            // This is for testing the hardening of the constellation.
            self.maybe_close_random_pipeline();
            self.handle_request();
        }
        self.handle_shutdown();
    }

    /// Generates a new unique `PipelineNamespaceId`.
    ///
    /// Post-condition: Returns a new `PipelineNamespaceId` and increments the internal counter.
    fn next_pipeline_namespace_id(&mut self) -> PipelineNamespaceId {
        let namespace_id = self.next_pipeline_namespace_id;
        let PipelineNamespaceId(ref mut i) = self.next_pipeline_namespace_id;
        *i += 1;
        namespace_id
    }

    /// Generates a new unique `BrowsingContextGroupId`.
    ///
    /// Post-condition: Returns a new `BrowsingContextGroupId` and increments the internal counter.
    fn next_browsing_context_group_id(&mut self) -> BrowsingContextGroupId {
        let id = self.browsing_context_group_next_id;
        self.browsing_context_group_next_id += 1;
        BrowsingContextGroupId(id)
    }

    /// Retrieves a weak reference to an `EventLoop` based on host, top-level browsing context,
    /// and opener information.
    ///
    /// # Arguments
    /// * `host` - The host for which to retrieve the event loop.
    /// * `top_level_browsing_context_id` - The ID of the top-level browsing context.
    /// * `opener` - An optional `BrowsingContextId` of the opener.
    ///
    /// Pre-condition: The browsing context group and event loop for the given host exist.
    /// Post-condition: Returns `Ok(Weak<EventLoop>)` if found, or `Err` if not.
    fn get_event_loop(
        &mut self,
        host: &Host,
        top_level_browsing_context_id: &TopLevelBrowsingContextId,
        opener: &Option<BrowsingContextId>,
    ) -> Result<Weak<EventLoop>, &'static str> {
        let bc_group = match opener {
            Some(browsing_context_id) => {
                let opener = self
                    .browsing_contexts
                    .get(browsing_context_id)
                    .ok_or("Opener was closed before the openee started")?;
                self.browsing_context_group_set
                    .get(&opener.bc_group_id)
                    .ok_or("Opener belongs to an unknown browsing context group")?
            },
            None => self
                .browsing_context_group_set
                .iter()
                .filter_map(|(_, bc_group)| {
                    if bc_group
                        .top_level_browsing_context_set
                        .contains(top_level_browsing_context_id)
                    {
                        Some(bc_group)
                    } else {
                        None
                    }
                })
                .last()
                .ok_or(
                    "Trying to get an event-loop for a top-level belonging to an unknown browsing context group",
                )?,
        };
        bc_group
            .event_loops
            .get(host)
            .ok_or("Trying to get an event-loop from an unknown browsing context group")
            .cloned()
    }

    /// Sets an `EventLoop` for a given host and browsing context.
    ///
    /// # Arguments
    /// * `event_loop` - A weak reference to the `EventLoop` to set.
    /// * `host` - The host associated with the event loop.
    /// * `top_level_browsing_context_id` - The ID of the top-level browsing context.
    /// * `opener` - An optional `BrowsingContextId` of the opener.
    ///
    /// Pre-condition: The browsing context group associated with the provided IDs exists.
    /// Post-condition: The `event_loop` is inserted into the appropriate browsing context group.
    fn set_event_loop(
        &mut self,
        event_loop: Weak<EventLoop>,
        host: Host,
        top_level_browsing_context_id: TopLevelBrowsingContextId,
        opener: Option<BrowsingContextId>,
    ) {
        let relevant_top_level = if let Some(opener) = opener {
            match self.browsing_contexts.get(&opener) {
                Some(opener) => opener.top_level_id,
                None => {
                    warn!("Setting event-loop for an unknown auxiliary");
                    return;
                },
            }
        } else {
            top_level_browsing_context_id
        };
        let maybe_bc_group_id = self
            .browsing_context_group_set
            .iter()
            .filter_map(|(id, bc_group)| {
                if bc_group
                    .top_level_browsing_context_set
                    .contains(&top_level_browsing_context_id)
                {
                    Some(*id)
                } else {
                    None
                }
            })
            .last();
        let bc_group_id = match maybe_bc_group_id {
            Some(id) => id,
            None => {
                warn!("Trying to add an event-loop to an unknown browsing context group");
                return;
            },
        };
        if let Some(bc_group) = self.browsing_context_group_set.get_mut(&bc_group_id) {
            if bc_group
                .event_loops
                .insert(host.clone(), event_loop)
                .is_some()
            {
                warn!(
                    "Double-setting an event-loop for {:?} at {:?}",
                    host, relevant_top_level
                );
            }
        }
    }

    /// Creates a new `Pipeline` and associates it with a browsing context.
    ///
    /// # Arguments
    /// * `pipeline_id` - The unique ID for the new pipeline.
    /// * `browsing_context_id` - The `BrowsingContextId` this pipeline belongs to.
    /// * `top_level_browsing_context_id` - The `TopLevelBrowsingContextId` of the current webview.
    /// * `parent_pipeline_id` - Optional ID of the parent pipeline.
    /// * `opener` - Optional `BrowsingContextId` of the opener.
    /// * `initial_window_size` - The initial size of the pipeline's window.
    /// * `load_data` - Data for the initial load of the pipeline.
    /// * `sandbox` - The sandbox state of the iframe, if applicable.
    /// * `is_private` - Whether the pipeline is for private browsing.
    /// * `throttled` - Whether the pipeline starts in a throttled state.
    ///
    /// Pre-condition: `pipeline_id` is unique and all input data is valid.
    /// Post-condition: A new `Pipeline` is spawned and registered with the Constellation.
    #[allow(clippy::too_many_arguments)]
    fn new_pipeline(
        &mut self,
        pipeline_id: PipelineId,
        browsing_context_id: BrowsingContextId,
        top_level_browsing_context_id: TopLevelBrowsingContextId,
        parent_pipeline_id: Option<PipelineId>,
        opener: Option<BrowsingContextId>,
        initial_window_size: Size2D<f32, CSSPixel>,
        // TODO: we have to provide ownership of the LoadData
        // here, because it will be send on an ipc channel,
        // and ipc channels take onership of their data.
        // https://github.com/servo/ipc-channel/issues/138
        load_data: LoadData,
        sandbox: IFrameSandboxState,
        is_private: bool,
        throttled: bool,
    ) {
        if self.shutting_down {
            return;
        }
        debug!(
            "{}: Creating new pipeline in {}",
            pipeline_id, browsing_context_id
        );

        let (event_loop, host) = match sandbox {
            IFrameSandboxState::IFrameSandboxed => (None, None),
            IFrameSandboxState::IFrameUnsandboxed => {
                // If this is an about:blank or about:srcdoc load, it must share the creator's
                // event loop. This must match the logic in the script thread when determining
                // the proper origin.
                if load_data.url.as_str() != "about:blank" &&
                    load_data.url.as_str() != "about:srcdoc"
                {
                    match reg_host(&load_data.url) {
                        None => (None, None),
                        Some(host) => {
                            match self.get_event_loop(
                                &host,
                                &top_level_browsing_context_id,
                                &opener,
                            ) {
                                Err(err) => {
                                    warn!("{}", err);
                                    (None, Some(host))
                                },
                                Ok(event_loop) => {
                                    if let Some(event_loop) = event_loop.upgrade() {
                                        (Some(event_loop), None)
                                    } else {
                                        (None, Some(host))
                                    }
                                },
                            }
                        },
                    }
                } else if let Some(parent) =
                    parent_pipeline_id.and_then(|pipeline_id| self.pipelines.get(&pipeline_id))
                {
                    (Some(parent.event_loop.clone()), None)
                } else if let Some(creator) = load_data
                    .creator_pipeline_id
                    .and_then(|pipeline_id| self.pipelines.get(&pipeline_id))
                {
                    (Some(creator.event_loop.clone()), None)
                } else {
                    (None, None)
                }
            },
        };

        let resource_threads = if is_private {
            self.private_resource_threads.clone()
        } else {
            self.public_resource_threads.clone()
        };

        let result = Pipeline::spawn::<STF>(InitialPipelineState {
            id: pipeline_id,
            browsing_context_id,
            top_level_browsing_context_id,
            parent_pipeline_id,
            opener,
            script_to_constellation_chan: ScriptToConstellationChan {
                sender: self.script_sender.clone(),
                pipeline_id,
            },
            namespace_request_sender: self.namespace_ipc_sender.clone(),
            pipeline_namespace_id: self.next_pipeline_namespace_id(),
            background_monitor_register: self.background_monitor_register.clone(),
            background_hang_monitor_to_constellation_chan: self
                .background_hang_monitor_sender
                .clone(),
            layout_to_constellation_chan: self.layout_sender.clone(),
            layout_factory: self.layout_factory.clone(),
            compositor_proxy: self.compositor_proxy.clone(),
            devtools_sender: self.devtools_sender.clone(),
            #[cfg(feature = "bluetooth")]
            bluetooth_thread: self.bluetooth_ipc_sender.clone(),
            swmanager_thread: self.swmanager_ipc_sender.clone(),
            system_font_service: self.system_font_service.clone(),
            resource_threads,
            time_profiler_chan: self.time_profiler_chan.clone(),
            mem_profiler_chan: self.mem_profiler_chan.clone(),
            window_size: WindowSizeData {
                initial_viewport: initial_window_size,
                device_pixel_ratio: self.window_size.device_pixel_ratio,
            },
            event_loop,
            load_data,
            prev_throttled: throttled,
            webrender_document: self.webrender_document,
            webgl_chan: self
                .webgl_threads
                .as_ref()
                .map(|threads| threads.pipeline()),
            webxr_registry: self.webxr_registry.clone(),
            player_context: WindowGLContext::get(),
            user_agent: self.user_agent.clone(),
            rippy_data: self.rippy_data.clone(),
        });

        let pipeline = match result {
            Ok(result) => result,
            Err(e) => return self.handle_send_error(pipeline_id, e),
        };

        if let Some(chan) = pipeline.bhm_control_chan {
            self.background_monitor_control_senders.push(chan);
        }

        if let Some(host) = host {
            debug!(
                "{}: Adding new host entry {}",
                top_level_browsing_context_id, host,
            );
            self.set_event_loop(
                Rc::downgrade(&pipeline.pipeline.event_loop),
                host,
                top_level_browsing_context_id,
                opener,
            );
        }

        assert!(!self.pipelines.contains_key(&pipeline_id));
        self.pipelines.insert(pipeline_id, pipeline.pipeline);
    }

    /// Returns an iterator over all fully active descendant browsing contexts of a given context.
    ///
    /// # Arguments
    /// * `browsing_context_id` - The ID of the browsing context from which to start iterating.
    ///
    /// Post-condition: Returns an `FullyActiveBrowsingContextsIterator`.
    fn fully_active_descendant_browsing_contexts_iter(
        &self,
        browsing_context_id: BrowsingContextId,
    ) -> FullyActiveBrowsingContextsIterator {
        FullyActiveBrowsingContextsIterator {
            stack: vec![browsing_context_id],
            pipelines: &self.pipelines,
            browsing_contexts: &self.browsing_contexts,
        }
    }

    /// Returns an iterator over all fully active browsing contexts in a tree rooted at a top-level context.
    ///
    /// # Arguments
    /// * `top_level_browsing_context_id` - The ID of the top-level browsing context.
    ///
    /// Post-condition: Returns an `FullyActiveBrowsingContextsIterator`.
    fn fully_active_browsing_contexts_iter(
        &self,
        top_level_browsing_context_id: TopLevelBrowsingContextId,
    ) -> FullyActiveBrowsingContextsIterator {
        self.fully_active_descendant_browsing_contexts_iter(BrowsingContextId::from(
            top_level_browsing_context_id,
        ))
    }

    /// Returns an iterator over all descendant browsing contexts of a given context.
    ///
    /// # Arguments
    /// * `browsing_context_id` - The ID of the browsing context from which to start iterating.
    ///
    /// Post-condition: Returns an `AllBrowsingContextsIterator`.
    fn all_descendant_browsing_contexts_iter(
        &self,
        browsing_context_id: BrowsingContextId,
    ) -> AllBrowsingContextsIterator {
        AllBrowsingContextsIterator {
            stack: vec![browsing_context_id],
            pipelines: &self.pipelines,
            browsing_contexts: &self.browsing_contexts,
        }
    }

    /// Creates a new browsing context and updates the internal bookkeeping.
    ///
    /// # Arguments
    /// * `browsing_context_id` - The ID of the new browsing context.
    /// * `top_level_id` - The ID of the top-level browsing context it belongs to.
    /// * `pipeline_id` - The `PipelineId` associated with this context.
    /// * `parent_pipeline_id` - Optional ID of the parent pipeline.
    /// * `size` - The size of the browsing context.
    /// * `is_private` - Whether this is a private browsing context.
    /// * `inherited_secure_context` - Whether this context inherits a secure context.
    /// * `throttled` - Whether the context starts in a throttled state.
    ///
    /// Pre-condition: `browsing_context_id` is unique and corresponds to valid parent/top-level IDs.
    /// Post-condition: A new `BrowsingContext` is created and registered.
    #[allow(clippy::too_many_arguments)]
    fn new_browsing_context(
        &mut self,
        browsing_context_id: BrowsingContextId,
        top_level_id: TopLevelBrowsingContextId,
        pipeline_id: PipelineId,
        parent_pipeline_id: Option<PipelineId>,
        size: Size2D<f32, CSSPixel>,
        is_private: bool,
        inherited_secure_context: Option<bool>,
        throttled: bool,
    ) {
        debug!("{}: Creating new browsing context", browsing_context_id);
        let bc_group_id = match self
            .browsing_context_group_set
            .iter_mut()
            .filter_map(|(id, bc_group)| {
                if bc_group
                    .top_level_browsing_context_set
                    .contains(&top_level_id)
                {
                    Some(id)
                } else {
                    None
                }
            })
            .last()
        {
            Some(id) => *id,
            None => {
                warn!("Top-level was unexpectedly removed from its top_level_browsing_context_set");
                return;
            },
        };
        let browsing_context = BrowsingContext::new(
            bc_group_id,
            browsing_context_id,
            top_level_id,
            pipeline_id,
            parent_pipeline_id,
            size,
            is_private,
            inherited_secure_context,
            throttled,
        );
        self.browsing_contexts
            .insert(browsing_context_id, browsing_context);

        // If this context is a nested container, attach it to parent pipeline.
        if let Some(parent_pipeline_id) = parent_pipeline_id {
            if let Some(parent) = self.pipelines.get_mut(&parent_pipeline_id) {
                parent.add_child(browsing_context_id);
            }
        }
    }

    /// Adds a pending `SessionHistoryChange` to the internal queue.
    ///
    /// # Arguments
    /// * `change` - The `SessionHistoryChange` to add.
    ///
    /// Post-condition: `change` is appended to `self.pending_changes`.
    fn add_pending_change(&mut self, change: SessionHistoryChange) {
        debug!(
            "adding pending session history change with {}",
            if change.replace.is_some() {
                "replacement"
            } else {
                "no replacement"
            },
        );
        self.pending_changes.push(change);
    }

    /// Handles incoming requests from various threads, including pipeline namespaces,
    /// script, background hang monitor, compositor, layout, and service worker manager.
    /// This is the central message processing loop for the Constellation.
    ///
    /// Pre-condition: The Constellation is running.
    /// Post-condition: An incoming request is processed by the appropriate handler.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(servo_profiling = true), level = "trace")
    )]
    fn handle_request(&mut self) {
        #[derive(Debug)]
        enum Request {
            PipelineNamespace(PipelineNamespaceRequest),
            Script((PipelineId, FromScriptMsg)),
            BackgroundHangMonitor(HangMonitorAlert),
            Compositor(FromCompositorMsg),
            Layout(FromLayoutMsg),
            FromSWManager(SWManagerMsg),
        }
        // Get one incoming request.
        // This is one of the few places where the compositor is
        // allowed to panic. If one of the receiver.recv() calls
        // fails, it is because the matching sender has been
        // reclaimed, but this can't happen in normal execution
        // because the constellation keeps a pointer to the sender,
        // so it should never be reclaimed. A possible scenario in
        // which receiver.recv() fails is if some unsafe code
        // produces undefined behaviour, resulting in the destructor
        // being called. If this happens, there's not much we can do
        // other than panic.
        let request = {
            #[cfg(feature = "tracing")]
            let _span =
                tracing::trace_span!("handle_request::select", servo_profiling = true).entered();
            select! {
                recv(self.namespace_receiver) -> msg => {
                    msg.expect("Unexpected script channel panic in constellation").map(Request::PipelineNamespace)
                }
                recv(self.script_receiver) -> msg => {
                    msg.expect("Unexpected script channel panic in constellation").map(Request::Script)
                }
                recv(self.background_hang_monitor_receiver) -> msg => {
                    msg.expect("Unexpected BHM channel panic in constellation").map(Request::BackgroundHangMonitor)
                }
                recv(self.compositor_receiver) -> msg => {
                    Ok(Request::Compositor(msg.expect("Unexpected compositor channel panic in constellation")))
                }
                recv(self.layout_receiver) -> msg => {
                    msg.expect("Unexpected layout channel panic in constellation").map(Request::Layout)
                }
                recv(self.swmanager_receiver) -> msg => {
                    msg.expect("Unexpected SW channel panic in constellation").map(Request::FromSWManager)
                }
            }
        };

        let request = match request {
            Ok(request) => request,
            Err(err) => return error!("Deserialization failed ({}).", err),
        };

        match request {
            Request::PipelineNamespace(message) => {
                self.handle_request_for_pipeline_namespace(message)
            },
            Request::Compositor(message) => self.handle_request_from_compositor(message),
            Request::Script(message) => {
                self.handle_request_from_script(message);
            },
            Request::BackgroundHangMonitor(message) => {
                self.handle_request_from_background_hang_monitor(message);
            },
            Request::Layout(message) => {
                self.handle_request_from_layout(message);
            },
            Request::FromSWManager(message) => {
                self.handle_request_from_swmanager(message);
            },
        }
    }

    /// Handles a `PipelineNamespaceRequest` by sending the next available pipeline namespace ID.
    ///
    /// # Arguments
    /// * `request` - The `PipelineNamespaceRequest` containing the sender for the response.
    ///
    /// Post-condition: The next `PipelineNamespaceId` is sent back to the requester.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(servo_profiling = true), level = "trace")
    )]
    fn handle_request_for_pipeline_namespace(&mut self, request: PipelineNamespaceRequest) {
        let PipelineNamespaceRequest(sender) = request;
        let _ = sender.send(self.next_pipeline_namespace_id());
    }

    /// Handles `HangMonitorAlert` messages from the background hang monitor.
    ///
    /// # Arguments
    /// * `message` - The `HangMonitorAlert` received.
    ///
    /// Post-condition: The alert is processed, potentially reporting a profile or logging a hang warning.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(servo_profiling = true), level = "trace")
    )]
    fn handle_request_from_background_hang_monitor(&self, message: HangMonitorAlert) {
        match message {
            HangMonitorAlert::Profile(bytes) => {
                self.embedder_proxy.send(EmbedderMsg::ReportProfile(bytes))
            },
            HangMonitorAlert::Hang(hang) => {
                // TODO: In case of a permanent hang being reported, add a "kill script" workflow,
                // via the embedder?
                warn!("Component hang alert: {:?}", hang);
            },
        }
    }

    /// Handles `SWManagerMsg` messages from the Service Worker Manager.
    ///
    /// # Arguments
    /// * `message` - The `SWManagerMsg` received.
    ///
    /// Post-condition: The message is processed, e.g., posting a message to a SW client.
    fn handle_request_from_swmanager(&mut self, message: SWManagerMsg) {
        match message {
            SWManagerMsg::PostMessageToClient => {
                // TODO: implement posting a message to a SW client.
                // https://github.com/servo/servo/issues/24660
            },
        }
    }

    /// Handles incoming `FromCompositorMsg` messages, dispatching them to various
    /// internal handlers based on the message type.
    ///
    /// # Arguments
    /// * `message` - The `FromCompositorMsg` received.
    ///
    /// Pre-condition: The Constellation is active.
    /// Post-condition: The message is processed, leading to state changes or actions.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(servo_profiling = true), level = "trace")
    )]
    fn handle_request_from_compositor(&mut self, message: FromCompositorMsg) {
        trace_msg_from_compositor!(message, "{message:?}");
        match message {
            FromCompositorMsg::Exit => {
                self.handle_exit();
            },
            FromCompositorMsg::GetBrowsingContext(pipeline_id, response_sender) => {
                self.handle_get_browsing_context(pipeline_id, response_sender);
            },
            FromCompositorMsg::GetPipeline(browsing_context_id, response_sender) => {
                self.handle_get_pipeline(browsing_context_id, response_sender);
            },
            FromCompositorMsg::GetFocusTopLevelBrowsingContext(resp_chan) => {
                let _ = resp_chan.send(self.webviews.focused_webview().map(|(id, _)| id));
            },
            // Perform a navigation previously requested by script, if approved by the embedder.
            // If there is already a pending page (self.pending_changes), it will not be overridden;
            // However, if the id is not encompassed by another change, it will be.
            FromCompositorMsg::AllowNavigationResponse(pipeline_id, allowed) => {
                let pending = self.pending_approval_navigations.remove(&pipeline_id);

                let top_level_browsing_context_id = match self.pipelines.get(&pipeline_id) {
                    Some(pipeline) => pipeline.top_level_browsing_context_id,
                    None => return warn!("{}: Attempted to navigate after closure", pipeline_id),
                };

                match pending {
                    Some((load_data, history_handling)) => {
                        if allowed {
                            self.load_url(
                                top_level_browsing_context_id,
                                pipeline_id,
                                load_data,
                                history_handling,
                            );
                        } else {
                            let pipeline_is_top_level_pipeline = self
                                .browsing_contexts
                                .get(&BrowsingContextId::from(top_level_browsing_context_id))
                                .map(|ctx| ctx.pipeline_id == pipeline_id)
                                .unwrap_or(false);
                            // If the navigation is refused, and this concerns an iframe,
                            // we need to take it out of it's "delaying-load-events-mode".
                            // https://html.spec.whatwg.org/multipage/#delaying-load-events-mode
                            if !pipeline_is_top_level_pipeline {
                                let msg =
                                    ScriptThreadMessage::StopDelayingLoadEventsMode(pipeline_id);
                                let result = match self.pipelines.get(&pipeline_id) {
                                    Some(pipeline) => pipeline.event_loop.send(msg),
                                    None => {
                                        return warn!(
                                            "{}: Attempted to navigate after closure",
                                            pipeline_id
                                        );
                                    },
                                };
                                if let Err(e) = result {
                                    self.handle_send_error(pipeline_id, e);
                                }
                            }
                        }
                    },
                    None => {
                        warn!(
                            "{}: AllowNavigationResponse for unknown request",
                            pipeline_id
                        )
                    },
                }
            },
            FromCompositorMsg::ClearCache => {
                self.public_resource_threads.clear_cache();
                self.private_resource_threads.clear_cache();
            },
            // Load a new page from a typed url
            // If there is already a pending page (self.pending_changes), it will not be overridden;
            // However, if the id is not encompassed by another change, it will be.
            FromCompositorMsg::LoadUrl(top_level_browsing_context_id, url) => {
                let load_data = LoadData::new(
                    LoadOrigin::Constellation,
                    url,
                    None,
                    Referrer::NoReferrer,
                    ReferrerPolicy::EmptyString,
                    None,
                    None,
                );
                let ctx_id = BrowsingContextId::from(top_level_browsing_context_id);
                let pipeline_id = match self.browsing_contexts.get(&ctx_id) {
                    Some(ctx) => ctx.pipeline_id,
                    None => {
                        return warn!(
                            "{}: LoadUrl for unknown browsing context",
                            top_level_browsing_context_id
                        );
                    },
                };
                // Since this is a top-level load, initiated by the embedder, go straight to load_url,
                // bypassing schedule_navigation.
                self.load_url(
                    top_level_browsing_context_id,
                    pipeline_id,
                    load_data,
                    NavigationHistoryBehavior::Push,
                );
            },
            FromCompositorMsg::IsReadyToSaveImage(pipeline_states) => {
                let is_ready = self.handle_is_ready_to_save_image(pipeline_states);
                debug!("Ready to save image {:?}.", is_ready);
                self.compositor_proxy
                    .send(CompositorMsg::IsReadyToSaveImageReply(
                        is_ready == ReadyToSave::Ready,
                    ));
            },
            // Create a new top level browsing context. Will use response_chan to return
            // the browsing context id.
            FromCompositorMsg::NewWebView(url, top_level_browsing_context_id) => {
                self.handle_new_top_level_browsing_context(
                    url,
                    top_level_browsing_context_id,
                    None,
                );
            },
            // A top level browsing context is created and opened in both constellation and
            // compositor.
            FromCompositorMsg::WebViewOpened(top_level_browsing_context_id) => {
                self.embedder_proxy
                    .send(EmbedderMsg::WebViewOpened(top_level_browsing_context_id));
            },
            // Close a top level browsing context.
            FromCompositorMsg::CloseWebView(top_level_browsing_context_id) => {
                self.handle_close_top_level_browsing_context(top_level_browsing_context_id);
            },
            // Panic a top level browsing context.
            FromCompositorMsg::SendError(top_level_browsing_context_id, error) => {
                debug!("constellation got SendError message");
                if top_level_browsing_context_id.is_none() {
                    warn!("constellation got a SendError message without top level id");
                }
                self.handle_panic(top_level_browsing_context_id, error, None);
            },
            FromCompositorMsg::FocusWebView(top_level_browsing_context_id) => {
                self.handle_focus_web_view(top_level_browsing_context_id);
            },
            FromCompositorMsg::BlurWebView => {
                self.webviews.unfocus();
                self.embedder_proxy.send(EmbedderMsg::WebViewBlurred);
            },
            // Handle a forward or back request
            FromCompositorMsg::TraverseHistory(top_level_browsing_context_id, direction) => {
                self.handle_traverse_history_msg(top_level_browsing_context_id, direction);
            },
            FromCompositorMsg::WindowSize(top_level_browsing_context_id, new_size, size_type) => {
                self.handle_window_size_msg(top_level_browsing_context_id, new_size, size_type);
            },
            FromCompositorMsg::ThemeChange(theme) => {
                self.handle_theme_change(theme);
            },
            FromCompositorMsg::TickAnimation(pipeline_id, tick_type) => {
                self.handle_tick_animation(pipeline_id, tick_type)
            },
            FromCompositorMsg::WebDriverCommand(command) => {
                self.handle_webdriver_msg(command);
            },
            FromCompositorMsg::Reload(top_level_browsing_context_id) => {
                self.handle_reload_msg(top_level_browsing_context_id);
            },
            FromCompositorMsg::LogEntry(top_level_browsing_context_id, thread_name, entry) => {
                self.handle_log_entry(top_level_browsing_context_id, thread_name, entry);
            },
            FromCompositorMsg::ForwardInputEvent(event, hit_test) => {
                self.forward_input_event(event, hit_test);
            },
            FromCompositorMsg::SetCursor(webview_id, cursor) => {
                self.handle_set_cursor_msg(webview_id, cursor)
            },
            FromCompositorMsg::ToggleProfiler(rate, max_duration) => {
                for background_monitor_control_sender in &self.background_monitor_control_senders {
                    if let Err(e) = background_monitor_control_sender.send(
                        BackgroundHangMonitorControlMsg::ToggleSampler(rate, max_duration),
                    ) {
                        warn!("error communicating with sampling profiler: {}", e);
                    }
                }
            },
            FromCompositorMsg::ExitFullScreen(top_level_browsing_context_id) => {
                self.handle_exit_fullscreen_msg(top_level_browsing_context_id);
            },
            FromCompositorMsg::MediaSessionAction(action) => {
                self.handle_media_session_action_msg(action);
            },
            FromCompositorMsg::SetWebViewThrottled(webview_id, throttled) => {
                self.set_webview_throttled(webview_id, throttled);
            },
        }
    }

    /// Handles incoming `FromScriptMsg` messages from a script thread, dispatching them to
    /// various internal handlers based on the message type.
    ///
    /// # Arguments
    /// * `message` - A tuple containing the `PipelineId` of the source and the `FromScriptMsg`.
    ///
    /// Pre-condition: The Constellation is active and `source_pipeline_id` is valid.
    /// Post-condition: The message is processed, leading to state changes or actions within the Constellation.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(servo_profiling = true), level = "trace")
    )]
    fn handle_request_from_script(&mut self, message: (PipelineId, FromScriptMsg)) {
        let (source_pipeline_id, content) = message;
        trace_script_msg!(content, "{source_pipeline_id}: {content:?}");

        let source_top_ctx_id = match self
            .pipelines
            .get(&source_pipeline_id)
            .map(|pipeline| pipeline.top_level_browsing_context_id)
        {
            None => return warn!("{}: ScriptMsg from closed pipeline", source_pipeline_id),
            Some(ctx) => ctx,
        };

        match content {
            FromScriptMsg::CompleteMessagePortTransfer(router_id, ports) => {
                self.handle_complete_message_port_transfer(router_id, ports);
            },
            FromScriptMsg::MessagePortTransferResult(router_id, succeeded, failed) => {
                self.handle_message_port_transfer_completed(router_id, succeeded);
                self.handle_message_port_transfer_failed(failed);
            },
            FromScriptMsg::RerouteMessagePort(port_id, task) => {
                self.handle_reroute_messageport(port_id, task);
            },
            FromScriptMsg::MessagePortShipped(port_id) => {
                self.handle_messageport_shipped(port_id);
            },
            FromScriptMsg::NewMessagePortRouter(router_id, ipc_sender) => {
                self.handle_new_messageport_router(router_id, ipc_sender);
            },
            FromScriptMsg::RemoveMessagePortRouter(router_id) => {
                self.handle_remove_messageport_router(router_id);
            },
            FromScriptMsg::NewMessagePort(router_id, port_id) => {
                self.handle_new_messageport(router_id, port_id);
            },
            FromScriptMsg::RemoveMessagePort(port_id) => {
                self.handle_remove_messageport(port_id);
            },
            FromScriptMsg::EntanglePorts(port1, port2) => {
                self.handle_entangle_messageports(port1, port2);
            },
            FromScriptMsg::NewBroadcastChannelRouter(router_id, response_sender, origin) => {
                self.handle_new_broadcast_channel_router(
                    source_pipeline_id,
                    router_id,
                    response_sender,
                    origin,
                );
            },
            FromScriptMsg::NewBroadcastChannelNameInRouter(router_id, channel_name, origin) => {
                self.handle_new_broadcast_channel_name_in_router(
                    source_pipeline_id,
                    router_id,
                    channel_name,
                    origin,
                );
            },
            FromScriptMsg::RemoveBroadcastChannelNameInRouter(router_id, channel_name, origin) => {
                self.handle_remove_broadcast_channel_name_in_router(
                    source_pipeline_id,
                    router_id,
                    channel_name,
                    origin,
                );
            },
            FromScriptMsg::RemoveBroadcastChannelRouter(router_id, origin) => {
                self.handle_remove_broadcast_channel_router(source_pipeline_id, router_id, origin);
            },
            FromScriptMsg::ScheduleBroadcast(router_id, message) => {
                self.handle_schedule_broadcast(source_pipeline_id, router_id, message);
            },
            FromScriptMsg::ForwardToEmbedder(embedder_msg) => {
                self.embedder_proxy.send(embedder_msg);
            },
            FromScriptMsg::PipelineExited => {
                self.handle_pipeline_exited(source_pipeline_id);
            },
            FromScriptMsg::DiscardDocument => {
                self.handle_discard_document(source_top_ctx_id, source_pipeline_id);
            },
            FromScriptMsg::DiscardTopLevelBrowsingContext => {
                self.handle_close_top_level_browsing_context(source_top_ctx_id);
            },
            FromScriptMsg::ScriptLoadedURLInIFrame(load_info) => {
                self.handle_script_loaded_url_in_iframe_msg(load_info);
            },
            FromScriptMsg::ScriptNewIFrame(load_info) => {
                self.handle_script_new_iframe(load_info);
            },
            FromScriptMsg::ScriptNewAuxiliary(load_info) => {
                self.handle_script_new_auxiliary(load_info);
            },
            FromScriptMsg::ChangeRunningAnimationsState(animation_state) => {
                self.handle_change_running_animations_state(source_pipeline_id, animation_state)
            },
            // Ask the embedder for permission to load a new page.
            FromScriptMsg::LoadUrl(load_data, history_handling) => {
                self.schedule_navigation(
                    source_top_ctx_id,
                    source_pipeline_id,
                    load_data,
                    history_handling,
                );
            },
            FromScriptMsg::AbortLoadUrl => {
                self.handle_abort_load_url_msg(source_pipeline_id);
            },
            // A page loaded has completed all parsing, script, and reflow messages have been sent.
            FromScriptMsg::LoadComplete => {
                self.handle_load_complete_msg(source_top_ctx_id, source_pipeline_id)
            },
            // Handle navigating to a fragment
            FromScriptMsg::NavigatedToFragment(new_url, replacement_enabled) => {
                self.handle_navigated_to_fragment(source_pipeline_id, new_url, replacement_enabled);
            },
            // Handle a forward or back request
            FromScriptMsg::TraverseHistory(direction) => {
                self.handle_traverse_history_msg(source_top_ctx_id, direction);
            },
            // Handle a push history state request.
            FromScriptMsg::PushHistoryState(history_state_id, url) => {
                self.handle_push_history_state_msg(source_pipeline_id, history_state_id, url);
            },
            FromScriptMsg::ReplaceHistoryState(history_state_id, url) => {
                self.handle_replace_history_state_msg(source_pipeline_id, history_state_id, url);
            },
            // Handle a joint session history length request.
            FromScriptMsg::JointSessionHistoryLength(response_sender) => {
                self.handle_joint_session_history_length(source_top_ctx_id, response_sender);
            },
            // Notification that the new document is ready to become active
            FromScriptMsg::ActivateDocument => {
                self.handle_activate_document_msg(source_pipeline_id);
            },
            // Update pipeline url after redirections
            FromScriptMsg::SetFinalUrl(final_url) => {
                // The script may have finished loading after we already started shutting down.
                if let Some(ref mut pipeline) = self.pipelines.get_mut(&source_pipeline_id) {
                    pipeline.url = final_url;
                } else {
                    warn!("constellation got set final url message for dead pipeline");
                }
            },
            FromScriptMsg::PostMessage {
                target: browsing_context_id,
                source: source_pipeline_id,
                target_origin: origin,
                source_origin,
                data,
            } => {
                self.handle_post_message_msg(
                    browsing_context_id,
                    source_pipeline_id,
                    origin,
                    source_origin,
                    data,
                );
            },
            FromScriptMsg::Focus => {
                self.handle_focus_msg(source_pipeline_id);
            },
            FromScriptMsg::SetThrottledComplete(throttled) => {
                self.handle_set_throttled_complete(source_pipeline_id, throttled);
            },
            FromScriptMsg::RemoveIFrame(browsing_context_id, response_sender) => {
                let removed_pipeline_ids = self.handle_remove_iframe_msg(browsing_context_id);
                if let Err(e) = response_sender.send(removed_pipeline_ids) {
                    warn!("Error replying to remove iframe ({})", e);
                }
            },
            FromScriptMsg::CreateCanvasPaintThread(size, response_sender) => {
                self.handle_create_canvas_paint_thread_msg(size, response_sender)
            },
            FromScriptMsg::SetDocumentState(state) => {
                self.document_states.insert(source_pipeline_id, state);
            },
            FromScriptMsg::SetLayoutEpoch(epoch, response_sender) => {
                if let Some(pipeline) = self.pipelines.get_mut(&source_pipeline_id) {
                    pipeline.layout_epoch = epoch;
                }

                response_sender.send(true).unwrap_or_default();
            },
            FromScriptMsg::LogEntry(thread_name, entry) => {
                self.handle_log_entry(Some(source_top_ctx_id), thread_name, entry);
            },
            FromScriptMsg::TouchEventProcessed(result) => self
                .compositor_proxy
                .send(CompositorMsg::TouchEventProcessed(result)),
            FromScriptMsg::GetBrowsingContextInfo(pipeline_id, response_sender) => {
                let result = self
                    .pipelines
                    .get(&pipeline_id)
                    .and_then(|pipeline| self.browsing_contexts.get(&pipeline.browsing_context_id))
                    .map(|ctx| (ctx.id, ctx.parent_pipeline_id));
                if let Err(e) = response_sender.send(result) {
                    warn!(
                        "Sending reply to get browsing context info failed ({:?}).",
                        e
                    );
                }
            },
            FromScriptMsg::GetTopForBrowsingContext(browsing_context_id, response_sender) => {
                let result = self
                    .browsing_contexts
                    .get(&browsing_context_id)
                    .map(|bc| bc.top_level_id);
                if let Err(e) = response_sender.send(result) {
                    warn!(
                        "Sending reply to get top for browsing context info failed ({:?}).",
                        e
                    );
                }
            },
            FromScriptMsg::GetChildBrowsingContextId(
                browsing_context_id,
                index,
                response_sender,
            ) => {
                let result = self
                    .browsing_contexts
                    .get(&browsing_context_id)
                    .and_then(|bc| self.pipelines.get(&bc.pipeline_id))
                    .and_then(|pipeline| pipeline.children.get(index))
                    .copied();
                if let Err(e) = response_sender.send(result) {
                    warn!(
                        "Sending reply to get child browsing context ID failed ({:?}).",
                        e
                    );
                }
            },
            FromScriptMsg::ScheduleJob(job) => {
                self.handle_schedule_serviceworker_job(source_pipeline_id, job);
            },
            FromScriptMsg::ForwardDOMMessage(msg_vec, scope_url) => {
                if let Some(mgr) = self.sw_managers.get(&scope_url.origin()) {
                    let _ = mgr.send(ServiceWorkerMsg::ForwardDOMMessage(msg_vec, scope_url));
                } else {
                    warn!("Unable to forward DOMMessage for postMessage call");
                }
            },
            FromScriptMsg::BroadcastStorageEvent(storage, url, key, old_value, new_value) => {
                self.handle_broadcast_storage_event(
                    source_pipeline_id,
                    storage,
                    url,
                    key,
                    old_value,
                    new_value,
                );
            },
            FromScriptMsg::MediaSessionEvent(pipeline_id, event) => {
                // Unlikely at this point, but we may receive events coming from
                // different media sessions, so we set the active media session based
                // on Playing events.
                // The last media session claiming to be in playing state is set to
                // the active media session.
                // Events coming from inactive media sessions are discarded.
                if self.active_media_session.is_some() {
                    if let MediaSessionEvent::PlaybackStateChange(ref state) = event {
                        if !matches!(
                            state,
                            MediaSessionPlaybackState::Playing | MediaSessionPlaybackState::Paused
                        ) {
                            return;
                        }
                    };
                }
                self.active_media_session = Some(pipeline_id);
                self.embedder_proxy
                    .send(EmbedderMsg::MediaSessionEvent(source_top_ctx_id, event));
            },
            #[cfg(feature = "webgpu")]
            FromScriptMsg::RequestAdapter(response_sender, options, ids) => self
                .handle_wgpu_request(
                    source_pipeline_id,
                    BrowsingContextId::from(source_top_ctx_id),
                    FromScriptMsg::RequestAdapter(response_sender, options, ids),
                ),
            #[cfg(feature = "webgpu")]
            FromScriptMsg::GetWebGPUChan(response_sender) => self.handle_wgpu_request(
                source_pipeline_id,
                BrowsingContextId::from(source_top_ctx_id),
                FromScriptMsg::GetWebGPUChan(response_sender),
            ),
            FromScriptMsg::TitleChanged(pipeline, title) => {
                if let Some(pipeline) = self.pipelines.get_mut(&pipeline) {
                    pipeline.title = title;
                }
            },
            FromScriptMsg::IFrameSizes(iframe_sizes) => self.handle_iframe_size_msg(iframe_sizes),
        }
    }

    /// Checks if the origin of a message matches the origin of its source pipeline.
    /// This is a security check, though limited.
    ///
    /// # Arguments
    /// * `pipeline_id` - The ID of the pipeline from which the message originated.
    /// * `origin` - The `ImmutableOrigin` of the message.
    ///
    /// Pre-condition: `pipeline_id` corresponds to an active pipeline.
    /// Post-condition: Returns `Ok(())` if the origins match, `Err(())` otherwise.
    /// See <https://github.com/servo/servo/issues/11722> for security limitations.
    fn check_origin_against_pipeline(
        &self,
        pipeline_id: &PipelineId,
        origin: &ImmutableOrigin,
    ) -> Result<(), ()> {
        let pipeline_origin = match self.pipelines.get(pipeline_id) {
            Some(pipeline) => pipeline.load_data.url.origin(),
            None => {
                warn!("Received message from closed or unknown pipeline.");
                return Err(());
            },
        };
        if &pipeline_origin == origin {
            return Ok(());
        }
        Err(())
    }

    /// Schedules a broadcast message to be sent to relevant broadcast channel routers.
    ///
    /// # Arguments
    /// * `pipeline_id` - The `PipelineId` from which the broadcast originated.
    /// * `router_id` - The `BroadcastChannelRouterId` of the sender.
    /// * `message` - The `BroadcastMsg` to be sent.
    ///
    /// Pre-condition: The message origin matches the pipeline's origin.
    /// Post-condition: The message is sent to all relevant broadcast routers, excluding the sender.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(servo_profiling = true), level = "trace")
    )]
    fn handle_schedule_broadcast(
        &self,
        pipeline_id: PipelineId,
        router_id: BroadcastChannelRouterId,
        message: BroadcastMsg,
    ) {
        if self
            .check_origin_against_pipeline(&pipeline_id, &message.origin)
            .is_err()
        {
            return warn!(
                "Attempt to schedule broadcast from an origin not matching the origin of the msg."
            );
        }
        if let Some(channels) = self.broadcast_channels.get(&message.origin) {
            let routers = match channels.get(&message.channel_name) {
                Some(routers) => routers,
                None => return warn!("Broadcast to channel name without active routers."),
            };
            for router in routers {
                // Exclude the sender of the broadcast.
                // Broadcasting locally is done at the point of sending.
                if router == &router_id {
                    continue;
                }

                if let Some(broadcast_ipc_sender) = self.broadcast_routers.get(router) {
                    if broadcast_ipc_sender.send(message.clone()).is_err() {
                        warn!("Failed to broadcast message to router: {:?}", router);
                    }
                } else {
                    warn!("No sender for broadcast router: {:?}", router);
                }
            }
        } else {
            warn!(
                "Attempt to schedule a broadcast for an origin without routers {:?}",
                message.origin
            );
        }
    }

    /// Removes a channel name associated with a broadcast router.
    ///
    /// # Arguments
    /// * `pipeline_id` - The `PipelineId` from which the request originated.
    /// * `router_id` - The `BroadcastChannelRouterId` to remove the channel name from.
    /// * `channel_name` - The name of the channel to remove.
    /// * `origin` - The `ImmutableOrigin` of the broadcast channel.
    ///
    /// Pre-condition: The origin matches the pipeline's origin.
    /// Post-condition: The `channel_name` is removed from the specified `router_id`'s channels.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(servo_profiling = true), level = "trace")
    )]
    fn handle_remove_broadcast_channel_name_in_router(
        &mut self,
        pipeline_id: PipelineId,
        router_id: BroadcastChannelRouterId,
        channel_name: String,
        origin: ImmutableOrigin,
    ) {
        if self
            .check_origin_against_pipeline(&pipeline_id, &origin)
            .is_err()
        {
            return warn!("Attempt to remove channel name from an unexpected origin.");
        }
        if let Some(channels) = self.broadcast_channels.get_mut(&origin) {
            let is_empty = if let Some(routers) = channels.get_mut(&channel_name) {
                routers.retain(|router| router != &router_id);
                routers.is_empty()
            } else {
                return warn!(
                    "Multiple attempts to remove name for broadcast-channel {:?} at {:?}",
                    channel_name, origin
                );
            };
            if is_empty {
                channels.remove(&channel_name);
            }
        } else {
            warn!(
                "Attempt to remove a channel-name for an origin without channels {:?}",
                origin
            );
        }
    }

    /// Notes a new channel name relevant to a given broadcast router.
    ///
    /// # Arguments
    /// * `pipeline_id` - The `PipelineId` from which the request originated.
    /// * `router_id` - The `BroadcastChannelRouterId` to associate the channel name with.
    /// * `channel_name` - The new channel name.
    /// * `origin` - The `ImmutableOrigin` of the broadcast channel.
    ///
    /// Pre-condition: The origin matches the pipeline's origin.
    /// Post-condition: The `channel_name` is added to the specified `router_id`'s channels.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(servo_profiling = true), level = "trace")
    )]
    fn handle_new_broadcast_channel_name_in_router(
        &mut self,
        pipeline_id: PipelineId,
        router_id: BroadcastChannelRouterId,
        channel_name: String,
        origin: ImmutableOrigin,
    ) {
        if self
            .check_origin_against_pipeline(&pipeline_id, &origin)
            .is_err()
        {
            return warn!("Attempt to add channel name from an unexpected origin.");
        }
        let channels = self.broadcast_channels.entry(origin).or_default();

        let routers = channels.entry(channel_name).or_default();

        routers.push(router_id);
    }

    /// Removes a broadcast router from the Constellation.
    ///
    /// # Arguments
    /// * `pipeline_id` - The `PipelineId` from which the request originated.
    /// * `router_id` - The `BroadcastChannelRouterId` to remove.
    /// * `origin` - The `ImmutableOrigin` of the broadcast channel.
    ///
    /// Pre-condition: The origin matches the pipeline's origin.
    /// Post-condition: The specified `router_id` and its associated channels are removed.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(servo_profiling = true), level = "trace")
    )]
    fn handle_remove_broadcast_channel_router(
        &mut self,
        pipeline_id: PipelineId,
        router_id: BroadcastChannelRouterId,
        origin: ImmutableOrigin,
    ) {
        if self
            .check_origin_against_pipeline(&pipeline_id, &origin)
            .is_err()
        {
            return warn!("Attempt to remove broadcast router from an unexpected origin.");
        }
        if self.broadcast_routers.remove(&router_id).is_none() {
            warn!("Attempt to remove unknown broadcast-channel router.");
        }
    }

    /// Adds a new broadcast router to the Constellation.
    ///
    /// # Arguments
    /// * `pipeline_id` - The `PipelineId` from which the request originated.
    /// * `router_id` - The `BroadcastChannelRouterId` to add.
    /// * `broadcast_ipc_sender` - The IPC sender for the new router.
    /// * `origin` - The `ImmutableOrigin` of the broadcast channel.
    ///
    /// Pre-condition: The origin matches the pipeline's origin.
    /// Post-condition: The `router_id` and its `ipc_sender` are added to `self.broadcast_routers`.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(servo_profiling = true), level = "trace")
    )]
    fn handle_new_broadcast_channel_router(
        &mut self,
        pipeline_id: PipelineId,
        router_id: BroadcastChannelRouterId,
        broadcast_ipc_sender: IpcSender<BroadcastMsg>,
        origin: ImmutableOrigin,
    ) {
        if self
            .check_origin_against_pipeline(&pipeline_id, &origin)
            .is_err()
        {
            return warn!("Attempt to add broadcast router from an unexpected origin.");
        }
        if self
            .broadcast_routers
            .insert(router_id, broadcast_ipc_sender)
            .is_some()
        {
            warn!("Multple attempt to add broadcast-channel router.");
        }
    }

    /// Handles WebGPU-related requests from script.
    ///
    /// # Arguments
    /// * `source_pipeline_id` - The `PipelineId` from which the request originated.
    /// * `browsing_context_id` - The `BrowsingContextId` associated with the request.
    /// * `request` - The `FromScriptMsg` containing the WebGPU request details.
    ///
    /// Pre-condition: `source_pipeline_id` and `browsing_context_id` are valid, and WebGPU is enabled.
    /// Post-condition: The WebGPU request is processed, potentially creating a new WebGPU channel or sending adapter information.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(servo_profiling = true), level = "trace")
    )]
    #[cfg(feature = "webgpu")]
    fn handle_wgpu_request(
        &mut self,
        source_pipeline_id: PipelineId,
        browsing_context_id: BrowsingContextId,
        request: FromScriptMsg,
    ) {
        let browsing_context_group_id = match self.browsing_contexts.get(&browsing_context_id) {
            Some(bc) => &bc.bc_group_id,
            None => return warn!("Browsing context not found"),
        };
        let source_pipeline = match self.pipelines.get(&source_pipeline_id) {
            Some(pipeline) => pipeline,
            None => return warn!("{}: ScriptMsg from closed pipeline", source_pipeline_id),
        };
        let host = match reg_host(&source_pipeline.url) {
            Some(host) => host,
            None => return warn!("Invalid host url"),
        };
        let browsing_context_group = if let Some(bcg) = self
            .browsing_context_group_set
            .get_mut(browsing_context_group_id)
        {
            bcg
        } else {
            return warn!("Browsing context group not found");
        };
        let webgpu_chan = match browsing_context_group.webgpus.entry(host) {
            Entry::Vacant(v) => WebGPU::new(
                self.webrender_wgpu.webrender_api.create_sender(),
                self.webrender_document,
                self.webrender_wgpu.webrender_external_images.clone(),
                self.webrender_wgpu.wgpu_image_map.clone(),
            )
            .map(|webgpu| {
                let msg = ScriptThreadMessage::SetWebGPUPort(webgpu.1);
                if let Err(e) = source_pipeline.event_loop.send(msg) {
                    warn!(
                        "{}: Failed to send SetWebGPUPort to pipeline ({:?})",
                        source_pipeline_id, e
                    );
                }
                v.insert(webgpu.0).clone()
            }),
            Entry::Occupied(o) => Some(o.get().clone()),
        };
        match request {
            FromScriptMsg::RequestAdapter(response_sender, options, adapter_id) => {
                match webgpu_chan {
                    None => {
                        if let Err(e) = response_sender.send(WebGPUResponse::None) {
                            warn!("Failed to send request adapter message: {}", e)
                        }
                    },
                    Some(webgpu_chan) => {
                        let adapter_request = WebGPURequest::RequestAdapter {
                            sender: response_sender,
                            options,
                            adapter_id,
                        };
                        if webgpu_chan.0.send(adapter_request).is_err() {
                            warn!("Failed to send request adapter message on WebGPU channel");
                        }
                    },
                }
            },
            FromScriptMsg::GetWebGPUChan(response_sender) => {
                if response_sender.send(webgpu_chan).is_err() {