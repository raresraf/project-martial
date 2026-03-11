
/**
 * @file window.rs
 * @brief Implementation of the `Window` object, the main global object in a browser's scripting environment.
 *
 * This module provides the Rust implementation for the `Window` object, which is the central global
 * object in a web browser's scripting environment for a given document. It serves as the primary
 * entry point for scripts to interact with the document, its content, and the browser itself.
 *
 * ## Core Functionality:
 *
 * - **Global Scope:** Acts as the global scope for JavaScript code running in the context of a
 *   document, meaning that all global variables and functions are properties of the `Window` object.
 * - **DOM Access:** Provides access to the document and its elements through properties like `document`,
 *   and methods like `getElementById()`.
 * - **Event Handling:** Implements the `EventTarget` interface, allowing for the dispatch and handling
 *   of events at the window level (e.g., `load`, `resize`, `scroll`).
 * - **Timers:** Provides the `setTimeout()`, `setInterval()`, `requestAnimationFrame()` methods for
 *   scheduling code execution.
 * - **Navigation and History:** Manages the browsing context's session history through the `history`
 *   object and allows for navigation via the `location` object.
 * - **Browser APIs:** Exposes a wide range of browser APIs, including `fetch()`, `postMessage()`,
 *   `getComputedStyle()`, and access to storage mechanisms like `localStorage` and `sessionStorage`.
 * - **Rendering and Layout:** Coordinates with the layout and rendering engines to update the
 *   display when the DOM changes.
 *
 * This implementation is based on numerous web standards, primarily the WHATWG HTML and DOM
 * specifications.
 *
 * @see https://html.spec.whatwg.org/multipage/window-object.html#the-window-object
 */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::borrow::ToOwned;
use std::cell::{Cell, RefCell, RefMut};
use std::cmp;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::io::{Write, stderr, stdout};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use app_units::Au;
use backtrace::Backtrace;
use base::cross_process_instant::CrossProcessInstant;
use base::id::{BrowsingContextId, PipelineId, WebViewId};
use base64::Engine;
#[cfg(feature = "bluetooth")]
use bluetooth_traits::BluetoothRequest;
use canvas_traits::webgl::WebGLChan;
use compositing_traits::CrossProcessCompositorApi;
use constellation_traits::{
    DocumentState, LoadData, LoadOrigin, NavigationHistoryBehavior, ScriptToConstellationChan,
    ScriptToConstellationMessage, ScrollState, StructuredSerializedData, WindowSizeType,
};
use crossbeam_channel::{Sender, unbounded};
use cssparser::{Parser, ParserInput, SourceLocation};
use devtools_traits::{ScriptToDevtoolsControlMsg, TimelineMarker, TimelineMarkerType};
use dom_struct::dom_struct;
use embedder_traits::user_content_manager::{UserContentManager, UserScript};
use embedder_traits::{
    AlertResponse, ConfirmResponse, EmbedderMsg, PromptResponse, SimpleDialog, Theme,
    ViewportDetails, WebDriverJSError, WebDriverJSResult,
};
use euclid::default::{Point2D as UntypedPoint2D, Rect as UntypedRect};
use euclid::{Point2D, Rect, Scale, Size2D, Vector2D};
use fonts::FontContext;
use ipc_channel::ipc::{self, IpcSender};
use js::conversions::ToJSValConvertible;
use js::glue::DumpJSStack;
use js::jsapi::{
    GCReason, Heap, JS_GC, JSAutoRealm, JSContext as RawJSContext, JSObject, JSPROP_ENUMERATE,
};
use js::jsval::{NullValue, UndefinedValue};
use js::rust::wrappers::JS_DefineProperty;
use js::rust::{
    CustomAutoRooter, CustomAutoRooterGuard, HandleObject, HandleValue, MutableHandleObject,
    MutableHandleValue,
};
use malloc_size_of::MallocSizeOf;
use media::WindowGLContext;
use net_traits::ResourceThreads;
use net_traits::image_cache::{
    ImageCache, ImageResponder, ImageResponse, PendingImageId, PendingImageResponse,
};
use net_traits::storage_thread::StorageType;
use num_traits::ToPrimitive;
use profile_traits::ipc as ProfiledIpc;
use profile_traits::mem::ProfilerChan as MemProfilerChan;
use profile_traits::time::ProfilerChan as TimeProfilerChan;
use script_bindings::interfaces::WindowHelpers;
use script_layout_interface::{
    FragmentType, Layout, PendingImageState, QueryMsg, Reflow, ReflowGoal, ReflowRequest,
    TrustedNodeAddress, combine_id_with_fragment_type,
};
use script_traits::ScriptThreadMessage;
use selectors::attr::CaseSensitivity;
use servo_arc::Arc as ServoArc;
use servo_config::{opts, pref};
use servo_geometry::{DeviceIndependentIntRect, MaxRect, f32_rect_to_au_rect};
use servo_url::{ImmutableOrigin, MutableOrigin, ServoUrl};
use style::dom::OpaqueNode;
use style::error_reporting::{ContextualParseError, ParseErrorReporter};
use style::media_queries;
use style::parser::ParserContext as CssParserContext;
use style::properties::PropertyId;
use style::properties::style_structs::Font;
use style::queries::values::PrefersColorScheme;
use style::selector_parser::PseudoElement;
use style::str::HTML_SPACE_CHARACTERS;
use style::stylesheets::{CssRuleType, Origin, UrlExtraData};
use style_traits::{CSSPixel, ParsingMode};
use stylo_atoms::Atom;
use url::Position;
use webrender_api::units::{DevicePixel, LayoutPixel};
use webrender_api::{DocumentId, ExternalScrollId};

use super::bindings::codegen::Bindings::MessagePortBinding::StructuredSerializeOptions;
use super::bindings::trace::HashMapTracedValues;
use crate::dom::bindings::cell::{DomRefCell, Ref};
use crate::dom::bindings::codegen::Bindings::DocumentBinding::{
    DocumentMethods, DocumentReadyState, NamedPropertyValue,
};
use crate::dom::bindings::codegen::Bindings::HTMLIFrameElementBinding::HTMLIFrameElementMethods;
use crate::dom::bindings::codegen::Bindings::HistoryBinding::History_Binding::HistoryMethods;
use crate::dom::bindings::codegen::Bindings::ImageBitmapBinding::{
    ImageBitmapOptions, ImageBitmapSource,
};
use crate::dom::bindings::codegen::Bindings::MediaQueryListBinding::MediaQueryList_Binding::MediaQueryListMethods;
use crate::dom::bindings::codegen::Bindings::RequestBinding::RequestInit;
use crate::dom::bindings::codegen::Bindings::VoidFunctionBinding::VoidFunction;
use crate::dom::bindings::codegen::Bindings::WindowBinding::{
    self, FrameRequestCallback, ScrollBehavior, ScrollToOptions, WindowMethods,
    WindowPostMessageOptions,
};
use crate::dom::bindings::codegen::UnionTypes::{RequestOrUSVString, StringOrFunction};
use crate::dom::bindings::error::{Error, ErrorResult, Fallible};
use crate::dom::bindings::inheritance::{Castable, ElementTypeId, HTMLElementTypeId, NodeTypeId};
use crate::dom::bindings::num::Finite;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::{DomGlobal, DomObject};
use crate::dom::bindings::root::{Dom, DomRoot, MutNullableDom};
use crate::dom::bindings::str::{DOMString, USVString};
use crate::dom::bindings::structuredclone;
use crate::dom::bindings::trace::{CustomTraceable, JSTraceable, RootedTraceableBox};
use crate::dom::bindings::utils::GlobalStaticData;
use crate::dom::bindings::weakref::DOMTracker;
#[cfg(feature = "bluetooth")]
use crate::dom::bluetooth::BluetoothExtraPermissionData;
use crate::dom::crypto::Crypto;
use crate::dom::cssstyledeclaration::{CSSModificationAccess, CSSStyleDeclaration, CSSStyleOwner};
use crate::dom::customelementregistry::CustomElementRegistry;
use crate::dom::document::{AnimationFrameCallback, Document, ReflowTriggerCondition};
use crate::dom::element::Element;
use crate::dom::event::{Event, EventBubbles, EventCancelable, EventStatus};
use crate::dom::eventtarget::EventTarget;
use crate::dom::globalscope::GlobalScope;
use crate::dom::hashchangeevent::HashChangeEvent;
use crate::dom::history::History;
use crate::dom::htmlcollection::{CollectionFilter, HTMLCollection};
use crate::dom::htmliframeelement::HTMLIFrameElement;
use crate::dom::location::Location;
use crate::dom::mediaquerylist::{MediaQueryList, MediaQueryListMatchState};
use crate::dom::mediaquerylistevent::MediaQueryListEvent;
use crate::dom::messageevent::MessageEvent;
use crate::dom::navigator::Navigator;
use crate::dom::node::{Node, NodeDamage, NodeTraits, from_untrusted_node_address};
use crate::dom::performance::Performance;
use crate::dom::promise::Promise;
use crate::dom::screen::Screen;
use crate::dom::selection::Selection;
use crate::dom::storage::Storage;
#[cfg(feature = "bluetooth")]
use crate::dom::testrunner::TestRunner;
use crate::dom::trustedtypepolicyfactory::TrustedTypePolicyFactory;
use crate::dom::types::UIEvent;
use crate::dom::webglrenderingcontext::WebGLCommandSender;
#[cfg(feature = "webgpu")]
use crate::dom::webgpu::identityhub::IdentityHub;
use crate::dom::windowproxy::{WindowProxy, WindowProxyHandler};
use crate::dom::worklet::Worklet;
use crate::dom::workletglobalscope::WorkletGlobalScopeType;
use crate::layout_image::fetch_image_for_layout;
use crate::messaging::{MainThreadScriptMsg, ScriptEventLoopReceiver, ScriptEventLoopSender};
use crate::microtask::MicrotaskQueue;
use crate::realms::{InRealm, enter_realm};
use crate::script_runtime::{CanGc, JSContext, Runtime};
use crate::script_thread::ScriptThread;
use crate::timers::{IsInterval, TimerCallback};
use crate::unminify::unminified_path;
use crate::webdriver_handlers::jsval_to_webdriver;
use crate::{fetch, window_named_properties};

/// A callback to call when a response comes back from the `ImageCache`.
///
/// This is wrapped in a struct so that we can implement `MallocSizeOf`
/// for this type.
#[derive(MallocSizeOf)]
pub struct PendingImageCallback(
    #[ignore_malloc_size_of = "dyn Fn is currently impossible to measure"]
    Box<dyn Fn(PendingImageResponse) + 'static>,
);

/// Current state of the window object
#[derive(Clone, Copy, Debug, JSTraceable, MallocSizeOf, PartialEq)]
enum WindowState {
    Alive,
    Zombie, // Pipeline is closed, but the window hasn't been GCed yet.
}

/// How long we should wait before performing the initial reflow after `<body>` is parsed,
/// assuming that `<body>` take this long to parse.
const INITIAL_REFLOW_DELAY: Duration = Duration::from_millis(200);

/// During loading and parsing, layouts are suppressed to avoid flashing incomplete page
/// contents.
///
/// Exceptions:
///  - Parsing the body takes so long, that layouts are no longer suppressed in order
///    to show the user that the page is loading.
///  - Script triggers a layout query or scroll event in which case, we want to layout
///    but not display the contents.
///
/// For more information see: <https://github.com/servo/servo/pull/6028>.
#[derive(Clone, Copy, MallocSizeOf)]
enum LayoutBlocker {
    /// The first load event hasn't been fired and we have not started to parse the `<body>` yet.
    WaitingForParse,
    /// The body is being parsed the `<body>` starting at the `Instant` specified.
    Parsing(Instant),
    /// The body finished parsing and the `load` event has been fired or parsing took so
    /// long, that we are going to do layout anyway. Note that subsequent changes to the body
    /// can trigger parsing again, but the `Window` stays in this state.
    FiredLoadEventOrParsingTimerExpired,
}

impl LayoutBlocker {
    fn layout_blocked(&self) -> bool {
        !matches!(self, Self::FiredLoadEventOrParsingTimerExpired)
    }
}

/**
 * @brief Represents the `Window` object, the global scope for a document.
 *
 * This struct holds the state for a `Window` object, including its document, history, location,
 * and all the other properties and methods that are part of the `Window` interface.
 */
#[dom_struct]
pub(crate) struct Window {
    globalscope: GlobalScope,
    /// The webview that contains this [`Window`].
    ///
    /// This may not be the top-level [`Window`], in the case of frames.
    #[no_trace]
    webview_id: WebViewId,
    script_chan: Sender<MainThreadScriptMsg>,
    #[no_trace]
    #[ignore_malloc_size_of = "TODO: Add MallocSizeOf support to layout"]
    layout: RefCell<Box<dyn Layout>>,
    /// A [`FontContext`] which is used to store and match against fonts for this `Window` and to
    /// trigger the download of web fonts.
    #[no_trace]
    #[conditional_malloc_size_of]
    font_context: Arc<FontContext>,
    navigator: MutNullableDom<Navigator>,
    #[ignore_malloc_size_of = "Arc"]
    #[no_trace]
    image_cache: Arc<dyn ImageCache>,
    #[no_trace]
    image_cache_sender: IpcSender<PendingImageResponse>,
    window_proxy: MutNullableDom<WindowProxy>,
    document: MutNullableDom<Document>,
    location: MutNullableDom<Location>,
    history: MutNullableDom<History>,
    custom_element_registry: MutNullableDom<CustomElementRegistry>,
    performance: MutNullableDom<Performance>,
    #[no_trace]
    navigation_start: Cell<CrossProcessInstant>,
    screen: MutNullableDom<Screen>,
    session_storage: MutNullableDom<Storage>,
    local_storage: MutNullableDom<Storage>,
    status: DomRefCell<DOMString>,
    trusted_types: MutNullableDom<TrustedTypePolicyFactory>,

    /// For sending timeline markers. Will be ignored if
    /// no devtools server
    #[no_trace]
    devtools_markers: DomRefCell<HashSet<TimelineMarkerType>>,
    #[no_trace]
    devtools_marker_sender: DomRefCell<Option<IpcSender<Option<TimelineMarker>>>>,

    /// Most recent unhandled resize event, if any.
    #[no_trace]
    unhandled_resize_event: DomRefCell<Option<(ViewportDetails, WindowSizeType)>>,

    /// Platform theme.
    #[no_trace]
    theme: Cell<PrefersColorScheme>,

    /// Parent id associated with this page, if any.
    #[no_trace]
    parent_info: Option<PipelineId>,

    /// Global static data related to the DOM.
    dom_static: GlobalStaticData,

    /// The JavaScript runtime.
    #[ignore_malloc_size_of = "Rc<T> is hard"]
    js_runtime: DomRefCell<Option<Rc<Runtime>>>,

    /// The [`ViewportDetails`] of this [`Window`]'s frame.
    #[no_trace]
    viewport_details: Cell<ViewportDetails>,

    /// A handle for communicating messages to the bluetooth thread.
    #[no_trace]
    #[cfg(feature = "bluetooth")]
    bluetooth_thread: IpcSender<BluetoothRequest>,

    #[cfg(feature = "bluetooth")]
    bluetooth_extra_permission_data: BluetoothExtraPermissionData,

    /// An enlarged rectangle around the page contents visible in the viewport, used
    /// to prevent creating display list items for content that is far away from the viewport.
    #[no_trace]
    page_clip_rect: Cell<UntypedRect<Au>>,

    /// See the documentation for [`LayoutBlocker`]. Essentially, this flag prevents
    /// layouts from happening before the first load event, apart from a few exceptional
    /// cases.
    #[no_trace]
    layout_blocker: Cell<LayoutBlocker>,

    /// A channel for communicating results of async scripts back to the webdriver server
    #[no_trace]
    webdriver_script_chan: DomRefCell<Option<IpcSender<WebDriverJSResult>>>,

    /// The current state of the window object
    current_state: Cell<WindowState>,

    #[no_trace]
    current_viewport: Cell<UntypedRect<Au>>,

    error_reporter: CSSErrorReporter,

    /// A list of scroll offsets for each scrollable element.
    #[no_trace]
    scroll_offsets: DomRefCell<HashMap<OpaqueNode, Vector2D<f32, LayoutPixel>>>,

    /// All the MediaQueryLists we need to update
    media_query_lists: DOMTracker<MediaQueryList>,

    #[cfg(feature = "bluetooth")]
    test_runner: MutNullableDom<TestRunner>,

    /// A handle for communicating messages to the WebGL thread, if available.
    #[ignore_malloc_size_of = "channels are hard"]
    #[no_trace]
    webgl_chan: Option<WebGLChan>,

    #[ignore_malloc_size_of = "defined in webxr"]
    #[no_trace]
    #[cfg(feature = "webxr")]
    webxr_registry: Option<webxr_api::Registry>,

    /// When an element triggers an image load or starts watching an image load from the
    /// `ImageCache` it adds an entry to this list. When those loads are triggered from
    /// layout, they also add an etry to [`Self::pending_layout_images`].
    #[no_trace]
    pending_image_callbacks: DomRefCell<HashMap<PendingImageId, Vec<PendingImageCallback>>>,

    /// All of the elements that have an outstanding image request that was
    /// initiated by layout during a reflow. They are stored in the script thread
    /// to ensure that the element can be marked dirty when the image data becomes
    /// available at some point in the future.
    pending_layout_images: DomRefCell<HashMapTracedValues<PendingImageId, Vec<Dom<Node>>>>,

    /// Directory to store unminified css for this window if unminify-css
    /// opt is enabled.
    unminified_css_dir: DomRefCell<Option<String>>,

    /// Directory with stored unminified scripts
    local_script_source: Option<String>,

    /// Worklets
    test_worklet: MutNullableDom<Worklet>,
    /// <https://drafts.css-houdini.org/css-paint-api-1/#paint-worklet>
    paint_worklet: MutNullableDom<Worklet>,
    /// The Webrender Document id associated with this window.
    #[ignore_malloc_size_of = "defined in webrender_api"]
    #[no_trace]
    webrender_document: DocumentId,

    /// Flag to identify whether mutation observers are present(true)/absent(false)
    exists_mut_observer: Cell<bool>,

    /// Cross-process access to the compositor.
    #[ignore_malloc_size_of = "Wraps an IpcSender"]
    #[no_trace]
    compositor_api: CrossProcessCompositorApi,

    /// Indicate whether a SetDocumentStatus message has been sent after a reflow is complete.
    /// It is used to avoid sending idle message more than once, which is unneccessary.
    has_sent_idle_message: Cell<bool>,

    /// Emits notifications when there is a relayout.
    relayout_event: bool,

    /// Unminify Css.
    unminify_css: bool,

    /// User content manager
    #[no_trace]
    user_content_manager: UserContentManager,

    /// Window's GL context from application
    #[ignore_malloc_size_of = "defined in script_thread"]
    #[no_trace]
    player_context: WindowGLContext,

    throttled: Cell<bool>,

    /// A shared marker for the validity of any cached layout values. A value of true
    /// indicates that any such values remain valid; any new layout that invalidates
    /// those values will cause the marker to be set to false.
    #[ignore_malloc_size_of = "Rc is hard"]
    layout_marker: DomRefCell<Rc<Cell<bool>>>,

    /// <https://dom.spec.whatwg.org/#window-current-event>
    current_event: DomRefCell<Option<Dom<Event>>>,
}

impl Window {
    pub(crate) fn webview_id(&self) -> WebViewId {
        self.webview_id
    }

    pub(crate) fn as_global_scope(&self) -> &GlobalScope {
        self.upcast::<GlobalScope>()
    }

    pub(crate) fn layout(&self) -> Ref<Box<dyn Layout>> {
        self.layout.borrow()
    }
...
        Self::create_named_properties_object(cx, proto, object)
    }
}
