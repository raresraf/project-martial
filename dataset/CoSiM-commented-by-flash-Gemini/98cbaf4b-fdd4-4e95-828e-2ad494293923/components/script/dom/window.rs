/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! @3f57f4bb-7c2b-4797-ab43-b75f5f704df1/components/script/dom/window.rs
//! @brief Main browsing context representation in Servo.
//!
//! This module defines the `Window` struct, which serves as the core
//! representation of a web browsing context within Servo. It encapsulates
//! a wide range of functionalities related to the Document Object Model (DOM),
//! event handling, layout management, and interactions with the embedding
//! application and rendering engine.
//!
//! The `Window` object is central to managing web page state, executing scripts,
//! handling user input, and coordinating the display of content. It integrates
//! various sub-components for networking, styling, graphics, and inter-process
//! communication to provide a comprehensive browsing environment.

// Block Logic: Standard library imports for fundamental data structures, concurrency, and I/O.
// Functional Utility: Provides core Rust language features such as smart pointers, collections,
//                     concurrency primitives, and basic input/output operations.
use std::borrow::{Cow, ToOwned};
use std::cell::{Cell, RefCell, RefMut};
use std::cmp;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::io::{stderr, stdout, Write};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Block Logic: External crate imports providing specialized functionalities.
// Functional Utility: Incorporates various third-party libraries for tasks such as
//                     unit management, stack tracing, base utilities, encoding,
//                     Bluetooth communication, canvas operations, inter-thread
//                     communication, CSS parsing, devtools integration, DOM structure
//                     definition, embedder interactions, 2D geometry, font handling,
//                     IPC, JavaScript engine interaction, memory profiling, media,
//                     network traits, numeric conversions, and performance profiling.
use app_units::Au;
use backtrace::Backtrace;
use base::cross_process_instant::CrossProcessInstant;
use base::id::{BrowsingContextId, PipelineId, WebViewId};
use base64::Engine;
#[cfg(feature = "bluetooth")]
use bluetooth_traits::BluetoothRequest;
use canvas_traits::webgl::WebGLChan;
use crossbeam_channel::{unbounded, Sender};
use cssparser::{Parser, ParserInput, SourceLocation};
use devtools_traits::{ScriptToDevtoolsControlMsg, TimelineMarker, TimelineMarkerType};
use dom_struct::dom_struct;
use embedder_traits::{EmbedderMsg, PromptDefinition, PromptOrigin, PromptResult, Theme};
use euclid::default::{Point2D as UntypedPoint2D, Rect as UntypedRect};
use euclid::{Point2D, Rect, Scale, Size2D, Vector2D};
use fonts::FontContext;
use ipc_channel::ipc::{self, IpcSender};
use js::conversions::ToJSValConvertible;
use js::glue::DumpJSStack;
use js::jsapi::{
    GCReason, Heap, JSAutoRealm, JSContext as RawJSContext, JSObject, JSPROP_ENUMERATE, JS_GC,
};
use js::jsval::{NullValue, UndefinedValue};
use js::rust::wrappers::JS_DefineProperty;
use js::rust::{
    CustomAutoRooter, CustomAutoRooterGuard, HandleObject, HandleValue, MutableHandleObject,
    MutableHandleValue,
};
use malloc_size_of::MallocSizeOf;
use media::WindowGLContext;
use net_traits::image_cache::{
    ImageCache, ImageResponder, ImageResponse, PendingImageId, PendingImageResponse,
};
use net_traits::request::{RequestId, Referrer};
use net_traits::response::ResponseInit;
use net_traits::storage_thread::StorageType;
use net_traits::ResourceThreads;
use num_traits::ToPrimitive;
use profile_traits::ipc as ProfiledIpc;
use profile_traits::mem::ProfilerChan as MemProfilerChan;
use profile_traits::time::ProfilerChan as TimeProfilerChan;
use script_layout_interface::{
    combine_id_with_fragment_type, FragmentType, Layout, PendingImageState, QueryMsg, Reflow,
    ReflowGoal, ReflowRequest, TrustedNodeAddress,
};
use script_traits::webdriver_msg::{WebDriverJSError, WebDriverJSResult};
use script_traits::{
    DocumentState, LoadData, LoadOrigin, NavigationHistoryBehavior, ScriptMsg, ScriptThreadMessage,
    ScriptToConstellationChan, ScrollState, StructuredSerializedData, WindowSizeData,
    WindowSizeType,
};
use selectors::attr::CaseSensitivity;
use servo_arc::Arc as ServoArc;
use servo_atoms::Atom;
use servo_config::{opts, pref};
use servo_geometry::{f32_rect_to_au_rect, DeviceIndependentIntRect, MaxRect};
use servo_url::{ImmutableOrigin, MutableOrigin, ServoUrl};
use style::dom::OpaqueNode;
use style::error_reporting::{ContextualParseError, ParseErrorReporter};
use style::media_queries;
use style::parser::ParserContext as CssParserContext;
use style::properties::style_structs::Font;
use style::properties::PropertyId;
use style::queries::values::PrefersColorScheme;
use style::selector_parser::PseudoElement;
use style::str::HTML_SPACE_CHARACTERS;
use style::stylesheets::{CssRuleType, Origin, UrlExtraData};
use style_traits::{CSSPixel, ParsingMode};
use url::Position;
use webrender_api::units::{DevicePixel, LayoutPixel};
use webrender_api::{DocumentId, ExternalScrollId};
use webrender_traits::CrossProcessCompositorApi;

// Block Logic: Internal crate imports for DOM bindings, core DOM elements, and script-related utilities.
// Functional Utility: Provides definitions and functionalities for various DOM components,
//                     event handling, styling, and inter-component communication within Servo.
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
    self, FrameRequestCallback, ScrollBehavior, ScrollToOptions, WindowPostMessageOptions,
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
use crate::dom::node::{from_untrusted_node_address, Node, NodeDamage, NodeTraits};
use crate::dom::performance::Performance;
use crate::dom::promise::Promise;
use crate::dom::screen::Screen;
use crate::dom::selection::Selection;
use crate::dom::storage::Storage;
#[cfg(feature = "bluetooth")]
use crate::dom::testrunner::TestRunner;
use crate::dom::types::UIEvent;
use crate::dom::webglrenderingcontext::WebGLCommandSender;
#[cfg(feature = "webgpu")]
use crate::dom::webgpu::identityhub::IdentityHub;
use crate::dom::window::Window;
use crate::dom::windowproxy::{WindowProxy, WindowProxyHandler};
use crate::dom::worklet::Worklet;
use crate::dom::workletglobalscope::WorkletGlobalScopeType;
use crate::layout_image::fetch_image_for_layout;
use crate::messaging::{MainThreadScriptMsg, ScriptEventLoopReceiver, ScriptEventLoopSender};
use crate::microtask::MicrotaskQueue;
use crate::realms::{enter_realm, InRealm};
use crate::script_runtime::{CanGc, JSContext, Runtime};
use crate::script_thread::ScriptThread;
use crate::timers::{IsInterval, TimerCallback};
use crate::unminify::unminified_path;
use crate::webdriver_handlers::jsval_to_webdriver;
use crate::{fetch, window_named_properties};

//! @brief A callback to call when a response comes back from the `ImageCache`.
//!
//! Functional Utility: This struct wraps a closure that is executed when an image
//!                     response is received from the `ImageCache`. It allows for
//!                     deferred processing of image load events, ensuring that
//!                     dependent operations, such as marking nodes dirty for relayout,
//!                     occur once image data becomes available.
//!
//! Implementation Detail: It is wrapped in a struct to enable the implementation of
//!                        `MallocSizeOf` for this type, which is crucial for memory
//!                        profiling within Servo. The inner `dyn Fn` is ignored
//!                        for `MallocSizeOf` due to the difficulty of measuring
//!                        its size dynamically.
#[derive(MallocSizeOf)]
pub struct PendingImageCallback(
    #[ignore_malloc_size_of = "dyn Fn is currently impossible to measure"]
    Box<dyn Fn(PendingImageResponse) + 'static>,
);

//! @brief Current state of the window object.
//!
//! Functional Utility: This enum tracks the lifecycle state of a `Window` object,
//!                     distinguishing between an actively functioning window and
//!                     one that has been closed but not yet garbage collected.
#[derive(Clone, Copy, Debug, JSTraceable, MallocSizeOf, PartialEq)]
enum WindowState {
    /// The window is active and fully functional.
    Alive,
    /// The window's pipeline is closed, but the JavaScript `Window` object
    /// has not yet been garbage collected. This state indicates a pending
    /// deallocation.
    Zombie, // Pipeline is closed, but the window hasn't been GCed yet.
}

//! @brief Delay for initial reflow after `<body>` parsing.
//!
//! Functional Utility: Defines the duration to wait before triggering the initial
//!                     reflow of the page once the `<body>` element has been parsed.
//!                     This delay aims to allow sufficient content to load, preventing
//!                     premature and potentially disruptive reflows that could lead
//!                     to a "flash of unstyled content" or an incomplete rendering
//!                     experience for the user. It is a heuristic to balance responsiveness
//!                     with visual stability during initial page load.
const INITIAL_REFLOW_DELAY: Duration = Duration::from_millis(200);

//! @brief Controls layout suppression during initial page loading.
//!
//! Functional Utility: This enum manages the state of layout blocking to prevent
//!                     premature or incomplete rendering updates during the initial
//!                     loading and parsing phases of a web page. It defines specific
//!                     conditions under which layout operations are suppressed, with
//!                     exceptions for long parsing times or script-initiated layout
//!                     queries/scroll events.
//!
//! For more information see: <https://github.com/servo/servo/pull/6028>.
#[derive(Clone, Copy, MallocSizeOf)]
enum LayoutBlocker {
    /// The first `load` event has not yet been fired, and the `<body>` element
    /// has not started parsing. Layouts are fully suppressed in this state.
    WaitingForParse,
    /// The `<body>` element is currently being parsed, starting at the specified
    /// `Instant`. Layouts are generally suppressed, but this state allows for
    /// exceptions if parsing takes an unusually long time.
    Parsing(Instant),
    /// The `<body>` element has finished parsing, and the `load` event has either
    /// been fired, or the parsing process took so long that layout suppression
    /// is no longer in effect. Subsequent changes to the body might re-trigger
    /// parsing, but the `Window` remains in this state for layout purposes.
    FiredLoadEventOrParsingTimerExpired,
}

impl LayoutBlocker {
    //! @brief Determines if layout is currently blocked.
    //!
    //! Functional Utility: This method provides a boolean indication of whether
    //!                     layout operations are currently being suppressed based
    //!                     on the `LayoutBlocker`'s current state. Layout is
    //!                     considered blocked in `WaitingForParse` and `Parsing` states.
    //!
    //! @return `true` if layout is blocked, `false` otherwise.
    fn layout_blocked(&self) -> bool {
        !matches!(self, Self::FiredLoadEventOrParsingTimerExpired)
    }
}

#[dom_struct]
pub(crate) struct Window {
    /// @brief The global scope associated with this window.
    ///
    /// Functional Utility: Manages global properties, functions, and object lifetimes
    ///                     for the JavaScript execution environment within this window.
    globalscope: GlobalScope,
    /// @brief Identifier for the WebView containing this window.
    ///
    /// Functional Utility: Uniquely identifies the rendering surface or browsing context
    ///                     to which this `Window` instance belongs. This is crucial for
    ///                     inter-process communication and coordinating rendering updates,
    ///                     especially when dealing with multiple tabs or frames.
    ///                     It may not be the top-level `Window` in the case of iframes.
    #[no_trace]
    webview_id: WebViewId,
    /// @brief Sender for messages to the main thread's script event loop.
    ///
    /// Functional Utility: Facilitates asynchronous communication from this `Window`'s
    ///                     script thread back to the main thread, allowing for actions
    ///                     like updating the UI, handling network responses, or managing
    ///                     system-level events.
    script_chan: Sender<MainThreadScriptMsg>,
    /// @brief The layout engine instance for this window.
    ///
    /// Functional Utility: Manages the layout computation and rendering tree for the
    ///                     document loaded within this window. It determines the
    ///                     position and size of all elements on the page.
    #[no_trace]
    #[ignore_malloc_size_of = "TODO: Add MallocSizeOf support to layout"]
    layout: RefCell<Box<dyn Layout>>,
    /// @brief Font context for this window.
    ///
    /// Functional Utility: Stores and manages fonts used within the window's document,
    ///                     including triggering the download of web fonts. It ensures
    ///                     consistent font rendering across the page.
    #[no_trace]
    #[conditional_malloc_size_of]
    font_context: Arc<FontContext>,
    /// @brief The `Navigator` object for this window.
    ///
    /// Functional Utility: Provides information about the user agent and the state
    ///                     of the browser, including details about the operating system,
    ///                     browser version, and network connection.
    navigator: MutNullableDom<Navigator>,
    /// @brief Image cache for this window.
    ///
    /// Functional Utility: Caches decoded image data to reduce redundant network requests
    ///                     and improve rendering performance. It manages the lifecycle
    ///                     of images loaded by the document.
    #[ignore_malloc_size_of = "Arc"]
    #[no_trace]
    image_cache: Arc<dyn ImageCache>,
    /// @brief Sender for image cache responses.
    ///
    /// Functional Utility: Used to send responses back from the image cache, indicating
    ///                     the status of image loading (e.g., loaded, placeholder, error).
    #[no_trace]
    image_cache_sender: IpcSender<PendingImageResponse>,
    /// @brief Proxy object for this window.
    ///
    /// Functional Utility: Represents the `Window` object in other browsing contexts
    ///                     (e.g., iframes, pop-ups), enabling safe cross-origin
    ///                     communication and access control.
    window_proxy: MutNullableDom<WindowProxy>,
    /// @brief The `Document` object loaded in this window.
    ///
    /// Functional Utility: Represents the web page content, providing the root of the
    ///                     DOM tree and acting as the entry point for all content-related
    ///                     operations and manipulations.
    document: MutNullableDom<Document>,
    /// @brief The `Location` object for this window.
    ///
    /// Functional Utility: Manages the current URL of the window and provides methods
    ///                     for navigating to new URLs.
    location: MutNullableDom<Location>,
    /// @brief The `History` object for this window.
    ///
    /// Functional Utility: Manages the browser's session history for this window,
    ///                     allowing for navigation backward and forward through visited pages.
    history: MutNullableDom<History>,
    /// @brief The `CustomElementRegistry` for this window.
    ///
    /// Functional Utility: Registers and manages custom elements defined within the
    ///                     document, enabling the creation of reusable, encapsulated
    ///                     HTML tags.
    custom_element_registry: MutNullableDom<CustomElementRegistry>,
    /// @brief The `Performance` object for this window.
    ///
    /// Functional Utility: Provides access to performance-related information for the
    ///                     current document, such as navigation timing, resource timing,
    ///                     and user timing marks.
    performance: MutNullableDom<Performance>,
    /// @brief Timestamp when navigation to the current document started.
    ///
    /// Functional Utility: Records the beginning of the navigation process, serving as
    ///                     a reference point for performance measurements.
    #[no_trace]
    navigation_start: Cell<CrossProcessInstant>,
    /// @brief The `Screen` object for this window.
    ///
    /// Functional Utility: Provides information about the user's screen, such as its
    ///                     dimensions, color depth, and pixel density.
    screen: MutNullableDom<Screen>,
    /// @brief The `sessionStorage` object for this window.
    ///
    /// Functional Utility: Provides a storage mechanism for key-value pairs that are
    ///                     persisted for the duration of the top-level browsing context's session.
    session_storage: MutNullableDom<Storage>,
    /// @brief The `localStorage` object for this window.
    ///
    /// Functional Utility: Provides a storage mechanism for key-value pairs that are
    ///                     persisted across browser sessions and tabs.
    local_storage: MutNullableDom<Storage>,
    /// @brief The status bar message for this window.
    ///
    /// Functional Utility: Holds the text displayed in the browser's status bar,
    ///                     often used to show link URLs or application messages.
    status: DomRefCell<DOMString>,

    /// @brief Set of active devtools timeline markers.
    ///
    /// Functional Utility: Tracks which types of timeline markers are currently
    ///                     enabled for emission to devtools, allowing for selective
    ///                     profiling and debugging.
    #[no_trace]
    devtools_markers: DomRefCell<HashSet<TimelineMarkerType>>,
    /// @brief Sender for devtools timeline markers.
    ///
    /// Functional Utility: Used to send `TimelineMarker` events to a connected devtools
    ///                     server, providing detailed timing and event information
    ///                     for debugging and performance analysis.
    #[no_trace]
    devtools_marker_sender: DomRefCell<Option<IpcSender<Option<TimelineMarker>>>>,

    /// @brief Most recent unhandled resize event data.
    ///
    /// Functional Utility: Stores information about a window resize event that has occurred
    ///                     but not yet been processed by the layout engine or script,
    ///                     ensuring that resize events are eventually handled.
    #[no_trace]
    unhandled_resize_event: DomRefCell<Option<(WindowSizeData, WindowSizeType)>>,

    /// @brief The current platform theme preference.
    ///
    /// Functional Utility: Reflects the user's preferred color scheme (e.g., light or dark mode),
    ///                     allowing web content to adapt its appearance accordingly.
    #[no_trace]
    theme: Cell<PrefersColorScheme>,

    /// @brief Information about the parent pipeline, if this is an iframe.
    ///
    /// Functional Utility: Stores the `PipelineId` of the parent browsing context,
    ///                     facilitating communication and hierarchical relationships
    ///                     between nested browsing contexts.
    #[no_trace]
    parent_info: Option<PipelineId>,

    /// @brief Global static data related to the DOM.
    ///
    /// Functional Utility: Provides access to static, globally accessible data
    ///                     structures and configurations pertinent to the DOM.
    dom_static: GlobalStaticData,

    /// @brief The JavaScript runtime associated with this window.
    ///
    /// Functional Utility: Manages the JavaScript execution environment, including
    ///                     the JavaScript context, heap, and garbage collection.
    #[ignore_malloc_size_of = "Rc<T> is hard"]
    js_runtime: DomRefCell<Option<Rc<Runtime>>>,

    /// @brief The current size of the window in device pixels.
    ///
    /// Functional Utility: Stores the current dimensions of the window's viewport,
    ///                     used for layout calculations and responsive design.
    #[no_trace]
    window_size: Cell<WindowSizeData>,

    /// @brief Handle for communicating messages to the Bluetooth thread.
    ///
    /// Functional Utility: Enables the `Window` to send requests and receive responses
    ///                     from the Bluetooth service, supporting Web Bluetooth APIs.
    #[no_trace]
    #[cfg(feature = "bluetooth")]
    bluetooth_thread: IpcSender<BluetoothRequest>,

    /// @brief Extra permission data for Bluetooth operations.
    ///
    /// Functional Utility: Stores additional context or permissions required for
    ///                     Bluetooth-related API calls, enhancing security and
    ///                     user privacy.
    #[cfg(feature = "bluetooth")]
    bluetooth_extra_permission_data: BluetoothExtraPermissionData,

    /// @brief Rectangular region clipping the page contents.
    ///
    /// Functional Utility: Defines an enlarged area around the visible viewport
    ///                     to optimize rendering by preventing the creation of
    ///                     display list items for content far outside this region.
    #[no_trace]
    page_clip_rect: Cell<UntypedRect<Au>>,

    /// @brief Flag to suppress layouts during initial loading.
    ///
    /// Functional Utility: Controls when layout operations are allowed to proceed,
    ///                     preventing premature rendering updates before critical
    ///                     page content has loaded. See `LayoutBlocker` for details.
    #[no_trace]
    layout_blocker: Cell<LayoutBlocker>,

    /// @brief Channel for sending async script results to the WebDriver server.
    ///
    /// Functional Utility: Provides a communication channel to report the outcomes
    ///                     of asynchronous JavaScript execution to a WebDriver
    ///                     instance, crucial for automated testing.
    #[no_trace]
    webdriver_script_chan: DomRefCell<Option<IpcSender<WebDriverJSResult>>>,

    /// @brief The current state of the window object.
    ///
    /// Functional Utility: Tracks the lifecycle state of the `Window` object,
    ///                     indicating whether it is active or has been closed.
    current_state: Cell<WindowState>,

    /// @brief The current viewport rectangle.
    ///
    /// Functional Utility: Stores the current visible area of the document,
    ///                     including its position and dimensions, in `Au` units.
    #[no_trace]
    current_viewport: Cell<UntypedRect<Au>>,

    /// @brief CSS error reporter for this window.
    ///
    /// Functional Utility: Handles and reports parsing errors encountered during
    ///                     the processing of CSS stylesheets, aiding in debugging
    ///                     and development.
    error_reporter: CSSErrorReporter,

    /// @brief List of scroll offsets for scrollable elements.
    ///
    /// Functional Utility: Maintains a map of `OpaqueNode` identifiers to their
    ///                     corresponding scroll offsets, enabling precise control
    ///                     and tracking of scrolling behavior for individual elements.
    #[no_trace]
    scroll_offsets: DomRefCell<HashMap<OpaqueNode, Vector2D<f32, LayoutPixel>>>,

    /// @brief Collection of `MediaQueryList` objects to be updated.
    ///
    /// Functional Utility: Tracks all active `MediaQueryList` instances within the
    ///                     window, ensuring they are correctly updated when
    ///                     media conditions change.
    media_query_lists: DOMTracker<MediaQueryList>,

    /// @brief Test runner instance for this window.
    ///
    /// Functional Utility: Provides an interface for executing web platform tests
    ///                     within this browsing context, typically used for
    ///                     development and compliance verification.
    #[cfg(feature = "bluetooth")]
    test_runner: MutNullableDom<TestRunner>,

    /// @brief Handle for communicating with the WebGL thread.
    ///
    /// Functional Utility: If WebGL is available and enabled, this channel
    ///                     facilitates sending commands and data to the WebGL
    ///                     rendering context.
    #[ignore_malloc_size_of = "channels are hard"]
    #[no_trace]
    webgl_chan: Option<WebGLChan>,

    /// @brief WebXR registry instance.
    ///
    /// Functional Utility: Manages WebXR device and session information,
    ///                     enabling augmented and virtual reality experiences.
    #[ignore_malloc_size_of = "defined in webxr"]
    #[no_trace]
    #[cfg(feature = "webxr")]
    webxr_registry: Option<webxr_api::Registry>,

    /// @brief Callbacks for pending image loads.
    ///
    /// Functional Utility: Stores a map of `PendingImageId` to a list of
    ///                     `PendingImageCallback` closures, ensuring that all
    ///                     registered callbacks are executed once an image
    ///                     load completes or progresses.
    #[no_trace]
    pending_image_callbacks: DomRefCell<HashMap<PendingImageId, Vec<PendingImageCallback>>>,

    /// @brief Elements with outstanding image requests initiated by layout.
    ///
    /// Functional Utility: Tracks DOM nodes that have triggered image loads during a reflow,
    ///                     allowing them to be marked as dirty when image data
    ///                     becomes available, prompting a re-layout.
    pending_layout_images: DomRefCell<HashMapTracedValues<PendingImageId, Vec<Dom<Node>>>>,

    /// @brief Directory for unminified CSS.
    ///
    /// Functional Utility: Specifies a directory where unminified CSS files
    ///                     for this window are stored, if the `unminify-css`
    ///                     option is enabled, aiding in debugging.
    unminified_css_dir: DomRefCell<Option<String>>,

    /// @brief Local source directory for unminified scripts.
    ///
    /// Functional Utility: Stores the path to a directory containing unminified
    ///                     script sources, used for debugging and development.
    local_script_source: Option<String>,

    /// @brief Test Worklet instance.
    ///
    /// Functional Utility: Provides a dedicated execution environment for
    ///                     test-related worklets.
    test_worklet: MutNullableDom<Worklet>,
    /// @brief Paint Worklet instance.
    ///
    /// Functional Utility: Provides a dedicated execution environment for
    ///                     Paint Worklets, enabling custom rendering effects.
    ///                     <https://drafts.css-houdini.org/css-paint-api-1/#paint-worklet>
    paint_worklet: MutNullableDom<Worklet>,
    /// @brief Webrender `DocumentId` for this window.
    ///
    /// Functional Utility: Uniquely identifies this window's document within the
    ///                     Webrender rendering engine, facilitating communication
    ///                     and rendering commands.
    #[ignore_malloc_size_of = "defined in webrender_api"]
    #[no_trace]
    webrender_document: DocumentId,

    /// @brief Flag indicating the presence of mutation observers.
    ///
    /// Functional Utility: A boolean flag that is true if any `MutationObserver`s
    ///                     are currently active in the document, optimizing the
    ///                     detection and notification of DOM changes.
    exists_mut_observer: Cell<bool>,

    /// @brief Cross-process compositor API for this window.
    ///
    /// Functional Utility: Provides an interface for sending rendering commands
    ///                     and receiving updates from the compositor process,
    ///                     enabling efficient and secure rendering.
    #[ignore_malloc_size_of = "Wraps an IpcSender"]
    #[no_trace]
    compositor_api: CrossProcessCompositorApi,

    /// @brief Flag indicating if an idle message has been sent after reflow.
    ///
    /// Functional Utility: Prevents redundant `SetDocumentStatus` messages
    ///                     from being sent when the window becomes idle after
    ///                     a reflow, optimizing message traffic.
    has_sent_idle_message: Cell<bool>,

    /// @brief Flag to emit relayout events.
    ///
    /// Functional Utility: When true, enables the emission of debug messages
    ///                     whenever a relayout occurs, aiding in performance
    ///                     analysis and debugging.
    relayout_event: bool,

    /// @brief Flag to enable CSS unminification.
    ///
    /// Functional Utility: When true, enables the process of unminifying CSS
    ///                     files, making them more readable for debugging purposes.
    unminify_css: bool,

    /// @brief Path to user scripts.
    ///
    /// Functional Utility: Specifies the directory from which user-defined
    ///                     scripts should be loaded. An empty string loads
    ///                     from `resources/user-agent-js`, while `None`
    ///                     disables user script loading.
    userscripts_path: Option<String>,

    /// @brief OpenGL context for the window.
    ///
    /// Functional Utility: Provides the OpenGL context used by the application
    ///                     for rendering graphics within this window.
    #[ignore_malloc_size_of = "defined in script_thread"]
    #[no_trace]
    player_context: WindowGLContext,

    /// @brief Flag indicating if the window is throttled.
    ///
    /// Functional Utility: When true, indicates that the window's activity
    ///                     (e.g., script execution, rendering updates) is
    ///                     being reduced to conserve resources, typically
    ///                     when the window is in the background or not visible.
    throttled: Cell<bool>,

    /// @brief Shared marker for cached layout values validity.
    ///
    /// Functional Utility: A reference-counted cell that, when set to `false`,
    ///                     invalidates any cached layout values, forcing a
    ///                     recomputation during the next layout pass.
    #[ignore_malloc_size_of = "Rc is hard"]
    layout_marker: DomRefCell<Rc<Cell<bool>>>,

    /// @brief The currently active event being dispatched.
    ///
    /// Functional Utility: Holds a reference to the `Event` object that is
    ///                     currently in the process of being dispatched through
    ///                     the event target chain, as per DOM specification.
    ///                     <https://dom.spec.whatwg.org/#window-current-event>
    current_event: DomRefCell<Option<Dom<Event>>>,
}

impl Window {
    //! @brief Returns the unique identifier of the WebView associated with this window.
    //!
    //! Functional Utility: Provides access to the `WebViewId` that uniquely identifies
    //!                     the browsing context (e.g., tab or iframe) in which this
    //!                     `Window` instance resides. This ID is essential for
    //!                     inter-process communication and for coordinating operations
    //!                     across different parts of the browser engine.
    //!
    //! @return The `WebViewId` of the containing WebView.
    pub(crate) fn webview_id(&self) -> WebViewId {
        self.webview_id
    }

    //! @brief Returns a reference to the `GlobalScope` associated with this window.
    //!
    //! Functional Utility: Provides a way to access the underlying `GlobalScope` object,
    //!                     which encapsulates global properties, functions, and object
    //!                     lifetimes for the JavaScript execution environment. This
    //!                     method allows treating the `Window` as its base `GlobalScope`
    //!                     for operations that apply to any global object.
    //!
    //! @return A shared reference to the `GlobalScope`.
    pub(crate) fn as_global_scope(&self) -> &GlobalScope {
        self.upcast::<GlobalScope>()
    }

    //! @brief Returns an immutable reference to the layout engine.
    //!
    //! Functional Utility: Provides read-only access to the window's layout engine.
    //!                     This allows for querying layout information without
    //!                     modifying the layout tree.
    //!
    //! @return An immutable `Ref` to the `Box<dyn Layout>`.
    pub(crate) fn layout(&self) -> Ref<Box<dyn Layout>> {
        self.layout.borrow()
    }

    //! @brief Returns a mutable reference to the layout engine.
    //!
    //! Functional Utility: Provides mutable access to the window's layout engine.
    //!                     This allows for triggering layout computations and
    //!                     modifying the layout tree.
    //!
    //! @return A mutable `RefMut` to the `Box<dyn Layout>`.
    pub(crate) fn layout_mut(&self) -> RefMut<Box<dyn Layout>> {
        self.layout.borrow_mut()
    }

    //! @brief Checks if mutation observers are present in the document.
    //!
    //! Functional Utility: Returns a boolean indicating whether there are any
    //!                     active `MutationObserver` instances attached to the
    //!                     document associated with this window. This can be
    //!                     used to optimize operations by avoiding unnecessary
    //!                     checks if no observers are present.
    //!
    //! @return `true` if mutation observers exist, `false` otherwise.
    pub(crate) fn get_exists_mut_observer(&self) -> bool {
        self.exists_mut_observer.get()
    }

    //! @brief Sets the flag indicating the presence of mutation observers.
    //!
    //! Functional Utility: Marks that a `MutationObserver` has been registered
    //!                     or is otherwise active within the window's document.
    //!                     This flag is used to enable or disable logic that
    //!                     depends on the presence of mutation observers,
    //!                     preventing redundant checks.
    pub(crate) fn set_exists_mut_observer(&self) {
        self.exists_mut_observer.set(true);
    }

    //! @brief Clears the JavaScript runtime and associated infrastructure for script deallocation.
    //!
    //! Functional Utility: This method is invoked during the deallocation process of a
    //!                     script environment. It systematically dismantles the JavaScript
    //!                     runtime, associated web messaging infrastructure, and dedicated
    //!                     workers, transitioning the window to a `Zombie` state. This
    //!                     ensures that no further script execution or interaction with
    //!                     the DOM occurs after the browsing context is discarded.
    //!
    //! Pre-condition: The browsing context is being discarded.
    //! Post-condition: JavaScript runtime is nullified, window state is `Zombie`, and
    //!                 all pending tasks are cancelled.
    #[allow(unsafe_code)]
    pub(crate) fn clear_js_runtime_for_script_deallocation(&self) {
        self.as_global_scope()
            .remove_web_messaging_and_dedicated_workers_infra();
        unsafe {
            *self.js_runtime.borrow_for_script_deallocation() = None;
            self.window_proxy.set(None);
            self.current_state.set(WindowState::Zombie);
            self.as_global_scope()
                .task_manager()
                .cancel_all_tasks_and_ignore_future_tasks();
        }
    }

    //! @brief Discards the browsing context associated with this window.
    //!
    //! Functional Utility: This method initiates the process of discarding a browsing
    //!                     context, as defined by the HTML specification. It primarily
    //!                     delegates the discarding operation to the associated
    //!                     `WindowProxy` and cancels any pending tasks, effectively
    //!                     shutting down the active browsing session for this window.
    //!                     Other cleanup steps are handled by the `ScriptThread`
    //!                     upon receiving a `PipelineExit` message.
    //!
    //! @see <https://html.spec.whatwg.org/multipage/#a-browsing-context-is-discarded>
    pub(crate) fn discard_browsing_context(&self) {
        let proxy = match self.window_proxy.get() {
            Some(proxy) => proxy,
            None => panic!("Discarding a BC from a window that has none"),
        };
        proxy.discard_browsing_context();
        // Step 4 of https://html.spec.whatwg.org/multipage/#discard-a-document
        // Other steps performed when the `PipelineExit` message
        // is handled by the ScriptThread.
        self.as_global_scope()
            .task_manager()
            .cancel_all_tasks_and_ignore_future_tasks();
    }

    //! @brief Retrieves the time profiler channel.
    //!
    //! Functional Utility: Provides access to the `TimeProfilerChan` instance from the
    //!                     global scope. This channel is used to send timing information
    //!                     and performance metrics to the time profiler thread, enabling
    //!                     detailed analysis of script execution and rendering performance.
    //!
    //! @return A shared reference to the `TimeProfilerChan`.
    pub(crate) fn time_profiler_chan(&self) -> &TimeProfilerChan {
        self.globalscope.time_profiler_chan()
    }

    //! @brief Retrieves the mutable origin of the document.
    //!
    //! Functional Utility: Provides access to the `MutableOrigin` object associated
    //!                     with the document loaded in this window. The origin
    //!                     defines the security context of the document, including
    //!                     its scheme, host, and port, and is fundamental for
    //!                     enforcing security policies like the Same-Origin Policy.
    //!
    //! @return A shared reference to the `MutableOrigin`.
    pub(crate) fn origin(&self) -> &MutableOrigin {
        self.globalscope.origin()
    }

    //! @brief Retrieves the raw JavaScript context associated with this window.
    //!
    //! Functional Utility: Provides direct access to the underlying `JSContext` pointer
    //!                     from the JavaScript runtime. This is primarily used for
    //!                     low-level interactions with the SpiderMonkey JavaScript engine,
    //!                     enabling operations that require direct manipulation of the
    //!                     JavaScript execution environment.
    //!
    //! @return A `JSContext` wrapper around the raw context pointer.
    //! Pre-condition: The JavaScript runtime must be initialized and present.
    #[allow(unsafe_code)]
    pub(crate) fn get_cx(&self) -> JSContext {
        unsafe { JSContext::from_ptr(self.js_runtime.borrow().as_ref().unwrap().cx()) }
    }

    //! @brief Retrieves an immutable reference to the JavaScript runtime.
    //!
    //! Functional Utility: Provides read-only access to the `Runtime` object
    //!                     that manages the JavaScript execution environment for this window.
    //!                     This allows other parts of the system to interact with the
    //!                     JavaScript engine, for instance, to create new JavaScript objects
    //!                     or evaluate scripts.
    //!
    //! @return A `Ref` containing an `Option<Rc<Runtime>>`, which is `Some` if the runtime
    //!         is active, and `None` otherwise.
    pub(crate) fn get_js_runtime(&self) -> Ref<Option<Rc<Runtime>>> {
        self.js_runtime.borrow()
    }

    //! @brief Returns a reference to the sender for the main thread script channel.
    //!
    //! Functional Utility: Provides the `Sender` endpoint for the `MainThreadScriptMsg`
    //!                     channel. This is used by the `Window`'s script thread to
    //!                     send messages and commands back to the main thread for
    //!                     processing, such as DOM updates, event handling, or
    //!                     rendering requests.
    //!
    //! @return A shared reference to the `Sender<MainThreadScriptMsg>`.
    pub(crate) fn main_thread_script_chan(&self) -> &Sender<MainThreadScriptMsg> {
        &self.script_chan
    }

    //! @brief Retrieves information about the parent pipeline.
    //!
    //! Functional Utility: If this `Window` is part of a nested browsing context (e.g., an iframe),
    //!                     this method returns an `Option` containing the `PipelineId` of its
    //!                     parent. This information is crucial for establishing hierarchical
    //!                     relationships and enabling communication between parent and child
    //!                     browsing contexts. Returns `None` if this is a top-level window.
    //!
    //! @return An `Option<PipelineId>` representing the parent pipeline, if any.
    pub(crate) fn parent_info(&self) -> Option<PipelineId> {
        self.parent_info
    }

    //! @brief Creates a new pair of script event loop sender and receiver.
    //!
    //! Functional Utility: This method generates a new, unbounded `crossbeam_channel` pair
    //!                     for script event communication. It provides a `ScriptEventLoopSender`
    //!                     configured for the main thread and a corresponding `ScriptEventLoopReceiver`,
    //!                     facilitating the creation of new communication channels within the
    //!                     scripting environment.
    //!
    //! @return A tuple containing a `ScriptEventLoopSender` and a `ScriptEventLoopReceiver`.
    pub(crate) fn new_script_pair(&self) -> (ScriptEventLoopSender, ScriptEventLoopReceiver) {
        let (sender, receiver) = unbounded();
        (
            ScriptEventLoopSender::MainThread(sender),
            ScriptEventLoopReceiver::MainThread(receiver),
        )
    }

    //! @brief Returns a sender for the script event loop.
    //!
    //! Functional Utility: Provides a `ScriptEventLoopSender` instance that
    //!                     allows sending messages to the main thread's script
    //!                     event loop. This is a convenient way to obtain a sender
    //!                     for existing communication channels without creating
    //!                     a new pair.
    //!
    //! @return A `ScriptEventLoopSender` configured for the main thread.
    pub(crate) fn event_loop_sender(&self) -> ScriptEventLoopSender {
        ScriptEventLoopSender::MainThread(self.script_chan.clone())
    }

    //! @brief Returns a cloned `Arc` to the image cache.
    //!
    //! Functional Utility: Provides shared access to the `ImageCache` instance
    //!                     associated with this window. This cache stores decoded
    //!                     image data, reducing the need for repeated network
    //!                     requests and improving rendering performance. Cloning
    //!                     the `Arc` allows multiple components to share ownership
    //!                     of the cache.
    //!
    //! @return An `Arc<dyn ImageCache>` for the image cache.
    pub(crate) fn image_cache(&self) -> Arc<dyn ImageCache> {
        self.image_cache.clone()
    }

    //! @brief Retrieves the `WindowProxy` for this window.
    //!
    //! Functional Utility: Returns the `WindowProxy` object associated with this
    //!                     `Window` instance. This proxy acts as an intermediary
    //!                     for cross-origin communication and provides a safe
    //!                     interface for other browsing contexts to interact
    //!                     with this window.
    //!
    //! Pre-condition: This method panics if called after the browsing context
    //!                has been discarded, as the `WindowProxy` would no longer
    //!                be available.
    //!
    //! @return A `DomRoot<WindowProxy>` representing the window's proxy.
    pub(crate) fn window_proxy(&self) -> DomRoot<WindowProxy> {
        self.window_proxy
            .get()
            .expect("Discarding a BC from a window that has none")
    }
    //! @brief Returns the `WindowProxy` if it has not been discarded.
    //!
    //! Functional Utility: This method safely retrieves the `WindowProxy` for
    //!                     this window, but only if the associated browsing
    //!                     context has not been discarded. It provides a way
    //!                     to interact with the window's proxy without risking
    //!                     accessing a defunct object.
    //!
    //! @see <https://html.spec.whatwg.org/multipage/#a-browsing-context-is-discarded>
    //! @return An `Option<DomRoot<WindowProxy>>`, which is `Some` if the proxy is
    //!         active and not discarded, and `None` otherwise.
    pub(crate) fn undiscarded_window_proxy(&self) -> Option<DomRoot<WindowProxy>> {
        self.window_proxy.get().and_then(|window_proxy| {
            if window_proxy.is_browsing_context_discarded() {
                None
            } else {
                Some(window_proxy)
            }
        })
    }

    //! @brief Retrieves a sender for the Bluetooth thread.
    //!
    //! Functional Utility: Provides an `IpcSender` to communicate with the Bluetooth
    //!                     thread. This channel is used to send `BluetoothRequest`
    //!                     messages for operations related to Web Bluetooth APIs,
    //!                     such as scanning for devices, connecting, and data transfer.
    //!                     This method is only available when the "bluetooth" feature
    //!                     is enabled.
    //!
    //! @return An `IpcSender<BluetoothRequest>` for the Bluetooth thread.
    #[cfg(feature = "bluetooth")]
    pub(crate) fn bluetooth_thread(&self) -> IpcSender<BluetoothRequest> {
        self.bluetooth_thread.clone()
    }

    //! @brief Retrieves the extra permission data for Bluetooth operations.
    //!
    //! Functional Utility: Provides read-only access to the `BluetoothExtraPermissionData`
    //!                     associated with this window. This data contains additional
    //!                     context or permissions that might be required for specific
    //!                     Bluetooth API calls, ensuring adherence to security and
    //!                     privacy policies. This method is only available when the
    //!                     "bluetooth" feature is enabled.
    //!
    //! @return A shared reference to `BluetoothExtraPermissionData`.
    #[cfg(feature = "bluetooth")]
    pub(crate) fn bluetooth_extra_permission_data(&self) -> &BluetoothExtraPermissionData {
        &self.bluetooth_extra_permission_data
    }

    //! @brief Retrieves the CSS error reporter for this window.
    //!
    //! Functional Utility: Provides access to the `ParseErrorReporter` instance responsible
    //!                     for handling and reporting errors encountered during CSS parsing
    //!                     within this window's context. This is crucial for debugging
    //!                     and ensuring the integrity of stylesheets.
    //!
    //! @return An `Option<&dyn ParseErrorReporter>`, which is `Some` if an error
    //!         reporter is available, and `None` otherwise.
    pub(crate) fn css_error_reporter(&self) -> Option<&dyn ParseErrorReporter> {
        Some(&self.error_reporter)
    }

    //! @brief Sets a new list of scroll offsets for scrollable elements.
    //!
    //! Functional Utility: Updates the internal map of scroll offsets for various
    //!                     scrollable elements within the document. This method is
    //!                     typically called by the layout engine when new scroll
    //!                     positions are computed or when WebRender is active,
    //!                     ensuring that the `Window` reflects the current scroll
    //!                     state accurately.
    //!
    //! @param offsets A `HashMap` where keys are `OpaqueNode` identifiers
    //!                and values are `Vector2D` representing the new scroll offsets.
    pub(crate) fn set_scroll_offsets(
        &self,
        offsets: HashMap<OpaqueNode, Vector2D<f32, LayoutPixel>>,
    ) {
        *self.scroll_offsets.borrow_mut() = offsets
    }

    //! @brief Retrieves the current viewport rectangle.
    //!
    //! Functional Utility: Returns the `UntypedRect<Au>` representing the current
    //!                     visible area of the document within the window. This
    //!                     rectangle defines the origin (scroll position) and
    //!                     dimensions of the viewport, which is essential for
    //!                     rendering and layout calculations.
    //!
    //! @return An `UntypedRect<Au>` representing the current viewport.
    pub(crate) fn current_viewport(&self) -> UntypedRect<Au> {
        self.current_viewport.clone().get()
    }

    //! @brief Retrieves the WebGL command sender, if available.
    //!
    //! Functional Utility: Provides an `Option` containing a `WebGLCommandSender`
    //!                     if WebGL is enabled and a communication channel to the
    //!                     WebGL thread exists. This sender allows the window to
    //!                     dispatch WebGL rendering commands to the dedicated
    //!                     WebGL rendering thread.
    //!
    //! @return An `Option<WebGLCommandSender>`.
    pub(crate) fn webgl_chan(&self) -> Option<WebGLCommandSender> {
        self.webgl_chan
            .as_ref()
            .map(|chan| WebGLCommandSender::new(chan.clone()))
    }

    //! @brief Retrieves the WebXR registry instance, if available.
    //!
    //! Functional Utility: Provides an `Option` containing a `webxr_api::Registry`
    //!                     instance if WebXR support is enabled. This registry
    //!                     manages information about WebXR devices and sessions,
    //!                     enabling augmented and virtual reality experiences
    //!                     within the window's context.
    //!
    //! @return An `Option<webxr_api::Registry>`.
    #[cfg(feature = "webxr")]
    pub(crate) fn webxr_registry(&self) -> Option<webxr_api::Registry> {
        self.webxr_registry.clone()
    }

    //! @brief Creates a new Paint Worklet.
    //!
    //! Functional Utility: Instantiates and returns a new `Worklet` configured
    //!                     as a `Paint` worklet. This is part of the CSS Houdini
    //!                     specification, allowing developers to programmatically
    //!                     generate images that can be used in CSS properties.
    //!
    //! @return A `DomRoot<Worklet>` representing the newly created Paint Worklet.
    fn new_paint_worklet(&self) -> DomRoot<Worklet> {
        debug!("Creating new paint worklet.");
        Worklet::new(self, WorkletGlobalScopeType::Paint)
    }

    //! @brief Registers a callback to be invoked when a response comes back from the `ImageCache`.
    //!
    //! Functional Utility: This method allows an external component to register a
    //!                     closure (`callback`) that will be executed when an image
    //!                     identified by `id` is processed by the `ImageCache`. It
    //!                     enables asynchronous handling of image loading events,
    //!                     such as updating the DOM or triggering layout changes
    //!                     once image data is available.
    //!
    //! @param id The `PendingImageId` identifying the image being tracked.
    //! @param callback The closure to be executed upon receiving an image response.
    //! @return An `IpcSender<PendingImageResponse>` that can be used to send
    //!         image responses back to the listener.
    pub(crate) fn register_image_cache_listener(
        &self,
        id: PendingImageId,
        callback: impl Fn(PendingImageResponse) + 'static,
    ) -> IpcSender<PendingImageResponse> {
        self.pending_image_callbacks
            .borrow_mut()
            .entry(id)
            .or_default()
            .push(PendingImageCallback(Box::new(callback)));
        self.image_cache_sender.clone()
    }

    //! @brief Handles notifications for pending layout images.
    //!
    //! Functional Utility: This method is called when a response for an image
    //!                     that was requested by the layout engine (during a reflow)
    //!                     becomes available. It identifies the DOM nodes associated
    //!                     with the image and marks them as dirty, triggering a
    //!                     re-layout to reflect the newly loaded image data.
    //!
    //! @param response The `PendingImageResponse` containing the image ID and status.
    fn pending_layout_image_notification(&self, response: PendingImageResponse) {
        let mut images = self.pending_layout_images.borrow_mut();
        let nodes = images.entry(response.id);
        let nodes = match nodes {
            Entry::Occupied(nodes) => nodes,
            Entry::Vacant(_) => return,
        };
        for node in nodes.get() {
            node.dirty(NodeDamage::OtherNodeDamage);
        }
        match response.response {
            ImageResponse::MetadataLoaded(_) => {},
            ImageResponse::Loaded(_, _) |
            ImageResponse::PlaceholderLoaded(_, _) |
            ImageResponse::None => {
                nodes.remove();
            },
        }
    }

    //! @brief Dispatches notifications for general pending image loads.
    //!
    //! Functional Utility: This method processes a `PendingImageResponse` by
    //!                     iterating through all registered `PendingImageCallback`s
    //!                     for the given image ID and executing them. This mechanism
    //!                     is central to updating the DOM and other components
    //!                     asynchronously once image data is fetched. It also
    //!                     manages the lifecycle of these callbacks, removing
    //!                     them once the image loading is complete.
    //!
    //! @param response The `PendingImageResponse` containing the image ID and status.
    pub(crate) fn pending_image_notification(&self, response: PendingImageResponse) {
        // We take the images here, in order to prevent maintaining a mutable borrow when
        // image callbacks are called. These, in turn, can trigger garbage collection.
        // Normally this shouldn't trigger more pending image notifications, but just in
        // case we do not want to cause a double borrow here.
        let mut images = std::mem::take(&mut *self.pending_image_callbacks.borrow_mut());
        let Entry::Occupied(callbacks) = images.entry(response.id) else {
            let _ = std::mem::replace(&mut *self.pending_image_callbacks.borrow_mut(), images);
            return;
        };

        for callback in callbacks.get() {
            callback.0(response.clone());
        }

        match response.response {
            ImageResponse::MetadataLoaded(_) => {},
            ImageResponse::Loaded(_, _) |
            ImageResponse::PlaceholderLoaded(_, _) |
            ImageResponse::None => {
                callbacks.remove();
            },
        }

        let _ = std::mem::replace(&mut *self.pending_image_callbacks.borrow_mut(), images);
    }

    //! @brief Retrieves a reference to the cross-process compositor API.
    //!
    //! Functional Utility: Provides access to the `CrossProcessCompositorApi`
    //!                     instance, which enables communication with the
    //!                     compositor process. This API is used to send rendering
    //!                     commands, synchronize rendering state, and receive
    //!                     feedback from the compositor, facilitating efficient
    //!                     and secure rendering across process boundaries.
    //!
    //! @return A shared reference to the `CrossProcessCompositorApi`.
    pub(crate) fn compositor_api(&self) -> &CrossProcessCompositorApi {
        &self.compositor_api
    }

    //! @brief Retrieves the path for loading user scripts.
    //!
    //! Functional Utility: Returns an `Option<String>` representing the directory
    //!                     path from which user-defined scripts should be loaded.
    //!                     This allows for customization and extension of web page
    //!                     behavior through client-side scripting.
    //!                     An empty string indicates loading from a default resource
    //!                     directory, while `None` means user scripts are not loaded.
    //!
    //! @return An `Option<String>` containing the user scripts path, or `None`.
    pub(crate) fn get_userscripts_path(&self) -> Option<String> {
        self.userscripts_path.clone()
    }

    //! @brief Retrieves the window's GL context from the application.
    //!
    //! Functional Utility: Provides access to the `WindowGLContext` that is
    //!                     used by the embedding application for rendering graphics
    //!                     within this window. This context is essential for
    //!                     WebGL and other hardware-accelerated rendering
    //!                     operations.
    //!
    //! @return A `WindowGLContext` for the window.
    pub(crate) fn get_player_context(&self) -> WindowGLContext {
        self.player_context.clone()
    }

    //! @brief Dispatches an event with a target override.
    //!
    //! Functional Utility: This method dispatches an `Event` through the event
    //!                     target chain, allowing the `Window` itself to act
    //!                     as the target of the event. It's used for scenarios
    //!                     where an event needs to be explicitly targeted at
    //!                     the window, overriding the default event flow.
    //!
    //! @param event The `Event` object to be dispatched.
    //! @param can_gc Indicates whether garbage collection is allowed during dispatch.
    //! @return The `EventStatus` after dispatching the event.
    //! @see https://dom.spec.whatwg.org/#concept-event-dispatch step 2
    pub(crate) fn dispatch_event_with_target_override(
        &self,
        event: &Event,
        can_gc: CanGc,
    ) -> EventStatus {
        event.dispatch(self.upcast(), true, can_gc)
    }

    //! @brief Retrieves a reference to the `FontContext` for this window.
    //!
    //! Functional Utility: Provides read-only access to the `FontContext` responsible
    //!                     for managing and matching fonts within this window's
    //!                     document. This context is also used to trigger the
    //!                     download of web fonts, ensuring consistent typography.
    //!
    //! @return A shared reference to the `Arc<FontContext>`.
    pub(crate) fn font_context(&self) -> &Arc<FontContext> {
        &self.font_context
    }
}

//! @brief Encodes a string in base64.
//!
//! Functional Utility: Implements the `btoa()` method as defined in the HTML
//!                     specification, converting a string of binary data (octets)
//!                     into its base64 ASCII string representation. It enforces
//!                     the constraint that the input string must only contain
//!                     characters with code points less than or equal to U+00FF.
//!
//! @param input The `DOMString` to be encoded.
//! @return A `Fallible<DOMString>` containing the base64 encoded string on success,
//!         or an `Error::InvalidCharacter` if the input contains characters
//!         outside the allowed range.
//! @see https://html.spec.whatwg.org/multipage/#atob
pub(crate) fn base64_btoa(input: DOMString) -> Fallible<DOMString> {
    // "The btoa() method must throw an InvalidCharacterError exception if
    //  the method's first argument contains any character whose code point
    //  is greater than U+00FF."
    if input.chars().any(|c: char| c > '\u{FF}') {
        Err(Error::InvalidCharacter)
    } else {
        // "Otherwise, the user agent must convert that argument to a
        //  sequence of octets whose nth octet is the eight-bit
        //  representation of the code point of the nth character of
        //  the argument,"
        let octets = input.chars().map(|c: char| c as u8).collect::<Vec<u8>>();

        // "and then must apply the base64 algorithm to that sequence of
        //  octets, and return the result. [RFC4648]"
        let config =
            base64::engine::general_purpose::GeneralPurposeConfig::new().with_encode_padding(true);
        let engine = base64::engine::GeneralPurpose::new(&base64::alphabet::STANDARD, config);
        Ok(DOMString::from(engine.encode(octets)))
    }
}

//! @brief Decodes a base64 encoded string.
//!
//! Functional Utility: Implements the `atob()` method as defined in the HTML
//!                     specification, decoding a base64 encoded ASCII string
//!                     back into its original binary data representation.
//!                     It handles space characters, optional padding, and validates
//!                     the input characters according to the base64 alphabet.
//!
//! @param input The `DOMString` to be decoded.
//! @return A `Fallible<DOMString>` containing the decoded string on success,
//!         or an `Error::InvalidCharacter` if the input is not valid base64.
//! @see https://html.spec.whatwg.org/multipage/#atob
pub(crate) fn base64_atob(input: DOMString) -> Fallible<DOMString> {
    // "Remove all space characters from input."
    fn is_html_space(c: char) -> bool {
        HTML_SPACE_CHARACTERS.iter().any(|&m| m == c)
    }
    let without_spaces = input
        .chars()
        .filter(|&c| !is_html_space(c))
        .collect::<String>();
    let mut input = &*without_spaces;

    // "If the length of input divides by 4 leaving no remainder, then:
    //  if input ends with one or two U+003D EQUALS SIGN (=) characters,
    //  remove them from input."
    if input.len() % 4 == 0 {
        if input.ends_with("==") {
            input = &input[..input.len() - 2]
        } else if input.ends_with('=') {
            input = &input[..input.len() - 1]
        }
    }

    // "If the length of input divides by 4 leaving a remainder of 1,
    //  throw an InvalidCharacterError exception and abort these steps."
    if input.len() % 4 == 1 {
        return Err(Error::InvalidCharacter);
    }

    // "If input contains a character that is not in the following list of
    //  characters and character ranges, throw an InvalidCharacterError
    //  exception and abort these steps:
    //
    //  U+002B PLUS SIGN (+)
    //  U+002F SOLIDUS (/)
    //  Alphanumeric ASCII characters"
    if input
        .chars()
        .any(|c| c != '+' && c != '/' && !c.is_alphanumeric())
    {
        return Err(Error::InvalidCharacter);
    }

    let config = base64::engine::general_purpose::GeneralPurposeConfig::new()
        .with_decode_padding_mode(base64::engine::DecodePaddingMode::RequireNone)
        .with_decode_allow_trailing_bits(true);
    let engine = base64::engine::GeneralPurpose::new(&base64::alphabet::STANDARD, config);

    let data = engine.decode(input).map_err(|_| Error::InvalidCharacter)?;
    Ok(data.iter().map(|&b| b as char).collect::<String>().into())
}

impl WindowMethods<crate::DomTypeHolder> for Window {
    //! @brief Displays an alert dialog with an empty message.
    //!
    //! Functional Utility: This method provides a default implementation of the
    //!                     `alert()` DOM method when no message is explicitly
    //!                     provided. It calls the more general `Alert` method
    //!                     with an empty `DOMString`.
    //!
    //! @see https://html.spec.whatwg.org/multipage/#dom-alert
    fn Alert_(&self) {
        self.Alert(DOMString::new());
    }

    //! @brief Displays an alert dialog with a specified message.
    //!
    //! Functional Utility: Implements the `alert(message)` DOM method, which
    //!                     displays a modal dialog box with the given `message`
    //!                     and an OK button. This method also handles flushing
    //!                     stdout/stderr and sending a `Prompt` message to the
    //!                     embedder to display the actual alert UI.
    //!
    //! @param s The `DOMString` message to display in the alert dialog.
    //! @see https://html.spec.whatwg.org/multipage/#dom-alert
    fn Alert(&self, s: DOMString) {
        // Print to the console.
        // Ensure that stderr doesn't trample through the alert() we use to
        // communicate test results (see executorservo.py in wptrunner).
        {
            let stderr = stderr();
            let mut stderr = stderr.lock();
            let stdout = stdout();
            let mut stdout = stdout.lock();
            writeln!(&mut stdout, "
ALERT: {}", s).unwrap();
            stdout.flush().unwrap();
            stderr.flush().unwrap();
        }
        let (sender, receiver) =
            ProfiledIpc::channel(self.global().time_profiler_chan().clone()).unwrap();
        let prompt = PromptDefinition::Alert(s.to_string(), sender);
        let msg = EmbedderMsg::Prompt(self.webview_id(), prompt, PromptOrigin::Untrusted);
        self.send_to_embedder(msg);
        receiver.recv().unwrap();
    }

    //! @brief Displays a confirmation dialog with a specified message.
    //!
    //! Functional Utility: Implements the `confirm(message)` DOM method, which
    //!                     displays a modal dialog box with the given `message`,
    //!                     an OK button, and a Cancel button. This method sends
    //!                     a `Prompt` message to the embedder and waits for
    //!                     the user's interaction to determine the return value.
    //!
    //! @param s The `DOMString` message to display in the confirmation dialog.
    //! @return `true` if the user clicks OK, `false` if the user clicks Cancel.
    //! @see https://html.spec.whatwg.org/multipage/#dom-confirm
    fn Confirm(&self, s: DOMString) -> bool {
        let (sender, receiver) =
            ProfiledIpc::channel(self.global().time_profiler_chan().clone()).unwrap();
        let prompt = PromptDefinition::OkCancel(s.to_string(), sender);
        let msg = EmbedderMsg::Prompt(self.webview_id(), prompt, PromptOrigin::Untrusted);
        self.send_to_embedder(msg);
        receiver.recv().unwrap() == PromptResult::Primary
    }

    //! @brief Displays a dialog box with a message, an input field, and OK/Cancel buttons.
    //!
    //! Functional Utility: Implements the `prompt(message, default)` DOM method,
    //!                     which allows the user to input a string value. This
    //!                     method sends a `Prompt` message to the embedder to
    //!                     display the dialog and waits for the user's input.
    //!
    //! @param message The `DOMString` message to display to the user.
    //! @param default The `DOMString` default value to pre-fill the input field.
    //! @return An `Option<DOMString>` containing the user's input on OK, or `None` on Cancel.
    //! @see https://html.spec.whatwg.org/multipage/#dom-prompt
    fn Prompt(&self, message: DOMString, default: DOMString) -> Option<DOMString> {
        let (sender, receiver) =
            ProfiledIpc::channel(self.global().time_profiler_chan().clone()).unwrap();
        let prompt = PromptDefinition::Input(message.to_string(), default.to_string(), sender);
        let msg = EmbedderMsg::Prompt(self.webview_id(), prompt, PromptOrigin::Untrusted);
        self.send_to_embedder(msg);
        receiver.recv().unwrap().map(|s| s.into())
    }

    //! @brief Stops the current document loading.
    //!
    //! Functional Utility: Implements the `stop()` DOM method, which aborts
    //!                     the loading of the current document in the window.
    //!                     This includes canceling any pending network requests
    //!                     and parsing operations for the document.
    //!
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @see https://html.spec.whatwg.org/multipage/#dom-window-stop
    fn Stop(&self, can_gc: CanGc) {
        // TODO: Cancel ongoing navigation.
        let doc = self.Document();
        doc.abort(can_gc);
    }

    //! @brief Opens a new browsing context.
    //!
    //! Functional Utility: Implements the `open()` DOM method, which allows for
    //!                     programmatically opening new browser windows or tabs,
    //!                     navigating an existing browsing context, or creating
    //!                     a new one. This method delegates the actual opening
    //!                     logic to the `WindowProxy` associated with this window.
    //!
    //! @param url The `USVString` representing the URL to navigate to.
    //! @param target The `DOMString` specifying the browsing context name (e.g., "_blank", "_self").
    //! @param features The `DOMString` containing a comma-separated list of window features.
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @return A `Fallible<Option<DomRoot<WindowProxy>>>` which, on success, contains
    //!         `Some` of the `WindowProxy` for the newly opened/navigated context,
    //!         or `None` if the operation fails or is not permitted.
    //! @see https://html.spec.whatwg.org/multipage/#dom-open
    fn Open(
        &self,
        url: USVString,
        target: DOMString,
        features: DOMString,
        can_gc: CanGc,
    ) -> Fallible<Option<DomRoot<WindowProxy>>> {
        self.window_proxy().open(url, target, features, can_gc)
    }

    //! @brief Retrieves the `WindowProxy` for the opening browsing context.
    //!
    //! Functional Utility: Implements the `window.opener` DOM property getter,
    //!                     which returns a reference to the window that opened
    //!                     the current window. This is subject to security
    //!                     restrictions (e.g., cross-origin policies).
    //!                     If the opening window's browsing context has been
    //!                     discarded or is null, it returns `null`.
    //!
    //! @param cx The JavaScript context.
    //! @param in_realm_proof Proof that the operation is within a valid realm.
    //! @param retval A mutable handle to the JavaScript value where the result will be stored.
    //! @return A `Fallible` indicating success or failure.
    //! @see https://html.spec.whatwg.org/multipage/#dom-opener
    fn GetOpener(
        &self,
        cx: JSContext,
        in_realm_proof: InRealm,
        mut retval: MutableHandleValue,
    ) -> Fallible<()> {
        // Step 1, Let current be this Window object's browsing context.
        let current = match self.window_proxy.get() {
            Some(proxy) => proxy,
            // Step 2, If current is null, then return null.
            None => {
                retval.set(NullValue());
                return Ok(());
            },
        };
        // Still step 2, since the window's BC is the associated doc's BC,
        // see https://html.spec.whatwg.org/multipage/#window-bc
        // and a doc's BC is null if it has been discarded.
        // see https://html.spec.whatwg.org/multipage/#concept-document-bc
        if current.is_browsing_context_discarded() {
            retval.set(NullValue());
            return Ok(());
        }
        // Step 3 to 5.
        current.opener(*cx, in_realm_proof, retval);
        Ok(())
    }

    //! @brief Sets the `window.opener` property.
    //!
    //! Functional Utility: Implements the setter for the `window.opener` DOM
    //!                     property. If the provided `value` is null, it
    //!                     "disowns" the current window's opener, effectively
    //!                     removing the reference to the opening window.
    //!                     Otherwise, it attempts to set the `opener` property
    //!                     on the JavaScript object.
    //!
    //! @param cx The JavaScript context.
    //! @param value The `HandleValue` to set as the new opener.
    //! @return An `ErrorResult` indicating success or failure of the operation.
    //! @see https://html.spec.whatwg.org/multipage/#dom-opener
    #[allow(unsafe_code)]
    // https://html.spec.whatwg.org/multipage/#dom-opener
    fn SetOpener(&self, cx: JSContext, value: HandleValue) -> ErrorResult {
        // Step 1.
        if value.is_null() {
            if let Some(proxy) = self.window_proxy.get() {
                proxy.disown();
            }
            return Ok(());
        }
        // Step 2.
        let obj = self.reflector().get_jsobject();
        unsafe {
            let result =
                JS_DefineProperty(*cx, obj, c"opener".as_ptr(), value, JSPROP_ENUMERATE as u32);

            if result {
                Ok(())
            } else {
                Err(Error::JSFailed)
            }
        }
    }

    //! @brief Checks if the window is closed.
    //!
    //! Functional Utility: Implements the `window.closed` DOM property, which
    //!                     returns a boolean indicating whether the window's
    //!                     browsing context has been discarded or if the
    //!                     `WindowProxy` is in the process of closing.
    //!
    //! @return `true` if the window is closed or closing, `false` otherwise.
    //! @see https://html.spec.whatwg.org/multipage/#dom-window-closed
    fn Closed(&self) -> bool {
        self.window_proxy
            .get()
            .map(|ref proxy| proxy.is_browsing_context_discarded() || proxy.is_closing())
            .unwrap_or(true)
    }

    //! @brief Closes the window.
    //!
    //! Functional Utility: Implements the `window.close()` DOM method, which
    //!                     attempts to close the current browsing context.
    //!                     This method performs checks to determine if the
    //!                     window is script-closable (e.g., opened by script,
    //!                     top-level with single history entry, or auxiliary).
    //!                     If closable, it sets the window's `is closing` flag
    //!                     and queues a task to unload the document and
    //!                     discard the browsing context.
    //!
    //! @see https://html.spec.whatwg.org/multipage/#dom-window-close
    fn Close(&self) {
        // Step 1, Let current be this Window object's browsing context.
        // Step 2, If current is null or its is closing is true, then return.
        let window_proxy = match self.window_proxy.get() {
            Some(proxy) => proxy,
            None => return,
        };
        if window_proxy.is_closing() {
            return;
        }
        // Note: check the length of the "session history", as opposed to the joint session history?
        // see https://github.com/whatwg/html/issues/3734
        if let Ok(history_length) = self.History().GetLength() {
            let is_auxiliary = window_proxy.is_auxiliary();

            // https://html.spec.whatwg.org/multipage/#script-closable
            let is_script_closable = (self.is_top_level() && history_length == 1) ||
                is_auxiliary ||
                pref!(dom_allow_scripts_to_close_windows);

            // TODO: rest of Step 3:
            // Is the incumbent settings object's responsible browsing context familiar with current?
            // Is the incumbent settings object's responsible browsing context allowed to navigate current?
            if is_script_closable {
                // Step 3.1, set current's is closing to true.
                window_proxy.close();

                // Step 3.2, queue a task on the DOM manipulation task source to close current.
                let this = Trusted::new(self);
                let task = task!(window_close_browsing_context: move || {
                    let window = this.root();
                    let document = window.Document();
                    // https://html.spec.whatwg.org/multipage/#closing-browsing-contexts
                    // Step 1, check if traversable is closing, was already done above.
                    // Steps 2 and 3, prompt to unload for all inclusive descendant navigables.
                    // TODO: We should be prompting for all inclusive descendant navigables,
                    // but we pass false here, which suggests we are not doing that. Why?
                    if document.prompt_to_unload(false, CanGc::note()) {
                        // Step 4, unload.
                        document.unload(false, CanGc::note());

                        // https://html.spec.whatwg.org/multipage/#a-browsing-context-is-discarded
                        // which calls into https://html.spec.whatwg.org/multipage/#discard-a-document.
                        window.discard_browsing_context();

                        window.send_to_constellation(ScriptMsg::DiscardTopLevelBrowsingContext);
                    }
                });
                self.as_global_scope()
                    .task_manager()
                    .dom_manipulation_task_source()
                    .queue(task);
            }
        }
    }

    //! @brief Retrieves the `Document` object for this window.
    //!
    //! Functional Utility: Implements the `window.document` DOM property getter,
    //!                     providing access to the `Document` object that
    //!                     represents the web page loaded in this window. This
    //!                     is the primary entry point for interacting with the
    //!                     content of the page, including its structure, style,
    //!                     and events.
    //!
    //! Pre-condition: Panics if the `Document` is accessed before it has been
    //!                properly initialized within the `Window`.
    //!
    //! @return A `DomRoot<Document>` representing the document of this window.
    //! @see https://html.spec.whatwg.org/multipage/#dom-document-2
    fn Document(&self) -> DomRoot<Document> {
        self.document
            .get()
            .expect("Document accessed before initialization.")
    }

    //! @brief Retrieves the `History` object for this window.
    //!
    //! Functional Utility: Implements the `window.history` DOM property getter,
    //!                     providing access to the `History` object. This object
    //!                     allows manipulation of the browser session history,
    //!                     enabling programmatic navigation (e.g., back, forward)
    //!                     and modification of the history stack.
    //!
    //! @return A `DomRoot<History>` representing the history of this window.
    //! @see https://html.spec.whatwg.org/multipage/#dom-history
    fn History(&self) -> DomRoot<History> {
        self.history.or_init(|| History::new(self))
    }

    //! @brief Retrieves the `CustomElementRegistry` object for this window.
    //!
    //! Functional Utility: Implements the `window.customElements` DOM property getter,
    //!                     providing access to the `CustomElementRegistry`. This
    //!                     registry allows developers to register new custom
    //!                     elements, query existing ones, and manage their lifecycle,
    //!                     enabling the extension of HTML with user-defined tags.
    //!
    //! @return A `DomRoot<CustomElementRegistry>` for this window.
    //! @see https://html.spec.whatwg.org/multipage/#dom-window-customelements
    fn CustomElements(&self) -> DomRoot<CustomElementRegistry> {
        self.custom_element_registry
            .or_init(|| CustomElementRegistry::new(self))
    }

    //! @brief Retrieves the `Location` object for this window.
    //!
    //! Functional Utility: Implements the `window.location` DOM property getter,
    //!                     providing access to the `Location` object. This object
    //!                     represents the current URL of the document displayed in
    //!                     the window and offers methods to navigate to new URLs,
    //!                     reload the current page, or manipulate URL components.
    //!
    //! @return A `DomRoot<Location>` representing the location of this window.
    //! @see https://html.spec.whatwg.org/multipage/#dom-location
    fn Location(&self) -> DomRoot<Location> {
        self.location.or_init(|| Location::new(self))
    }

    //! @brief Retrieves the `sessionStorage` object for this window.
    //!
    //! Functional Utility: Implements the `window.sessionStorage` DOM property getter,
    //!                     providing access to the `Storage` object for session-level
    //!                     data. This storage mechanism allows web applications to
    //!                     store key-value pairs that are maintained for the duration
    //!                     of the top-level browsing context's session.
    //!
    //! @return A `DomRoot<Storage>` representing the session storage of this window.
    //! @see https://html.spec.whatwg.org/multipage/#dom-sessionstorage
    fn SessionStorage(&self) -> DomRoot<Storage> {
        self.session_storage
            .or_init(|| Storage::new(self, StorageType::Session))
    }

    //! @brief Retrieves the `localStorage` object for this window.
    //!
    //! Functional Utility: Implements the `window.localStorage` DOM property getter,
    //!                     providing access to the `Storage` object for local
    //!                     persistent data. This storage mechanism allows web
    //!                     applications to store key-value pairs that are preserved
    //!                     across browser sessions and tabs.
    //!
    //! @return A `DomRoot<Storage>` representing the local storage of this window.
    //! @see https://html.spec.whatwg.org/multipage/#dom-localstorage
    fn LocalStorage(&self) -> DomRoot<Storage> {
        self.local_storage
            .or_init(|| Storage::new(self, StorageType::Local))
    }

    //! @brief Retrieves the `Crypto` object for this window.
    //!
    //! Functional Utility: Implements the `window.crypto` DOM property getter,
    //!                     providing access to the `Crypto` object. This object
    //!                     provides web pages with a number of cryptographic
    //!                     primitives, such as strong random number generation,
    //!                     and cryptographic functions (e.g., hashing, encryption).
    //!
    //! @return A `DomRoot<Crypto>` representing the crypto object of this window.
    //! @see https://dvcs.w3.org/hg/webcrypto-api/raw-file/tip/spec/Overview.html#dfn-GlobalCrypto
    fn Crypto(&self) -> DomRoot<Crypto> {
        self.as_global_scope().crypto()
    }

    //! @brief Retrieves the `Element` that represents the frame containing this window.
    //!
    //! Functional Utility: Implements the `window.frameElement` DOM property getter.
    //!                     This method returns the `Element` (e.g., an `<iframe>`,
    //!                     `<frame>`, or `<object>`) that directly embeds the current
    //!                     browsing context. It includes a security check to ensure
    //!                     that the origin of the containing document is the same-origin
    //!                     domain as the current document, preventing cross-origin
    //!                     information leakage.
    //!
    //! @return An `Option<DomRoot<Element>>` which is `Some` if a frame element is found
    //!         and passes security checks, and `None` otherwise.
    //! @see https://html.spec.whatwg.org/multipage/#dom-frameelement
    fn GetFrameElement(&self) -> Option<DomRoot<Element>> {
        // Steps 1-3.
        let window_proxy = self.window_proxy.get()?;

        // Step 4-5.
        let container = window_proxy.frame_element()?;

        // Step 6.
        let container_doc = container.owner_document();
        let current_doc = GlobalScope::current()
            .expect("No current global object")
            .as_window()
            .Document();
        if !current_doc
            .origin()
            .same_origin_domain(container_doc.origin())
        {
            return None;
        }
        // Step 7.
        Some(DomRoot::from_ref(container))
    }

    //! @brief Retrieves the `Navigator` object for this window.
    //!
    //! Functional Utility: Implements the `window.navigator` DOM property getter,
    //!                     providing access to the `Navigator` object. This object
    //!                     contains information about the web browser and the
    //!                     user agent, such as its name, version, and supported
    //!                     MIME types and plugins.
    //!
    //! @return A `DomRoot<Navigator>` representing the navigator object of this window.
    //! @see https://html.spec.whatwg.org/multipage/#dom-navigator
    fn Navigator(&self) -> DomRoot<Navigator> {
        self.navigator.or_init(|| Navigator::new(self))
    }

    //! @brief Schedules a function or code snippet to be executed after a specified delay.
    //!
    //! Functional Utility: Implements the `window.setTimeout()` DOM method.
    //!                     This method schedules a `callback` (either a function
    //!                     or a string of code) to be executed once after a
    //!                     given `timeout` (in milliseconds). Any additional
    //!                     `args` are passed to the callback function.
    //!                     The returned integer ID can be used to cancel the timeout.
    //!
    //! @param _cx The JavaScript context (unused in this implementation).
    //! @param callback The `StringOrFunction` representing the code to execute.
    //! @param timeout The delay in milliseconds before the callback is executed.
    //! @param args A vector of `HandleValue` arguments to pass to the callback.
    //! @return An integer ID that uniquely identifies the scheduled timeout.
    //! @see https://html.spec.whatwg.org/multipage/#dom-windowtimers-settimeout
    fn SetTimeout(
        &self,
        _cx: JSContext,
        callback: StringOrFunction,
        timeout: i32,
        args: Vec<HandleValue>,
    ) -> i32 {
        let callback = match callback {
            StringOrFunction::String(i) => TimerCallback::StringTimerCallback(i),
            StringOrFunction::Function(i) => TimerCallback::FunctionTimerCallback(i),
        };
        self.as_global_scope().set_timeout_or_interval(
            callback,
            args,
            Duration::from_millis(timeout.max(0) as u64),
            IsInterval::NonInterval,
        )
    }

    //! @brief Clears a scheduled timeout.
    //!
    //! Functional Utility: Implements the `window.clearTimeout()` DOM method.
    //!                     This method cancels a timeout previously established
    //!                     by a call to `setTimeout()`, preventing the scheduled
    //!                     callback from executing.
    //!
    //! @param handle The integer ID returned by `setTimeout()` that identifies
    //!               the timeout to be cleared.
    //! @see https://html.spec.whatwg.org/multipage/#dom-windowtimers-cleartimeout
    fn ClearTimeout(&self, handle: i32) {
        self.as_global_scope().clear_timeout_or_interval(handle);
    }

    //! @brief Schedules a function or code snippet to be executed repeatedly at a fixed interval.
    //!
    //! Functional Utility: Implements the `window.setInterval()` DOM method.
    //!                     This method repeatedly calls a `callback` (either a function
    //!                     or a string of code) with a fixed time delay between each call.
    //!                     Any additional `args` are passed to the callback function.
    //!                     The returned integer ID can be used to clear the interval.
    //!
    //! @param _cx The JavaScript context (unused in this implementation).
    //! @param callback The `StringOrFunction` representing the code to execute.
    //! @param timeout The delay in milliseconds between each execution of the callback.
    //! @param args A vector of `HandleValue` arguments to pass to the callback.
    //! @return An integer ID that uniquely identifies the scheduled interval.
    //! @see https://html.spec.whatwg.org/multipage/#dom-windowtimers-setinterval
    fn SetInterval(
        &self,
        _cx: JSContext,
        callback: StringOrFunction,
        timeout: i32,
        args: Vec<HandleValue>,
    ) -> i32 {
        let callback = match callback {
            StringOrFunction::String(i) => TimerCallback::StringTimerCallback(i),
            StringOrFunction::Function(i) => TimerCallback::FunctionTimerCallback(i),
        };
        self.as_global_scope().set_timeout_or_interval(
            callback,
            args,
            Duration::from_millis(timeout.max(0) as u64),
            IsInterval::Interval,
        )
    }

    //! @brief Clears a scheduled interval.
    //!
    //! Functional Utility: Implements the `window.clearInterval()` DOM method.
    //!                     This method cancels a repeating action which was
    //!                     originally set up by a call to `setInterval()`,
    //!                     preventing further executions of its callback.
    //!
    //! @param handle The integer ID returned by `setInterval()` that identifies
    //!               the interval to be cleared.
    //! @see https://html.spec.whatwg.org/multipage/#dom-windowtimers-clearinterval
    fn ClearInterval(&self, handle: i32) {
        self.ClearTimeout(handle);
    }

    //! @brief Queues a microtask to be executed.
    //!
    //! Functional Utility: Implements the `window.queueMicrotask()` DOM method.
    //!                     This method schedules a `callback` function to be
    //!                     executed as a microtask. Microtasks are typically
    //!                     executed after the current script task completes
    //!                     and before the browser's rendering updates or
    //!                     the next macrotask (e.g., event loop iteration).
    //!                     This is crucial for ensuring timely execution of
    //!                     certain asynchronous operations.
    //!
    //! @param callback An `Rc<VoidFunction>` representing the function to be executed.
    //! @see https://html.spec.whatwg.org/multipage/#dom-queuemicrotask
    fn QueueMicrotask(&self, callback: Rc<VoidFunction>) {
        self.as_global_scope().queue_function_as_microtask(callback);
    }

    //! @brief Creates an `ImageBitmap` from a given image source.
    //!
    //! Functional Utility: Implements the `window.createImageBitmap()` DOM method.
    //!                     This asynchronous method efficiently creates a bitmap
    //!                     image from a variety of sources (e.g., `<img>` element,
    //!                     `<canvas>`, `ImageData`). The `ImageBitmap` object
    //!                     is optimized for high-performance drawing to a canvas.
    //!
    //! @param image The `ImageBitmapSource` from which to create the bitmap.
    //! @param options Configuration `ImageBitmapOptions` for the creation process.
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @return A `Rc<Promise>` that resolves with the `ImageBitmap` or rejects on failure.
    //! @see https://html.spec.whatwg.org/multipage/#dom-createimagebitmap
    fn CreateImageBitmap(
        &self,
        image: ImageBitmapSource,
        options: &ImageBitmapOptions,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        let p = self
            .as_global_scope()
            .create_image_bitmap(image, options, can_gc);
        p
    }

    //! @brief Retrieves the `WindowProxy` object for this window.
    //!
    //! Functional Utility: Implements the `window.window` DOM property getter,
    //!                     which returns the `WindowProxy` object itself. This
    //!                     property is essentially a self-reference, providing
    //!                     a consistent way to access the current window's
    //!                     global object, including its properties and methods.
    //!
    //! @return A `DomRoot<WindowProxy>` representing the window's proxy.
    //! @see https://html.spec.whatwg.org/multipage/#dom-window
    fn Window(&self) -> DomRoot<WindowProxy> {
        self.window_proxy()
    }

    //! @brief Retrieves the `WindowProxy` object for this window (self-reference).
    //!
    //! Functional Utility: Implements the `window.self` DOM property getter,
    //!                     which returns the `WindowProxy` object itself. This
    //!                     property is an alias for `window.window`, providing
    //!                     another way to refer to the current window's global
    //!                     object.
    //!
    //! @return A `DomRoot<WindowProxy>` representing the window's proxy.
    //! @see https://html.spec.whatwg.org/multipage/#dom-self
    fn Self_(&self) -> DomRoot<WindowProxy> {
        self.window_proxy()
    }

    //! @brief Retrieves the `WindowProxy` object for this window (frames collection).
    //!
    //! Functional Utility: Implements the `window.frames` DOM property getter,
    //!                     which returns the `WindowProxy` object itself. Historically,
    //!                     this property represented a collection of child frames.
    //!                     In modern DOM, it typically acts as a self-reference to
    //!                     the current window's global object.
    //!
    //! @return A `DomRoot<WindowProxy>` representing the window's proxy.
    //! @see https://html.spec.whatwg.org/multipage/#dom-frames
    fn Frames(&self) -> DomRoot<WindowProxy> {
        self.window_proxy()
    }

    //! @brief Returns the number of child frames in the current window.
    //!
    //! Functional Utility: Implements the `window.length` DOM property getter,
    //!                     which returns the number of `<iframe>` elements
    //!                     directly nested within the current document. This
    //!                     provides a count of immediate child browsing contexts.
    //!
    //! @return A `u32` representing the number of child frames.
    //! @see https://html.spec.whatwg.org/multipage/#accessing-other-browsing-contexts
    fn Length(&self) -> u32 {
        self.Document().iframes().iter().count() as u32