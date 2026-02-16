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
            self.window_proxy.get().expect("Discarding a BC from a window that has none")
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
            writeln!(&mut stdout, "\nALERT: {}", s).unwrap();
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
    //!                     across browser sessions, enabling offline capabilities
    //!                     and persistent user preferences.
    //!
    //! @return A `DomRoot<Storage>` representing the local storage of this window.
    //! @see https://html.spec.whatwg.org/multipage/#dom-localstorage
    fn LocalStorage(&self) -> DomRoot<Storage> {
        self.local_storage
            .or_init(|| Storage::new(self, StorageType::Local))
    }

    // https://dvcs.w3.org/hg/webcrypto-api/raw-file/tip/spec/Overview.html#dfn-GlobalCrypto
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
    }

    //! @brief Retrieves the parent `WindowProxy` for this window.
    //!
    //! Functional Utility: Implements the `window.parent` DOM property getter.
    //!                     If the current window is an iframe or nested browsing
    //!                     context, this method returns a `WindowProxy` for its
    //!                     immediate parent. If the window is a top-level browsing
    //!                     context, it returns a self-reference.
    //!
    //! @return An `Option<DomRoot<WindowProxy>>` representing the parent window, or self if top-level.
    //! @see https://html.spec.whatwg.org/multipage/#dom-parent
    fn GetParent(&self) -> Option<DomRoot<WindowProxy>> {
        // Steps 1-3.
        let window_proxy = self.undiscarded_window_proxy()?;

        // Step 4.
        if let Some(parent) = window_proxy.parent() {
            return Some(DomRoot::from_ref(parent));
        }
        // Step 5.
        Some(window_proxy)
    }

    //! @brief Retrieves the topmost `WindowProxy` in the browsing context hierarchy.
    //!
    //! Functional Utility: Implements the `window.top` DOM property getter.
    //!                     This method returns a `WindowProxy` for the topmost
    //!                     browsing context in the current hierarchy. This is
    //!                     useful for breaking out of nested frames and
    //!                     interacting with the main window.
    //!
    //! @return An `Option<DomRoot<WindowProxy>>` representing the top-level window.
    //! @see https://html.spec.whatwg.org/multipage/#dom-top
    fn GetTop(&self) -> Option<DomRoot<WindowProxy>> {
        // Steps 1-3.
        let window_proxy = self.undiscarded_window_proxy()?;

        // Steps 4-5.
        Some(DomRoot::from_ref(window_proxy.top()))
    }

    //! @brief Retrieves the `Performance` object for this window.
    //!
    //! Functional Utility: Implements the `window.performance` DOM property getter,
    //!                     providing access to the `Performance` object. This object
    //!                     offers a wide range of performance-related metrics, such
    //!                     as navigation timings, resource loading times, and user
    //!                     timing marks, crucial for analyzing and optimizing web
    //!                     application performance.
    //!
    //! @return A `DomRoot<Performance>` representing the performance object of this window.
    //! @see https://dvcs.w3.org/hg/webperf/raw-file/tip/specs/
    //!      NavigationTiming/Overview.html#sec-window.performance-attribute
    fn Performance(&self) -> DomRoot<Performance> {
        self.performance
            .or_init(|| Performance::new(self.as_global_scope(), self.navigation_start.get()))
    }

    //! @brief Retrieves the `Crypto` object for this window.
    //!
    //! Functional Utility: Implements the `window.crypto` DOM property getter,
    //!                     providing access to the `Crypto` object. This object
    //!                     offers cryptographic primitives for secure operations
    //!                     such as generating cryptographically strong random
    //!                     numbers, hashing, and signature verification.
    //!
    //! @return A `DomRoot<Crypto>` representing the crypto object of this window.
    //! @see https://dvcs.w3.org/hg/webcrypto-api/raw-file/tip/spec/Overview.html#dfn-GlobalCrypto
    fn Crypto(&self) -> DomRoot<Crypto> {
        self.as_global_scope().crypto()
    }

    //! @brief Implements standard DOM `GlobalEventHandlers` and `WindowEventHandlers` interfaces.
    //! 
    //! Functional Utility: These macros inject implementations for a wide range of
    //!                     event handler properties (e.g., `onclick`, `onload`, `onresize`)
    //!                     directly into the `Window` object. This provides the standard
    //!                     mechanism for web content to register and respond to user
    //!                     interactions and browser events.
    //! 
    //! @see https://html.spec.whatwg.org/multipage/#globaleventhandlers
    //! @see https://html.spec.whatwg.org/multipage/#windoweventhandlers
    global_event_handlers!();

    window_event_handlers!();

    //! @brief Retrieves the `Screen` object for this window.
    //!
    //! Functional Utility: Implements the `window.screen` DOM property getter,
    //!                     providing access to the `Screen` object. This object
    //!                     contains information about the user's screen, such as
    //!                     its dimensions, available screen space, color depth,
    //!                     and pixel density, enabling web applications to adapt
    //!                     to different display environments.
    //!
    //! @return A `DomRoot<Screen>` representing the screen object of this window.
    //! @see https://developer.mozilla.org/en-US/docs/Web/API/Window/screen
    fn Screen(&self) -> DomRoot<Screen> {
        self.screen.or_init(|| Screen::new(self))
    }

    //! @brief Encodes a string in base64.
    //!
    //! Functional Utility: Implements the `window.btoa()` DOM method, which
    //!                     encodes a string of binary data (octets) into a
    //!                     base64-encoded ASCII string. This method is typically
    //!                     used for encoding data that needs to be transmitted
    //!                     over mediums that only support ASCII characters.
    //!
    //! @param btoa The `DOMString` to be encoded.
    //! @return A `Fallible<DOMString>` containing the base64 encoded string on success.
    //! @see https://html.spec.whatwg.org/multipage/#dom-windowbase64-btoa
    fn Btoa(&self, btoa: DOMString) -> Fallible<DOMString> {
        base64_btoa(btoa)
    }

    //! @brief Decodes a base64 encoded string.
    //!
    //! Functional Utility: Implements the `window.atob()` DOM method, which
    //!                     decodes a base64-encoded ASCII string back into
    //!                     its original binary data representation. This is
    //!                     the counterpart to `btoa()`, used for decoding
    //!                     base64 data received from various sources.
    //!
    //! @param atob The `DOMString` to be decoded.
    //! @return A `Fallible<DOMString>` containing the decoded string on success.
    //! @see https://html.spec.whatwg.org/multipage/#dom-windowbase64-atob
    fn Atob(&self, atob: DOMString) -> Fallible<DOMString> {
        base64_atob(atob)
    }

    //! @brief Schedules a function to run before the browser's next repaint.
    //!
    //! Functional Utility: Implements the `window.requestAnimationFrame()` DOM method.
    //!                     This method tells the browser that you wish to perform
    //!                     an animation and requests that the browser call a
    //!                     specified function to update an animation before the
    //!                     next repaint. This is the preferred method for animations
    //!                     to ensure they are synchronized with the browser's
    //!                     rendering cycle, leading to smoother animations.
    //!
    //! @param callback An `Rc<FrameRequestCallback>` representing the function
    //!                 to be called before the next repaint.
    //! @return A `u32` ID that uniquely identifies the request.
    //! @see https://html.spec.whatwg.org/multipage/#dom-window-requestanimationframe
    fn RequestAnimationFrame(&self, callback: Rc<FrameRequestCallback>) -> u32 {
        self.Document()
            .request_animation_frame(AnimationFrameCallback::FrameRequestCallback { callback })
    }

    //! @brief Cancels a previously scheduled animation frame request.
    //!
    //! Functional Utility: Implements the `window.cancelAnimationFrame()` DOM method.
    //!                     This method cancels an animation frame request that was
    //!                     previously scheduled using `requestAnimationFrame()`,
    //!                     preventing its callback function from being executed
    //!                     before the next repaint.
    //!
    //! @param ident The `u32` ID returned by `requestAnimationFrame()` that identifies
    //!               the request to be canceled.
    //! @see https://html.spec.whatwg.org/multipage/#dom-window-cancelanimationframe
    fn CancelAnimationFrame(&self, ident: u32) {
        let doc = self.Document();
        doc.cancel_animation_frame(ident);
    }

    //! @brief Dispatches a message to other browsing contexts.
    //!
    //! Functional Utility: Implements the `window.postMessage()` DOM method,
    //!                     enabling secure cross-origin communication between
    //!                     `Window` objects. It sends a `message` to a specified
    //!                     `target_origin` and optionally transfers ownership of
    //!                     `JSObject`s (`transfer`). The message is first
    //!                     structured-cloned to ensure it's a valid transferable
    //!                     object.
    //!
    //! @param cx The JavaScript context.
    //! @param message The `HandleValue` representing the message to be sent.
    //! @param target_origin The `USVString` specifying the origin of the target
    //!                      window. Can be `"*"` for any origin.
    //! @param transfer A `CustomAutoRooterGuard` containing `Vec<*mut JSObject>`
    //!                 that are to be transferred.
    //! @return An `ErrorResult` indicating success or failure.
    //! @see https://html.spec.whatwg.org/multipage/#dom-window-postmessage
    fn PostMessage(
        &self,
        cx: JSContext,
        message: HandleValue,
        target_origin: USVString,
        transfer: CustomAutoRooterGuard<Vec<*mut JSObject>>,
    ) -> ErrorResult {
        let incumbent = GlobalScope::incumbent().expect("no incumbent global?");
        let source = incumbent.as_window();
        let source_origin = source.Document().origin().immutable().clone();

        self.post_message_impl(&target_origin, source_origin, source, cx, message, transfer)
    }

    //! @brief Dispatches a message to other browsing contexts using an options dictionary.
    //!
    //! Functional Utility: This is an overloaded version of the `postMessage()`
    //!                     DOM method that accepts an `options` dictionary,
    //!                     providing a more structured way to specify the
    //!                     `targetOrigin` and `transfer` list. It internally
    //!                     prepares the `transfer` list and delegates to the
    //!                     `post_message_impl` for the core message dispatch logic.
    //!
    //! @param cx The JavaScript context.
    //! @param message The `HandleValue` representing the message to be sent.
    //! @param options A `RootedTraceableBox<WindowPostMessageOptions>` containing
    //!                the target origin and transferable objects.
    //! @return An `ErrorResult` indicating success or failure.
    //! @see https://html.spec.whatwg.org/multipage/#dom-messageport-postmessage
    fn PostMessage_(
        &self,
        cx: JSContext,
        message: HandleValue,
        options: RootedTraceableBox<WindowPostMessageOptions>,
    ) -> ErrorResult {
        let mut rooted = CustomAutoRooter::new(
            options
                .parent
                .transfer
                .iter()
                .map(|js: &RootedTraceableBox<Heap<*mut JSObject>>| js.get())
                .collect(),
        );
        let transfer = CustomAutoRooterGuard::new(*cx, &mut rooted);

        let incumbent = GlobalScope::incumbent().expect("no incumbent global?");
        let source = incumbent.as_window();

        let source_origin = source.Document().origin().immutable().clone();

        self.post_message_impl(
            &options.targetOrigin,
            source_origin,
            source,
            cx,
            message,
            transfer,
        )
    }

    //! @brief Intentionally does nothing, part of a legacy API.
    //!
    //! Functional Utility: Implements the `window.captureEvents()` DOM method,
    //!                     which is a legacy method from older DOM event models.
    //!                     In modern web development, this method has no effect
    //!                     and is typically retained for backward compatibility
    //!                     with older scripts.
    //!
    //! @see https://html.spec.whatwg.org/multipage/#dom-window-captureevents
    fn CaptureEvents(&self) {
        // This method intentionally does nothing
    }

    //! @brief Intentionally does nothing, part of a legacy API.
    //!
    //! Functional Utility: Implements the `window.releaseEvents()` DOM method,
    //!                     which is a legacy method from older DOM event models.
    //!                     Similar to `captureEvents()`, this method has no effect
    //!                     in modern web environments and is retained primarily
    //!                     for backward compatibility.
    //!
    //! @see https://html.spec.whatwg.org/multipage/#dom-window-releaseevents
    fn ReleaseEvents(&self) {
        // This method intentionally does nothing
    }

    //! @brief Logs a debug message to the console.
    //!
    //! Functional Utility: This method provides a simple way to output `message`
    //!                     to the debug console. It is primarily used for
    //!                     development and debugging purposes, allowing
    //!                     developers to inspect values or track execution flow.
    //!
    //! @param message The `DOMString` to be logged.
    //! check-tidy: no specs after this line
    fn Debug(&self, message: DOMString) {
        debug!("{}", message);
    }

    //! @brief Forces a JavaScript garbage collection cycle.
    //!
    //! Functional Utility: This method explicitly triggers a garbage collection
    //!                     cycle within the JavaScript runtime associated with
    //!                     this window. It's primarily used for debugging and
    //!                     performance testing to observe memory management
    //!                     behavior, as garbage collection is typically handled
    //!                     automatically by the JavaScript engine.
    //!
    //! Pre-condition: Requires `unsafe_code` to directly interact with the
    //!                JavaScript engine's garbage collector.
    #[allow(unsafe_code)]
    fn Gc(&self) {
        unsafe {
            JS_GC(*self.get_cx(), GCReason::API);
        }
    }

    //! @brief Prints JavaScript and Rust backtraces to the console.
    //!
    //! Functional Utility: This method is a debugging utility that outputs
    //!                     the current JavaScript call stack and the Rust
    //!                     backtrace to the console. It's invaluable for
    //!                     diagnosing issues that span both the JavaScript
    //!                     and Rust layers of the browser engine.
    //!
    //! Pre-condition: Requires `unsafe_code` to interact directly with the
    //!                JavaScript engine's context and to obtain a Rust backtrace.
    #[allow(unsafe_code)]
    fn Js_backtrace(&self) {
        unsafe {
            println!("Current JS stack:");
            dump_js_stack(*self.get_cx());
            let rust_stack = Backtrace::new();
            println!("Current Rust stack:\n{:?}", rust_stack);
        }
    }

    //! @brief Sends JavaScript evaluation results to the WebDriver server.
    //!
    //! Functional Utility: This method is used to transmit the result of a
    //!                     JavaScript execution (`val`) back to a connected
    //!                     WebDriver server. It converts the JavaScript value
    //!                     into a WebDriver-compatible format and sends it
    //!                     through the `webdriver_script_chan`. This is a
    //!                     critical component for automated browser testing.
    //!
    //! @param cx The JavaScript context.
    //! @param val The `HandleValue` representing the JavaScript result to be sent.
    //! @see The WebDriver specification for asynchronous script execution.
    #[allow(unsafe_code)]
    fn WebdriverCallback(&self, cx: JSContext, val: HandleValue) {
        let rv = unsafe { jsval_to_webdriver(*cx, &self.globalscope, val) };
        let opt_chan = self.webdriver_script_chan.borrow_mut().take();
        if let Some(chan) = opt_chan {
            chan.send(rv).unwrap();
        }
    }

    //! @brief Notifies the WebDriver server of a script execution timeout.
    //!
    //! Functional Utility: This method is invoked when a script executed
    //!                     under WebDriver control exceeds its allocated
    //!                     timeout duration. It takes ownership of the
    //!                     `webdriver_script_chan` and sends a `WebDriverJSError::Timeout`
    //!                     error, informing the WebDriver client about the
    //!                     timeout condition.
    //!
    //! @see The WebDriver specification for asynchronous script execution timeouts.
    fn WebdriverTimeout(&self) {
        let opt_chan = self.webdriver_script_chan.borrow_mut().take();
        if let Some(chan) = opt_chan {
            chan.send(Err(WebDriverJSError::Timeout)).unwrap();
        }
    }

    //! @brief Retrieves the computed style of an element.
    //!
    //! Functional Utility: Implements the `window.getComputedStyle()` DOM method.
    //!                     This method returns a `CSSStyleDeclaration` object
    //!                     containing the final resolved values of all CSS
    //!                     properties for the given `element`, after applying
    //!                     all active stylesheets and resolving any conflicts.
    //!                     It also supports querying the style of pseudo-elements.
    //!
    //! @param element A reference to the `Element` for which to get the computed style.
    //! @param pseudo An `Option<DOMString>` specifying the pseudo-element to style (e.g., ":before", "::after").
    //! @return A `DomRoot<CSSStyleDeclaration>` representing the computed style.
    //! @see https://drafts.csswg.org/cssom/#dom-window-getcomputedstyle
    fn GetComputedStyle(
        &self,
        element: &Element,
        pseudo: Option<DOMString>,
    ) -> DomRoot<CSSStyleDeclaration> {
        // Steps 1-4.
        let pseudo = pseudo.map(|mut s| {
            s.make_ascii_lowercase();
            s
        });
        let pseudo = match pseudo {
            Some(ref pseudo) if pseudo == ":before" || pseudo == "::before" => {
                Some(PseudoElement::Before)
            },
            Some(ref pseudo) if pseudo == ":after" || pseudo == "::after" => {
                Some(PseudoElement::After)
            },
            _ => None,
        };

        // Step 5.
        CSSStyleDeclaration::new(
            self,
            CSSStyleOwner::Element(Dom::from_ref(element)),
            pseudo,
            CSSModificationAccess::Readonly,
        )
    }

    //! @brief Returns the inner height of the window's layout viewport.
    //!
    //! Functional Utility: Implements the `window.innerHeight` DOM property getter.
    //!                     This method returns the height of the layout viewport
    //!                     of the window, in CSS pixels, excluding any scrollbars
    //!                     (although the implementation notes a TODO to include them).
    //!                     It's crucial for responsive design and layout calculations.
    //!
    //! @return An `i32` representing the inner height of the window.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-innerheight
    //TODO Include Scrollbar
    fn InnerHeight(&self) -> i32 {
        self.window_size
            .get()
            .initial_viewport
            .height
            .to_i32()
            .unwrap_or(0)
    }

    //! @brief Returns the inner width of the window's layout viewport.
    //!
    //! Functional Utility: Implements the `window.innerWidth` DOM property getter.
    //!                     This method returns the width of the layout viewport
    //!                     of the window, in CSS pixels, excluding any scrollbars
    //!                     (although the implementation notes a TODO to include them).
    //!                     It's crucial for responsive design and layout calculations.
    //!
    //! @return An `i32` representing the inner width of the window.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-innerwidth
    //TODO Include Scrollbar
    fn InnerWidth(&self) -> i32 {
        self.window_size
            .get()
            .initial_viewport
            .width
            .to_i32()
            .unwrap_or(0)
    }

    //! @brief Returns the x-coordinate of the current viewport's scroll position.
    //!
    //! Functional Utility: Implements the `window.scrollX` DOM property getter.
    //!                     This method returns the number of pixels that the
    //!                     document has been scrolled horizontally from the
    //!                     left edge of its initial containing block.
    //!
    //! @return An `i32` representing the horizontal scroll offset.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-scrollx
    fn ScrollX(&self) -> i32 {
        self.current_viewport.get().origin.x.to_px()
    }

    //! @brief Returns the x-coordinate of the current viewport's scroll position (alias for `scrollX`).
    //!
    //! Functional Utility: Implements the `window.pageXOffset` DOM property getter,
    //!                     which is an alias for `window.scrollX`. It returns the
    //!                     number of pixels that the document has been scrolled
    //!                     horizontally from the left edge.
    //!
    //! @return An `i32` representing the horizontal scroll offset.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-pagexoffset
    fn PageXOffset(&self) -> i32 {
        self.ScrollX()
    }

    //! @brief Returns the y-coordinate of the current viewport's scroll position.
    //!
    //! Functional Utility: Implements the `window.scrollY` DOM property getter.
    //!                     This method returns the number of pixels that the
    //!                     document has been scrolled vertically from the
    //!                     top edge of its initial containing block.
    //!
    //! @return An `i32` representing the vertical scroll offset.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-scrolly
    fn ScrollY(&self) -> i32 {
        self.current_viewport.get().origin.y.to_px()
    }

    //! @brief Returns the y-coordinate of the current viewport's scroll position (alias for `scrollY`).
    //!
    //! Functional Utility: Implements the `window.pageYOffset` DOM property getter,
    //!                     which is an alias for `window.scrollY`. It returns the
    //!                     number of pixels that the document has been scrolled
    //!                     vertically from the top edge.
    //!
    //! @return An `i32` representing the vertical scroll offset.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-pageyoffset
    fn PageYOffset(&self) -> i32 {
        self.ScrollY()
    }

    //! @brief Scrolls the window to a specific position using options.
    //!
    //! Functional Utility: Implements the `window.scroll(options)` DOM method.
    //!                     This method scrolls the document to the coordinates
    //!                     specified in the `options` dictionary, taking into
    //!                      account the desired scrolling `behavior` (e.g., "auto", "smooth").
    //!
    //! @param options A reference to `ScrollToOptions` containing target `left`, `top`, and `behavior`.
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-scroll
    fn Scroll(&self, options: &ScrollToOptions, can_gc: CanGc) {
        // Step 1
        let left = options.left.unwrap_or(0.0f64);
        let top = options.top.unwrap_or(0.0f64);
        self.scroll(left, top, options.parent.behavior, can_gc);
    }

    //! @brief Scrolls the window to a specific position using coordinates.
    //!
    //! Functional Utility: This is an overloaded version of the `window.scroll()`
    //!                     DOM method that accepts `x` and `y` coordinates
    //!                     directly. It performs a scroll with the default
    //!                     "auto" behavior.
    //!
    //! @param x The horizontal coordinate to scroll to.
    //! @param y The vertical coordinate to scroll to.
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-scroll
    fn Scroll_(&self, x: f64, y: f64, can_gc: CanGc) {
        self.scroll(x, y, ScrollBehavior::Auto, can_gc);
    }

    //! @brief Scrolls the window to a specific position using options.
    //!
    //! Functional Utility: Implements the `window.scrollTo(options)` DOM method.
    //!                     This method scrolls the document to the position
    //!                     specified by the `options` dictionary. It is functionally
    //!                     equivalent to `window.scroll(options)`.
    //!
    //! @param options A reference to `ScrollToOptions` containing target `left`, `top`, and `behavior`.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-scrollto
    fn ScrollTo(&self, options: &ScrollToOptions) {
        self.Scroll(options, CanGc::note());
    }

    //! @brief Scrolls the window to a specific position using coordinates.
    //!
    //! Functional Utility: This is an overloaded version of the `window.scrollTo()`
    //!                     DOM method that accepts `x` and `y` coordinates
    //!                     directly. It performs a scroll with the default
    //!                     "auto" behavior. It is functionally equivalent to
    //!                     `window.scroll(x, y)`.
    //!
    //! @param x The horizontal coordinate to scroll to.
    //! @param y The vertical coordinate to scroll to.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-scrollto
    fn ScrollTo_(&self, x: f64, y: f64) {
        self.scroll(x, y, ScrollBehavior::Auto, CanGc::note());
    }

    //! @brief Scrolls the window by a specified amount using options.
    //!
    //! Functional Utility: Implements the `window.scrollBy(options)` DOM method.
    //!                     This method scrolls the document by the amounts
    //!                     specified in the `options` dictionary, relative to
    //!                     its current scroll position. It then performs the
    //!                     scroll with the specified `behavior`.
    //!
    //! @param options A reference to `ScrollToOptions` containing `left`, `top`
    //!                (amounts to scroll by), and `behavior`.
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-scrollby
    fn ScrollBy(&self, options: &ScrollToOptions, can_gc: CanGc) {
        // Step 1
        let x = options.left.unwrap_or(0.0f64);
        let y = options.top.unwrap_or(0.0f64);
        self.ScrollBy_(x, y, can_gc);
        self.scroll(x, y, options.parent.behavior, can_gc);
    }

    //! @brief Scrolls the window by a specified amount using coordinates.
    //!
    //! Functional Utility: This is an overloaded version of the `window.scrollBy()`
    //!                     DOM method that accepts `x` and `y` coordinates
    //!                     directly. It calculates the new scroll position by
    //!                     adding `x` to the current `scrollX` and `y` to the
    //!                     current `scrollY`, then performs a scroll with
    //!                     "auto" behavior.
    //!
    //! @param x The horizontal amount to scroll by.
    //! @param y The vertical amount to scroll by.
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-scrollby
    fn ScrollBy_(&self, x: f64, y: f64, can_gc: CanGc) {
        // Step 3
        let left = x + self.ScrollX() as f64;
        // Step 4
        let top = y + self.ScrollY() as f64;

        // Step 5
        self.scroll(left, top, ScrollBehavior::Auto, can_gc);
    }

    //! @brief Resizes the window to the specified dimensions.
    //!
    //! Functional Utility: Implements the `window.resizeTo()` DOM method.
    //!                     This method resizes the current window to a new
    //!                     width and height in CSS pixels. The actual resizing
    //!                     is delegated to the embedder process after converting
    //!                     the CSS pixel dimensions to device pixels.
    //!
    //! @param width The new width for the window, in CSS pixels.
    //! @param height The new height for the window, in CSS pixels.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-resizeto
    fn ResizeTo(&self, width: i32, height: i32) {
        // Step 1
        //TODO determine if this operation is allowed
        let dpr = self.device_pixel_ratio();
        let size = Size2D::new(width, height).to_f32() * dpr;
        self.send_to_embedder(EmbedderMsg::ResizeTo(self.webview_id(), size.to_i32()));
    }

    //! @brief Resizes the window by the specified amounts.
    //!
    //! Functional Utility: Implements the `window.resizeBy()` DOM method.
    //!                     This method resizes the current window by adding
    //!                     the specified `x` and `y` amounts to its current
    //!                     width and height, respectively. The resizing is
    //!                     delegated to the `ResizeTo` method after calculating
    //!                     the new absolute dimensions.
    //!
    //! @param x The amount to add to the window's current width.
    //! @param y The amount to add to the window's current height.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-resizeby
    fn ResizeBy(&self, x: i32, y: i32) {
        let (size, _) = self.client_window();
        // Step 1
        self.ResizeTo(
            x + size.width.to_i32().unwrap_or(1),
            y + size.height.to_i32().unwrap_or(1),
        )
    }

    //! @brief Moves the window to a specified position on the screen.
    //!
    //! Functional Utility: Implements the `window.moveTo()` DOM method.
    //!                     This method moves the top-left corner of the
    //!                     window to the absolute screen coordinates (`x`, `y`).
    //!                     The actual moving operation is delegated to the
    //!                     embedder process after converting CSS pixel
    //!                     coordinates to device pixels.
    //!
    //! @param x The new x-coordinate for the window's top-left corner.
    //! @param y The new y-coordinate for the window's top-left corner.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-moveto
    fn MoveTo(&self, x: i32, y: i32) {
        // Step 1
        //TODO determine if this operation is allowed
        let dpr = self.device_pixel_ratio();
        let point = Point2D::new(x, y).to_f32() * dpr;
        let msg = EmbedderMsg::MoveTo(self.webview_id(), point.to_i32());
        self.send_to_embedder(msg);
    }

    //! @brief Moves the window by a specified offset relative to its current position.
    //!
    //! Functional Utility: Implements the `window.moveBy()` DOM method.
    //!                     This method moves the window by adding the specified
    //!                     `x` and `y` offsets to its current screen position.
    //!                     The actual movement is delegated to the `MoveTo` method
    //!                     after calculating the new absolute screen coordinates.
    //!
    //! @param x The horizontal offset to move the window by.
    //! @param y The vertical offset to move the window by.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-moveby
    fn MoveBy(&self, x: i32, y: i32) {
        let (_, origin) = self.client_window();
        // Step 1
        self.MoveTo(x + origin.x, y + origin.y)
    }

    //! @brief Returns the horizontal coordinate of the window relative to the screen.
    //!
    //! Functional Utility: Implements the `window.screenX` DOM property getter.
    //!                     This method returns the horizontal (x) coordinate
    //!                     of the window's left border relative to the left
    //!                     edge of the screen, in CSS pixels.
    //!
    //! @return An `i32` representing the x-coordinate of the window on the screen.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-screenx
    fn ScreenX(&self) -> i32 {
        let (_, origin) = self.client_window();
        origin.x
    }

    //! @brief Returns the vertical coordinate of the window relative to the screen.
    //!
    //! Functional Utility: Implements the `window.screenY` DOM property getter.
    //!                     This method returns the vertical (y) coordinate
    //!                     of the window's top border relative to the top
    //!                     edge of the screen, in CSS pixels.
    //!
    //! @return An `i32` representing the y-coordinate of the window on the screen.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-screeny
    fn ScreenY(&self) -> i32 {
        let (_, origin) = self.client_window();
        origin.y
    }

    //! @brief Returns the outer height of the window.
    //!
    //! Functional Utility: Implements the `window.outerHeight` DOM property getter.
    //!                     This method returns the height of the entire browser
    //!                     window, including toolbars, scrollbars, and window
    //!                     decorations, in CSS pixels.
    //!
    //! @return An `i32` representing the outer height of the window.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-outerheight
    fn OuterHeight(&self) -> i32 {
        let (size, _) = self.client_window();
        size.height.to_i32().unwrap_or(1)
    }

    //! @brief Returns the outer width of the window.
    //!
    //! Functional Utility: Implements the `window.outerWidth` DOM property getter.
    //!                     This method returns the width of the entire browser
    //!                     window, including toolbars, scrollbars, and window
    //!                     decorations, in CSS pixels.
    //!
    //! @return An `i32` representing the outer width of the window.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-outerwidth
    fn OuterWidth(&self) -> i32 {
        let (size, _) = self.client_window();
        size.width.to_i32().unwrap_or(1)
    }

    //! @brief Returns the device pixel ratio of the current display.
    //!
    //! Functional Utility: Implements the `window.devicePixelRatio` DOM property getter.
    //!                     This method returns the ratio between the physical pixels
    //!                     of the display device and the CSS pixels used by the browser.
    //!                     It is crucial for high-DPI displays to ensure that content
    //!                     is rendered sharply.
    //!
    //! @return A `Finite<f64>` representing the device pixel ratio.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-devicepixelratio
    fn DevicePixelRatio(&self) -> Finite<f64> {
        Finite::wrap(self.device_pixel_ratio().get() as f64)
    }

    //! @brief Retrieves the status bar message for the window.
    //!
    //! Functional Utility: Implements the `window.status` DOM property getter.
    //!                     This method returns the text message currently set
    //!                     for display in the browser's status bar. In modern
    //!                     browsers, direct manipulation of the status bar is
    //!                     often restricted for security reasons.
    //!
    //! @return A `DOMString` representing the status bar message.
    //! @see https://html.spec.whatwg.org/multipage/#dom-window-status
    fn Status(&self) -> DOMString {
        self.status.borrow().clone()
    }

    //! @brief Sets the status bar message for the window.
    //!
    //! Functional Utility: Implements the `window.status` DOM property setter.
    //!                     This method attempts to set the text message for
    //!                     display in the browser's status bar. Due to security
    //!                     restrictions in modern browsers, direct modification
    //!                     of the status bar content by scripts is often limited
    //!                     or ignored.
    //!
    //! @param status The `DOMString` to set as the new status bar message.
    //! @see https://html.spec.whatwg.org/multipage/#dom-window-status
    fn SetStatus(&self, status: DOMString) {
        *self.status.borrow_mut() = status
    }

    //! @brief Creates a `MediaQueryList` object for a given media query string.
    //!
    //! Functional Utility: Implements the `window.matchMedia()` DOM method.
    //!                     This method allows web content to programmatically
    //!                     test a media query string and receive notifications
    //!                     when the media query's state changes (i.e., when it
    //!                     starts or stops matching the current environment).
    //!                     It parses the query, creates a `MediaQueryList` instance,
    //!                     and tracks it for updates.
    //!
    //! @param query The `DOMString` representing the media query to be evaluated.
    //! @return A `DomRoot<MediaQueryList>` representing the created media query list.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-matchmedia
    fn MatchMedia(&self, query: DOMString) -> DomRoot<MediaQueryList> {
        let mut input = ParserInput::new(&query);
        let mut parser = Parser::new(&mut input);
        let url_data = UrlExtraData(self.get_url().get_arc());
        let quirks_mode = self.Document().quirks_mode();
        let context = CssParserContext::new(
            Origin::Author,
            &url_data,
            Some(CssRuleType::Media),
            ParsingMode::DEFAULT,
            quirks_mode,
            /* namespaces = */ Default::default(),
            self.css_error_reporter(),
            None,
        );
        let media_query_list = media_queries::MediaList::parse(&context, &mut parser);
        let document = self.Document();
        let mql = MediaQueryList::new(&document, media_query_list);
        self.media_query_lists.track(&*mql);
        mql
    }

    //! @brief Initiates a network request to fetch a resource.
    //!
    //! Functional Utility: Implements the `window.fetch()` DOM method,
    //!                     providing a modern, promise-based interface
    //!                     for making network requests. This method can
    //!                     handle various types of `input` (URL or `Request` object)
    //!                     and `init` options (e.g., method, headers, body),
    //!                     returning a `Promise` that resolves to the `Response`.
    //!
    //! @param input The `RequestOrUSVString` representing the resource to fetch.
    //! @param init A `RootedTraceableBox<RequestInit>` containing options for the request.
    //! @param comp An `InRealm` proof for the current realm.
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @return A `Rc<Promise>` that resolves to the `Response` object.
    //! @see https://fetch.spec.whatwg.org/#fetch-method
    fn Fetch(
        &self,
        input: RequestOrUSVString,
        init: RootedTraceableBox<RequestInit>,
        comp: InRealm,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        fetch::Fetch(self.upcast(), input, init, comp, can_gc)
    }

    //! @brief Retrieves the `TestRunner` object for this window.
    //!
    //! Functional Utility: This method provides access to the `TestRunner`
    //!                     instance, which is used for executing web platform
    //!                     tests within the context of this browsing window.
    //!                     It's typically enabled under specific build
    //!                     configurations (e.g., when the "bluetooth" feature
    //!                     is active) for automated testing purposes.
    //!
    //! @return A `DomRoot<TestRunner>` representing the test runner for this window.
    #[cfg(feature = "bluetooth")]
    fn TestRunner(&self) -> DomRoot<TestRunner> {
        self.test_runner.or_init(|| TestRunner::new(self.upcast()))
    }

    //! @brief Returns the number of running animations in the document.
    //!
    //! Functional Utility: This method provides a count of all currently active
    //!                     animations within the document associated with this
    //!                     window. It's useful for determining the animation
    //!                     state of the page, potentially for performance
    //!                     monitoring or to control when other operations can
    //!                     safely occur without interfering with animations.
    //!
    //! @return A `u32` representing the count of running animations.
    fn RunningAnimationCount(&self) -> u32 {
        self.document
            .get()
            .map_or(0, |d| d.animations().running_animation_count() as u32)
    }

    //! @brief Sets the name of the window.
    //!
    //! Functional Utility: Implements the `window.name` DOM property setter.
    //!                     This method allows a script to set the name of the
    //!                     current browsing context. The name can be used as
    //!                     a target for hyperlinks or form submissions from
    //!                     other browsing contexts. The name is persistent
    //!                     across navigations within the same browsing context.
    //!
    //! @param name The `DOMString` to set as the new name of the window.
    //! @see https://html.spec.whatwg.org/multipage/#dom-name
    fn SetName(&self, name: DOMString) {
        if let Some(proxy) = self.undiscarded_window_proxy() {
            proxy.set_name(name);
        }
    }

    //! @brief Retrieves the name of the window.
    //!
    //! Functional Utility: Implements the `window.name` DOM property getter.
    //!                     This method returns the name of the current browsing
    //!                     context as a `DOMString`. If the window's browsing
    //!                     context has been discarded, an empty string is returned.
    //!
    //! @return A `DOMString` representing the name of the window.
    //! @see https://html.spec.whatwg.org/multipage/#dom-name
    fn Name(&self) -> DOMString {
        match self.undiscarded_window_proxy() {
            Some(proxy) => proxy.get_name(),
            None => "".into(),
        }
    }

    //! @brief Retrieves the origin of the document.
    //!
    //! Functional Utility: Implements the `window.origin` DOM property getter.
    //!                     This method returns the `USVString` representation
    //!                     of the origin of the document currently loaded in
    //!                     the window. The origin is a fundamental security
    //!                     concept used to enforce the Same-Origin Policy.
    //!
    //! @return A `USVString` representing the document's origin.
    //! @see https://html.spec.whatwg.org/multipage/#dom-origin
    fn Origin(&self) -> USVString {
        USVString(self.origin().immutable().ascii_serialization())
    }

    //! @brief Retrieves the `Selection` object for the current document.
    //!
    //! Functional Utility: Implements the `window.getSelection()` DOM method.
    //!                     This method returns a `Selection` object that
    //!                     represents the range of text selected by the user
    //!                     or the current position of the caret in the document.
    //!                     It provides methods to programmatically manipulate
    //!                     the document's selection.
    //!
    //! @return An `Option<DomRoot<Selection>>` representing the current selection, or `None`.
    //! @see https://w3c.github.io/selection-api/#dom-window-getselection
    fn GetSelection(&self) -> Option<DomRoot<Selection>> {
        self.document.get().and_then(|d| d.GetSelection())
    }

    //! @brief Retrieves the currently dispatched `Event` object.
    //!
    //! Functional Utility: Implements the `window.event` DOM property getter.
    //!                     This method returns the `Event` object that is
    //!                     currently being dispatched, allowing event handlers
    //!                     to access properties of the event (e.g., target, type).
    //!                     It leverages `unsafe_code` to convert the `JSObject`
    //!                     of the event to a `JSVal`.
    //!
    //! @param cx The JavaScript context.
    //! @param rval A mutable handle to the JavaScript value where the event object will be stored.
    //! @see https://dom.spec.whatwg.org/#dom-window-event
    #[allow(unsafe_code)]
    fn Event(&self, cx: JSContext, rval: MutableHandleValue) {
        if let Some(ref event) = *self.current_event.borrow() {
            unsafe {
                event.reflector().get_jsobject().to_jsval(*cx, rval);
            }
        }
    }

    //! @brief Checks if the current browsing context is secure.
    //!
    //! Functional Utility: This method indicates whether the current browsing
    //!                     context (i.e., the environment in which the document
    //!                     is running) is considered secure. A secure context
    //!                     is one that meets certain minimum standards of
    //!                     authenticity and confidentiality (e.g., loaded
    //!                     via HTTPS). Many powerful web platform features
    //!                     are only available in secure contexts.
    //!
    //! @return `true` if the context is secure, `false` otherwise.
    fn IsSecureContext(&self) -> bool {
        self.as_global_scope().is_secure_context()
    }

    //! @brief Implements the named property getter for the `Window` object.
    //! 
    //! Functional Utility: This method handles property access on the `Window`
    //!                     object using a name (e.g., `window.myFrame`). It
    //!                     searches for a browsing context (iframe) with the
    //!                     given name, then for elements with a matching `name`
    //!                     or `id` attribute. If multiple matches are found,
    //!                     it returns an `HTMLCollection`.
    //! 
    //! @param name The `DOMString` representing the name of the property to retrieve.
    //! @return An `Option<NamedPropertyValue>` which can be a `WindowProxy`,
    //!         an `Element`, or an `HTMLCollection`, or `None` if no match is found.
    //! @see https://html.spec.whatwg.org/multipage/#named-access-on-the-window-object
    //! @brief Implements the named property getter for the `Window` object.
    //!
    //! Functional Utility: This method handles property access on the `Window`
    //!                     object using a name (e.g., `window.myFrame`). It
    //!                     searches for a browsing context (iframe) with the
    //!                     given name, then for elements with a matching `name`
    //!                     or `id` attribute. If multiple matches are found,
    //!                     it returns an `HTMLCollection`.
    //!
    //! @param name The `DOMString` representing the name of the property to retrieve.
    //! @return An `Option<NamedPropertyValue>` which can be a `WindowProxy`,
    //!         an `Element`, or an `HTMLCollection`, or `None` if no match is found.
    //! @see https://html.spec.whatwg.org/multipage/#named-access-on-the-window-object
    fn NamedGetter(&self, name: DOMString) -> Option<NamedPropertyValue> {
        if name.is_empty() {
            return None;
        }
        let document = self.Document();

        // https://html.spec.whatwg.org/multipage/#document-tree-child-browsing-context-name-property-set
        let iframes: Vec<_> = document
            .iframes()
            .iter()
            .filter(|iframe| {
                if let Some(window) = iframe.GetContentWindow() {
                    return window.get_name() == name;
                }
                false
            })
            .collect();

        let iframe_iter = iframes.iter().map(|iframe| iframe.upcast::<Element>());

        let name = Atom::from(&*name);

        // Step 1.
        let elements_with_name = document.get_elements_with_name(&name);
        let name_iter = elements_with_name
            .iter()
            .map(|element| &**element)
            .filter(|elem| is_named_element_with_name_attribute(elem));
        let elements_with_id = document.get_elements_with_id(&name);
        let id_iter = elements_with_id
            .iter()
            .map(|element| &**element)
            .filter(|elem| is_named_element_with_id_attribute(elem));

        // Step 2.
        for elem in iframe_iter.clone() {
            if let Some(nested_window_proxy) = elem
                .downcast::<HTMLIFrameElement>()
                .and_then(|iframe| iframe.GetContentWindow())
            {
                return Some(NamedPropertyValue::WindowProxy(nested_window_proxy));
            }
        }

        let mut elements = iframe_iter.chain(name_iter).chain(id_iter);

        let first = elements.next()?;

        if elements.next().is_none() {
            // Step 3.
            return Some(NamedPropertyValue::Element(DomRoot::from_ref(first)));
        }

        // Step 4.
        #[derive(JSTraceable, MallocSizeOf)]
        struct WindowNamedGetter {
            #[no_trace]
            name: Atom,
        }
        impl CollectionFilter for WindowNamedGetter {
            fn filter(&self, elem: &Element, _root: &Node) -> bool {
                let type_ = match elem.upcast::<Node>().type_id() {
                    NodeTypeId::Element(t) => t,
                    _ => return false,
                };
                if elem.get_id().as_ref() == Some(&self.name) {
                    return true;
                }
                match type_ {
                    HTMLElementTypeId::HTMLEmbedElement |
                    HTMLElementTypeId::HTMLFormElement |
                    HTMLElementTypeId::HTMLImageElement |
                    HTMLElementTypeId::HTMLObjectElement => {
                        elem.get_name().as_ref() == Some(&self.name)
                    },
                    _ => false,
                }
            }
        }
        let collection = HTMLCollection::create(
            self,
            document.upcast(),
            Box::new(WindowNamedGetter { name }),
        );
        Some(NamedPropertyValue::HTMLCollection(collection))
    }

    //! @brief Returns a list of supported named properties for the `Window` object.
    //!
    //! Functional Utility: Implements the `window.SupportedPropertyNames()` method,
    //!                     which enumerates the names of properties that can be
    //!                     accessed via named access on the `Window` object.
    //!                     This includes the `name` and `id` attributes of certain
    //!                     HTML elements, as well as the names of browsing contexts
    //!                     (iframes). The method collects and sorts these names.
    //!
    //! @return A `Vec<DOMString>` containing the sorted list of supported property names.
    //! @see https://html.spec.whatwg.org/multipage/#dom-tree-accessors:supported-property-names
    fn SupportedPropertyNames(&self) -> Vec<DOMString> {
        let mut names_with_first_named_element_map: HashMap<&Atom, &Element> = HashMap::new();

        let document = self.Document();
        let name_map = document.name_map();
        for (name, elements) in &name_map.0 {
            if name.is_empty() {
                continue;
            }
            let mut name_iter = elements
                .iter()
                .filter(|elem| is_named_element_with_name_attribute(elem));
            if let Some(first) = name_iter.next() {
                names_with_first_named_element_map.insert(name, first);
            }
        }
        let id_map = document.id_map();
        for (id, elements) in &id_map.0 {
            if id.is_empty() {
                continue;
            }
            let mut id_iter = elements
                .iter()
                .filter(|elem| is_named_element_with_id_attribute(elem));
            if let Some(first) = id_iter.next() {
                match names_with_first_named_element_map.entry(id) {
                    Entry::Vacant(entry) => drop(entry.insert(first)),
                    Entry::Occupied(mut entry) => {
                        if first.upcast::<Node>().is_before(entry.get().upcast()) {
                            *entry.get_mut() = first;
                        }
                    },
                }
            }
        }

        let mut names_with_first_named_element_vec: Vec<(&Atom, &Element)> =
            names_with_first_named_element_map
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect();
        names_with_first_named_element_vec.sort_unstable_by(|a, b| {
            if a.1 == b.1 {
                // This can happen if an img has an id different from its name,
                // spec does not say which string to put first.
                a.0.cmp(b.0)
            } else if a.1.upcast::<Node>().is_before(b.1.upcast::<Node>()) {
                cmp::Ordering::Less
            } else {
                cmp::Ordering::Greater
            }
        });

        names_with_first_named_element_vec
            .iter()
            .map(|(k, _v)| DOMString::from(&***k))
            .collect()
    }

    //! @brief Creates a structured clone of a JavaScript value.
    //! 
    //! Functional Utility: Implements the `window.structuredClone()` DOM method.
    //!                     This method creates a deep copy of a given JavaScript
    //!                     `value`, which can include complex objects like `Date`,
    //!                     `RegExp`, `Map`, `Set`, and `ArrayBuffer`, and
    //!                     optionally transfers ownership of certain objects
    //!                     (e.g., `ArrayBuffer`s). It's crucial for cross-origin
    //!                     messaging and IndexedDB storage.
    //! 
    //! @param cx The JavaScript context.
    //! @param value The `HandleValue` to be structured-cloned.
    //! @param options A `RootedTraceableBox<StructuredSerializeOptions>` containing
    //!                options for the cloning process (e.g., `transfer` list).
    //! @param retval A `MutableHandleValue` where the cloned value will be stored.
    //! @return A `Fallible` indicating success or failure.
    //! @see https://html.spec.whatwg.org/multipage/#dom-structuredclone
    fn StructuredClone(
        &self,
        cx: JSContext,
        value: HandleValue,
        options: RootedTraceableBox<StructuredSerializeOptions>,
        retval: MutableHandleValue,
    ) -> Fallible<()> {
        self.as_global_scope()
            .structured_clone(cx, value, options, retval)
    }
}

impl Window {
    // https://heycam.github.io/webidl/#named-properties-object
    // https://html.spec.whatwg.org/multipage/#named-access-on-the-window-object
    //! @brief Creates the named properties object for the `Window`.
    //!
    //! Functional Utility: This method initializes and constructs the JavaScript
    //!                     object that handles named property access on the
    //!                     `Window` object. It configures the object with a
    //!                     given prototype and sets it up to manage properties
    //!                     like frame names and element IDs, as defined by
    //!                     the WebIDL and HTML specifications for named access.
    //!
    //! @param cx The JavaScript context.
    //! @param proto The `HandleObject` representing the prototype for the new object.
    //! @param object A `MutableHandleObject` where the new named properties object will be stored.
    #[allow(unsafe_code)]
    pub(crate) fn create_named_properties_object(
        cx: JSContext,
        proto: HandleObject,
        object: MutableHandleObject,
    ) {
        window_named_properties::create(cx, proto, object)
    }

    //! @brief Retrieves the currently dispatched `Event` object.
    //!
    //! Functional Utility: Returns an `Option` containing a `DomRoot<Event>`
    //!                     representing the `Event` object that is currently
    //!                     in the process of being dispatched through the
    //!                     event target chain. This allows event handlers
    //!                     to access properties of the event.
    //!
    //! @return An `Option<DomRoot<Event>>` representing the current event, or `None`.
    pub(crate) fn current_event(&self) -> Option<DomRoot<Event>> {
        self.current_event
            .borrow()
            .as_ref()
            .map(|e| DomRoot::from_ref(&**e))
    }

    //! @brief Sets the currently dispatched `Event` object.
    //!
    //! Functional Utility: This method updates the `current_event` field to
    //!                     reflect the `Event` object that is currently being
    //!                     dispatched. It also returns the previously active
    //!                     event, allowing for proper nesting and restoration
    //!                     of event contexts during event handling.
    //!
    //! @param event An `Option<&Event>` representing the new current event, or `None` to clear.
    //! @return An `Option<DomRoot<Event>>` representing the event that was previously current.
    pub(crate) fn set_current_event(&self, event: Option<&Event>) -> Option<DomRoot<Event>> {
        let current = self.current_event();
        *self.current_event.borrow_mut() = event.map(Dom::from_ref);
        current
    }

    //! @brief Internal implementation for dispatching messages to other browsing contexts.
    //!
    //! Functional Utility: This private method encapsulates the core logic for
    //!                     handling `postMessage` operations. It performs
    //!                     structured cloning of the message, validates and
    //!                     resolves the `target_origin`, and then dispatches
    //!                     the message to the appropriate browsing context.
    //!                     This method is called by the public `PostMessage`
    //!                     and `PostMessage_` methods.
    //!
    //! @param target_origin The `USVString` specifying the origin of the target
    //!                      window. Can be `"*"` for any origin, or `"/"` for
    //!                      the source origin.
    //! @param source_origin The `ImmutableOrigin` of the sending window.
    //! @param source A reference to the `Window` object that is sending the message.
    //! @param cx The JavaScript context.
    //! @param message The `HandleValue` representing the message to be sent.
    //! @param transfer A `CustomAutoRooterGuard` containing `Vec<*mut JSObject>`
    //!                 that are to be transferred.
    //! @return An `ErrorResult` indicating success or failure.
    //! @see https://html.spec.whatwg.org/multipage/#window-post-message-steps
    fn post_message_impl(
        &self,
        target_origin: &USVString,
        source_origin: ImmutableOrigin,
        source: &Window,
        cx: JSContext,
        message: HandleValue,
        transfer: CustomAutoRooterGuard<Vec<*mut JSObject>>,
    ) -> ErrorResult {
        // Step 1-2, 6-8.
        let data = structuredclone::write(cx, message, Some(transfer))?;

        // Step 3-5.
        let target_origin = match target_origin.0[..].as_ref() {
            "*" => None,
            "/" => Some(source_origin.clone()),
            url => match ServoUrl::parse(url) {
                Ok(url) => Some(url.origin().clone()),
                Err(_) => return Err(Error::Syntax),
            },
        };

        // Step 9.
        self.post_message(target_origin, source_origin, &source.window_proxy(), data);
        Ok(())
    }

    //! @brief Retrieves or creates the Paint Worklet instance.
    //!
    //! Functional Utility: This method implements the `window.paintWorklet`
    //!                     property, which provides access to a `Worklet`
    //!                     instance dedicated to the CSS Paint API. If the
    //!                     Paint Worklet has not yet been initialized, it
    //!                     creates a new one.
    //!
    //! @return A `DomRoot<Worklet>` representing the Paint Worklet.
    //! @see https://drafts.css-houdini.org/css-paint-api-1/#paint-worklet
    pub(crate) fn paint_worklet(&self) -> DomRoot<Worklet> {
        self.paint_worklet.or_init(|| self.new_paint_worklet())
    }

    //! @brief Checks if a document is currently associated with this window.
    //!
    //! Functional Utility: Returns `true` if the `document` field of the
    //!                     `Window` object contains a `Document` instance,
    //!                     indicating that a web page has been loaded or
    //!                     is in the process of being loaded. Returns `false`
    //!                     otherwise.
    //!
    //! @return `true` if a document exists, `false` otherwise.
    pub(crate) fn has_document(&self) -> bool {
        self.document.get().is_some()
    }

    //! @brief Clears and disposes of the JavaScript runtime and associated resources.
    //!
    //! Functional Utility: This method performs a comprehensive cleanup of the
    //!                     JavaScript execution environment and related DOM
    //!                     infrastructure when the `Window` is being torn down.
    //!                     It removes web messaging, dedicated workers, tears down
    //!                     custom elements, triggers garbage collection, and
    //!                     nullifies the JavaScript runtime, transitioning the
    //!                     window to a `Zombie` state. It also handles cleanup
    //!                     of performance entries.
    //!
    //! Pre-condition: This method is typically called during the shutdown
    //!                or destruction of a browsing context.
    pub(crate) fn clear_js_runtime(&self) {
        self.as_global_scope()
            .remove_web_messaging_and_dedicated_workers_infra();

        // Clean up any active promises
        // https://github.com/servo/servo/issues/15318
        if let Some(custom_elements) = self.custom_element_registry.get() {
            custom_elements.teardown();
        }

        // The above code may not catch all DOM objects (e.g. DOM
        // objects removed from the tree that haven't been collected
        // yet). There should not be any such DOM nodes with layout
        // data, but if there are, then when they are dropped, they
        // will attempt to send a message to layout.
        // This causes memory safety issues, because the DOM node uses
        // the layout channel from its window, and the window has
        // already been GC'd.  For nodes which do not have a live
        // pointer, we can avoid this by GCing now:
        self.Gc();
        // but there may still be nodes being kept alive by user
        // script.
        // TODO: ensure that this doesn't happen!

        self.current_state.set(WindowState::Zombie);
        *self.js_runtime.borrow_mut() = None;

        // If this is the currently active pipeline,
        // nullify the window_proxy.
        if let Some(proxy) = self.window_proxy.get() {
            let pipeline_id = self.pipeline_id();
            if let Some(currently_active) = proxy.currently_active() {
                if currently_active == pipeline_id {
                    self.window_proxy.set(None);
                }
            }
        }

        if let Some(performance) = self.performance.get() {
            performance.clear_and_disable_performance_entry_buffer();
        }
        self.as_global_scope()
            .task_manager()
            .cancel_all_tasks_and_ignore_future_tasks();
    }

    //! @brief Scrolls the window to a specified position with optional behavior.
    //!
    //! Functional Utility: This method handles the core logic for scrolling the
    //!                     window's document to a target `x` and `y` coordinate.
    //!                     It clamps the target coordinates within the valid
    //!                     scrolling area, updates the internal viewport state,
    //!                     and then delegates to `perform_a_scroll` to initiate
    //!                     the actual scrolling animation or immediate jump
    //!                     based on the provided `behavior`.
    //!
    //! @param x_ The target horizontal coordinate (in CSS pixels).
    //! @param y_ The target vertical coordinate (in CSS pixels).
    //! @param behavior The `ScrollBehavior` (e.g., "auto", "smooth") for the scroll operation.
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @see https://drafts.csswg.org/cssom-view/#dom-window-scroll
    pub(crate) fn scroll(&self, x_: f64, y_: f64, behavior: ScrollBehavior, can_gc: CanGc) {
        // Step 3
        let xfinite = if x_.is_finite() { x_ } else { 0.0f64 };
        let yfinite = if y_.is_finite() { y_ } else { 0.0f64 };

        // TODO Step 4 - determine if a window has a viewport

        // Step 5 & 6
        // TODO: Remove scrollbar dimensions.
        let viewport = self.window_size.get().initial_viewport;

        // Step 7 & 8
        // TODO: Consider `block-end` and `inline-end` overflow direction.
        let scrolling_area = self.scrolling_area_query(None, can_gc);
        let x = xfinite
            .min(scrolling_area.width() as f64 - viewport.width as f64)
            .max(0.0f64);
        let y = yfinite
            .min(scrolling_area.height() as f64 - viewport.height as f64)
            .max(0.0f64);

        // Step 10
        //TODO handling ongoing smooth scrolling
        if x == self.ScrollX() as f64 && y == self.ScrollY() as f64 {
            return;
        }

        //TODO Step 11
        //let document = self.Document();
        // Step 12
        let x = x.to_f32().unwrap_or(0.0f32);
        let y = y.to_f32().unwrap_or(0.0f32);
        self.update_viewport_for_scroll(x, y);
        self.perform_a_scroll(
            x,
            y,
            self.pipeline_id().root_scroll_id(),
            behavior,
            None,
            can_gc,
        );
    }

    //! @brief Initiates a scroll operation for a given scroll ID.
    //!
    //! Functional Utility: This method orchestrates the actual scrolling action
    //!                     within the layout engine. It takes the target `x` and `y`
    //!                     scroll offsets, an `ExternalScrollId` to identify the
    //!                     scrolling element, and a `ScrollBehavior`. It then
    //!                     triggers a reflow with the `UpdateScrollNode` goal to
    //!                     apply the new scroll position.
    //!
    //! @param x The target horizontal scroll offset.
    //! @param y The target vertical scroll offset.
    //! @param scroll_id The `ExternalScrollId` of the scrolling element.
    //! @param _behavior The `ScrollBehavior` (e.g., "auto", "smooth") for the operation (currently unused).
    //! @param _element An optional reference to the `Element` being scrolled (currently unused).
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @see https://drafts.csswg.org/cssom-view/#perform-a-scroll
    pub(crate) fn perform_a_scroll(
        &self,
        x: f32,
        y: f32,
        scroll_id: ExternalScrollId,
        _behavior: ScrollBehavior,
        _element: Option<&Element>,
        can_gc: CanGc,
    ) {
        // TODO Step 1
        // TODO(mrobinson, #18709): Add smooth scrolling support to WebRender so that we can
        // properly process ScrollBehavior here.
        self.reflow(
            ReflowGoal::UpdateScrollNode(ScrollState {
                scroll_id,
                scroll_offset: Vector2D::new(-x, -y),
            }),
            can_gc,
        );
    }

    //! @brief Updates the internal viewport representation after a scroll.
    //!
    //! Functional Utility: This method updates the `current_viewport` field
    //!                     of the `Window` object to reflect the new scroll
    //!                     position (`x`, `y`). It creates a new `Rect` with
    //!                     the updated origin but retains the existing size,
    //!                     ensuring that the window's view of the document
    //!                     is synchronized with the scroll operation.
    //!
    //! @param x The new horizontal scroll position (in CSS pixels).
    //! @param y The new vertical scroll position (in CSS pixels).
    //! @see https://drafts.csswg.org/cssom-view/#perform-a-scroll
    pub(crate) fn update_viewport_for_scroll(&self, x: f32, y: f32) {
        let size = self.current_viewport.get().size;
        let new_viewport = Rect::new(Point2D::new(Au::from_f32_px(x), Au::from_f32_px(y)), size);
        self.current_viewport.set(new_viewport)
    }

    //! @brief Retrieves the device pixel ratio for the window.
    //!
    //! Functional Utility: This method returns the `Scale` factor between
    //!                     CSS pixels and device pixels for the current display.
    //!                     It is derived from the `window_size` information and
    //!                     is crucial for correctly scaling content to high-DPI
    //!                     (Retina) screens, ensuring crisp rendering of text
    //!                     and graphics.
    //!
    //! @return A `Scale<f32, CSSPixel, DevicePixel>` representing the device pixel ratio.
    pub(crate) fn device_pixel_ratio(&self) -> Scale<f32, CSSPixel, DevicePixel> {
        self.window_size.get().device_pixel_ratio
    }

    //! @brief Retrieves the size and origin of the client window rectangle.
    //!
    //! Functional Utility: This method communicates with the compositor API to
    //!                     obtain the `DeviceIndependentIntRect` of the client
    //!                     window. It then extracts and returns the `Size2D`
    //!                     (width and height) and `Point2D` (top-left origin)
    //!                     in CSS pixels, providing accurate dimensions and
    //!                     positioning of the window within the screen.
    //!
    //! @return A tuple containing the `Size2D<u32, CSSPixel>` and `Point2D<i32, CSSPixel>`
    //!         of the client window.
    fn client_window(&self) -> (Size2D<u32, CSSPixel>, Point2D<i32, CSSPixel>) {
        let timer_profile_chan = self.global().time_profiler_chan().clone();
        let (send, recv) =
            ProfiledIpc::channel::<DeviceIndependentIntRect>(timer_profile_chan).unwrap();
        let _ = self
            .compositor_api
            .sender()
            .send(webrender_traits::CrossProcessCompositorMessage::GetClientWindowRect(send));
        let rect = recv.recv().unwrap_or_default();
        (
            Size2D::new(rect.size().width as u32, rect.size().height as u32),
            Point2D::new(rect.min.x, rect.min.y),
        )
    }

    //! @brief Advances the animation clock and triggers a reflow.
    //!
    //! Functional Utility: This method is used for testing purposes to manually
    //!                     advance the animation timeline by a specified `delta_ms`.
    //!                     It then ensures that all animations within the document
    //!                     are ticked and triggers a reflow to update the layout
    //!                     and rendering based on the new animation states.
    //!
    //! @param delta_ms The time in milliseconds by which to advance the animation clock.
    #[allow(unsafe_code)]
    pub(crate) fn advance_animation_clock(&self, delta_ms: i32) {
        self.Document()
            .advance_animation_timeline_for_testing(delta_ms as f64 / 1000.);
        ScriptThread::handle_tick_all_animations_for_testing(self.pipeline_id());
    }

    //! @brief Forces an unconditional reflow of the page.
    //!
    //! Functional Utility: This method triggers a layout computation (reflow) for
    //!                     the document, irrespective of whether the page is
    //!                     marked as dirty. It also advances the layout animation
    //!                     clock. Reflows can be suppressed based on `LayoutBlocker`
    //!                     state, particularly for display-only goals before the
    //!                     first load event. It handles invalidating layout caches,
    //!                     flushing stylesheets and WebGL canvases, and processing
    //!                     pending restyles.
    //!
    //! @param reflow_goal The `ReflowGoal` indicating the purpose of the reflow.
    //! @param condition An `Option<ReflowTriggerCondition>` describing what triggered the reflow.
    //! @return `true` if a layout actually happened, `false` otherwise.
    //!
    //! Note: This method should almost never be called directly; layout and rendering
    //! updates should occur as part of the HTML event loop via "update the rendering".
    #[allow(unsafe_code)]
    fn force_reflow(
        &self,
        reflow_goal: ReflowGoal,
        condition: Option<ReflowTriggerCondition>,
    ) -> bool {
        self.Document().ensure_safe_to_run_script_or_layout();

        // If layouts are blocked, we block all layouts that are for display only. Other
        // layouts (for queries and scrolling) are not blocked, as they do not display
        // anything and script excpects the layout to be up-to-date after they run.
        let layout_blocked = self.layout_blocker.get().layout_blocked();
        let pipeline_id = self.pipeline_id();
        if reflow_goal == ReflowGoal::UpdateTheRendering && layout_blocked {
            debug!("Suppressing pre-load-event reflow pipeline {pipeline_id}");
            return false;
        }

        if condition != Some(ReflowTriggerCondition::PaintPostponed) {
            debug!(
                "Invalidating layout cache due to reflow condition {:?}",
                condition
            );
            // Invalidate any existing cached layout values.
            self.layout_marker.borrow().set(false);
            // Create a new layout caching token.
            *self.layout_marker.borrow_mut() = Rc::new(Cell::new(true));
        } else {
            debug!("Not invalidating cached layout values for paint-only reflow.");
        }

        debug!("script: performing reflow for goal {reflow_goal:?}");
        let marker = if self.need_emit_timeline_marker(TimelineMarkerType::Reflow) {
            Some(TimelineMarker::start("Reflow".to_owned()))
        } else {
            None
        };

        // On debug mode, print the reflow event information.
        if self.relayout_event {
            debug_reflow_events(pipeline_id, &reflow_goal);
        }

        let document = self.Document();

        let stylesheets_changed = document.flush_stylesheets_for_reflow();

        // If this reflow is for display, ensure webgl canvases are composited with
        // up-to-date contents.
        let for_display = reflow_goal.needs_display();
        if for_display {
            document.flush_dirty_webgl_canvases();
        }

        let pending_restyles = document.drain_pending_restyles();

        let dirty_root = document
            .take_dirty_root()
            .filter(|_| !stylesheets_changed)
            .or_else(|| document.GetDocumentElement())
            .map(|root| root.upcast::<Node>().to_trusted_node_address());

        // Send new document and relevant styles to layout.
        let reflow = ReflowRequest {
            reflow_info: Reflow {
                page_clip_rect: self.page_clip_rect.get(),
            },
            document: document.upcast::<Node>().to_trusted_node_address(),
            dirty_root,
            stylesheets_changed,
            window_size: self.window_size.get(),
            origin: self.origin().immutable().clone(),
            reflow_goal,
            dom_count: document.dom_count(),
            pending_restyles,
            animation_timeline_value: document.current_animation_timeline_value(),
            animations: document.animations().sets.clone(),
            theme: self.theme.get(),
        };

        let Some(results) = self.layout.borrow_mut().reflow(reflow) else {
            return false;
        };

        debug!("script: layout complete");
        if let Some(marker) = marker {
            self.emit_timeline_marker(marker.end());
        }

        // Either this reflow caused new contents to be displayed or on the next
        // full layout attempt a reflow should be forced in order to update the
        // visual contents of the page. A case where full display might be delayed
        // is when reflowing just for the purpose of doing a layout query.
        document.set_needs_paint(!for_display);

        for image in results.pending_images {
            let id = image.id;
            let node = unsafe { from_untrusted_node_address(image.node) };

            if let PendingImageState::Unrequested(ref url) = image.state {
                fetch_image_for_layout(url.clone(), &node, id, self.image_cache.clone());
            }

            let mut images = self.pending_layout_images.borrow_mut();
            let nodes = images.entry(id).or_default();
            if !nodes.iter().any(|n| std::ptr::eq(&**n, &*node)) {
                let trusted_node = Trusted::new(&*node);
                let sender = self.register_image_cache_listener(id, move |response| {
                    trusted_node
                        .root()
                        .owner_window()
                        .pending_layout_image_notification(response);
                });

                self.image_cache
                    .add_listener(ImageResponder::new(sender, self.pipeline_id(), id));
                nodes.push(Dom::from_ref(&*node));
            }
        }

        let size_messages = self
            .Document()
            .iframes_mut()
            .handle_new_iframe_sizes_after_layout(results.iframe_sizes, self.device_pixel_ratio());
        if !size_messages.is_empty() {
            self.send_to_constellation(ScriptMsg::IFrameSizes(size_messages));
        }

        document.update_animations_post_reflow();
        self.update_constellation_epoch();

        true
    }

    //! @brief Triggers a conditional reflow of the page.
    //!
    //! Functional Utility: This method initiates a reflow of the page if the
    //!                     document is currently "dirty" (i.e., requires a layout
    //!                     update) and if reflows are not suppressed. It ensures
    //!                     that layout computations are performed only when
    //!                     necessary, optimizing performance by avoiding redundant
    //!                     work.
    //!
    //! @return `true` if a layout actually happened, `false` otherwise.
    pub(crate) fn reflow(
        &self,
        reflow_goal: ReflowGoal,
        can_gc: CanGc,
    ) -> bool {
        // TODO: Figure out if this is strictly necessary given current usages.
        let document = self.Document();
        if !document.is_dirty() {
            return false;
        }

        self.force_reflow(reflow_goal, Some(ReflowTriggerCondition::Forced));
        true
    }
    //! @brief Conditionally triggers a reflow of the page.
    //!
    //! Functional Utility: This method orchestrates the layout computation for the
    //!                     document, but only if certain conditions are met,
    //!                     optimizing performance by preventing unnecessary reflows.
    //!                     It considers the document's dirty state, pending web fonts,
    //!                     and the current `ReflowGoal`. It also integrates logic
    //!                     for WebDriver screenshot stability checks.
    //!
    //! Pre-condition: Layout and rendering updates should primarily occur via the
    //!                HTML event loop's "update the rendering" mechanism. Direct
    //!                calls to this method are typically for script queries or
    //!                scroll requests that require immediate layout information.
    //!
    //! @param reflow_goal The `ReflowGoal` specifying the reason for the reflow.
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @return `true` if a reflow was issued, `false` otherwise.
    pub(crate) fn reflow(&self, reflow_goal: ReflowGoal, can_gc: CanGc) -> bool {
        // Count the pending web fonts before layout, in case a font loads during the layout.
        let waiting_for_web_fonts_to_load = self.font_context.web_fonts_still_loading() != 0;

        self.Document().ensure_safe_to_run_script_or_layout();

        let mut issued_reflow = false;
        let condition = self.Document().needs_reflow();
        let updating_the_rendering = reflow_goal == ReflowGoal::UpdateTheRendering;
        let for_display = reflow_goal.needs_display();
        if !updating_the_rendering || condition.is_some() {
            debug!("Reflowing document ({:?})", self.pipeline_id());
            issued_reflow = self.force_reflow(reflow_goal, condition);

            // We shouldn't need a reflow immediately after a completed reflow, unless the reflow didn't
            // display anything and it wasn't for display. Queries can cause this to happen.
            if issued_reflow {
                let condition = self.Document().needs_reflow();
                let display_is_pending = condition == Some(ReflowTriggerCondition::PaintPostponed);
                assert!(
                    condition.is_none() || (display_is_pending && !for_display),
                    "Needed reflow after reflow: {:?}",
                    condition
                );
            }
        } else {
            debug!(
                "Document ({:?}) doesn't need reflow - skipping it (goal {reflow_goal:?})",
                self.pipeline_id()
            );
        }

        let document = self.Document();
        let font_face_set = document.Fonts(can_gc);
        let is_ready_state_complete = document.ReadyState() == DocumentReadyState::Complete;

        // From https://drafts.csswg.org/css-font-loading/#font-face-set-ready:
        // > A FontFaceSet is pending on the environment if any of the following are true:
        // >  - the document is still loading
        // >  - the document has pending stylesheet requests
        // >  - the document has pending layout operations which might cause the user agent to request
        // >    a font, or which depend on recently-loaded fonts
        //
        // Thus, we are queueing promise resolution here. This reflow should have been triggered by
        // a "rendering opportunity" in `ScriptThread::handle_web_font_loaded, which should also
        // make sure a microtask checkpoint happens, triggering the promise callback.
        if !waiting_for_web_fonts_to_load && is_ready_state_complete {
            font_face_set.fulfill_ready_promise_if_needed();
        }

        // If writing a screenshot, check if the script has reached a state
        // where it's safe to write the image. This means that:
        // 1) The reflow is for display (otherwise it could be a query)
        // 2) The html element doesn't contain the 'reftest-wait' class
        // 3) The load event has fired.
        // When all these conditions are met, notify the constellation
        // that this pipeline is ready to write the image (from the script thread
        // perspective at least).
        if opts::get().wait_for_stable_image && updating_the_rendering {
            // Checks if the html element has reftest-wait attribute present.
            // See http://testthewebforward.org/docs/reftests.html
            // and https://web-platform-tests.org/writing-tests/crashtest.html
            let html_element = document.GetDocumentElement();
            let reftest_wait = html_element.is_some_and(|elem| {
                elem.has_class(&atom!("reftest-wait"), CaseSensitivity::CaseSensitive) ||
                    elem.has_class(&Atom::from("test-wait"), CaseSensitivity::CaseSensitive)
            });

            let has_sent_idle_message = self.has_sent_idle_message.get();
            let pending_images = !self.pending_layout_images.borrow().is_empty();

            if !has_sent_idle_message &&
                is_ready_state_complete &&
                !reftest_wait &&
                !pending_images &&
                !waiting_for_web_fonts_to_load
            {
                debug!(
                    "{:?}: Sending DocumentState::Idle to Constellation",
                    self.pipeline_id()
                );
                let event = ScriptMsg::SetDocumentState(DocumentState::Idle);
                self.send_to_constellation(event);
                self.has_sent_idle_message.set(true);
            }
        }

        issued_reflow
    }

    //! @brief Initiates a reflow if the reflow timer has expired due to long parsing.
    //!
    //! Functional Utility: This method checks if parsing has taken an exceptionally
    //!                     long time and if the `LayoutBlocker` indicates that
    //!                     reflows are still being suppressed. If these conditions
    //!                     are met, it allows layouts to proceed by calling
    //!                     `allow_layout_if_necessary`, ensuring that users
    //!                     see progress even with very slow-loading pages.
    //!
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @see <https://github.com/servo/servo/pull/6028> for more details on layout suppression.
    pub(crate) fn reflow_if_reflow_timer_expired(&self, can_gc: CanGc) {
        // Only trigger a long parsing time reflow if we are in the first parse of `<body>`
        // and it started more than `INITIAL_REFLOW_DELAY` ago.
        if !matches!(
            self.layout_blocker.get(),
            LayoutBlocker::Parsing(instant) if instant + INITIAL_REFLOW_DELAY < Instant::now()
        ) {
            return;
        }
        self.allow_layout_if_necessary(can_gc);
    }

    //! @brief Prevents layout until the `load` event fires or a timeout occurs.
    //!
    //! Functional Utility: This method sets the `layout_blocker` state to prevent
    //!                     layouts from occurring until the document's `load` event
    //!                     has fired. If parsing takes an excessively long time,
    //!                     a scheduled timer (`INITIAL_REFLOW_DELAY`) will eventually
    //!                     allow layouts to resume, ensuring progress is visible.
    //!                     This prevents "flash of unstyled content" and visual
    //!                     instability during initial page load.
    //!
    //! @see <https://github.com/servo/servo/pull/6028> for more details on layout suppression.
    pub(crate) fn prevent_layout_until_load_event(&self) {
        // If we have already started parsing or have already fired a load event, then
        // don't delay the first layout any longer.
        if !matches!(self.layout_blocker.get(), LayoutBlocker::WaitingForParse) {
            return;
        }

        self.layout_blocker
            .set(LayoutBlocker::Parsing(Instant::now()));
    }

    //! @brief Allows layout to proceed, if currently blocked, due to load event or parsing timeout.
    //!
    //! Functional Utility: This method updates the `layout_blocker` state to
    //!                     `FiredLoadEventOrParsingTimerExpired`, effectively
    //!                     allowing layout operations to proceed. It is called
    //!                     either when the `load` event for the document has
    //!                     fired or when parsing of the `<body>` has taken so
    //!                     long that layout suppression is no longer desired.
    //!                     It also triggers a synchronous reflow to update the
    //!                     display.
    //!
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @see <https://github.com/servo/servo/issues/14719> for context on synchronous updates.
    pub(crate) fn allow_layout_if_necessary(&self, can_gc: CanGc) {
        if matches!(
            self.layout_blocker.get(),
            LayoutBlocker::FiredLoadEventOrParsingTimerExpired
        ) {
            return;
        }

        self.layout_blocker
            .set(LayoutBlocker::FiredLoadEventOrParsingTimerExpired);
        self.Document().set_needs_paint(true);

        // We do this immediately instead of scheduling a future task, because this can
        // happen if parsing is taking a very long time, which means that the
        // `ScriptThread` is busy doing the parsing and not doing layouts.
        //
        // TOOD(mrobinson): It's expected that this is necessary when in the process of
        // parsing, as we need to interrupt it to update contents, but why is this
        // necessary when parsing finishes? Not doing the synchronous update in that case
        // causes iframe tests to become flaky. It seems there's an issue with the timing of
        // iframe size updates.
        //
        // See <https://github.com/servo/servo/issues/14719>
        self.reflow(ReflowGoal::UpdateTheRendering, can_gc);
    }

    //! @brief Checks if layout operations are currently blocked.
    //!
    //! Functional Utility: This method returns a boolean indicating whether
    //!                     the current `LayoutBlocker` state is suppressing
    //!                     layout computations. It provides a quick way to
    //!                     determine if the window is in a state where layouts
    //!                     are intentionally deferred, typically during
    //!                     initial page loading.
    //!
    //! @return `true` if layout is blocked, `false` otherwise.
    pub(crate) fn layout_blocked(&self) -> bool {
        self.layout_blocker.get().layout_blocked()
    }

    //! @brief Synchronously updates the layout epoch in the constellation.
    //!
    //! Functional Utility: This method sends the current layout epoch to the
    //!                     constellation if the `wait_for_stable_image` option
    //!                     is enabled. This is particularly relevant when generating
    //!                     screenshots, as it provides a synchronization point
    //!                     indicating a stable layout state for image capture.
    //!                     The update is performed synchronously, blocking until
    //!                     the message is processed.
    pub(crate) fn update_constellation_epoch(&self) {
        if !opts::get().wait_for_stable_image {
            return;
        }

        let epoch = self.layout.borrow().current_epoch();
        debug!(
            "{:?}: Updating constellation epoch: {epoch:?}",
            self.pipeline_id()
        );
        let (sender, receiver) = ipc::channel().expect("Failed to create IPC channel!");
        let event = ScriptMsg::SetLayoutEpoch(epoch, sender);
        self.send_to_constellation(event);
        let _ = receiver.recv();
    }

    //! @brief Triggers a reflow specifically for layout queries.
    //!
    //! Functional Utility: This method initiates a reflow operation with the
    //!                     `ReflowGoal::LayoutQuery` to ensure that the layout
    //!                     engine computes fresh layout information needed
    //!                     to answer a specific query (e.g., element dimensions,
    //!                     scroll positions). This ensures that scripts
    //!                     requesting layout-dependent values receive up-to-date
    //!                     data.
    //!
    //! @param query_msg The `QueryMsg` specifying the type of layout query being performed.
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @return `true` if the reflow was successfully issued, `false` otherwise.
    pub(crate) fn layout_reflow(&self, query_msg: QueryMsg, can_gc: CanGc) -> bool {
        self.reflow(ReflowGoal::LayoutQuery(query_msg), can_gc)
    }

    //! @brief Queries the resolved font style for a given node.
    //!
    //! Functional Utility: This method first ensures that the layout is up-to-date
    //!                     by calling `layout_reflow` with a `ResolvedFontStyleQuery`.
    //!                     Then, it queries the layout engine to retrieve the
    //!                     resolved `Font` style for a specific `node` and `value`.
    //!                     This is important for rendering text with the correct
    //!                     font properties after all CSS rules and layout
    //!                     calculations have been applied.
    //!
    //! @param node The `Node` for which to query the font style.
    //! @param value A `String` representing the font style property.
    //! @param can_gc Indicates whether garbage collection is allowed during this operation.
    //! @return An `Option<ServoArc<Font>>` containing the resolved font style, or `None` if not found.
    pub(crate) fn resolved_font_style_query(
        &self,
        node: &Node,
        value: String,
        can_gc: CanGc,
    ) -> Option<ServoArc<Font>> {
        if !self.layout_reflow(QueryMsg::ResolvedFontStyleQuery, can_gc) {
            return None;
        }

        let document = self.Document();
        let animations = document.animations().sets.clone();
        self.layout.borrow().query_resolved_font_style(
            node.to_trusted_node_address(),
            &value,
            animations,
            document.current_animation_timeline_value(),
        )
    }

    pub(crate) fn content_box_query(&self, node: &Node, can_gc: CanGc) -> Option<UntypedRect<Au>> {
        if !self.layout_reflow(QueryMsg::ContentBox, can_gc) {
            return None;
        }
        self.layout.borrow().query_content_box(node.to_opaque())
    }

    pub(crate) fn content_boxes_query(&self, node: &Node, can_gc: CanGc) -> Vec<UntypedRect<Au>> {
        if !self.layout_reflow(QueryMsg::ContentBoxes, can_gc) {
            return vec![];
        }
        self.layout.borrow().query_content_boxes(node.to_opaque())
    }

    pub(crate) fn client_rect_query(&self, node: &Node, can_gc: CanGc) -> UntypedRect<i32> {
        if !self.layout_reflow(QueryMsg::ClientRectQuery, can_gc) {
            return Rect::zero();
        }
        self.layout.borrow().query_client_rect(node.to_opaque())
    }

    /// Find the scroll area of the given node, if it is not None. If the node
    /// is None, find the scroll area of the viewport.
    pub(crate) fn scrolling_area_query(
        &self,
        node: Option<&Node>,
        can_gc: CanGc,
    ) -> UntypedRect<i32> {
        let opaque = node.map(|node| node.to_opaque());
        if !self.layout_reflow(QueryMsg::ScrollingAreaQuery, can_gc) {
            return Rect::zero();
        }
        self.layout.borrow().query_scrolling_area(opaque)
    }

    pub(crate) fn scroll_offset_query(&self, node: &Node) -> Vector2D<f32, LayoutPixel> {
        if let Some(scroll_offset) = self.scroll_offsets.borrow().get(&node.to_opaque()) {
            return *scroll_offset;
        }
        Vector2D::new(0.0, 0.0)
    }

    // https://drafts.csswg.org/cssom-view/#element-scrolling-members
    pub(crate) fn scroll_node(
        &self,
        node: &Node,
        x_: f64,
        y_: f64,
        behavior: ScrollBehavior,
        can_gc: CanGc,
    ) {
        // The scroll offsets are immediatly updated since later calls
        // to topScroll and others may access the properties before
        // webrender has a chance to update the offsets.
        self.scroll_offsets
            .borrow_mut()
            .insert(node.to_opaque(), Vector2D::new(x_ as f32, y_ as f32));
        let scroll_id = ExternalScrollId(
            combine_id_with_fragment_type(node.to_opaque().id(), FragmentType::FragmentBody),
            self.pipeline_id().into(),
        );

        // Step 12
        self.perform_a_scroll(
            x_.to_f32().unwrap_or(0.0f32),
            y_.to_f32().unwrap_or(0.0f32),
            scroll_id,
            behavior,
            None,
            can_gc,
        );
    }

    pub(crate) fn resolved_style_query(
        &self,
        element: TrustedNodeAddress,
        pseudo: Option<PseudoElement>,
        property: PropertyId,
        can_gc: CanGc,
    ) -> DOMString {
        if !self.layout_reflow(QueryMsg::ResolvedStyleQuery, can_gc) {
            return DOMString::new();
        }

        let document = self.Document();
        let animations = document.animations().sets.clone();
        DOMString::from(self.layout.borrow().query_resolved_style(
            element,
            pseudo,
            property,
            animations,
            document.current_animation_timeline_value(),
        ))
    }

    /// If the given |browsing_context_id| refers to an `<iframe>` that is an element
    /// in this [`Window`] and that `<iframe>` has been laid out, return its size.
    /// Otherwise, return `None`.
    pub(crate) fn get_iframe_size_if_known(
        &self,
        browsing_context_id: BrowsingContextId,
        can_gc: CanGc,
    ) -> Option<Size2D<f32, CSSPixel>> {
        // Reflow might fail, but do a best effort to return the right size.
        self.layout_reflow(QueryMsg::InnerWindowDimensionsQuery, can_gc);
        self.Document()
            .iframes()
            .get(browsing_context_id)
            .and_then(|iframe| iframe.size)
    }

    #[allow(unsafe_code)]
    pub(crate) fn offset_parent_query(
        &self,
        node: &Node,
        can_gc: CanGc,
    ) -> (Option<DomRoot<Element>>, UntypedRect<Au>) {
        if !self.layout_reflow(QueryMsg::OffsetParentQuery, can_gc) {
            return (None, Rect::zero());
        }

        let response = self.layout.borrow().query_offset_parent(node.to_opaque());
        let element = response.node_address.and_then(|parent_node_address| {
            let node = unsafe { from_untrusted_node_address(parent_node_address) };
            DomRoot::downcast(node)
        });
        (element, response.rect)
    }

    pub(crate) fn text_index_query(
        &self,
        node: &Node,
        point_in_node: UntypedPoint2D<f32>,
        can_gc: CanGc,
    ) -> Option<usize> {
        if !self.layout_reflow(QueryMsg::TextIndexQuery, can_gc) {
            return None;
        }
        self.layout
            .borrow()
            .query_text_indext(node.to_opaque(), point_in_node)
    }

    #[allow(unsafe_code)]
    pub(crate) fn init_window_proxy(&self, window_proxy: &WindowProxy) {
        assert!(self.window_proxy.get().is_none());
        self.window_proxy.set(Some(window_proxy));
    }

    #[allow(unsafe_code)]
    pub(crate) fn init_document(&self, document: &Document) {
        assert!(self.document.get().is_none());
        assert!(document.window() == self);
        self.document.set(Some(document));

        if self.unminify_css {
            *self.unminified_css_dir.borrow_mut() = Some(unminified_path("unminified-css"));
        }
    }

    /// Commence a new URL load which will either replace this window or scroll to a fragment.
    ///
    /// <https://html.spec.whatwg.org/multipage/#navigating-across-documents>
    pub(crate) fn load_url(
        &self,
        history_handling: NavigationHistoryBehavior,
        force_reload: bool,
        load_data: LoadData,
        can_gc: CanGc,
    ) {
        let doc = self.Document();

        // Step 3. Let initiatorOriginSnapshot be sourceDocument's origin.
        let initiator_origin_snapshot = &load_data.load_origin;

        // TODO: Important re security. See https://github.com/servo/servo/issues/23373
        // Step 5. check that the source browsing-context is "allowed to navigate" this window.
        if !force_reload &&
            load_data.url.as_url()[..Position::AfterQuery] ==
                doc.url().as_url()[..Position::AfterQuery]
        {
            // Step 6
            // TODO: Fragment handling appears to have moved to step 13
            if let Some(fragment) = load_data.url.fragment() {
                self.send_to_constellation(ScriptMsg::NavigatedToFragment(
                    load_data.url.clone(),
                    history_handling,
                ));
                doc.check_and_scroll_fragment(fragment, can_gc);
                let this = Trusted::new(self);
                let old_url = doc.url().into_string();
                let new_url = load_data.url.clone().into_string();
                let task = task!(hashchange_event: move || {
                    let this = this.root();
                    let event = HashChangeEvent::new(
                        &this,
                        atom!("hashchange"),
                        false,
                        false,
                        old_url,
                        new_url,
                        CanGc::note());
                    event.upcast::<Event>().fire(this.upcast::<EventTarget>(), CanGc::note());
                });
                self.as_global_scope()
                    .task_manager()
                    .dom_manipulation_task_source()
                    .queue(task);
                doc.set_url(load_data.url.clone());
                return;
            }
        }

        // Step 4 and 5
        let pipeline_id = self.pipeline_id();
        let window_proxy = self.window_proxy();
        if let Some(active) = window_proxy.currently_active() {
            if pipeline_id == active && doc.is_prompting_or_unloading() {
                return;
            }
        }

        // Step 8
        if doc.prompt_to_unload(false, can_gc) {
            let window_proxy = self.window_proxy();
            if window_proxy.parent().is_some() {
                // Step 10
                // If browsingContext is a nested browsing context,
                // then put it in the delaying load events mode.
                window_proxy.start_delaying_load_events_mode();
            }

            // Step 11. If historyHandling is "auto", then:
            let resolved_history_handling = if history_handling == NavigationHistoryBehavior::Auto {
                // Step 11.1. If url equals navigable's active document's URL, and
                // initiatorOriginSnapshot is same origin with targetNavigable's active document's
                // origin, then set historyHandling to "replace".
                // Note: `targetNavigable` is not actually defined in the spec, "active document" is
                // assumed to be the correct reference based on WPT results
                if let LoadOrigin::Script(initiator_origin) = initiator_origin_snapshot {
                    if load_data.url == doc.url() && initiator_origin.same_origin(doc.origin()) {
                        NavigationHistoryBehavior::Replace
                    } else {
                        NavigationHistoryBehavior::Push
                    }
                } else {
                    // Step 11.2. Otherwise, set historyHandling to "push".
                    NavigationHistoryBehavior::Push
                }
            // Step 12. If the navigation must be a replace given url and navigable's active
            // document, then set historyHandling to "replace".
            } else if load_data.url.scheme() == "javascript" || doc.is_initial_about_blank() {
                NavigationHistoryBehavior::Replace
            } else {
                NavigationHistoryBehavior::Push
            };

            // Step 13
            ScriptThread::navigate(
                window_proxy.browsing_context_id(),
                pipeline_id,
                load_data,
                resolved_history_handling,
            );
        };
    }

    pub(crate) fn set_window_size(&self, size: WindowSizeData) {
        self.window_size.set(size);
    }

    pub(crate) fn window_size(&self) -> WindowSizeData {
        self.window_size.get()
    }

    /// Handle a theme change request, triggering a reflow is any actual change occured.
    pub(crate) fn handle_theme_change(&self, new_theme: Theme) {
        let new_theme = match new_theme {
            Theme::Light => PrefersColorScheme::Light,
            Theme::Dark => PrefersColorScheme::Dark,
        };

        if self.theme.get() == new_theme {
            return;
        }
        self.theme.set(new_theme);
        self.Document().set_needs_paint(true);
    }

    pub(crate) fn get_url(&self) -> ServoUrl {
        self.Document().url()
    }

    pub(crate) fn windowproxy_handler(&self) -> &'static WindowProxyHandler {
        self.dom_static.windowproxy_handler
    }

    pub(crate) fn add_resize_event(&self, event: WindowSizeData, event_type: WindowSizeType) {
        // Whenever we receive a new resize event we forget about all the ones that came before
        // it, to avoid unnecessary relayouts
        *self.unhandled_resize_event.borrow_mut() = Some((event, event_type))
    }

    pub(crate) fn take_unhandled_resize_event(&self) -> Option<(WindowSizeData, WindowSizeType)> {
        self.unhandled_resize_event.borrow_mut().take()
    }

    pub(crate) fn set_page_clip_rect_with_new_viewport(&self, viewport: UntypedRect<f32>) -> bool {
        let rect = f32_rect_to_au_rect(viewport);
        self.current_viewport.set(rect);
        // We use a clipping rectangle that is five times the size of the of the viewport,
        // so that we don't collect display list items for areas too far outside the viewport,
        // but also don't trigger reflows every time the viewport changes.
        static VIEWPORT_EXPANSION: f32 = 2.0; // 2 lengths on each side plus original length is 5 total.
        let proposed_clip_rect = f32_rect_to_au_rect(viewport.inflate(
            viewport.size.width * VIEWPORT_EXPANSION,
            viewport.size.height * VIEWPORT_EXPANSION,
        ));
        let clip_rect = self.page_clip_rect.get();
        if proposed_clip_rect == clip_rect {
            return false;
        }

        let had_clip_rect = clip_rect != MaxRect::max_rect();
        if had_clip_rect && !should_move_clip_rect(clip_rect, viewport) {
            return false;
        }

        self.page_clip_rect.set(proposed_clip_rect);

        // The document needs to be repainted, because the initial containing block
        // is now a different size.
        self.Document().set_needs_paint(true);

        // If we didn't have a clip rect, the previous display doesn't need rebuilding
        // because it was built for infinite clip (MaxRect::amax_rect()).
        had_clip_rect
    }

    pub(crate) fn suspend(&self) {
        // Suspend timer events.
        self.as_global_scope().suspend();

        // Set the window proxy to be a cross-origin window.
        if self.window_proxy().currently_active() == Some(self.global().pipeline_id()) {
            self.window_proxy().unset_currently_active();
        }

        // A hint to the JS runtime that now would be a good time to
        // GC any unreachable objects generated by user script,
        // or unattached DOM nodes. Attached DOM nodes can't be GCd yet,
        // as the document might be reactivated later.
        self.Gc();
    }

    pub(crate) fn resume(&self) {
        // Resume timer events.
        self.as_global_scope().resume();

        // Set the window proxy to be this object.
        self.window_proxy().set_currently_active(self);

        // Push the document title to the compositor since we are
        // activating this document due to a navigation.
        self.Document().title_changed();
    }

    pub(crate) fn need_emit_timeline_marker(&self, timeline_type: TimelineMarkerType) -> bool {
        let markers = self.devtools_markers.borrow();
        markers.contains(&timeline_type)
    }

    pub(crate) fn emit_timeline_marker(&self, marker: TimelineMarker) {
        let sender = self.devtools_marker_sender.borrow();
        let sender = sender.as_ref().expect("There is no marker sender");
        sender.send(Some(marker)).unwrap();
    }

    pub(crate) fn set_devtools_timeline_markers(
        &self,
        markers: Vec<TimelineMarkerType>,
        reply: IpcSender<Option<TimelineMarker>>,
    ) {
        *self.devtools_marker_sender.borrow_mut() = Some(reply);
        self.devtools_markers.borrow_mut().extend(markers);
    }

    pub(crate) fn drop_devtools_timeline_markers(&self, markers: Vec<TimelineMarkerType>) {
        let mut devtools_markers = self.devtools_markers.borrow_mut();
        for marker in markers {
            devtools_markers.remove(&marker);
        }
        if devtools_markers.is_empty() {
            *self.devtools_marker_sender.borrow_mut() = None;
        }
    }

    pub(crate) fn set_webdriver_script_chan(&self, chan: Option<IpcSender<WebDriverJSResult>>) {
        *self.webdriver_script_chan.borrow_mut() = chan;
    }

    pub(crate) fn is_alive(&self) -> bool {
        self.current_state.get() == WindowState::Alive
    }

    // https://html.spec.whatwg.org/multipage/#top-level-browsing-context
    pub(crate) fn is_top_level(&self) -> bool {
        self.parent_info.is_none()
    }

    /// An implementation of:
    /// <https://drafts.csswg.org/cssom-view/#document-run-the-resize-steps>
    ///
    /// Returns true if there were any pending resize events.
    pub(crate) fn run_the_resize_steps(&self, can_gc: CanGc) -> bool {
        let Some((new_size, size_type)) = self.take_unhandled_resize_event() else {
            return false;
        };

        if self.window_size() == new_size {
            return false;
        }

        let _realm = enter_realm(self);
        debug!(
            "Resizing Window for pipeline {:?} from {:?} to {new_size:?}",
            self.pipeline_id(),
            self.window_size(),
        );
        self.set_window_size(new_size);

        // http://dev.w3.org/csswg/cssom-view/#resizing-viewports
        if size_type == WindowSizeType::Resize {
            let uievent = UIEvent::new(
                self,
                DOMString::from("resize"),
                EventBubbles::DoesNotBubble,
                EventCancelable::NotCancelable,
                Some(self),
                0i32,
                can_gc,
            );
            uievent.upcast::<Event>().fire(self.upcast(), can_gc);
        }

        // The document needs to be repainted, because the initial containing block
        // is now a different size.
        self.Document().set_needs_paint(true);

        true
    }

    /// Evaluate media query lists and report changes
    /// <https://drafts.csswg.org/cssom-view/#evaluate-media-queries-and-report-changes>
    pub(crate) fn evaluate_media_queries_and_report_changes(&self, can_gc: CanGc) {
        let _realm = enter_realm(self);

        rooted_vec!(let mut mql_list);
        self.media_query_lists.for_each(|mql| {
            if let MediaQueryListMatchState::Changed = mql.evaluate_changes() {
                // Recording list of changed Media Queries
                mql_list.push(Dom::from_ref(&*mql));
            }
        });
        // Sending change events for all changed Media Queries
        for mql in mql_list.iter() {
            let event = MediaQueryListEvent::new(
                &mql.global(),
                atom!("change"),
                false,
                false,
                mql.Media(),
                mql.Matches(),
                can_gc,
            );
            event
                .upcast::<Event>()
                .fire(mql.upcast::<EventTarget>(), can_gc);
        }
    }

    /// Set whether to use less resources by running timers at a heavily limited rate.
    pub(crate) fn set_throttled(&self, throttled: bool) {
        self.throttled.set(throttled);
        if throttled {
            self.as_global_scope().slow_down_timers();
        } else {
            self.as_global_scope().speed_up_timers();
        }
    }

    pub(crate) fn throttled(&self) -> bool {
        self.throttled.get()
    }

    pub(crate) fn unminified_css_dir(&self) -> Option<String> {
        self.unminified_css_dir.borrow().clone()
    }

    pub(crate) fn local_script_source(&self) -> &Option<String> {
        &self.local_script_source
    }

    pub(crate) fn set_navigation_start(&self) {
        self.navigation_start.set(CrossProcessInstant::now());
    }

    pub(crate) fn send_to_embedder(&self, msg: EmbedderMsg) {
        self.send_to_constellation(ScriptMsg::ForwardToEmbedder(msg));
    }

    pub(crate) fn send_to_constellation(&self, msg: ScriptMsg) {
        self.as_global_scope()
            .script_to_constellation_chan()
            .send(msg)
            .unwrap();
    }

    pub(crate) fn webrender_document(&self) -> DocumentId {
        self.webrender_document
    }

    #[cfg(feature = "webxr")]
    pub(crate) fn in_immersive_xr_session(&self) -> bool {
        self.navigator
            .get()
            .as_ref()
            .and_then(|nav| nav.xr())
            .is_some_and(|xr| xr.pending_or_active_session())
    }

    #[cfg(not(feature = "webxr"))]
    pub(crate) fn in_immersive_xr_session(&self) -> bool {
        false
    }
}

impl Window {
    #[allow(unsafe_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        webview_id: WebViewId,
        runtime: Rc<Runtime>,
        script_chan: Sender<MainThreadScriptMsg>,
        layout: Box<dyn Layout>,
        font_context: Arc<FontContext>,
        image_cache_sender: IpcSender<PendingImageResponse>,
        image_cache: Arc<dyn ImageCache>,
        resource_threads: ResourceThreads,
        #[cfg(feature = "bluetooth")] bluetooth_thread: IpcSender<BluetoothRequest>,
        mem_profiler_chan: MemProfilerChan,
        time_profiler_chan: TimeProfilerChan,
        devtools_chan: Option<IpcSender<ScriptToDevtoolsControlMsg>>,
        constellation_chan: ScriptToConstellationChan,
        control_chan: IpcSender<ScriptThreadMessage>,
        pipeline_id: PipelineId,
        parent_info: Option<PipelineId>,
        window_size: WindowSizeData,
        origin: MutableOrigin,
        creator_url: ServoUrl,
        navigation_start: CrossProcessInstant,
        webgl_chan: Option<WebGLChan>,
        #[cfg(feature = "webxr")] webxr_registry: Option<webxr_api::Registry>,
        microtask_queue: Rc<MicrotaskQueue>,
        webrender_document: DocumentId,
        compositor_api: CrossProcessCompositorApi,
        relayout_event: bool,
        unminify_js: bool,
        unminify_css: bool,
        local_script_source: Option<String>,
        userscripts_path: Option<String>,
        user_agent: Cow<'static, str>,
        player_context: WindowGLContext,
        #[cfg(feature = "webgpu")] gpu_id_hub: Arc<IdentityHub>,
        inherited_secure_context: Option<bool>,
    ) -> DomRoot<Self> {
        let error_reporter = CSSErrorReporter {
            pipelineid: pipeline_id,
            script_chan: Arc::new(Mutex::new(control_chan)),
        };

        let initial_viewport = f32_rect_to_au_rect(UntypedRect::new(
            Point2D::zero(),
            window_size.initial_viewport.to_untyped(),
        ));

        let win = Box::new(Self {
            webview_id,
            globalscope: GlobalScope::new_inherited(
                pipeline_id,
                devtools_chan,
                mem_profiler_chan,
                time_profiler_chan,
                constellation_chan,
                resource_threads,
                origin,
                Some(creator_url),
                microtask_queue,
                user_agent,
                #[cfg(feature = "webgpu")]
                gpu_id_hub,
                inherited_secure_context,
                unminify_js,
            ),
            script_chan,
            layout: RefCell::new(layout),
            font_context,
            image_cache_sender,
            image_cache,
            navigator: Default::default(),
            location: Default::default(),
            history: Default::default(),
            custom_element_registry: Default::default(),
            window_proxy: Default::default(),
            document: Default::default(),
            performance: Default::default(),
            navigation_start: Cell::new(navigation_start),
            screen: Default::default(),
            session_storage: Default::default(),
            local_storage: Default::default(),
            status: DomRefCell::new(DOMString::new()),
            parent_info,
            dom_static: GlobalStaticData::new(),
            js_runtime: DomRefCell::new(Some(runtime.clone())),
            #[cfg(feature = "bluetooth")]
            bluetooth_thread,
            #[cfg(feature = "bluetooth")]
            bluetooth_extra_permission_data: BluetoothExtraPermissionData::new(),
            page_clip_rect: Cell::new(MaxRect::max_rect()),
            unhandled_resize_event: Default::default(),
            window_size: Cell::new(window_size),
            current_viewport: Cell::new(initial_viewport.to_untyped()),
            layout_blocker: Cell::new(LayoutBlocker::WaitingForParse),
            current_state: Cell::new(WindowState::Alive),
            devtools_marker_sender: Default::default(),
            devtools_markers: Default::default(),
            webdriver_script_chan: Default::default(),
            error_reporter,
            scroll_offsets: Default::default(),
            media_query_lists: DOMTracker::new(),
            #[cfg(feature = "bluetooth")]
            test_runner: Default::default(),
            webgl_chan,
            #[cfg(feature = "webxr")]
            webxr_registry,
            pending_image_callbacks: Default::default(),
            pending_layout_images: Default::default(),
            unminified_css_dir: Default::default(),
            local_script_source,
            test_worklet: Default::default(),
            paint_worklet: Default::default(),
            webrender_document,
            exists_mut_observer: Cell::new(false),
            compositor_api,
            has_sent_idle_message: Cell::new(false),
            relayout_event,
            unminify_css,
            userscripts_path,
            player_context,
            throttled: Cell::new(false),
            layout_marker: DomRefCell::new(Rc::new(Cell::new(true))),
            current_event: DomRefCell::new(None),
            theme: Cell::new(PrefersColorScheme::Light),
        });

        unsafe {
            WindowBinding::Wrap::<crate::DomTypeHolder>(JSContext::from_ptr(runtime.cx()), win)
        }
    }

    pub(crate) fn pipeline_id(&self) -> PipelineId {
        self.as_global_scope().pipeline_id()
    }

    /// Create a new cached instance of the given value.
    pub(crate) fn cache_layout_value<T>(&self, value: T) -> LayoutValue<T>
    where
        T: Copy + MallocSizeOf,
    {
        LayoutValue::new(self.layout_marker.borrow().clone(), value)
    }
}

/// An instance of a value associated with a particular snapshot of layout. This stored
/// value can only be read as long as the associated layout marker that is considered
/// valid. It will automatically become unavailable when the next layout operation is
/// performed.
#[derive(MallocSizeOf)]
pub(crate) struct LayoutValue<T: MallocSizeOf> {
    #[ignore_malloc_size_of = "Rc is hard"]
    is_valid: Rc<Cell<bool>>,
    value: T,
}

#[allow(unsafe_code)]
unsafe impl<T: JSTraceable + MallocSizeOf> JSTraceable for LayoutValue<T> {
    unsafe fn trace(&self, trc: *mut js::jsapi::JSTracer) {
        self.value.trace(trc)
    }
}

impl<T: Copy + MallocSizeOf> LayoutValue<T> {
    fn new(marker: Rc<Cell<bool>>, value: T) -> Self {
        LayoutValue {
            is_valid: marker,
            value,
        }
    }

    /// Retrieve the stored value if it is still valid.
    pub(crate) fn get(&self) -> Result<T, ()> {
        if self.is_valid.get() {
            return Ok(self.value);
        }
        Err(())
    }
}

fn should_move_clip_rect(clip_rect: UntypedRect<Au>, new_viewport: UntypedRect<f32>) -> bool {
    let clip_rect = UntypedRect::new(
        Point2D::new(
            clip_rect.origin.x.to_f32_px(),
            clip_rect.origin.y.to_f32_px(),
        ),
        Size2D::new(
            clip_rect.size.width.to_f32_px(),
            clip_rect.size.height.to_f32_px(),
        ),
    );

    // We only need to move the clip rect if the viewport is getting near the edge of
    // our preexisting clip rect. We use half of the size of the viewport as a heuristic
    // for "close."
    static VIEWPORT_SCROLL_MARGIN_SIZE: f32 = 0.5;
    let viewport_scroll_margin = new_viewport.size * VIEWPORT_SCROLL_MARGIN_SIZE;

    (clip_rect.origin.x - new_viewport.origin.x).abs() <= viewport_scroll_margin.width ||
        (clip_rect.max_x() - new_viewport.max_x()).abs() <= viewport_scroll_margin.width ||
        (clip_rect.origin.y - new_viewport.origin.y).abs() <= viewport_scroll_margin.height ||
        (clip_rect.max_y() - new_viewport.max_y()).abs() <= viewport_scroll_margin.height
}

fn debug_reflow_events(id: PipelineId, reflow_goal: &ReflowGoal) {
    let goal_string = match *reflow_goal {
        ReflowGoal::UpdateTheRendering => "\tFull",
        ReflowGoal::UpdateScrollNode(_) => "\tUpdateScrollNode",
        ReflowGoal::LayoutQuery(ref query_msg) => match *query_msg {
            QueryMsg::ContentBox => "\tContentBoxQuery",
            QueryMsg::ContentBoxes => "\tContentBoxesQuery",
            QueryMsg::NodesFromPointQuery => "\tNodesFromPointQuery",
            QueryMsg::ClientRectQuery => "\tClientRectQuery",
            QueryMsg::ScrollingAreaQuery => "\tNodeScrollGeometryQuery",
            QueryMsg::ResolvedStyleQuery => "\tResolvedStyleQuery",
            QueryMsg::ResolvedFontStyleQuery => "\nResolvedFontStyleQuery",
            QueryMsg::OffsetParentQuery => "\tOffsetParentQuery",
            QueryMsg::StyleQuery => "\tStyleQuery",
            QueryMsg::TextIndexQuery => "\tTextIndexQuery",
            QueryMsg::ElementInnerOuterTextQuery => "\tElementInnerOuterTextQuery",
            QueryMsg::InnerWindowDimensionsQuery => "\tInnerWindowDimensionsQuery",
        },
    };

    println!("**** pipeline={id}\t{goal_string}");
}

impl Window {
    // https://html.spec.whatwg.org/multipage/#dom-window-postmessage step 7.
    pub(crate) fn post_message(
        &self,
        target_origin: Option<ImmutableOrigin>,
        source_origin: ImmutableOrigin,
        source: &WindowProxy,
        data: StructuredSerializedData,
    ) {
        let this = Trusted::new(self);
        let source = Trusted::new(source);
        let task = task!(post_serialised_message: move || {
            let this = this.root();
            let source = source.root();
            let document = this.Document();

            // Step 7.1.
            if let Some(ref target_origin) = target_origin {
                if !target_origin.same_origin(document.origin()) {
                    return;
                }
            }

            // Steps 7.2.-7.5.
            let cx = this.get_cx();
            let obj = this.reflector().get_jsobject();
            let _ac = JSAutoRealm::new(*cx, obj.get());
            rooted!(in(*cx) let mut message_clone = UndefinedValue());
            if let Ok(ports) = structuredclone::read(this.upcast(), data, message_clone.handle_mut()) {
                // Step 7.6, 7.7
                MessageEvent::dispatch_jsval(
                    this.upcast(),
                    this.upcast(),
                    message_clone.handle(),
                    Some(&source_origin.ascii_serialization()),
                    Some(&*source),
                    ports,
                    CanGc::note()
                );
            } else {
                // Step 4, fire messageerror.
                MessageEvent::dispatch_error(
                    this.upcast(),
                    this.upcast(),
                    CanGc::note()
                );
            }
        });
        // TODO(#12718): Use the "posted message task source".
        self.as_global_scope()
            .task_manager()
            .dom_manipulation_task_source()
            .queue(task);
    }
}

#[derive(Clone, MallocSizeOf)]
pub(crate) struct CSSErrorReporter {
    pub(crate) pipelineid: PipelineId,
    // Arc+Mutex combo is necessary to make this struct Sync,
    // which is necessary to fulfill the bounds required by the
    // uses of the ParseErrorReporter trait.
    #[ignore_malloc_size_of = "Arc is defined in libstd"]
    pub(crate) script_chan: Arc<Mutex<IpcSender<ScriptThreadMessage>>>,
}
unsafe_no_jsmanaged_fields!(CSSErrorReporter);

impl ParseErrorReporter for CSSErrorReporter {
    fn report_error(
        &self,
        url: &UrlExtraData,
        location: SourceLocation,
        error: ContextualParseError,
    ) {
        if log_enabled!(log::Level::Info) {
            info!(
                "Url:\t{}\n{}:{} {}",
                url.0.as_str(),
                location.line,
                location.column,
                error
            )
        }

        //TODO: report a real filename
        let _ = self
            .script_chan
            .lock()
            .unwrap()
            .send(ScriptThreadMessage::ReportCSSError(
                self.pipelineid,
                url.0.to_string(),
                location.line,
                location.column,
                error.to_string(),
            ));
    }
}

fn is_named_element_with_name_attribute(elem: &Element) -> bool {
    let type_ = match elem.upcast::<Node>().type_id() {
        NodeTypeId::Element(ElementTypeId::HTMLElement(type_)) => type_,
        _ => return false,
    };
    matches!(
        type_,
        HTMLElementTypeId::HTMLEmbedElement |
            HTMLElementTypeId::HTMLFormElement |
            HTMLElementTypeId::HTMLImageElement |
            HTMLElementTypeId::HTMLObjectElement
    )
}

fn is_named_element_with_id_attribute(elem: &Element) -> bool {
    elem.is_html_element()
}

#[allow(unsafe_code)]
#[no_mangle]
/// Helper for interactive debugging sessions in lldb/gdb.
unsafe extern "C" fn dump_js_stack(cx: *mut RawJSContext) {
    DumpJSStack(cx, true, false, false);
}
