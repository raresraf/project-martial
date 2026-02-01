/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// @file document.rs
/// @brief This file implements the Document object, representing a web page loaded in the browser.
/// It encapsulates the DOM tree, manages its lifecycle, handles user interactions (mouse, keyboard, touch),
/// and integrates with various browser subsystems like styling, layout, networking, and scripting.
/// Functional Utility: Centralizes the management and manipulation of web content, serving as the
/// primary entry point for programmatic access to page content and functionality.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{HashMap, HashSet, VecDeque};
use std::default::Default;
use std::f64::consts::PI; // Mathematical constant PI for calculations.
use std::mem;
use std::rc::Rc;
use std::slice::from_ref;
use std::str::FromStr;
use std::sync::{LazyLock, Mutex}; // LazyLock for lazy static initialization, Mutex for synchronization.
use std::time::{Duration, Instant}; // Duration and Instant for time-related operations.

use base::cross_process_instant::CrossProcessInstant;
use base::id::WebViewId;
use canvas_traits::canvas::CanvasId;
use canvas_traits::webgl::{self, WebGLContextId, WebGLMsg};
use chrono::Local;
use constellation_traits::{
    AnimationTickType,
    NavigationHistoryBehavior,
    ScriptToConstellationMessage,
};
use content_security_policy::{self as csp, CspList, PolicyDisposition};
use cookie::Cookie;
use cssparser::match_ignore_ascii_case;
use devtools_traits::{
    AttrModification,
    AutoMargins,
    ComputedNodeLayout,
    CssDatabaseProperty,
    EvaluateJSReply,
    NodeInfo,
    NodeStyle,
    RuleModification,
    TimelineMarker,
    TimelineMarkerType,
};
use embedder_traits::{
    AllowOrDeny,
    AnimationState,
    CompositorHitTestResult,
    ContextMenuResult,
    EditingActionEvent,
    EmbedderMsg,
    ImeEvent,
    InputEvent,
    LoadStatus,
    MouseButton,
    MouseButtonAction,
    MouseButtonEvent,
    TouchEvent,
    TouchEventType,
    TouchId,
    WheelEvent,
};
use encoding_rs::{Encoding, UTF_8};
use euclid::default::{Point2D, Rect, Size2D};
use html5ever::{LocalName, Namespace, QualName, local_name, namespace_url, ns};
use hyper_serde::Serde;
use ipc_channel::ipc;
use js::rust::{HandleObject, HandleValue};
use keyboard_types::{Code, Key, KeyState, Modifiers};
use metrics::{InteractiveFlag, InteractiveWindow, ProgressiveWebMetrics};
use mime::{self, Mime};
use net_traits::CookieSource::NonHTTP;
use net_traits::CoreResourceMsg::{GetCookiesForUrl, SetCookiesForUrl};
use net_traits::policy_container::PolicyContainer;
use net_traits::pub_domains::is_pub_domain;
use net_traits::request::{InsecureRequestsPolicy, RequestBuilder};
use net_traits::response::HttpsState;
use net_traits::{FetchResponseListener, IpcSend, ReferrerPolicy};
use num_traits::ToPrimitive;
use percent_encoding::percent_decode;
use profile_traits::ipc as profile_ipc;
use profile_traits::time::TimerMetadataFrameType;
use regex::bytes::Regex;
use script_bindings::interfaces::DocumentHelpers;
use script_layout_interface::{PendingRestyle, TrustedNodeAddress};
use script_traits::{ConstellationInputEvent, DocumentActivity, ProgressiveWebMetricType};
use servo_arc::Arc;
use servo_config::pref;
use servo_media::{ClientContextId, ServoMedia};
use servo_url::{ImmutableOrigin, MutableOrigin, ServoUrl};
use style::attr::AttrValue;
use style::context::QuirksMode;
use style::invalidation::element::restyle_hints::RestyleHint;
use style::selector_parser::Snapshot;
use style::shared_lock::SharedRwLock as StyleSharedRwLock;
use style::str::{split_html_space_chars, str_join};
use style::stylesheet_set::DocumentStylesheetSet;
use style::stylesheets::{Origin, OriginSet, Stylesheet};
use stylo_atoms::Atom;
use url::Host;
use uuid::Uuid;
#[cfg(feature = "webgpu")]
use webgpu_traits::WebGPUContextId;
use webrender_api::units::DeviceIntRect;

use crate::animation_timeline::AnimationTimeline;
use crate::animations::Animations;
use crate::canvas_context::CanvasContext as _;
use crate::document_loader::{DocumentLoader, LoadType};
use crate::dom::attr::Attr;
use crate::dom::beforeunloadevent::BeforeUnloadEvent;
use crate::dom::bindings::callback::ExceptionHandling;
use crate::dom::bindings::cell::{DomRefCell, Ref, RefMut};
use crate::dom::bindings::codegen::Bindings::BeforeUnloadEventBinding::BeforeUnloadEvent_Binding::BeforeUnloadEventMethods;
use crate::dom::bindings::codegen::Bindings::DocumentBinding::{
    DocumentMethods,
    DocumentReadyState,
    DocumentVisibilityState,
    NamedPropertyValue,
};
use crate::dom::bindings::codegen::Bindings::EventBinding::Event_Binding::EventMethods;
use crate::dom::bindings::codegen::Bindings::HTMLIFrameElementBinding::HTMLIFrameElement_Binding::HTMLIFrameElementMethods;
use crate::dom::bindings::codegen::Bindings::HTMLInputElementBinding::HTMLInputElementMethods;
use crate::dom::bindings::codegen::Bindings::HTMLTextAreaElementBinding::HTMLTextAreaElementMethods;
use crate::dom::bindings::codegen::Bindings::NavigatorBinding::Navigator_Binding::NavigatorMethods;
use crate::dom::bindings::codegen::Bindings::NodeBinding::NodeMethods;
use crate::dom::bindings::codegen::Bindings::NodeFilterBinding::NodeFilter;
use crate::dom::bindings::codegen::Bindings::PerformanceBinding::PerformanceMethods;
use crate::dom::bindings::codegen::Bindings::PermissionStatusBinding::PermissionName;
use crate::dom::bindings::codegen::Bindings::ShadowRootBinding::ShadowRootMethods;
use crate::dom::bindings::codegen::Bindings::TouchBinding::TouchMethods;
use crate::dom::bindings::codegen::Bindings::WindowBinding::{
    FrameRequestCallback,
    ScrollBehavior,
    WindowMethods,
};
use crate::dom::bindings::codegen::Bindings::XPathEvaluatorBinding::XPathEvaluatorMethods;
use crate::dom::bindings::codegen::Bindings::XPathNSResolverBinding::XPathNSResolver;
use crate::dom::bindings::codegen::UnionTypes::{NodeOrString, StringOrElementCreationOptions};
use crate::dom::bindings::error::{Error, ErrorInfo, ErrorResult, Fallible};
use crate::dom::bindings::inheritance::{Castable, ElementTypeId, HTMLElementTypeId, NodeTypeId};
use crate::dom::bindings::num::Finite;
use crate::dom::bindings::refcounted::{Trusted, TrustedPromise};
use crate::dom::bindings::reflector::{DomGlobal, reflect_dom_object_with_proto};
use crate::dom::bindings::root::{Dom, DomRoot, DomSlice, LayoutDom, MutNullableDom, ToLayout};
use crate::dom::bindings::str::{DOMString, USVString};
use crate::dom::bindings::trace::{HashMapTracedValues, NoTrace};
#[cfg(feature = "webgpu")]
use crate::dom::bindings::weakref::WeakRef;
use crate::dom::bindings::xmlname::{
    matches_name_production,
    namespace_from_domstring,
    validate_and_extract,
};
use crate::dom::canvasrenderingcontext2d::CanvasRenderingContext2D;
use crate::dom::cdatasection::CDATASection;
use crate::dom::clipboardevent::{ClipboardEvent, ClipboardEventType};
use crate::dom::comment::Comment;
use crate::dom::compositionevent::CompositionEvent;
use crate::dom::cssstylesheet::CSSStyleSheet;
use crate::dom::customelementregistry::CustomElementDefinition;
use crate::dom::customevent::CustomEvent;
use crate::dom::datatransfer::DataTransfer;
use crate::dom::documentfragment::DocumentFragment;
use crate::dom::documentorshadowroot::{DocumentOrShadowRoot, StyleSheetInDocument};
use crate::dom::documenttype::DocumentType;
use crate::dom::domimplementation::DOMImplementation;
use crate::dom::element::{
    CustomElementCreationMode,
    Element,
    ElementCreator,
    ElementPerformFullscreenEnter,
    ElementPerformFullscreenExit,
};
use crate::dom::event::{Event, EventBubbles, EventCancelable, EventDefault, EventStatus};
use crate::dom::eventtarget::EventTarget;
use crate::dom::focusevent::FocusEvent;
use crate::dom::fontfaceset::FontFaceSet;
use crate::dom::globalscope::GlobalScope;
use crate::dom::hashchangeevent::HashChangeEvent;
use crate::dom::htmlanchorelement::HTMLAnchorElement;
use crate::dom::htmlareaelement::HTMLAreaElement;
use crate::dom::htmlbaseelement::HTMLBaseElement;
use crate::dom::htmlbodyelement::HTMLBodyElement;
use crate::dom::htmlcollection::{CollectionFilter, HTMLCollection};
use crate::dom::htmlelement::HTMLElement;
use crate::dom::htmlembedelement::HTMLEmbedElement;
use crate::dom::htmlformelement::{FormControl, FormControlElementHelpers, HTMLFormElement};
use crate::dom::htmlheadelement::HTMLHeadElement;
use crate::dom::htmlhtmlelement::HTMLHtmlElement;
use crate::dom::htmliframeelement::HTMLIFrameElement;
use crate::dom::htmlimageelement::HTMLImageElement;
use crate::dom::htmlinputelement::HTMLInputElement;
use crate::dom::htmlscriptelement::{HTMLScriptElement, ScriptResult};
use crate::dom::htmltextareaelement::HTMLTextAreaElement;
use crate::dom::htmltitleelement::HTMLTitleElement;
use crate::dom::intersectionobserver::IntersectionObserver;
use crate::dom::keyboardevent::KeyboardEvent;
use crate::dom::location::{Location, NavigationType};
use crate::dom::messageevent::MessageEvent;
use crate::dom::mouseevent::MouseEvent;
use crate::dom::node::{
    self,
    CloneChildrenFlag,
    Node,
    NodeDamage,
    NodeFlags,
    NodeTraits,
    ShadowIncluding,
};
use crate::dom::nodeiterator::NodeIterator;
use crate::dom::nodelist::NodeList;
use crate::dom::pagetransitionevent::PageTransitionEvent;
use crate::dom::performanceentry::PerformanceEntry;
use crate::dom::performancepainttiming::PerformancePaintTiming;
use crate::dom::pointerevent::{PointerEvent, PointerId};
use crate::dom::processinginstruction::ProcessingInstruction;
use crate::dom::promise::Promise;
use crate::dom::range::Range;
use crate::dom::resizeobserver::{ResizeObservationDepth, ResizeObserver};
use crate::dom::selection::Selection;
use crate::dom::servoparser::ServoParser;
use crate::dom::shadowroot::ShadowRoot;
use crate::dom::storageevent::StorageEvent;
use crate::dom::stylesheetlist::{StyleSheetList, StyleSheetListOwner};
use crate::dom::text::Text;
use crate::dom::touch::Touch;
use crate::dom::touchevent::TouchEvent as DomTouchEvent;
use crate::dom::touchlist::TouchList;
use crate::dom::treewalker::TreeWalker;
use crate::dom::types::VisibilityStateEntry;
use crate::dom::uievent::UIEvent;
use crate::dom::virtualmethods::vtable_for;
use crate::dom::webglrenderingcontext::WebGLRenderingContext;
#[cfg(feature = "webgpu")]
use crate::dom::webgpu::gpucanvascontext::GPUCanvasContext;
use crate::dom::wheelevent::WheelEvent as DomWheelEvent;
use crate::dom::window::Window;
use crate::dom::windowproxy::WindowProxy;
use crate::dom::xpathevaluator::XPathEvaluator;
use crate::drag_data_store::{DragDataStore, Kind, Mode};
use crate::fetch::FetchCanceller;
use crate::iframe_collection::IFrameCollection;
use crate::image_animation::ImageAnimationManager;
use crate::messaging::{CommonScriptMsg, MainThreadScriptMsg};
use crate::network_listener::{NetworkListener, PreInvoke};
use crate::realms::{AlreadyInRealm, InRealm, enter_realm};
use crate::script_runtime::{CanGc, ScriptThreadEventCategory};
use crate::script_thread::{ScriptThread, with_script_thread};
use crate::stylesheet_set::StylesheetSetRef;
use crate::task::TaskBox;
use crate::task_source::TaskSourceName;
use crate::timers::OneshotTimerCallback;

/// The number of times we are allowed to see spurious `requestAnimationFrame()` calls before
/// falling back to fake ones.
///
/// A spurious `requestAnimationFrame()` call is defined as one that does not change the DOM.
const SPURIOUS_ANIMATION_FRAME_THRESHOLD: u8 = 5;

/// The amount of time between fake `requestAnimationFrame()`s.
const FAKE_REQUEST_ANIMATION_FRAME_DELAY: u64 = 16;

/// @enum TouchEventResult
/// @brief Represents the outcome of processing a touch event.
pub(crate) enum TouchEventResult {
    Processed(bool), //!< Touch event was processed, with a boolean indicating if it was consumed.
    Forwarded,       //!< Touch event was forwarded to another handler.
}

/// @enum FireMouseEventType
/// @brief Defines the types of mouse events that can be fired.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum FireMouseEventType {
    Move,  //!< Mouse move event.
    Over,  //!< Mouse over event.
    Out,   //!< Mouse out event.
    Enter, //!< Mouse enter event.
    Leave, //!< Mouse leave event.
}

impl FireMouseEventType {
    /// @brief Converts the `FireMouseEventType` enum variant to its corresponding string representation.
    /// Functional Utility: Provides the DOM event name for each mouse event type.
    /// @return A string slice representing the mouse event type.
    pub(crate) fn as_str(&self) -> &str {
        match *self {
            FireMouseEventType::Move => "mousemove",
            FireMouseEventType::Over => "mouseover",
            FireMouseEventType::Out => "mouseout",
            FireMouseEventType::Enter => "mouseenter",
            FireMouseEventType::Leave => "mouseleave",
        }
    }
}

/// @struct RefreshRedirectDue
/// @brief Represents a pending refresh redirect, including the target URL and the window.
#[derive(JSTraceable, MallocSizeOf)]
pub(crate) struct RefreshRedirectDue {
    #[no_trace]
    pub(crate) url: ServoUrl, //!< The URL to redirect to.
    #[ignore_malloc_size_of = "non-owning"]
    pub(crate) window: DomRoot<Window>, //!< The window associated with the redirect.
}
impl RefreshRedirectDue {
    /// @brief Invokes the refresh redirect, navigating the associated window to the target URL.
    /// Functional Utility: Executes a declarative refresh navigation action.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn invoke(self, can_gc: CanGc) {
        self.window.Location().navigate(
            self.url.clone(),
            NavigationHistoryBehavior::Replace,
            NavigationType::DeclarativeRefresh,
            can_gc,
        );
    }
}

/// @enum IsHTMLDocument
/// @brief Indicates whether a document is an HTML document or a non-HTML document.
#[derive(Clone, Copy, Debug, JSTraceable, MallocSizeOf, PartialEq)]
pub(crate) enum IsHTMLDocument {
    HTMLDocument,    //!< The document is an HTML document.
    NonHTMLDocument, //!< The document is not an HTML document.
}

/// @enum FocusTransaction
/// @brief Represents the state of a focus operation transaction.
#[derive(JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
enum FocusTransaction {
    /// No focus operation is in effect.
    NotInTransaction,
    /// A focus operation is in effect.
    /// Contains the element that has most recently requested focus for itself.
    InTransaction(Option<Dom<Element>>),
}

/// @enum DeclarativeRefresh
/// @brief Information about a declarative refresh, including pending loads or creation status.
#[derive(JSTraceable, MallocSizeOf)]
pub(crate) enum DeclarativeRefresh {
    PendingLoad {
        #[no_trace]
        url: ServoUrl, //!< The URL of the pending load.
        time: u64,     //!< The time remaining for the refresh.
    },
    CreatedAfterLoad, //!< The refresh was created after the document finished loading.
}
#[cfg(feature = "webgpu")]
pub(crate) type WebGPUContextsMap =
    Rc<RefCell<HashMapTracedValues<WebGLContextId, WeakRef<GPUCanvasContext>>>>;

/// @struct Document
/// @brief Represents the Document object as defined by the DOM and HTML specifications.
/// This struct holds all the state and functionality related to a web page, including
/// its DOM tree, manages its lifecycle, handles user interactions (mouse, keyboard, touch),
/// and integrates with various browser subsystems like styling, layout, networking, and scripting.
/// Functional Utility: Centralizes the management and manipulation of web content, serving as the
/// primary entry point for programmatic access to page content and functionality.
///
/// <https://dom.spec.whatwg.org/#document>
#[dom_struct]
pub(crate) struct Document {
    node: Node,                                     //!< The underlying Node object for the document.
    document_or_shadow_root: DocumentOrShadowRoot,  //!< Common properties for Document and ShadowRoot.
    window: Dom<Window>,                            //!< The Window object that this document belongs to.
    implementation: MutNullableDom<DOMImplementation>, //!< The DOMImplementation object for this document.
    #[ignore_malloc_size_of = "type from external crate"]
    #[no_trace]
    content_type: Mime, //!< The MIME type of the document.
    last_modified: Option<String>, //!< The last modification date of the document.
    #[no_trace]
    encoding: Cell<&'static Encoding>, //!< The character encoding of the document.
    has_browsing_context: bool, //!< Indicates if this document is associated with a browsing context.
    is_html_document: bool,     //!< Indicates if this document is an HTML document.
    #[no_trace]
    activity: Cell<DocumentActivity>, //!< The activity state of the document (e.g., active, inactive).
    #[no_trace]
    url: DomRefCell<ServoUrl>, //!< The URL of the document.
    #[ignore_malloc_size_of = "defined in selectors"]
    #[no_trace]
    quirks_mode: Cell<QuirksMode>, //!< The quirks mode of the document.
    /// Caches for the getElement methods
    id_map: DomRefCell<HashMapTracedValues<Atom, Vec<Dom<Element>>>>, //!< Cache for elements by ID.
    name_map: DomRefCell<HashMapTracedValues<Atom, Vec<Dom<Element>>>>, //!< Cache for elements by name.
    tag_map: DomRefCell<HashMapTracedValues<LocalName, Dom<HTMLCollection>>>, //!< Cache for elements by tag name.
    tagns_map: DomRefCell<HashMapTracedValues<QualName, Dom<HTMLCollection>>>, //!< Cache for elements by tag name and namespace.
    classes_map: DomRefCell<HashMapTracedValues<Vec<Atom>, Dom<HTMLCollection>>>, //!< Cache for elements by class names.
    images: MutNullableDom<HTMLCollection>, //!< Collection of image elements.
    embeds: MutNullableDom<HTMLCollection>, //!< Collection of embed elements.
    links: MutNullableDom<HTMLCollection>, //!< Collection of link elements.
    forms: MutNullableDom<HTMLCollection>, //!< Collection of form elements.
    scripts: MutNullableDom<HTMLCollection>, //!< Collection of script elements.
    anchors: MutNullableDom<HTMLCollection>, //!< Collection of anchor elements.
    applets: MutNullableDom<HTMLCollection>, //!< Collection of applet elements.
    /// Information about the `<iframes>` in this [`Document`].
    iframes: RefCell<IFrameCollection>, //!< Collection of iframe elements.
    /// Lock use for style attributes and author-origin stylesheet objects in this document.
    /// Can be acquired once for accessing many objects.
    #[no_trace]
    style_shared_lock: StyleSharedRwLock, //!< Shared lock for style attributes and stylesheets.
    /// List of stylesheets associated with nodes in this document. |None| if the list needs to be refreshed.
    #[custom_trace]
    stylesheets: DomRefCell<DocumentStylesheetSet<StyleSheetInDocument>>, //!< Set of stylesheets for the document.
    stylesheet_list: MutNullableDom<StyleSheetList>, //!< The StyleSheetList object.
    ready_state: Cell<DocumentReadyState>, //!< The ready state of the document.
    /// Whether the DOMContentLoaded event has already been dispatched.
    domcontentloaded_dispatched: Cell<bool>, //!< Flag indicating if DOMContentLoaded has been dispatched.
    /// The state of this document's focus transaction.
    focus_transaction: DomRefCell<FocusTransaction>, //!< Tracks the current focus transaction.
    /// The element that currently has the document focus context.
    focused: MutNullableDom<Element>, //!< The currently focused element.
    /// The script element that is currently executing.
    current_script: MutNullableDom<HTMLScriptElement>, //!< The script element currently being executed.
    /// <https://html.spec.whatwg.org/multipage/#pending-parsing-blocking-script>
    pending_parsing_blocking_script: DomRefCell<Option<PendingScript>>, //!< A script that is blocking parsing.
    /// Number of stylesheets that block executing the next parser-inserted script
    script_blocking_stylesheets_count: Cell<u32>, //!< Count of stylesheets blocking script execution.
    /// <https://html.spec.whatwg.org/multipage/#list-of-scripts-that-will-execute-when-the-document-has-finished-parsing>
    deferred_scripts: PendingInOrderScriptVec, //!< List of scripts to execute after parsing.
    /// <https://html.spec.whatwg.org/multipage/#list-of-scripts-that-will-execute-in-order-as-soon-as-possible>
    asap_in_order_scripts_list: PendingInOrderScriptVec, //!< List of scripts to execute ASAP in order.
    /// <https://html.spec.whatwg.org/multipage/#set-of-scripts-that-will-execute-as-soon-as-possible>
    asap_scripts_set: DomRefCell<Vec<Dom<HTMLScriptElement>>>, //!< Set of scripts to execute ASAP.
    /// <https://html.spec.whatwg.org/multipage/#concept-n-noscript>
    /// True if scripting is enabled for all scripts in this document
    scripting_enabled: bool, //!< Flag indicating if scripting is enabled.
    /// <https://html.spec.whatwg.org/multipage/#animation-frame-callback-identifier>
    /// Current identifier of animation frame callback
    animation_frame_ident: Cell<u32>, //!< Current identifier for `requestAnimationFrame` callbacks.
    /// <https://html.spec.whatwg.org/multipage/#list-of-animation-frame-callbacks>
    /// List of animation frame callbacks
    animation_frame_list: DomRefCell<VecDeque<(u32, Option<AnimationFrameCallback>)>>, //!< Queue of animation frame callbacks.
    /// Whether we're in the process of running animation callbacks.
    ///
    /// Tracking this is not necessary for correctness. Instead, it is an optimization to avoid
    /// sending needless `ChangeRunningAnimationsState` messages to the compositor.
    running_animation_callbacks: Cell<bool>, //!< Flag indicating if animation callbacks are currently running.
    /// Tracks all outstanding loads related to this document.
    loader: DomRefCell<DocumentLoader>, //!< The document loader for tracking resource loads.
    /// The current active HTML parser, to allow resuming after interruptions.
    current_parser: MutNullableDom<ServoParser>, //!< The HTML parser currently associated with this document.
    /// The cached first `base` element with an `href` attribute.
    base_element: MutNullableDom<HTMLBaseElement>, //!< Cached reference to the first `<base>` element.
    /// This field is set to the document itself for inert documents.
    /// <https://html.spec.whatwg.org/multipage/#appropriate-template-contents-owner-document>
    appropriate_template_contents_owner_document: MutNullableDom<Document>, //!< Owner document for template contents.
    /// Information on elements needing restyle to ship over to layout when the
    /// time comes.
    pending_restyles: DomRefCell<HashMap<Dom<Element>, NoTrace<PendingRestyle>>>, //!< Elements waiting for restyle.
    /// This flag will be true if the `Document` needs to be painted again
    /// during the next full layout attempt due to some external change such as
    /// the web view changing size, or because the previous layout was only for
    /// layout queries (which do not trigger display).
    needs_paint: Cell<bool>, //!< Flag indicating if the document needs to be repainted.
    /// <http://w3c.github.io/touch-events/#dfn-active-touch-point>
    active_touch_points: DomRefCell<Vec<Dom<Touch>>>, //!< List of currently active touch points.
    /// Navigation Timing properties:
    /// <https://w3c.github.io/navigation-timing/#sec-PerformanceNavigationTiming>
    #[no_trace]
    dom_interactive: Cell<Option<CrossProcessInstant>>, //!< Timestamp for `domInteractive`.
    #[no_trace]
    dom_content_loaded_event_start: Cell<Option<CrossProcessInstant>>, //!< Timestamp for `DOMContentLoaded` event start.
    #[no_trace]
    dom_content_loaded_event_end: Cell<Option<CrossProcessInstant>>, //!< Timestamp for `DOMContentLoaded` event end.
    #[no_trace]
    dom_complete: Cell<Option<CrossProcessInstant>>, //!< Timestamp for `domComplete`.
    #[no_trace]
    top_level_dom_complete: Cell<Option<CrossProcessInstant>>, //!< Timestamp for top-level `domComplete`.
    #[no_trace]
    load_event_start: Cell<Option<CrossProcessInstant>>, //!< Timestamp for `load` event start.
    #[no_trace]
    load_event_end: Cell<Option<CrossProcessInstant>>, //!< Timestamp for `load` event end.
    #[no_trace]
    unload_event_start: Cell<Option<CrossProcessInstant>>, //!< Timestamp for `unload` event start.
    #[no_trace]
    unload_event_end: Cell<Option<CrossProcessInstant>>, //!< Timestamp for `unload` event end.
    /// <https://html.spec.whatwg.org/multipage/#concept-document-https-state>
    #[no_trace]
    https_state: Cell<HttpsState>, //!< The HTTPS state of the document.
    /// The document's origin.
    #[no_trace]
    origin: MutableOrigin, //!< The mutable origin of the document.
    /// <https://html.spec.whatwg.org/multipage/#dom-document-referrer>
    referrer: Option<String>, //!< The referrer URL.
    /// <https://html.spec.whatwg.org/multipage/#target-element>
    target_element: MutNullableDom<Element>, //!< The target element for fragment navigation.
    /// <https://html.spec.whatwg.org/multipage/#concept-document-policy-container>
    #[no_trace]
    policy_container: DomRefCell<PolicyContainer>, //!< The policy container for the document.
    /// <https://w3c.github.io/uievents/#event-type-dblclick>
    #[ignore_malloc_size_of = "Defined in std"]
    #[no_trace]
    last_click_info: DomRefCell<Option<(Instant, Point2D<f32>)>>, //!< Information about the last click for double-click detection.
    /// <https://html.spec.whatwg.org/multipage/#ignore-destructive-writes-counter>
    ignore_destructive_writes_counter: Cell<u32>, //!< Counter for ignoring destructive writes.
    /// <https://html.spec.whatwg.org/multipage/#ignore-opens-during-unload-counter>
    ignore_opens_during_unload_counter: Cell<u32>, //!< Counter for ignoring opens during unload.
    /// The number of spurious `requestAnimationFrame()` requests we've received.
    ///
    /// A rAF request is considered spurious if nothing was actually reflowed.
    spurious_animation_frames: Cell<u8>, //!< Counter for spurious `requestAnimationFrame` calls.

    /// Track the total number of elements in this DOM's tree.
    /// This is sent to layout every time a reflow is done;
    /// layout uses this to determine if the gains from parallel layout will be worth the overhead.
    ///
    /// See also: <https://github.com/servo/servo/issues/10110>
    dom_count: Cell<u32>, //!< Total number of elements in the DOM tree.
    /// Entry node for fullscreen.
    fullscreen_element: MutNullableDom<Element>, //!< The element currently in fullscreen mode.
    /// Map from ID to set of form control elements that have that ID as
    /// their 'form' content attribute. Used to reset form controls
    /// whenever any element with the same ID as the form attribute
    /// is inserted or removed from the document.
    /// See <https://html.spec.whatwg.org/multipage/#form-owner>
    form_id_listener_map: DomRefCell<HashMapTracedValues<Atom, HashSet<Dom<Element>>>>, //!< Maps form IDs to listening form control elements.
    #[no_trace]
    interactive_time: DomRefCell<ProgressiveWebMetrics>, //!< Metrics related to interactive time.
    #[no_trace]
    tti_window: DomRefCell<InteractiveWindow>, //!< Window for Time To Interactive (TTI) metrics.
    /// RAII canceller for Fetch
    canceller: FetchCanceller, //!< Handles cancellation of fetch requests.
    /// <https://html.spec.whatwg.org/multipage/#throw-on-dynamic-markup-insertion-counter>
    throw_on_dynamic_markup_insertion_counter: Cell<u64>, //!< Counter for dynamic markup insertion.
    /// <https://html.spec.whatwg.org/multipage/#page-showing>
    page_showing: Cell<bool>, //!< Flag indicating if the page is currently showing.
    /// Whether the document is salvageable.
    salvageable: Cell<bool>, //!< Flag indicating if the document is salvageable.
    /// Whether the document was aborted with an active parser
    active_parser_was_aborted: Cell<bool>, //!< Flag indicating if the active parser was aborted.
    /// Whether the unload event has already been fired.
    fired_unload: Cell<bool>, //!< Flag indicating if the unload event has been fired.
    /// List of responsive images
    responsive_images: DomRefCell<Vec<Dom<HTMLImageElement>>>, //!< List of responsive image elements.
    /// Number of redirects for the document load
    redirect_count: Cell<u16>, //!< Count of redirects during document load.
    /// Number of outstanding requests to prevent JS or layout from running.
    script_and_layout_blockers: Cell<u32>, //!< Count of script and layout blocking requests.
    /// List of tasks to execute as soon as last script/layout blocker is removed.
    #[ignore_malloc_size_of = "Measuring trait objects is hard"]
    delayed_tasks: DomRefCell<Vec<Box<dyn TaskBox>>>, //!< Queue of tasks to execute later.
    /// <https://html.spec.whatwg.org/multipage/#completely-loaded>
    completely_loaded: Cell<bool>, //!< Flag indicating if the document is completely loaded.
    /// Set of shadow roots connected to the document tree.
    shadow_roots: DomRefCell<HashSet<Dom<ShadowRoot>>>, //!< Set of connected shadow roots.
    /// Whether any of the shadow roots need the stylesheets flushed.
    shadow_roots_styles_changed: Cell<bool>, //!< Flag indicating if shadow root styles have changed.
    /// List of registered media controls.
    /// We need to keep this list to allow the media controls to
    /// access the "privileged" document.servoGetMediaControls(id) API,
    /// where `id` needs to match any of the registered ShadowRoots
    /// hosting the media controls UI.
    media_controls: DomRefCell<HashMap<String, Dom<ShadowRoot>>>, //!< Maps media control IDs to shadow roots.
    /// List of all context 2d IDs that need flushing.
    dirty_2d_contexts: DomRefCell<HashMapTracedValues<CanvasId, Dom<CanvasRenderingContext2D>>>, //!< Dirty 2D canvas contexts.
    /// List of all WebGL context IDs that need flushing.
    dirty_webgl_contexts:
        DomRefCell<HashMapTracedValues<WebGLContextId, Dom<WebGLRenderingContext>>>, //!< Dirty WebGL contexts.
    /// List of all WebGPU contexts.
    #[cfg(feature = "webgpu")]
    #[ignore_malloc_size_of = "Rc are hard"]
    webgpu_contexts: WebGPUContextsMap, //!< WebGPU contexts.
    /// <https://w3c.github.io/slection-api/#dfn-selection>
    selection: MutNullableDom<Selection>, //!< The current selection object.
    /// A timeline for animations which is used for synchronizing animations.
    /// <https://drafts.csswg.org/web-animations/#timeline>
    animation_timeline: DomRefCell<AnimationTimeline>, //!< The animation timeline for the document.
    /// Animations for this Document
    animations: DomRefCell<Animations>, //!< Manages animations for this document.
    /// Image Animation Manager for this Document
    image_animation_manager: DomRefCell<ImageAnimationManager>, //!< Manages image animations for this document.
    /// The nearest inclusive ancestors to all the nodes that require a restyle.
    dirty_root: MutNullableDom<Element>, //!< The root element of the dirty region for restyle.
    /// <https://html.spec.whatwg.org/multipage/#will-declaratively-refresh>
    declarative_refresh: DomRefCell<Option<DeclarativeRefresh>>, //!< Information about a declarative refresh.
    /// Pending input events, to be handled at the next rendering opportunity.
    #[no_trace]
    #[ignore_malloc_size_of = "CompositorEvent contains data from outside crates"]
    pending_input_events: DomRefCell<Vec<ConstellationInputEvent>>, //!< Queue of pending input events.
    /// The index of the last mouse move event in the pending compositor events queue.
    mouse_move_event_index: DomRefCell<Option<usize>>, //!< Index of the last mouse move event.
    /// Pending animation ticks, to be handled at the next rendering opportunity.
    #[no_trace]
    #[ignore_malloc_size_of = "AnimationTickType contains data from an outside crate"]
    pending_animation_ticks: DomRefCell<AnimationTickType>, //!< Pending animation ticks.
    /// <https://drafts.csswg.org/resize-observer/#dom-document-resizeobservers-slot>
    ///
    /// Note: we are storing, but never removing, resize observers.
    /// The lifetime of resize observers is specified at
    /// <https://drafts.csswg.org/resize-observer/#lifetime>.
    /// But implementing it comes with known problems:
    /// - <https://bugzilla.mozilla.org/show_bug.cgi?id=1596992>
    /// - <https://github.com/w3c/csswg-drafts/issues/4518>
    resize_observers: DomRefCell<Vec<Dom<ResizeObserver>>>, //!< List of registered resize observers.
    /// The set of all fonts loaded by this document.
    /// <https://drafts.csswg.org/css-font-loading/#font-face-source>
    fonts: MutNullableDom<FontFaceSet>, //!< Set of fonts loaded by the document.
    /// <https://html.spec.whatwg.org/multipage/#visibility-state>
    visibility_state: Cell<DocumentVisibilityState>, //!< The visibility state of the document.
    /// <https://www.iana.org/assignments/http-status-codes/http-status-codes.xhtml>
    status_code: Option<u16>, //!< The HTTP status code of the document.
    /// <https://html.spec.whatwg.org/multipage/#is-initial-about:blank>
    is_initial_about_blank: Cell<bool>, //!< Flag indicating if the document is an initial `about:blank` page.
    /// <https://dom.spec.whatwg.org/#document-allow-declarative-shadow-roots>
    allow_declarative_shadow_roots: Cell<bool>, //!< Flag indicating if declarative shadow roots are allowed.
    /// <https://w3c.github.io/webappsec-upgrade-insecure-requests/#insecure-requests-policy>
    #[no_trace]
    inherited_insecure_requests_policy: Cell<Option<InsecureRequestsPolicy>>, //!< The inherited insecure requests policy.
    //// <https://w3c.github.io/webappsec-mixed-content/#categorize-settings-object>
    has_trustworthy_ancestor_origin: Cell<bool>, //!< Flag indicating if the document has a trustworthy ancestor origin.
    /// <https://w3c.github.io/IntersectionObserver/#document-intersectionobservertaskqueued>
    intersection_observer_task_queued: Cell<bool>, //!< Flag indicating if an Intersection Observer task is queued.
    /// Active intersection observers that should be processed by this document in
    /// the update intersection observation steps.
    /// <https://w3c.github.io/IntersectionObserver/#run-the-update-intersection-observations-steps>
    /// > Let observer list be a list of all IntersectionObservers whose root is in the DOM tree of document.
    /// > For the top-level browsing context, this includes implicit root observers.
    ///
    /// Details of which document that should process an observers is discussed further at
    /// <https://github.com/w3c/IntersectionObserver/issues/525>.
    ///
    /// The lifetime of an intersection observer is specified at
    /// <https://github.com/w3c/IntersectionObserver/issues/525>.
    intersection_observers: DomRefCell<Vec<Dom<IntersectionObserver>>>, //!< List of active Intersection Observers.
    /// The active keyboard modifiers for the WebView. This is updated when receiving any input event.
    #[no_trace]
    active_keyboard_modifiers: Cell<Modifiers>, //!< Currently active keyboard modifiers.
}

#[allow(non_snake_case)]
impl Document {
    /// @brief Marks a node as having dirty descendants, triggering a potential restyle or reflow.
    /// Functional Utility: Propagates a "dirty" state up the DOM tree to ensure that layout
    /// and rendering accurately reflect changes in the subtree.
    ///
    /// @param node The node whose descendants have changed.
    pub(crate) fn note_node_with_dirty_descendants(&self, node: &Node) {
        debug_assert!(*node.owner_doc() == *self); // Ensure the node belongs to this document.
        if !node.is_connected() {
            return; // If the node is not connected to the document tree, no action is needed.
        }

        let parent = match node.parent_in_flat_tree() {
            Some(parent) => parent,
            None => {
                // Block Logic: If the node has no parent, it means it's likely the Document node itself.
                // In this case, we treat the Document Element as the dirty root.
                let document_element = match self.GetDocumentElement() {
                    Some(element) => element,
                    None => return, // If no document element, nothing to do.
                };
                if let Some(dirty_root) = self.dirty_root.get() {
                    // Block Logic: If there's an existing dirty root, mark its ancestors up to
                    // the document element as having dirty descendants.
                    for ancestor in dirty_root
                        .upcast::<Node>()
                        .inclusive_ancestors_in_flat_tree()
                    {
                        if ancestor.is::<Element>() {
                            ancestor.set_flag(NodeFlags::HAS_DIRTY_DESCENDANTS, true);
                        }
                    }
                }
                self.dirty_root.set(Some(&document_element)); // Set the document element as the new dirty root.
                return;
            },
        };

        // Block Logic: Check if the parent is an element and should prevent dirty propagation.
        if parent.is::<Element>() {
            if !parent.is_styled() {
                return; // If the parent element is not styled, it won't affect layout.
            }

            if parent.is_display_none() {
                return; // If the parent element has display: none, its children won't be rendered.
            }
        }

        let element_parent: DomRoot<Element>;
        let element = match node.downcast::<Element>() {
            Some(element) => element,
            None => {
                // Block Logic: If the current node is not an element (e.g., text node),
                // try to get its closest element parent.
                match DomRoot::downcast::<Element>(parent) {
                    Some(parent) => {
                        element_parent = parent;
                        &element_parent
                    },
                    None => {
                        // Parent is not an element either (e.g., DocumentFragment), so no element to mark dirty.
                        return;
                    },
                }
            },
        };

        let dirty_root = match self.dirty_root.get() {
            None => {
                // Block Logic: If there's no existing dirty root, set this element as having
                // dirty descendants and make it the new dirty root.
                element
                    .upcast::<Node>()
                    .set_flag(NodeFlags::HAS_DIRTY_DESCENDANTS, true);
                self.dirty_root.set(Some(element));
                return;
            },
            Some(root) => root,
        };

        // Block Logic: Propagate the HAS_DIRTY_DESCENDANTS flag up from the 'element'
        // until an ancestor that already has the flag is found.
        for ancestor in element.upcast::<Node>().inclusive_ancestors_in_flat_tree() {
            if ancestor.get_flag(NodeFlags::HAS_DIRTY_DESCENDANTS) {
                return; // Stop if an ancestor is already marked as dirty.
            }

            if ancestor.is::<Element>() {
                ancestor.set_flag(NodeFlags::HAS_DIRTY_DESCENDANTS, true);
            }
        }

        // Block Logic: Reconcile the new dirty root with the old one to find a common ancestor.
        let new_dirty_root = element
            .upcast::<Node>()
            .common_ancestor_in_flat_tree(dirty_root.upcast())
            .expect("Couldn't find common ancestor"); // There should always be a common ancestor (at least the Document).

        // Block Logic: Reset the HAS_DIRTY_DESCENDANTS flag for nodes between the old dirty root
        // and the new common ancestor, as the common ancestor now covers the dirty region.
        let mut has_dirty_descendants = true;
        for ancestor in dirty_root
            .upcast::<Node>()
            .inclusive_ancestors_in_flat_tree()
        {
            ancestor.set_flag(NodeFlags::HAS_DIRTY_DESCENDANTS, has_dirty_descendants);
            has_dirty_descendants &= *ancestor != *new_dirty_root;
        }

        // Block Logic: Set the dirty root to the element or its shadow host if applicable.
        let maybe_shadow_host = new_dirty_root
            .downcast::<ShadowRoot>()
            .map(ShadowRootMethods::Host);
        let new_dirty_root_element = new_dirty_root
            .downcast::<Element>()
            .or(maybe_shadow_host.as_deref());

        self.dirty_root.set(new_dirty_root_element);
    }

    /// @brief Takes (consumes) the currently set dirty root element.
    /// Functional Utility: Allows the layout engine to retrieve the root of the subtree
    /// that needs relayout/restyle and then clears the `dirty_root` field.
    /// @return An `Option<DomRoot<Element>>` containing the dirty root element if set, otherwise `None`.
    pub(crate) fn take_dirty_root(&self) -> Option<DomRoot<Element>> {
        self.dirty_root.take()
    }

    /// @brief Borrows an immutable reference to the DocumentLoader.
    /// @return A `Ref` to the `DocumentLoader`.
    #[inline]
    pub(crate) fn loader(&self) -> Ref<DocumentLoader> {
        self.loader.borrow()
    }

    /// @brief Borrows a mutable reference to the DocumentLoader.
    /// @return A `RefMut` to the `DocumentLoader`.
    #[inline]
    pub(crate) fn loader_mut(&self) -> RefMut<DocumentLoader> {
        self.loader.borrow_mut()
    }

    /// @brief Checks if the document has an associated browsing context.
    /// @return `true` if a browsing context exists, `false` otherwise.
    #[inline]
    pub(crate) fn has_browsing_context(&self) -> bool {
        self.has_browsing_context
    }

    /// @brief Returns the browsing context associated with this document.
    /// Functional Utility: Provides access to the `WindowProxy` for the browsing context,
    /// enabling interaction with the browser's UI and navigation.
    /// <https://html.spec.whatwg.org/multipage/#concept-document-bc>
    /// @return An `Option<DomRoot<WindowProxy>>` containing the browsing context or `None` if not available.
    #[inline]
    pub(crate) fn browsing_context(&self) -> Option<DomRoot<WindowProxy>> {
        if self.has_browsing_context {
            self.window.undiscarded_window_proxy()
        } else {
            None
        }
    }

    /// @brief Returns the WebView ID associated with this document's window.
    /// @return The `WebViewId`.
    pub(crate) fn webview_id(&self) -> WebViewId {
        self.window.webview_id()
    }

    /// @brief Returns an immutable reference to the Window object this document belongs to.
    /// @return A reference to the `Window`.
    #[inline]
    pub(crate) fn window(&self) -> &Window {
        &self.window
    }

    /// @brief Checks if the document is an HTML document.
    /// @return `true` if it's an HTML document, `false` otherwise.
    #[inline]
    pub(crate) fn is_html_document(&self) -> bool {
        self.is_html_document
    }

    /// @brief Checks if the document is an XHTML document.
    /// Functional Utility: Determines if the document's content type explicitly indicates XHTML,
    /// which affects parsing and rendering rules.
    /// @return `true` if it's an XHTML document, `false` otherwise.
    pub(crate) fn is_xhtml_document(&self) -> bool {
        self.content_type.type_() == mime::APPLICATION &&
            self.content_type.subtype().as_str() == "xhtml" &&
            self.content_type.suffix() == Some(mime::XML)
    }

    /// @brief Sets the HTTPS state of the document.
    /// Functional Utility: Updates the security indicator for the document, influencing
    /// how mixed content and other security-related features are handled.
    /// @param https_state The new `HttpsState` to set.
    pub(crate) fn set_https_state(&self, https_state: HttpsState) {
        self.https_state.set(https_state);
    }

    /// @brief Checks if the document is fully active.
    /// Functional Utility: Determines if the document is visible, has focus, and is able to run scripts
    /// and render animations.
    /// @return `true` if fully active, `false` otherwise.
    pub(crate) fn is_fully_active(&self) -> bool {
        self.activity.get() == DocumentActivity::FullyActive
    }

    /// @brief Checks if the document is active (not inactive).
    /// @return `true` if active, `false` if inactive.
    pub(crate) fn is_active(&self) -> bool {
        self.activity.get() != DocumentActivity::Inactive
    }

    /// @brief Sets the activity state of the document.
    /// Functional Utility: Manages the lifecycle of the document, including suspending/resuming
    /// timers, media, and re-evaluating layout based on its visibility and interaction state.
    ///
    /// @param activity The new `DocumentActivity` state.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn set_activity(&self, activity: DocumentActivity, can_gc: CanGc) {
        // This function should only be called on documents with a browsing context
        assert!(self.has_browsing_context);
        if activity == self.activity.get() {
            return; // No change in activity, so do nothing.
        }

        // Set the document's activity level, reflow if necessary, and suspend or resume timers.
        self.activity.set(activity); // Update the activity state.
        let media = ServoMedia::get(); // Get the ServoMedia instance.
        let pipeline_id = self.window().pipeline_id(); // Get the pipeline ID of the window.
        let client_context_id =
            ClientContextId::build(pipeline_id.namespace_id.0, pipeline_id.index.0.get()); // Build client context ID.

        // Block Logic: Suspend timers and media if the document becomes not fully active.
        if activity != DocumentActivity::FullyActive {
            self.window().suspend(can_gc); // Suspend window timers and tasks.
            media.suspend(&client_context_id); // Suspend media playback.
            return;
        }

        // Block Logic: Resume activity when the document becomes fully active.
        self.title_changed(); // Notify about title change.
        self.dirty_all_nodes(); // Mark all nodes as dirty to trigger re-rendering.
        self.window().resume(can_gc); // Resume window timers and tasks.
        media.resume(&client_context_id); // Resume media playback.

        // Block Logic: If the document is not yet complete, return early.
        if self.ready_state.get() != DocumentReadyState::Complete {
            return;
        }

        // Block Logic: Queue a task to fire a `pageshow` event if conditions are met.
        // This handles reactivating a document after it becomes fully active.
        // TODO: See #32687 for more information.
        let document = Trusted::new(self); // Create a trusted reference to the document.
        self.owner_global()
            .task_manager()
            .dom_manipulation_task_source()
            .queue(task!(fire_pageshow_event: move || { // Queue a task for page show event.
                let document = document.root(); // Get the document root.
                let window = document.window(); // Get the window.
                // Step 4.6.1: If document's page showing flag is true, return.
                if document.page_showing.get() {
                    return;
                }
                // Step 4.6.2 Set document's page showing flag to true.
                document.page_showing.set(true);
                // Step 4.6.3 Update the visibility state of document to "visible".
                document.update_visibility_state(DocumentVisibilityState::Visible, CanGc::note());
                // Step 4.6.4 Fire a page transition event named pageshow at document's relevant
                // global object with true.
                let event = PageTransitionEvent::new(
                    window,
                    atom!("pageshow"),
                    false, // bubbles
                    false, // cancelable
                    true, // persisted
                    CanGc::note(),
                );
                let event = event.upcast::<Event>(); // Upcast to a generic Event.
                event.set_trusted(true); // Mark as trusted event.
                window.dispatch_event_with_target_override(event, CanGc::note()); // Dispatch the event.
            }))
    }

    /// @brief Returns the mutable origin of the document.
    /// @return A reference to `MutableOrigin`.
    pub(crate) fn origin(&self) -> &MutableOrigin {
        &self.origin
    }

    /// @brief Returns the current URL of the document.
    /// <https://dom.spec.whatwg.org/#concept-document-url>
    /// @return A cloned `ServoUrl` representing the document's URL.
    pub(crate) fn url(&self) -> ServoUrl {
        self.url.borrow().clone()
    }

    /// @brief Sets the URL of the document.
    /// @param url The new `ServoUrl` to set.
    pub(crate) fn set_url(&self, url: ServoUrl) {
        *self.url.borrow_mut() = url;
    }

    /// @brief Determines the fallback base URL for the document.
    /// Functional Utility: Provides a base URL for resolving relative URLs, considering
    /// `iframe srcdoc` and `about:blank` special cases.
    /// <https://html.spec.whatwg.org/multipage/#fallback-base-url>
    /// @return The computed fallback base URL.
    pub(crate) fn fallback_base_url(&self) -> ServoUrl {
        let document_url = self.url(); // Get the document's current URL.
        if let Some(browsing_context) = self.browsing_context() {
            // Step 1: If document is an iframe srcdoc document, then return the
            // document base URL of document's browsing context's container document.
            let container_base_url = browsing_context
                .parent()
                .and_then(|parent| parent.document())
                .map(|document| document.base_url());
            if document_url.as_str() == "about:srcdoc" {
                if let Some(base_url) = container_base_url {
                    return base_url;
                }
            }
            // Step 2: If document's URL is about:blank, and document's browsing
            // context's creator base URL is non-null, then return that creator base URL.
            if document_url.as_str() == "about:blank" && browsing_context.has_creator_base_url() {
                return browsing_context.creator_base_url().unwrap();
            }
        }
        // Step 3: Return document's URL.
        document_url
    }

    /// @brief Determines the base URL for the document.
    /// Functional Utility: Resolves the base URL for relative URL resolution,
    /// prioritizing a `<base>` element's `href` attribute over the document's URL.
    /// <https://html.spec.whatwg.org/multipage/#document-base-url>
    /// @return The computed base URL.
    pub(crate) fn base_url(&self) -> ServoUrl {
        match self.base_element() {
            // Step 1. If there is no base element, use the fallback base URL.
            None => self.fallback_base_url(),
            // Step 2. If a base element exists, use its frozen base URL.
            Some(base) => base.frozen_base_url(),
        }
    }

    /// @brief Sets the `needs_paint` flag for the document.
    /// Functional Utility: Informs the rendering engine that the document needs
    /// to be repainted during the next rendering cycle.
    /// @param value The boolean value to set the flag to.
    pub(crate) fn set_needs_paint(&self, value: bool) {
        self.needs_paint.set(value)
    }

    /// @brief Determines if the document needs a reflow (re-layout).
    /// Functional Utility: Checks various conditions (stylesheet changes, dirty DOM, pending restyles,
    /// postponed paint) to decide if a reflow is necessary.
    /// @return An `Option<ReflowTriggerCondition>` indicating why a reflow is needed, or `None` if not.
    pub(crate) fn needs_reflow(&self) -> Option<ReflowTriggerCondition> {
        // FIXME: This should check the dirty bit on the document,
        // not the document element. Needs some layout changes to make
        // that workable.
        if self.stylesheets.borrow().has_changed() {
            return Some(ReflowTriggerCondition::StylesheetsChanged);
        }

        let root = self.GetDocumentElement()?; // Get the document element.
        if root.upcast::<Node>().has_dirty_descendants() {
            return Some(ReflowTriggerCondition::DirtyDescendants);
        }

        if !self.pending_restyles.borrow().is_empty() {
            return Some(ReflowTriggerCondition::PendingRestyles);
        }

        if self.needs_paint.get() {
            return Some(ReflowTriggerCondition::PaintPostponed);
        }

        None
    }

    /// @brief Returns the first `base` element in the DOM that has an `href` attribute.
    /// Functional Utility: Provides the authoritative `<base>` element for the document,
    /// which affects how relative URLs are resolved.
    /// @return An `Option<DomRoot<HTMLBaseElement>>` containing the base element if found, otherwise `None`.
    pub(crate) fn base_element(&self) -> Option<DomRoot<HTMLBaseElement>> {
        self.base_element.get()
    }

    /// @brief Refreshes the cached first `base` element in the DOM.
    /// Functional Utility: Scans the document for the first `<base>` element with an `href`
    /// attribute and updates the internal cache. This is necessary when `base` elements
    /// are added, removed, or modified.
    /// <https://github.com/w3c/web-platform-tests/issues/2122>
    pub(crate) fn refresh_base_element(&self) {
        let base = self
            .upcast::<Node>()
            .traverse_preorder(ShadowIncluding::No) // Traverse all nodes in preorder, excluding shadow DOM.
            .filter_map(DomRoot::downcast::<HTMLBaseElement>) // Filter for HTMLBaseElement nodes.
            .find(|element| {
                element
                    .upcast::<Element>()
                    .has_attribute(&local_name!("href")) // Find the first one with an 'href' attribute.
            });
        self.base_element.set(base.as_deref()); // Update the cached base element.
    }

    /// @brief Returns the current count of DOM nodes in the document's tree.
    /// Functional Utility: Provides a metric used by the layout engine to decide
    /// between sequential and parallel layout strategies.
    /// @return The total number of DOM nodes.
    pub(crate) fn dom_count(&self) -> u32 {
        self.dom_count.get()
    }

    /// @brief Increments the internal count of DOM nodes.
    /// Functional Utility: Called when a new node is added to the DOM tree,
    /// updating the `dom_count` for layout optimization.
    /// This is called by `bind_to_tree` when a node is added to the DOM.
    /// The internal count is used by layout to determine whether to be sequential or parallel.
    /// (it's sequential for small DOMs)
    ///
    /// See also: <https://github.com/servo/servo/issues/10110>
    pub(crate) fn increment_dom_count(&self) {
        self.dom_count.set(self.dom_count.get() + 1);
    }

    /// @brief Decrements the internal count of DOM nodes.
    /// Functional Utility: Called when a node is removed from the DOM tree,
    /// updating the `dom_count`.
    /// This is called by `unbind_from_tree` when a node is removed from the DOM.
    pub(crate) fn decrement_dom_count(&self) {
        self.dom_count.set(self.dom_count.get() - 1);
    }

    /// @brief Returns the quirks mode of the document.
    /// @return The `QuirksMode` of the document.
    pub(crate) fn quirks_mode(&self) -> QuirksMode {
        self.quirks_mode.get()
    }

    /// @brief Sets the quirks mode of the document.
    /// Functional Utility: Updates the document's quirks mode and propagates
    /// this change to the layout engine, affecting how CSS and rendering rules are applied.
    /// @param new_mode The new `QuirksMode` to set.
    pub(crate) fn set_quirks_mode(&self, new_mode: QuirksMode) {
        let old_mode = self.quirks_mode.replace(new_mode); // Atomically set new mode and get old.

        if old_mode != new_mode {
            self.window.layout_mut().set_quirks_mode(new_mode); // Notify layout if mode changed.
        }
    }

    /// @brief Returns the character encoding of the document.
    /// @return A static reference to the `Encoding`.
    pub(crate) fn encoding(&self) -> &'static Encoding {
        self.encoding.get()
    }

    /// @brief Sets the character encoding of the document.
    /// @param encoding The new static reference to the `Encoding`.
    pub(crate) fn set_encoding(&self, encoding: &'static Encoding) {
        self.encoding.set(encoding);
    }

    /// @brief Notifies the document that a node's content or heritage has changed.
    /// Functional Utility: Triggers a dirty state for the node and its ancestors
    /// if the node is connected to the document, ensuring re-rendering.
    /// @param node The node whose content or heritage has changed.
    pub(crate) fn content_and_heritage_changed(&self, node: &Node) {
        if node.is_connected() {
            node.note_dirty_descendants(); // Mark descendants as dirty if connected.
        }

        // FIXME(emilio): This is very inefficient, ideally the flag above would
        // be enough and incremental layout could figure out from there.
        node.dirty(NodeDamage::OtherNodeDamage); // Mark the node itself as dirty.
    }

    /// @brief Unregisters an element from the document's ID map.
    /// Functional Utility: Removes the association between an element and a given ID,
    /// and triggers a reset for form owners listening to changes for that ID.
    ///
    /// @param to_unregister The `Element` to unregister.
    /// @param id The `Atom` representing the ID.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn unregister_element_id(&self, to_unregister: &Element, id: Atom, can_gc: CanGc) {
        self.document_or_shadow_root
            .unregister_named_element(&self.id_map, to_unregister, &id);
        self.reset_form_owner_for_listeners(&id, can_gc); // Reset form owner for listeners.
    }

    /// @brief Registers an element with the document's ID map.
    /// Functional Utility: Associates an element with a given ID, enabling it to be
    /// retrieved by methods like `getElementById`. Also triggers a reset for form
    /// owners listening to changes for that ID.
    ///
    /// @param element The `Element` to register.
    /// @param id The `Atom` representing the ID.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn register_element_id(&self, element: &Element, id: Atom, can_gc: CanGc) {
        let root = self.GetDocumentElement().expect(
            "The element is in the document, so there must be a document \ 
             element."
        );
        self.document_or_shadow_root.register_named_element(
            &self.id_map,
            element,
            &id,
            DomRoot::from_ref(root.upcast::<Node>()),
        );
        self.reset_form_owner_for_listeners(&id, can_gc); // Reset form owner for listeners.
    }

    /// @brief Unregisters an element from the document's name map.
    /// Functional Utility: Removes the association between an element and a given name.
    ///
    /// @param to_unregister The `Element` to unregister.
    /// @param name The `Atom` representing the name.
    pub(crate) fn unregister_element_name(&self, to_unregister: &Element, name: Atom) {
        self.document_or_shadow_root
            .unregister_named_element(&self.name_map, to_unregister, &name);
    }

    /// @brief Registers an element with the document's name map.
    /// Functional Utility: Associates an element with a given name.
    ///
    /// @param element The `Element` to register.
    /// @param name The `Atom` representing the name.
    pub(crate) fn register_element_name(&self, element: &Element, name: Atom) {
        let root = self.GetDocumentElement().expect(
            "The element is in the document, so there must be a document \ 
             element."
        );
        self.document_or_shadow_root.register_named_element(
            &self.name_map,
            element,
            &name,
            DomRoot::from_ref(root.upcast::<Node>()),
        );
    }

    /// @brief Registers a form control element as a listener for changes to a specific form ID.
    /// Functional Utility: Allows form controls to be dynamically associated with forms
    /// identified by their `id` attribute, facilitating form submission and reset behavior.
    ///
    /// @param id The `DOMString` representing the form ID to listen for.
    /// @param listener A reference to the `FormControl` element.
    pub(crate) fn register_form_id_listener<T: ?Sized + FormControl>(
        &self,
        id: DOMString,
        listener: &T,
    ) {
        let mut map = self.form_id_listener_map.borrow_mut(); // Get mutable reference to the map.
        let listener = listener.to_element(); // Get the underlying Element from the form control.
        let set = map.entry(Atom::from(id)).or_default(); // Get or create a HashSet for the given ID.
        set.insert(Dom::from_ref(listener)); // Insert the form control element into the set.
    }

    /// @brief Unregisters a form control element from listening to changes for a specific form ID.
    /// Functional Utility: Removes an association between a form control and a form ID.
    ///
    /// @param id The `DOMString` representing the form ID.
    /// @param listener A reference to the `FormControl` element to unregister.
    pub(crate) fn unregister_form_id_listener<T: ?Sized + FormControl>(
        &self,
        id: DOMString,
        listener: &T,
    ) {
        let mut map = self.form_id_listener_map.borrow_mut(); // Get mutable reference to the map.
        // Block Logic: If an entry for the ID exists, remove the listener. If the set becomes empty, remove the entry.
        if let Occupied(mut entry) = map.entry(Atom::from(id)) {
            entry
                .get_mut()
                .remove(&Dom::from_ref(listener.to_element())); // Remove the listener.
            if entry.get().is_empty() {
                entry.remove(); // Remove the entry if the set is empty.
            }
        }
    }

    /// @brief Attempts to find a named element in this document based on a fragment identifier.
    /// Functional Utility: Resolves a URL fragment identifier to a specific DOM element,
    /// first by ID and then by anchor name.
    /// <https://html.spec.whatwg.org/multipage/#the-indicated-part-of-the-document>
    ///
    /// @param fragid The fragment identifier string.
    /// @return An `Option<DomRoot<Element>>` containing the found element, or `None`.
    pub(crate) fn find_fragment_node(&self, fragid: &str) -> Option<DomRoot<Element>> {
        // Step 1 is not handled here; the fragid is already obtained by the calling function
        // Step 2: Simply use None to indicate the top of the document.
        // Step 3 & 4: Decode the percent-encoded fragment ID.
        percent_decode(fragid.as_bytes())
            .decode_utf8()
            .ok()
            // Step 5: Try to get an element by its ID.
            .and_then(|decoded_fragid| self.get_element_by_id(&Atom::from(decoded_fragid)))
            // Step 6: If not found by ID, try to get an anchor by its name.
            .or_else(|| self.get_anchor_by_name(fragid))
        // Step 7 & 8: Further steps in the spec not implemented here.
    }

    /// @brief Checks and scrolls the viewport to a fragment identifier.
    /// Functional Utility: Scrolls the document to the element identified by the fragment,
    /// or to the top of the document if the fragment is empty or "top".
    /// <https://html.spec.whatwg.org/multipage/#scroll-to-the-fragment-identifier>
    ///
    /// @param fragment The fragment identifier string.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn check_and_scroll_fragment(&self, fragment: &str, can_gc: CanGc) {
        let target = self.find_fragment_node(fragment); // Find the target element.

        // Step 1: Set the target element in the document.
        self.set_target_element(target.as_deref());

        // Block Logic: Determine the scroll point based on the target element or fragment.
        let point = target
            .as_ref()
            .map(|element| {
                // TODO: This strategy is completely wrong if the element we are scrolling to in
                // inside other scrollable containers. Ideally this should use an implementation of
                // `scrollIntoView` when that is available:
                // See https://github.com/servo/servo/issues/24059.
                let rect = element
                    .upcast::<Node>()
                    .bounding_content_box_or_zero(can_gc); // Get bounding box of the element.

                // In order to align with element edges, we snap to unscaled pixel boundaries, since
                // the paint thread currently does the same for drawing elements. This is important
                // for pages that require pixel perfect scroll positioning for proper display
                // (like Acid2).
                let device_pixel_ratio = self.window.device_pixel_ratio().get(); // Get device pixel ratio.
                (
                    rect.origin.x.to_nearest_pixel(device_pixel_ratio), // X-coordinate snapped to nearest pixel.
                    rect.origin.y.to_nearest_pixel(device_pixel_ratio), // Y-coordinate snapped to nearest pixel.
                )
            })
            .or_else(|| {
                // Block Logic: If no target element and fragment is empty or "top", scroll to (0,0).
                if fragment.is_empty() || fragment.eq_ignore_ascii_case("top") {
                    // FIXME(stshine): this should be the origin of the stacking context space,
                    // which may differ under the influence of writing mode.
                    Some((0.0, 0.0))
                } else {
                    None // No valid scroll target.
                }
            });

        // Block Logic: Perform the scroll if a point was determined.
        if let Some((x, y)) = point {
            self.window
                .scroll(x as f64, y as f64, ScrollBehavior::Instant, can_gc) // Scroll the window.
        }
    }

    /// @brief Retrieves an anchor element by its `name` attribute.
    /// Functional Utility: Provides a way to find specific anchor points within the document
    /// for navigation or linking.
    /// @param name The `name` attribute value to search for.
    /// @return An `Option<DomRoot<Element>>` containing the found anchor element, or `None`.
    fn get_anchor_by_name(&self, name: &str) -> Option<DomRoot<Element>> {
        let name = Atom::from(name); // Convert name to an Atom.
        self.name_map.borrow().get(&name).and_then(|elements| {
            // Block Logic: Iterate through elements with the given name and find the first HTMLAnchorElement.
            elements
                .iter()
                .find(|e| e.is::<HTMLAnchorElement>())
                .map(|e| DomRoot::from_ref(&**e))
        })
    }

    /// @brief Sets the ready state of the document.
    /// Functional Utility: Manages the document's loading progression, dispatches `readystatechange`
    /// events, and updates performance timing metrics.
    /// <https://html.spec.whatwg.org/multipage/#current-document-readiness>
    ///
    /// @param state The new `DocumentReadyState` to set.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn set_ready_state(&self, state: DocumentReadyState, can_gc: CanGc) {
        // Block Logic: Handle state-specific actions and update performance timings.
        match state {
            DocumentReadyState::Loading => {
                if self.window().is_top_level() {
                    self.send_to_embedder(EmbedderMsg::NotifyLoadStatusChanged(
                        self.webview_id(),
                        LoadStatus::Started,
                    ));
                    self.send_to_embedder(EmbedderMsg::Status(self.webview_id(), None));
                }
            },
            DocumentReadyState::Complete => {
                if self.window().is_top_level() {
                    self.send_to_embedder(EmbedderMsg::NotifyLoadStatusChanged(
                        self.webview_id(),
                        LoadStatus::Complete,
                    ));
                }
                update_with_current_instant(&self.dom_complete); // Update `domComplete` timestamp.
            },
            DocumentReadyState::Interactive => update_with_current_instant(&self.dom_interactive), // Update `domInteractive` timestamp.
        };

        self.ready_state.set(state); // Set the new ready state.

        // Functional Utility: Fire a `readystatechange` event.
        self.upcast::<EventTarget>()
            .fire_event(atom!("readystatechange"), can_gc);
    }

    /// @brief Returns whether scripting is enabled for this document.
    /// @return `true` if scripting is enabled, `false` otherwise.
    pub(crate) fn is_scripting_enabled(&self) -> bool {
        self.scripting_enabled
    }

    /// @brief Returns the element that currently has focus within the document.
    /// <https://w3c.github.io/uievents/#events-focusevent-doc-focus>
    /// @return An `Option<DomRoot<Element>>` containing the focused element, or `None`.
    pub(crate) fn get_focused_element(&self) -> Option<DomRoot<Element>> {
        self.focused.get()
    }

    /// @brief Initiates a new round of checking for elements requesting focus.
    /// Functional Utility: Prepares the document for a focus change operation,
    /// ensuring that only the last element requesting focus during the transaction
    /// will ultimately receive it.
    fn begin_focus_transaction(&self) {
        *self.focus_transaction.borrow_mut() = FocusTransaction::InTransaction(Default::default()); // Set state to `InTransaction`.
    }

    /// @brief Performs the focus fixup rule according to the HTML specification.
    /// Functional Utility: Adjusts focus if the currently focused element becomes
    /// non-focusable, typically by moving focus to the document body.
    /// <https://html.spec.whatwg.org/multipage/#focus-fixup-rule>
    ///
    /// @param not_focusable The element that has become non-focusable.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn perform_focus_fixup_rule(&self, not_focusable: &Element, can_gc: CanGc) {
        if Some(not_focusable) != self.focused.get().as_deref() {
            return; // If the non_focusable element is not the currently focused one, do nothing.
        }
        // Block Logic: Request focus for the document body.
        self.request_focus(
            self.GetBody().as_ref().map(|e| e.upcast()), // Attempt to get the body element.
            FocusType::Element, // Focus type is element.
            can_gc,
        )
    }

    /// @brief Requests that a given element receive focus.
    /// Functional Utility: Manages the focus state of the document, allowing elements
    /// to programmatically request focus, either implicitly or explicitly within a transaction.
    ///
    /// @param elem An `Option<&Element>` indicating the element to focus, or `None` to unfocus.
    /// @param focus_type The `FocusType` of the request (e.g., element, mouse).
    /// @param can_gc A `CanGc` token.
    pub(crate) fn request_focus(
        &self,
        elem: Option<&Element>,
        focus_type: FocusType,
        can_gc: CanGc,
    ) {
        // Block Logic: Determine if an implicit focus transaction should be started.
        let implicit_transaction = matches!(
            *self.focus_transaction.borrow(),
            FocusTransaction::NotInTransaction
        );
        if implicit_transaction {
            self.begin_focus_transaction(); // Start a new focus transaction.
        }
        // Block Logic: If the element is focusable, update the focus transaction.
        // Invariant: If `elem` is `None` or refers to a focusable area, it updates `focus_transaction`.
        if elem.is_none_or(|e| e.is_focusable_area()) {
            *self.focus_transaction.borrow_mut() =
                FocusTransaction::InTransaction(elem.map(Dom::from_ref)); // Store the requested element.
        }
        if implicit_transaction {
            self.commit_focus_transaction(focus_type, can_gc); // Commit the focus transaction if it was implicit.
        }
    }

    /// @brief Commits a focus transaction, applying the requested focus change.
    /// Functional Utility: Finalizes a focus operation, dispatches blur/focus events,
    /// and interacts with the embedder for IME (Input Method Editor) management.
    ///
    /// @param focus_type The `FocusType` of the request.
    /// @param can_gc A `CanGc` token.
    fn commit_focus_transaction(&self, focus_type: FocusType, can_gc: CanGc) {
        // Block Logic: Retrieve the element that was requested to be focused during the transaction.
        let possibly_focused = match *self.focus_transaction.borrow() {
            FocusTransaction::NotInTransaction => unreachable!(), // Should not be in this state.
            FocusTransaction::InTransaction(ref elem) => {
                elem.as_ref().map(|e| DomRoot::from_ref(&**e))
            },
        };
        *self.focus_transaction.borrow_mut() = FocusTransaction::NotInTransaction; // Reset transaction state.

        // Block Logic: If the focus hasn't changed, return early.
        if self.focused == possibly_focused.as_deref() {
            return;
        }
        // Block Logic: Handle blur event for the previously focused element.
        if let Some(ref elem) = self.focused.get() {
            let node = elem.upcast::<Node>(); // Get node from element.
            elem.set_focus_state(false); // Set focus state to false.
            // FIXME: pass appropriate relatedTarget
            self.fire_focus_event(FocusEventType::Blur, node, None, can_gc); // Fire blur event.

            // Notify the embedder to hide the input method.
            if elem.input_method_type().is_some() {
                self.send_to_embedder(EmbedderMsg::HideIME(self.webview_id()));
            }
        }

        self.focused.set(possibly_focused.as_deref()); // Update the focused element.

        // Block Logic: Handle focus event for the newly focused element.
        if let Some(ref elem) = self.focused.get() {
            elem.set_focus_state(true); // Set focus state to true.
            let node = elem.upcast::<Node>(); // Get node from element.
            // FIXME: pass appropriate relatedTarget
            self.fire_focus_event(FocusEventType::Focus, node, None, can_gc); // Fire focus event.
            // Update the focus state for all elements in the focus chain.
            // https://html.spec.whatwg.org/multipage/#focus-chain
            if focus_type == FocusType::Element {
                self.window()
                    .send_to_constellation(ScriptToConstellationMessage::Focus);
            }

            // Block Logic: Notify the embedder to display an input method if applicable.
            if let Some(kind) = elem.input_method_type() {
                let rect = elem.upcast::<Node>().bounding_content_box_or_zero(can_gc); // Get bounding box.
                let rect = Rect::new(
                    Point2D::new(rect.origin.x.to_px(), rect.origin.y.to_px()),
                    Size2D::new(rect.size.width.to_px(), rect.size.height.to_px()),
                );
                let (text, multiline) = if let Some(input) = elem.downcast::<HTMLInputElement>() {
                    // For HTMLInputElement.
                    (
                        Some((
                            input.Value().to_string(),
                            input.GetSelectionEnd().unwrap_or(0) as i32,
                        )),
                        false,
                    )
                } else if let Some(textarea) = elem.downcast::<HTMLTextAreaElement>() {
                    // For HTMLTextAreaElement.
                    (
                        Some((
                            textarea.Value().to_string(),
                            textarea.GetSelectionEnd().unwrap_or(0) as i32,
                        )),
                        true,
                    )
                } else {
                    // Other elements.
                    (None, false)
                };
                self.send_to_embedder(EmbedderMsg::ShowIME(
                    self.webview_id(),
                    kind,
                    text,
                    multiline,
                    DeviceIntRect::from_untyped(&rect.to_box2d()), // Send IME info to embedder.
                ));
            }
        }
    }

    /// @brief Handles any updates when the document's title has changed.
    /// Functional Utility: Propagates title changes to the embedder and constellation
    /// for UI updates and devtools integration.
    pub(crate) fn title_changed(&self) {
        if self.browsing_context().is_some() {
            self.send_title_to_embedder(); // Send title to embedder.
            let title = String::from(self.Title()); // Get the document title.
            self.window
                .send_to_constellation(ScriptToConstellationMessage::TitleChanged(
                    self.window.pipeline_id(),
                    title.clone(),
                )); // Send title to constellation.
            if let Some(chan) = self.window.as_global_scope().devtools_chan() {
                let _ = chan.send(ScriptToDevtoolsControlMsg::TitleChanged(
                    self.window.pipeline_id(),
                    title,
                )); // Send title to devtools.
            }
        }
    }

    /// @brief Determines the title of the [`Document`] according to the specification.
    /// Functional Utility: Extracts the document title, either from an `<svg:title>` element
    /// in SVG or the first `<title>` element in HTML.
    /// <https://html.spec.whatwg.org/multipage/#document.title>
    ///
    /// @return An `Option<DOMString>` containing the document title if specified, otherwise `None`.
    fn title(&self) -> Option<DOMString> {
        let title = self.GetDocumentElement().and_then(|root| {
            if root.namespace() == &ns!(svg) && root.local_name() == &local_name!("svg") {
                // Step 1: If root is an SVG element, find <svg:title>.
                root.upcast::<Node>()
                    .child_elements()
                    .find(|node| {
                        node.namespace() == &ns!(svg) && node.local_name() == &local_name!("title")
                    })
                    .map(DomRoot::upcast::<Node>)
            } else {
                // Step 2: For other documents, find the first <title> element.
                root.upcast::<Node>()
                    .traverse_preorder(ShadowIncluding::No)
                    .find(|node| node.is::<HTMLTitleElement>())
            }
        });

        title.map(|title| {
            // Steps 3-4: Get child text content and normalize whitespace.
            let value = title.child_text_content();
            DOMString::from(str_join(split_html_space_chars(&value), " "))
        })
    }

    /// @brief Sends this document's title to the embedder.
    /// Functional Utility: Communicates the document's title to the embedding application
    /// for display in the browser's UI (e.g., tab title).
    pub(crate) fn send_title_to_embedder(&self) {
        let window = self.window(); // Get the window.
        if window.is_top_level() {
            let title = self.title().map(String::from); // Get the title as a String.
            self.send_to_embedder(EmbedderMsg::ChangePageTitle(self.webview_id(), title)); // Send to embedder.
        }
    }

    /// @brief Sends an `EmbedderMsg` to the embedder.
    /// Functional Utility: Provides a generic mechanism for the document to send messages
    /// to the embedding application.
    /// @param msg The `EmbedderMsg` to send.
    pub(crate) fn send_to_embedder(&self, msg: EmbedderMsg) {
        let window = self.window(); // Get the window.
        window.send_to_embedder(msg); // Delegate to window's send_to_embedder.
    }

    /// @brief Marks all nodes in the document as dirty.
    /// Functional Utility: Forces a full re-evaluation of layout and rendering for the entire document,
    /// typically after a significant structural or stylistic change.
    pub(crate) fn dirty_all_nodes(&self) {
        let root = match self.GetDocumentElement() {
            Some(root) => root,
            None => return, // If no document element, there are no nodes to mark dirty.
        };
        // Block Logic: Traverse all nodes in preorder (including shadow DOM) and mark them dirty.
        for node in root
            .upcast::<Node>()
            .traverse_preorder(ShadowIncluding::Yes)
        {
            node.dirty(NodeDamage::OtherNodeDamage) // Apply generic node damage.
        }
    }

    /// @brief Handles a mouse button event.
    /// Functional Utility: Processes mouse click, down, and up events, including hit testing,
    /// focus management, click/double-click detection, and context menu triggering.
    ///
    /// @param event The `MouseButtonEvent` to handle.
    /// @param hit_test_result An `Option<CompositorHitTestResult>` from the compositor.
    /// @param pressed_mouse_buttons A bitmask of currently pressed mouse buttons.
    /// @param can_gc A `CanGc` token.
    #[allow(unsafe_code)]
    pub(crate) fn handle_mouse_button_event(
        &self,
        event: MouseButtonEvent,
        hit_test_result: Option<CompositorHitTestResult>,
        pressed_mouse_buttons: u16,
        can_gc: CanGc,
    ) {
        // Block Logic: Ignore events without a valid hit test result.
        let Some(hit_test_result) = hit_test_result else {
            return;
        };

        debug!(
            "{}: at {:?}",
            event.action,
            hit_test_result.point_in_viewport
        );

        // Block Logic: Convert untrusted node address to a trusted `DomRoot<Node>` and find the target element.
        let node = unsafe { node::from_untrusted_node_address(hit_test_result.node) };
        let Some(el) = node
            .inclusive_ancestors(ShadowIncluding::Yes)
            .filter_map(DomRoot::downcast::<Element>)
            .next()
        else {
            return; // If no element found, return.
        };

        let node = el.upcast::<Node>(); // Upcast the element to a Node.
        debug!("{:?} on {:?}", event.action, node.debug_str());
        // Block Logic: Handle click-specific logic, including disabled element check and focus.
        if let MouseButtonAction::Click = event.action {
            // The click event is filtered by the disabled state.
            if el.is_actually_disabled() {
                return; // Ignore click if element is disabled.
            }

            self.begin_focus_transaction(); // Start a focus transaction.
            self.request_focus(Some(&*el), FocusType::Element, can_gc); // Request focus for the clicked element.
        }

        // Functional Utility: Create a new `MouseEvent` from the platform event.
        let dom_event = DomRoot::upcast::<Event>(MouseEvent::for_platform_mouse_event(
            event,
            pressed_mouse_buttons,
            &self.window,
            &hit_test_result,
            can_gc,
        ));

        // Block Logic: Dispatch events based on mouse button action.
        // https://html.spec.whatwg.org/multipage/#run-authentic-click-activation-steps
        let activatable = el.as_maybe_activatable(); // Get activatable interface for the element.
        match event.action {
            MouseButtonAction::Click => {
                el.set_click_in_progress(true); // Set click in progress flag.
                dom_event.fire(node.upcast(), can_gc); // Fire the click event.
                el.set_click_in_progress(false); // Clear click in progress flag.
            },
            MouseButtonAction::Down => {
                if let Some(a) = activatable {
                    a.enter_formal_activation_state(); // Enter formal activation state.
                }

                let target = node.upcast(); // Get the event target.
                dom_event.fire(target, can_gc); // Fire the mousedown event.
            },
            MouseButtonAction::Up => {
                if let Some(a) = activatable {
                    a.exit_formal_activation_state(); // Exit formal activation state.
                }

                let target = node.upcast(); // Get the event target.
                dom_event.fire(target, can_gc); // Fire the mouseup event.
            },
        }

        // Block Logic: Perform post-click actions if it was a click event.
        if let MouseButtonAction::Click = event.action {
            self.commit_focus_transaction(FocusType::Element, can_gc); // Commit the focus transaction.
            self.maybe_fire_dblclick(
                hit_test_result.point_in_viewport,
                node,
                pressed_mouse_buttons,
                can_gc,
            ); // Check and fire double-click event.
        }

        // Block Logic: Trigger context menu for right-click down events.
        // When the contextmenu event is triggered by right mouse button
        // the contextmenu event MUST be dispatched after the mousedown event.
        if let (MouseButtonAction::Down, MouseButton::Right) = (event.action, event.button) {
            self.maybe_show_context_menu(
                node.upcast(),
                pressed_mouse_buttons,
                hit_test_result.point_in_viewport,
                can_gc,
            ); // Check and show context menu.
        }
    }

    /// @brief Checks conditions and potentially shows a context menu.
    /// Functional Utility: Dispatches a `contextmenu` event and, if not prevented,
    /// notifies the embedder to display the native context menu.
    /// <https://www.w3.org/TR/uievents/#maybe-show-context-menu>
    ///
    /// @param target The `EventTarget` for the context menu event.
    /// @param pressed_mouse_buttons A bitmask of currently pressed mouse buttons.
    /// @param client_point The client coordinates where the event occurred.
    /// @param can_gc A `CanGc` token.
    fn maybe_show_context_menu(
        &self,
        target: &EventTarget,
        pressed_mouse_buttons: u16,
        client_point: Point2D<f32>,
        can_gc: CanGc,
    ) {
        let client_x = client_point.x.to_i32().unwrap_or(0); // Convert x-coordinate to i32.
        let client_y = client_point.y.to_i32().unwrap_or(0); // Convert y-coordinate to i32.

        // Block Logic: Create and fire a `contextmenu` event.
        // <https://w3c.github.io/uievents/#contextmenu>
        let menu_event = PointerEvent::new(
            &self.window,
            DOMString::from("contextmenu"), // Event type string.
            EventBubbles::Bubbles,
            EventCancelable::Cancelable,
            Some(&self.window),             // view
            0,                              // detail
            client_x,                       // screen_x
            client_y,                       // screen_y
            client_x,                       // client_x
            client_y,                       // client_y
            false,                          // ctrl_key
            false,                          // alt_key
            false,                          // shift_key
            false,                          // meta_key
            2i16,                           // button, right mouse button
            pressed_mouse_buttons,          // buttons
            None,                           // related_target
            None,                           // point_in_target
            PointerId::Mouse as i32,        // pointer_id
            1,                              // width
            1,                              // height
            0.5,                            // pressure
            0.0,                            // tangential_pressure
            0,                              // tilt_x
            0,                              // tilt_y
            0,                              // twist
            PI / 2.0,                       // altitude_angle
            0.0,                            // azimuth_angle
            DOMString::from("mouse"),       // pointer_type
            true,                           // is_primary
            vec![],                         // coalesced_events
            vec![],                         // predicted_events
            can_gc,
        );
        let event = menu_event.upcast::<Event>(); // Upcast to a generic Event.
        event.fire(target, can_gc); // Fire the `contextmenu` event.

        // Block Logic: If the event was not canceled, notify the embedder to show the context menu.
        // if the event was not canceled, notify the embedder to show the context menu
        if event.status() == EventStatus::NotCanceled {
            let (sender, receiver) = 
                ipc::channel::<ContextMenuResult>().expect("Failed to create IPC channel."); // Create IPC channel.
            self.send_to_embedder(EmbedderMsg::ShowContextMenu(
                self.webview_id(),
                sender,
                None,
                vec![],
            )); // Send message to embedder.
            let _ = receiver.recv().unwrap(); // Wait for reply from embedder.
        };
    }

    /// @brief Checks conditions and potentially fires a `dblclick` event.
    /// Functional Utility: Implements the double-click detection logic based on
    /// time and distance thresholds between consecutive clicks.
    /// <https://w3c.github.io/uievents/#event-type-dblclick>
    ///
    /// @param click_pos The client coordinates of the current click.
    /// @param target The `Node` that received the click.
    /// @param pressed_mouse_buttons A bitmask of currently pressed mouse buttons.
    /// @param can_gc A `CanGc` token.
    fn maybe_fire_dblclick(
        &self,
        click_pos: Point2D<f32>,
        target: &Node,
        pressed_mouse_buttons: u16,
        can_gc: CanGc,
    ) {
        // <https://w3c.github.io/uievents/#event-type-dblclick>
        let now = Instant::now(); // Current time.

        let opt = self.last_click_info.borrow_mut().take(); // Take the last click info.

        // Block Logic: Check if a double-click has occurred based on time and distance.
        if let Some((last_time, last_pos)) = opt {
            let DBL_CLICK_TIMEOUT =
                Duration::from_millis(pref!(dom_document_dblclick_timeout) as u64); // Double-click timeout preference.
            let DBL_CLICK_DIST_THRESHOLD = pref!(dom_document_dblclick_dist) as u64; // Double-click distance threshold preference.

            // Calculate distance between this click and the previous click.
            let line = click_pos - last_pos; // Vector between click positions.
            let dist = (line.dot(line) as f64).sqrt(); // Euclidean distance.

            if now.duration_since(last_time) < DBL_CLICK_TIMEOUT &&
                dist < DBL_CLICK_DIST_THRESHOLD as f64
            {
                // A double click has occurred if this click is within a certain time and dist. of previous click.
                let click_count = 2; // Click count for double-click event.
                let client_x = click_pos.x as i32; // Client X coordinate.
                let client_y = click_pos.y as i32; // Client Y coordinate.

                // Block Logic: Create and fire a `dblclick` event.
                let event = MouseEvent::new(
                    &self.window,
                    DOMString::from("dblclick"),
                    EventBubbles::Bubbles,
                    EventCancelable::Cancelable,
                    Some(&self.window),
                    click_count,
                    client_x,
                    client_y,
                    client_x,
                    client_y,
                    false,
                    false,
                    false,
                    false,
                    0i16,
                    pressed_mouse_buttons,
                    None,
                    None,
                    can_gc,
                );
                event.upcast::<Event>().fire(target.upcast(), can_gc); // Fire the `dblclick` event.

                // When a double click occurs, self.last_click_info is left as None so that a
                // third sequential click will not cause another double click.
                return;
            }
        }

        // Update last_click_info with the time and position of the click.
        *self.last_click_info.borrow_mut() = Some((now, click_pos));
    }

    /// @brief Fires a generic mouse event (e.g., `mousemove`, `mouseover`, `mouseout`).
    /// Functional Utility: Constructs and dispatches a `MouseEvent` with specified properties
    /// to a target.
    ///
    /// @param client_point The client coordinates of the event.
    /// @param target The `EventTarget` for the event.
    /// @param event_name The `FireMouseEventType` to dispatch.
    /// @param can_bubble `EventBubbles` enum for bubbling behavior.
    /// @param cancelable `EventCancelable` enum for cancelable behavior.
    /// @param pressed_mouse_buttons A bitmask of currently pressed mouse buttons.
    /// @param can_gc A `CanGc` token.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fire_mouse_event(
        &self,
        client_point: Point2D<f32>,
        target: &EventTarget,
        event_name: FireMouseEventType,
        can_bubble: EventBubbles,
        cancelable: EventCancelable,
        pressed_mouse_buttons: u16,
        can_gc: CanGc,
    ) {
        let client_x = client_point.x.to_i32().unwrap_or(0); // Convert x-coordinate to i32.
        let client_y = client_point.y.to_i32().unwrap_or(0); // Convert y-coordinate to i32.

        MouseEvent::new(
            &self.window,
            DOMString::from(event_name.as_str()), // Event type string.
            can_bubble,
            cancelable,
            Some(&self.window),
            0i32,
            client_x,
            client_y,
            client_x,
            client_y,
            false,
            false,
            false,
            false,
            0i16,
            pressed_mouse_buttons,
            None,
            None,
            can_gc,
        )
        .upcast::<Event>() // Upcast to a generic Event.
        .fire(target, can_gc); // Fire the event.
    }

    /// @brief Handles an editing action (copy, cut, paste).
    /// Functional Utility: Delegates to `handle_clipboard_action` to process clipboard operations.
    /// @param action The `EditingActionEvent` (Copy, Cut, or Paste).
    /// @param can_gc A `CanGc` token.
    /// @return `true` if the action was handled, `false` otherwise.
    pub(crate) fn handle_editing_action(&self, action: EditingActionEvent, can_gc: CanGc) -> bool {
        let clipboard_event = match action {
            EditingActionEvent::Copy => ClipboardEventType::Copy,
            EditingActionEvent::Cut => ClipboardEventType::Cut,
            EditingActionEvent::Paste => ClipboardEventType::Paste,
        };
        self.handle_clipboard_action(clipboard_event, can_gc) // Delegate to clipboard action handler.
    }

    /// @brief Handles a clipboard action (copy, cut, paste).
    /// Functional Utility: Orchestrates the firing of clipboard events, manages data transfer
    /// to/from the clipboard, and interacts with the embedder for native clipboard operations.
    /// <https://www.w3.org/TR/clipboard-apis/#clipboard-actions>
    ///
    /// @param action The `ClipboardEventType` to handle.
    /// @param can_gc A `CanGc` token.
    /// @return `true` if the action was successfully processed, `false` otherwise.
    fn handle_clipboard_action(&self, action: ClipboardEventType, can_gc: CanGc) -> bool {
        // The script_triggered flag is set if the action runs because of a script, e.g. document.execCommand()
        let script_triggered = false;

        // The script_may_access_clipboard flag is set
        // if action is paste and the script thread is allowed to read from clipboard or
        // if action is copy or cut and the script thread is allowed to modify the clipboard
        let script_may_access_clipboard = false;

        // Step 1 If the script-triggered flag is set and the script-may-access-clipboard flag is unset
        if script_triggered && !script_may_access_clipboard {
            return false;
        }

        // Step 2 Fire a clipboard event
        let event = ClipboardEvent::new(
            &self.window,
            None,
            DOMString::from(action.as_str()), // Event type string.
            EventBubbles::Bubbles,
            EventCancelable::Cancelable,
            None,
            can_gc,
        );
        self.fire_clipboard_event(&event, action, can_gc); // Fire the clipboard event.

        // Step 3 If a script doesn't call preventDefault()
        // the event will be handled inside target's VirtualMethods::handle_event

        let e = event.upcast::<Event>(); // Upcast to a generic Event.

        if !e.IsTrusted() {
            return false;
        }

        // Step 4 If the event was canceled, then
        if e.DefaultPrevented() {
            match e.Type().str() {
                "copy" => {
                    // Step 4.1 Call the write content to the clipboard algorithm,
                    // passing on the DataTransferItemList items, a clear-was-called flag and a types-to-clear list.
                    if let Some(clipboard_data) = event.get_clipboard_data() {
                        let drag_data_store =
                            clipboard_data.data_store().expect("This shouldn't fail");
                        self.write_content_to_the_clipboard(&drag_data_store); // Write to clipboard.
                    }
                },
                "cut" => {
                    // Step 4.1 Call the write content to the clipboard algorithm,
                    // passing on the DataTransferItemList items, a clear-was-called flag and a types-to-clear list.
                    if let Some(clipboard_data) = event.get_clipboard_data() {
                        let drag_data_store =
                            clipboard_data.data_store().expect("This shouldn't fail");
                        self.write_content_to_the_clipboard(&drag_data_store); // Write to clipboard.
                    }

                    // Step 4.2 Fire a clipboard event named clipboardchange
                    self.fire_clipboardchange_event(can_gc); // Fire `clipboardchange` event.
                },
                "paste" => return false,
                _ => (),
            }
        }
        //Step 5
        true
    }

    /// @brief Fires a clipboard event.
    /// Functional Utility: Constructs and dispatches a `ClipboardEvent` with appropriate data
    /// and flags, handling the interaction with the native clipboard.
    /// <https://www.w3.org/TR/clipboard-apis/#fire-a-clipboard-event>
    ///
    /// @param event A reference to the `ClipboardEvent` to fire.
    /// @param action The `ClipboardEventType` of the event.
    /// @param can_gc A `CanGc` token.
    fn fire_clipboard_event(
        &self,
        event: &ClipboardEvent,
        action: ClipboardEventType,
        can_gc: CanGc,
    ) {
        // Step 1 Let clear_was_called be false
        // Step 2 Let types_to_clear an empty list
        let mut drag_data_store = DragDataStore::new(); // Create a new data transfer store.

        // Step 4 let clipboard-entry be the sequence number of clipboard content, null if the OS doesn't support it.

        // Step 5 let trusted be true if the event is generated by the user agent, false otherwise
        let trusted = true; // Event is considered trusted (generated by user agent).

        // Step 6 if the context is editable:
        let focused = self.get_focused_element(); // Get the currently focused element.
        let body = self.GetBody(); // Get the document body.

        // Block Logic: Determine the event target. Prioritize focused element, then body, then window.
        let target = match (&focused, &body) {
            (Some(focused), _) => focused.upcast(), // If an element is focused, use it.
            (&None, Some(body)) => body.upcast(), // Otherwise, if body exists, use it.
            (&None, &None) => self.window.upcast(), // Otherwise, use the window.
        };
        // Step 6.2 else TODO require Selection see https://github.com/w3c/clipboard-apis/issues/70

        // Step 7
        match action {
            ClipboardEventType::Copy | ClipboardEventType::Cut => {
                // Step 7.2.1: Set mode to ReadWrite for copy/cut.
                drag_data_store.set_mode(Mode::ReadWrite);
            },
            ClipboardEventType::Paste => {
                let (sender, receiver) = ipc::channel().unwrap(); // Create IPC channel for clipboard data.
                self.window
                    .send_to_constellation(ScriptToConstellationMessage::ForwardToEmbedder(
                        EmbedderMsg::GetClipboardText(self.window.webview_id(), sender),
                    )); // Request clipboard text from embedder.
                let text_contents = receiver
                    .recv()
                    .map(Result::unwrap_or_default)
                    .unwrap_or_default(); // Receive clipboard text.

                // Step 7.1.1: Set mode to ReadOnly for paste.
                drag_data_store.set_mode(Mode::ReadOnly);
                // Step 7.1.2 If trusted or the implementation gives script-generated events access to the clipboard
                if trusted {
                    // Step 7.1.2.1 For each clipboard-part on the OS clipboard:

                    // Step 7.1.2.1.1 If clipboard-part contains plain text, then
                    let data = DOMString::from(text_contents.to_string()); // Convert text to DOMString.
                    let type_ = DOMString::from("text/plain"); // MIME type.
                    let _ = drag_data_store.add(Kind::Text { data, type_ }); // Add plain text to data store.

                    // Step 7.1.2.1.2 TODO If clipboard-part represents file references, then for each file reference
                    // Step 7.1.2.1.3 TODO If clipboard-part contains HTML- or XHTML-formatted text then

                    // Step 7.1.3 Update clipboard-event-data’s files to match clipboard-event-data’s items
                    // Step 7.1.4 Update clipboard-event-data’s types to match clipboard-event-data’s items
                }
            },
            ClipboardEventType::Change => (), // No specific action for change event here.
        }

        // Step 3: Create DataTransfer object for the clipboard event.
        let clipboard_event_data = DataTransfer::new(
            &self.window,
            Rc::new(RefCell::new(Some(drag_data_store))), // Wrap data store in Rc<RefCell>.
            can_gc,
        );

        // Step 8: Set the clipboard data on the event.
        event.set_clipboard_data(Some(&clipboard_event_data));
        let event = event.upcast::<Event>(); // Upcast to generic Event.
        // Step 9: Set event's trusted flag.
        event.set_trusted(trusted);
        // Step 10 Set event’s composed to true.
        event.set_composed(true);
        // Step 11: Dispatch the event.
        event.dispatch(target, false, can_gc);
    }

    /// @brief Fires a `clipboardchange` event.
    /// Functional Utility: Notifies listeners that the system clipboard contents may have changed.
    ///
    /// @param can_gc A `CanGc` token.
    pub(crate) fn fire_clipboardchange_event(&self, can_gc: CanGc) {
        let clipboardchange_event = ClipboardEvent::new(
            &self.window,
            None,
            DOMString::from("clipboardchange"), // Event type string.
            EventBubbles::Bubbles,
            EventCancelable::Cancelable,
            None,
            can_gc,
        );
        self.fire_clipboard_event(&clipboardchange_event, ClipboardEventType::Change, can_gc); // Fire the `clipboardchange` event.
    }

    /// @brief Writes content from a `DragDataStore` to the system clipboard.
    /// Functional Utility: Interacts with the embedder to perform native clipboard write operations,
    /// handling different data kinds (e.g., plain text).
    /// <https://www.w3.org/TR/clipboard-apis/#write-content-to-the-clipboard>
    ///
    /// @param drag_data_store The `DragDataStore` containing the data to write.
    fn write_content_to_the_clipboard(&self, drag_data_store: &DragDataStore) {
        // Step 1
        if drag_data_store.list_len() > 0 {
            // Step 1.1 Clear the clipboard.
            self.send_to_embedder(EmbedderMsg::ClearClipboard(self.webview_id())); // Clear native clipboard.
            // Step 1.2
            for item in drag_data_store.iter_item_list() {
                match item {
                    Kind::Text { data, .. } => {
                        // Step 1.2.1.1 Ensure encoding is correct per OS and locale conventions
                        // Step 1.2.1.2 Normalize line endings according to platform conventions
                        // Step 1.2.1.3
                        self.send_to_embedder(EmbedderMsg::SetClipboardText(
                            self.webview_id(),
                            data.to_string(),
                        )); // Set clipboard text.
                    },
                    Kind::File { .. } => {
                        // Step 1.2.2 If data is of a type listed in the mandatory data types list, then
                        // Step 1.2.2.1 Place part on clipboard with the appropriate OS clipboard format description
                        // Step 1.2.3 Else this is left to the implementation
                    },
                }
            }
        } else {
            // Step 2.1
            if drag_data_store.clear_was_called {
                // Step 2.1.1 If types-to-clear list is empty, clear the clipboard
                self.send_to_embedder(EmbedderMsg::ClearClipboard(self.webview_id())); // Clear native clipboard.
                // Step 2.1.2 Else remove the types in the list from the clipboard
                // As of now this can't be done with Arboard, and it's possible that will be removed from the spec
            }
        }
    }

    /// @brief Handles a mouse move event, including hover state changes and event dispatch.
    /// Functional Utility: Processes mouse movement, determines the new `mouseover` and `mouseout`
    /// targets, dispatches corresponding events, and updates element hover states.
    ///
    /// @param hit_test_result An `Option<CompositorHitTestResult>` from the compositor.
    /// @param pressed_mouse_buttons A bitmask of currently pressed mouse buttons.
    /// @param prev_mouse_over_target A mutable nullable DOM reference to the previously hovered element.
    /// @param can_gc A `CanGc` token.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn handle_mouse_move_event(
        &self,
        hit_test_result: Option<CompositorHitTestResult>,
        pressed_mouse_buttons: u16,
        prev_mouse_over_target: &MutNullableDom<Element>,
        can_gc: CanGc,
    ) {
        // Block Logic: Ignore events without a valid hit test result.
        let Some(hit_test_result) = hit_test_result else {
            return;
        };

        // Block Logic: Convert untrusted node address to a trusted `DomRoot<Node>` and find the new target element.
        let node = unsafe { node::from_untrusted_node_address(hit_test_result.node) };
        let Some(new_target) = node
            .inclusive_ancestors(ShadowIncluding::No)
            .filter_map(DomRoot::downcast::<Element>)
            .next()
        else {
            return; // If no element found, return.
        };

        // Block Logic: Determine if the mouse target has changed.
        let target_has_changed = prev_mouse_over_target
            .get()
            .as_ref()
            .is_none_or(|old_target| old_target != &new_target);

        // Block Logic: If the target has changed, dispatch `mouseout`/`mouseleave` and `mouseover`/`mouseenter` events.
        // Here we know the target has changed, so we must update the state,
        // dispatch mouseout to the previous one, mouseover to the new one.
        if target_has_changed {
            // Dispatch mouseout and mouseleave to previous target.
            if let Some(old_target) = prev_mouse_over_target.get() {
                let old_target_is_ancestor_of_new_target = old_target
                    .upcast::<Node>()
                    .is_ancestor_of(new_target.upcast::<Node>());

                // If the old target is an ancestor of the new target, this can be skipped
                // completely, since the node's hover state will be reset below.
                if !old_target_is_ancestor_of_new_target {
                    // Block Logic: Iterate through ancestors of the old target and reset hover/active states.
                    for element in old_target
                        .upcast::<Node>()
                        .inclusive_ancestors(ShadowIncluding::No)
                        .filter_map(DomRoot::downcast::<Element>)
                    {
                        element.set_hover_state(false); // Clear hover state.
                        element.set_active_state(false); // Clear active state.
                    }
                }

                // Functional Utility: Fire a `mouseout` event.
                self.fire_mouse_event(
                    hit_test_result.point_in_viewport,
                    old_target.upcast(),
                    FireMouseEventType::Out,
                    EventBubbles::Bubbles,
                    EventCancelable::Cancelable,
                    pressed_mouse_buttons,
                    can_gc,
                );

                if !old_target_is_ancestor_of_new_target {
                    let event_target = DomRoot::from_ref(old_target.upcast::<Node>()); // Get event target.
                    let moving_into = Some(DomRoot::from_ref(new_target.upcast::<Node>())); // Get related target.
                    // Functional Utility: Handle `mouseleave` event.
                    self.handle_mouse_enter_leave_event(
                        hit_test_result.point_in_viewport,
                        FireMouseEventType::Leave,
                        moving_into,
                        event_target,
                        pressed_mouse_buttons,
                        can_gc,
                    );
                }
            }

            // Dispatch mouseover and mouseenter to new target.
            // Block Logic: Iterate through ancestors of the new target and set hover state.
            for element in new_target
                .upcast::<Node>()
                .inclusive_ancestors(ShadowIncluding::No)
                .filter_map(DomRoot::downcast::<Element>)
            {
                if element.hover_state() {
                    break; // Stop if an ancestor is already hovered.
                }
                element.set_hover_state(true); // Set hover state.
            }

            // Functional Utility: Fire a `mouseover` event.
            self.fire_mouse_event(
                hit_test_result.point_in_viewport,
                new_target.upcast(),
                FireMouseEventType::Over,
                EventBubbles::Bubbles,
                EventCancelable::Cancelable,
                pressed_mouse_buttons,
                can_gc,
            );

            let moving_from = prev_mouse_over_target
                .get()
                .map(|old_target| DomRoot::from_ref(old_target.upcast::<Node>())); // Get related target.
            let event_target = DomRoot::from_ref(new_target.upcast::<Node>()); // Get event target.
            // Functional Utility: Handle `mouseenter` event.
            self.handle_mouse_enter_leave_event(
                hit_test_result.point_in_viewport,
                FireMouseEventType::Enter,
                moving_from,
                event_target,
                pressed_mouse_buttons,
                can_gc,
            );
        }

        // Send mousemove event to topmost target, unless it's an iframe, in which case the
        // compositor should have also sent an event to the inner document.
        // Functional Utility: Fire a `mousemove` event.
        self.fire_mouse_event(
            hit_test_result.point_in_viewport,
            new_target.upcast(),
            FireMouseEventType::Move,
            EventBubbles::Bubbles,
            EventCancelable::Cancelable,
            pressed_mouse_buttons,
            can_gc,
        );

        // If the target has changed then store the current mouse over target for next frame.
        if target_has_changed {
            prev_mouse_over_target.set(Some(&new_target)); // Update `prev_mouse_over_target`.
        }
    }

    /// @brief Handles `mouseenter` and `mouseleave` events, ensuring correct event flow.
    /// Functional Utility: Dispatches `mouseenter` or `mouseleave` events to the appropriate
    /// elements in the hierarchy, based on the common ancestor between the related and event targets.
    ///
    /// @param client_point The client coordinates of the event.
    /// @param event_type The `FireMouseEventType` (Enter or Leave).
    /// @param related_target An `Option<DomRoot<Node>>` representing the related target.
    /// @param event_target The `DomRoot<Node>` that is the event target.
    /// @param pressed_mouse_buttons A bitmask of currently pressed mouse buttons.
    /// @param can_gc A `CanGc` token.
    fn handle_mouse_enter_leave_event(
        &self,
        client_point: Point2D<f32>,
        event_type: FireMouseEventType,
        related_target: Option<DomRoot<Node>>,
        event_target: DomRoot<Node>,
        pressed_mouse_buttons: u16,
        can_gc: CanGc,
    ) {
        assert!(matches!(
            event_type,
            FireMouseEventType::Enter | FireMouseEventType::Leave
        ));

        // Block Logic: Find the common ancestor between the event target and related target.
        let common_ancestor = match related_target.as_ref() {
            Some(related_target) => event_target
                .common_ancestor(related_target, ShadowIncluding::No)
                .unwrap_or_else(|| DomRoot::from_ref(&*event_target)), // If no common ancestor, use event target.
            None => DomRoot::from_ref(&*event_target), // If no related target, use event target.
        };

        // We need to create a target chain in case the event target shares
        // its boundaries with its ancestors.
        let mut targets = vec![]; // Vector to store the event targets.
        let mut current = Some(event_target); // Start with the event target.
        // Block Logic: Build a chain of ancestors from the event target up to (but not including) the common ancestor.
        while let Some(node) = current {
            if node == common_ancestor {
                break; // Stop at the common ancestor.
            }
            current = node.GetParentNode(); // Get parent node.
            targets.push(node); // Add node to targets.
        }

        // The order for dispatching mouseenter events starts from the topmost
        // common ancestor of the event target and the related target.
        if event_type == FireMouseEventType::Enter {
            targets = targets.into_iter().rev().collect(); // Reverse order for `mouseenter`.
        }

        // Block Logic: Fire `mouseenter` or `mouseleave` events to the collected targets.
        for target in targets {
            self.fire_mouse_event(
                client_point,
                target.upcast(),
                event_type,
                EventBubbles::DoesNotBubble, // These events do not bubble.
                EventCancelable::NotCancelable, // These events are not cancelable.
                pressed_mouse_buttons,
                can_gc,
            );
        }
    }

    /// @brief Handles a wheel event, including hit testing and event dispatch.
    /// Functional Utility: Processes mouse wheel input, identifies the target element,
    /// and dispatches a `WheelEvent` to it.
    ///
    /// @param event The `WheelEvent` to handle.
    /// @param hit_test_result An `Option<CompositorHitTestResult>` from the compositor.
    /// @param can_gc A `CanGc` token.
    #[allow(unsafe_code)]
    pub(crate) fn handle_wheel_event(
        &self,
        event: WheelEvent,
        hit_test_result: Option<CompositorHitTestResult>,
        can_gc: CanGc,
    ) {
        // Block Logic: Ignore events without a valid hit test result.
        let Some(hit_test_result) = hit_test_result else {
            return;
        };

        // Block Logic: Convert untrusted node address to a trusted `DomRoot<Node>` and find the target element.
        let node = unsafe { node::from_untrusted_node_address(hit_test_result.node) };
        let Some(el) = node
            .inclusive_ancestors(ShadowIncluding::No)
            .filter_map(DomRoot::downcast::<Element>)
            .next()
        else {
            return; // If no element found, return.
        };

        let node = el.upcast::<Node>(); // Upcast element to node.
        let wheel_event_type_string = "wheel".to_owned(); // Event type string.
        debug!(
            "{}: on {:?} at {:?}",
            wheel_event_type_string,
            node.debug_str(),
            hit_test_result.point_in_viewport
        );

        // Block Logic: Create and fire a DOM `WheelEvent`.
        // https://w3c.github.io/uievents/#event-wheelevents
        let dom_event = DomWheelEvent::new(
            &self.window,
            DOMString::from(wheel_event_type_string),
            EventBubbles::Bubbles, // Wheel events bubble.