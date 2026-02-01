/*!
This module defines the `Document` struct, which represents a web page in the
DOM tree. It is the entry point for all content in a web page, and provides
methods for accessing and manipulating the document's content.

The `Document` struct implements the `Node` trait, and is the root of the
DOM tree. It also implements the `DocumentOrShadowRoot` trait, which provides
methods for accessing the document's stylesheets, and the `GlobalScope`
trait, which provides methods for accessing the document's window and other
global properties.

The `Document` struct is responsible for:
- Managing the document's lifecycle, including loading, parsing, and
  rendering.
- Handling events, including mouse, keyboard, and focus events.
- Managing the document's stylesheets and scripts.
- Providing access to the document's elements and other nodes.
- Managing the document's focus and selection.
- Handling the document's history and navigation.
*/

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{HashMap, HashSet, VecDeque};
use std::default::Default;
use std::f64::consts::PI;
use std::mem;
use std::rc::Rc;
use std::slice::from_ref;
use std::str::FromStr;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use base::cross_process_instant::CrossProcessInstant;
use base::id::WebViewId;
use canvas_traits::canvas::CanvasId;
use canvas_traits::webgl::{self, WebGLContextId, WebGLMsg};
use chrono::Local;
use constellation_traits::{NavigationHistoryBehavior, ScriptToConstellationMessage};
use content_security_policy::{self as csp, CspList, PolicyDisposition};
use cookie::Cookie;
use cssparser::match_ignore_ascii_case;
use data_url::mime::Mime;
use devtools_traits::ScriptToDevtoolsControlMsg;
use dom_struct::dom_struct;
use embedder_traits::{
    AllowOrDeny, AnimationState, CompositorHitTestResult, ContextMenuResult, EditingActionEvent,
    EmbedderMsg, FocusSequenceNumber, ImeEvent, InputEvent, LoadStatus, MouseButton,
    MouseButtonAction, MouseButtonEvent, ScrollEvent, TouchEvent, TouchEventType, TouchId,
    UntrustedNodeAddress, WheelEvent,
};
use encoding_rs::{Encoding, UTF_8};
use euclid::default::{Point2D, Rect, Size2D};
use html5ever::{LocalName, Namespace, QualName, local_name, ns};
use hyper_serde::Serde;
use ipc_channel::ipc;
use js::rust::{HandleObject, HandleValue};
use keyboard_types::{Code, Key, KeyState, Modifiers};
use layout_api::{PendingRestyle, TrustedNodeAddress, node_id_from_scroll_id};
use metrics::{InteractiveFlag, InteractiveWindow, ProgressiveWebMetrics};
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
use crate::dom::bindings::codegen::UnionTypes::{
    NodeOrString,
    StringOrElementCreationOptions,
    TrustedHTMLOrString,
};
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
use crate::dom::csp::report_csp_violations;
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
use crate::dom::trustedhtml::TrustedHTML;
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
use crate::mime::{APPLICATION, CHARSET, MimeExt};
use crate::network_listener::{NetworkListener, PreInvoke};
use crate::realms::{AlreadyInRealm, InRealm, enter_realm};
use crate::script_runtime::{CanGc, ScriptThreadEventCategory};
use crate::script_thread::{ScriptThread, with_script_thread};
use crate::stylesheet_set::StylesheetSetRef;
use crate::task::TaskBox;
use crate::task_source::TaskSourceName;
use crate::timers::OneshotTimerCallback;

/// The result of a touch event.
pub(crate) enum TouchEventResult {
    /// The event was processed.
    Processed(bool),
    /// The event was forwarded.
    Forwarded,
}

/// The type of a mouse event to fire.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum FireMouseEventType {
    /// A mouse move event.
    Move,
    /// A mouse over event.
    Over,
    /// A mouse out event.
    Out,
    /// A mouse enter event.
    Enter,
    /// A mouse leave event.
    Leave,
}

impl FireMouseEventType {
    /// Returns the string representation of the event type.
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

/// A refresh redirect that is due to be performed.
#[derive(JSTraceable, MallocSizeOf)]
pub(crate) struct RefreshRedirectDue {
    #[no_trace]
    /// The URL to redirect to.
    pub(crate) url: ServoUrl,
    #[ignore_malloc_size_of = "non-owning"]
    /// The window to perform the redirect in.
    pub(crate) window: DomRoot<Window>,
}
impl RefreshRedirectDue {
    /// Invokes the refresh redirect.
    pub(crate) fn invoke(self, can_gc: CanGc) {
        self.window.Location().navigate(
            self.url.clone(),
            NavigationHistoryBehavior::Replace,
            NavigationType::DeclarativeRefresh,
            can_gc,
        );
    }
}

/// Whether a document is an HTML document or not.
#[derive(Clone, Copy, Debug, JSTraceable, MallocSizeOf, PartialEq)]
pub(crate) enum IsHTMLDocument {
    /// The document is an HTML document.
    HTMLDocument,
    /// The document is not an HTML document.
    NonHTMLDocument,
}

/// A focus transaction.
#[derive(JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
struct FocusTransaction {
    /// The focused element of this document.
    element: Option<Dom<Element>>,
    /// See [`Document::has_focus`].
    has_focus: bool,
}

/// Information about a declarative refresh
#[derive(JSTraceable, MallocSizeOf)]
pub(crate) enum DeclarativeRefresh {
    /// A refresh that is pending.
    PendingLoad {
        #[no_trace]
        /// The URL to refresh to.
        url: ServoUrl,
        /// The time to wait before refreshing.
        time: u64,
    },
    /// A refresh that was created after the document was loaded.
    CreatedAfterLoad,
}
#[cfg(feature = "webgpu")]
/// A map of WebGPU context IDs to their contexts.
pub(crate) type WebGPUContextsMap =
    Rc<RefCell<HashMapTracedValues<WebGPUContextId, WeakRef<GPUCanvasContext>>>>;

/// <https://dom.spec.whatwg.org/#document>
#[dom_struct]
pub(crate) struct Document {
    node: Node,
    document_or_shadow_root: DocumentOrShadowRoot,
    window: Dom<Window>,
    implementation: MutNullableDom<DOMImplementation>,
    #[ignore_malloc_size_of = "type from external crate"]
    #[no_trace]
    content_type: Mime,
    last_modified: Option<String>,
    #[no_trace]
    encoding: Cell<&'static Encoding>,
    has_browsing_context: bool,
    is_html_document: bool,
    #[no_trace]
    activity: Cell<DocumentActivity>,
    #[no_trace]
    url: DomRefCell<ServoUrl>,
    #[ignore_malloc_size_of = "defined in selectors"]
    #[no_trace]
    quirks_mode: Cell<QuirksMode>,
    /// Caches for the getElement methods
    id_map: DomRefCell<HashMapTracedValues<Atom, Vec<Dom<Element>>>>,
    name_map: DomRefCell<HashMapTracedValues<Atom, Vec<Dom<Element>>>>,
    tag_map: DomRefCell<HashMapTracedValues<LocalName, Dom<HTMLCollection>>>,
    tagns_map: DomRefCell<HashMapTracedValues<QualName, Dom<HTMLCollection>>>,
    classes_map: DomRefCell<HashMapTracedValues<Vec<Atom>, Dom<HTMLCollection>>>,
    images: MutNullableDom<HTMLCollection>,
    embeds: MutNullableDom<HTMLCollection>,
    links: MutNullableDom<HTMLCollection>,
    forms: MutNullableDom<HTMLCollection>,
    scripts: MutNullableDom<HTMLCollection>,
    anchors: MutNullableDom<HTMLCollection>,
    applets: MutNullableDom<HTMLCollection>,
    /// Information about the `<iframes>` in this [`Document`].
    iframes: RefCell<IFrameCollection>,
    /// Lock use for style attributes and author-origin stylesheet objects in this document.
    /// Can be acquired once for accessing many objects.
    #[no_trace]
    style_shared_lock: StyleSharedRwLock,
    /// List of stylesheets associated with nodes in this document. |None| if the list needs to be refreshed.
    #[custom_trace]
    stylesheets: DomRefCell<DocumentStylesheetSet<StyleSheetInDocument>>,
    stylesheet_list: MutNullableDom<StyleSheetList>,
    ready_state: Cell<DocumentReadyState>,
    /// Whether the DOMContentLoaded event has already been dispatched.
    domcontentloaded_dispatched: Cell<bool>,
    /// The state of this document's focus transaction.
    focus_transaction: DomRefCell<Option<FocusTransaction>>,
    /// The element that currently has the document focus context.
    focused: MutNullableDom<Element>,
    /// The last sequence number sent to the constellation.
    #[no_trace]
    focus_sequence: Cell<FocusSequenceNumber>,
    /// Indicates whether the container is included in the top-level browsing
    /// context's focus chain (not considering system focus). Permanently `true`
    /// for a top-level document.
    has_focus: Cell<bool>,
    /// The script element that is currently executing.
    current_script: MutNullableDom<HTMLScriptElement>,
    /// <https://html.spec.whatwg.org/multipage/#pending-parsing-blocking-script>
    pending_parsing_blocking_script: DomRefCell<Option<PendingScript>>,
    /// Number of stylesheets that block executing the next parser-inserted script
    script_blocking_stylesheets_count: Cell<u32>,
    /// <https://html.spec.whatwg.org/multipage/#list-of-scripts-that-will-execute-when-the-document-has-finished-parsing>
    deferred_scripts: PendingInOrderScriptVec,
    /// <https://html.spec.whatwg.org/multipage/#list-of-scripts-that-will-execute-in-order-as-soon-as-possible>
    asap_in_order_scripts_list: PendingInOrderScriptVec,
    /// <https://html.spec.whatwg.org/multipage/#set-of-scripts-that-will-execute-as-soon-as-possible>
    asap_scripts_set: DomRefCell<Vec<Dom<HTMLScriptElement>>>,
    /// <https://html.spec.whatwg.org/multipage/#concept-n-noscript>
    /// True if scripting is enabled for all scripts in this document
    scripting_enabled: bool,
    /// <https://html.spec.whatwg.org/multipage/#animation-frame-callback-identifier>
    /// Current identifier of animation frame callback
    animation_frame_ident: Cell<u32>,
    /// <https://html.spec.whatwg.org/multipage/#list-of-animation-frame-callbacks>
    /// List of animation frame callbacks
    animation_frame_list: DomRefCell<VecDeque<(u32, Option<AnimationFrameCallback>)>>,
    /// Whether we're in the process of running animation callbacks.
    ///
    /// Tracking this is not necessary for correctness. Instead, it is an optimization to avoid
    /// sending needless `ChangeRunningAnimationsState` messages to the compositor.
    running_animation_callbacks: Cell<bool>,
    /// Tracks all outstanding loads related to this document.
    loader: DomRefCell<DocumentLoader>,
    /// The current active HTML parser, to allow resuming after interruptions.
    current_parser: MutNullableDom<ServoParser>,
    /// The cached first `base` element with an `href` attribute.
    base_element: MutNullableDom<HTMLBaseElement>,
    /// This field is set to the document itself for inert documents.
    /// <https://html.spec.whatwg.org/multipage/#appropriate-template-contents-owner-document>
    appropriate_template_contents_owner_document: MutNullableDom<Document>,
    /// Information on elements needing restyle to ship over to layout when the
    /// time comes.
    pending_restyles: DomRefCell<HashMap<Dom<Element>, NoTrace<PendingRestyle>>>,
    /// This flag will be true if the `Document` needs to be painted again
    /// during the next full layout attempt due to some external change such as
    /// the web view changing size, or because the previous layout was only for
    /// layout queries (which do not trigger display).
    needs_paint: Cell<bool>,
    /// <http://w3c.github.io/touch-events/#dfn-active-touch-point>
    active_touch_points: DomRefCell<Vec<Dom<Touch>>>,
    /// Navigation Timing properties:
    /// <https://w3c.github.io/navigation-timing/#sec-PerformanceNavigationTiming>
    #[no_trace]
    dom_interactive: Cell<Option<CrossProcessInstant>>,
    #[no_trace]
    dom_content_loaded_event_start: Cell<Option<CrossProcessInstant>>,
    #[no_trace]
    dom_content_loaded_event_end: Cell<Option<CrossProcessInstant>>,
    #[no_trace]
    dom_complete: Cell<Option<CrossProcessInstant>>,
    #[no_trace]
    top_level_dom_complete: Cell<Option<CrossProcessInstant>>,
    #[no_trace]
    load_event_start: Cell<Option<CrossProcessInstant>>,
    #[no_trace]
    load_event_end: Cell<Option<CrossProcessInstant>>,
    #[no_trace]
    unload_event_start: Cell<Option<CrossProcessInstant>>,
    #[no_trace]
    unload_event_end: Cell<Option<CrossProcessInstant>>,
    /// <https://html.spec.whatwg.org/multipage/#concept-document-https-state>
    #[no_trace]
    https_state: Cell<HttpsState>,
    /// The document's origin.
    #[no_trace]
    origin: MutableOrigin,
    /// <https://html.spec.whatwg.org/multipage/#dom-document-referrer>
    referrer: Option<String>,
    /// <https://html.spec.whatwg.org/multipage/#target-element>
    target_element: MutNullableDom<Element>,
    /// <https://html.spec.whatwg.org/multipage/#concept-document-policy-container>
    #[no_trace]
    policy_container: DomRefCell<PolicyContainer>,
    /// <https://w3c.github.io/uievents/#event-type-dblclick>
    #[ignore_malloc_size_of = "Defined in std"]
    #[no_trace]
    last_click_info: DomRefCell<Option<(Instant, Point2D<f32>)>>,
    /// <https://html.spec.whatwg.org/multipage/#ignore-destructive-writes-counter>
    ignore_destructive_writes_counter: Cell<u32>,
    /// <https://html.spec.whatwg.org/multipage/#ignore-opens-during-unload-counter>
    ignore_opens_during_unload_counter: Cell<u32>,
    /// The number of spurious `requestAnimationFrame()` requests we've received.
    ///
    /// A rAF request is considered spurious if nothing was actually reflowed.
    spurious_animation_frames: Cell<u8>,

    /// Track the total number of elements in this DOM's tree.
    /// This is sent to layout every time a reflow is done;
    /// layout uses this to determine if the gains from parallel layout will be worth the overhead.
    ///
    /// See also: <https://github.com/servo/servo/issues/10110>
    dom_count: Cell<u32>,
    /// Entry node for fullscreen.
    fullscreen_element: MutNullableDom<Element>,
    /// Map from ID to set of form control elements that have that ID as
    /// their 'form' content attribute. Used to reset form controls
    /// whenever any element with the same ID as the form attribute
    /// is inserted or removed from the document.
    /// See <https://html.spec.whatwg.org/multipage/#form-owner>
    form_id_listener_map: DomRefCell<HashMapTracedValues<Atom, HashSet<Dom<Element>>>>,
    #[no_trace]
    interactive_time: DomRefCell<ProgressiveWebMetrics>,
    #[no_trace]
    tti_window: DomRefCell<InteractiveWindow>,
    /// RAII canceller for Fetch
    canceller: FetchCanceller,
    /// <https://html.spec.whatwg.org/multipage/#throw-on-dynamic-markup-insertion-counter>
    throw_on_dynamic_markup_insertion_counter: Cell<u64>,
    /// <https://html.spec.whatwg.org/multipage/#page-showing>
    page_showing: Cell<bool>,
    /// Whether the document is salvageable.
    salvageable: Cell<bool>,
    /// Whether the document was aborted with an active parser
    active_parser_was_aborted: Cell<bool>,
    /// Whether the unload event has already been fired.
    fired_unload: Cell<bool>,
    /// List of responsive images
    responsive_images: DomRefCell<Vec<Dom<HTMLImageElement>>>,
    /// Number of redirects for the document load
    redirect_count: Cell<u16>,
    /// Number of outstanding requests to prevent JS or layout from running.
    script_and_layout_blockers: Cell<u32>,
    /// List of tasks to execute as soon as last script/layout blocker is removed.
    #[ignore_malloc_size_of = "Measuring trait objects is hard"]
    delayed_tasks: DomRefCell<Vec<Box<dyn TaskBox>>>,
    /// <https://html.spec.whatwg.org/multipage/#completely-loaded>
    completely_loaded: Cell<bool>,
    /// Set of shadow roots connected to the document tree.
    shadow_roots: DomRefCell<HashSet<Dom<ShadowRoot>>>,
    /// Whether any of the shadow roots need the stylesheets flushed.
    shadow_roots_styles_changed: Cell<bool>,
    /// List of registered media controls.
    /// We need to keep this list to allow the media controls to
    /// access the "privileged" document.servoGetMediaControls(id) API,
    /// where `id` needs to match any of the registered ShadowRoots
    /// hosting the media controls UI.
    media_controls: DomRefCell<HashMap<String, Dom<ShadowRoot>>>,
    /// List of all context 2d IDs that need flushing.
    dirty_2d_contexts: DomRefCell<HashMapTracedValues<CanvasId, Dom<CanvasRenderingContext2D>>>,
    /// List of all WebGL context IDs that need flushing.
    dirty_webgl_contexts:
        DomRefCell<HashMapTracedValues<WebGLContextId, Dom<WebGLRenderingContext>>>,
    /// List of all WebGPU contexts.
    #[cfg(feature = "webgpu")]
    #[ignore_malloc_size_of = "Rc are hard"]
    webgpu_contexts: WebGPUContextsMap,
    /// <https://w3c.github.io/slection-api/#dfn-selection>
    selection: MutNullableDom<Selection>,
    /// A timeline for animations which is used for synchronizing animations.
    /// <https://drafts.csswg.org/web-animations/#timeline>
    animation_timeline: DomRefCell<AnimationTimeline>,
    /// Animations for this Document
    animations: DomRefCell<Animations>,
    /// Image Animation Manager for this Document
    image_animation_manager: DomRefCell<ImageAnimationManager>,
    /// The nearest inclusive ancestors to all the nodes that require a restyle.
    dirty_root: MutNullableDom<Element>,
    /// <https://html.spec.whatwg.org/multipage/#will-declaratively-refresh>
    declarative_refresh: DomRefCell<Option<DeclarativeRefresh>>,
    /// Pending input events, to be handled at the next rendering opportunity.
    #[no_trace]
    #[ignore_malloc_size_of = "CompositorEvent contains data from outside crates"]
    pending_input_events: DomRefCell<Vec<ConstellationInputEvent>>,
    /// The index of the last mouse move event in the pending compositor events queue.
    mouse_move_event_index: DomRefCell<Option<usize>>,
    /// <https://drafts.csswg.org/resize-observer/#dom-document-resizeobservers-slot>
    ///
    /// Note: we are storing, but never removing, resize observers.
    /// The lifetime of resize observers is specified at
    /// <https://drafts.csswg.org/resize-observer/#lifetime>.
    /// But implementing it comes with known problems:
    /// - <https://bugzilla.mozilla.org/show_bug.cgi?id=1596992>
    /// - <https://github.com/w3c/csswg-drafts/issues/4518>
    resize_observers: DomRefCell<Vec<Dom<ResizeObserver>>>,
    /// The set of all fonts loaded by this document.
    /// <https://drafts.csswg.org/css-font-loading/#font-face-source>
    fonts: MutNullableDom<FontFaceSet>,
    /// <https://html.spec.whatwg.org/multipage/#visibility-state>
    visibility_state: Cell<DocumentVisibilityState>,
    /// <https://www.iana.org/assignments/http-status-codes/http-status-codes.xhtml>
    status_code: Option<u16>,
    /// <https://html.spec.whatwg.org/multipage/#is-initial-about:blank>
    is_initial_about_blank: Cell<bool>,
    /// <https://dom.spec.whatwg.org/#document-allow-declarative-shadow-roots>
    allow_declarative_shadow_roots: Cell<bool>,
    /// <https://w3c.github.io/webappsec-upgrade-insecure-requests/#insecure-requests-policy>
    #[no_trace]
    inherited_insecure_requests_policy: Cell<Option<InsecureRequestsPolicy>>,
    //// <https://w3c.github.io/webappsec-mixed-content/#categorize-settings-object>
    has_trustworthy_ancestor_origin: Cell<bool>,
    /// <https://w3c.github.io/IntersectionObserver/#document-intersectionobservertaskqueued>
    intersection_observer_task_queued: Cell<bool>,
    /// Active intersection observers that should be processed by this document in
    /// the update intersection observation steps.
    /// <https://w3c.github.io/IntersectionObserver/#run-the-update-intersection-observations-steps>
    /// > Let observer list be a list of all IntersectionObservers whose root is in the DOM tree of document.
    /// > For the top-level browsing context, this includes implicit root observers.
    /// \
    /// Details of which document that should process an observers is discussed further at
    /// <https://github.com/w3c/IntersectionObserver/issues/525>.
    ///
    /// The lifetime of an intersection observer is specified at
    /// <https://github.com/w3c/IntersectionObserver/issues/525>`.
    intersection_observers: DomRefCell<Vec<Dom<IntersectionObserver>>>,
    /// The active keyboard modifiers for the WebView. This is updated when receiving any input event.
    #[no_trace]
    active_keyboard_modifiers: Cell<Modifiers>,
    /// The node that is currently highlighted by the devtools
    highlighted_dom_node: MutNullableDom<Node>,
}

#[allow(non_snake_case)]
impl Document {
    /// Notes that a node has dirty descendants.
    pub(crate) fn note_node_with_dirty_descendants(&self, node: &Node) {
        debug_assert!(*node.owner_doc() == *self);
        if !node.is_connected() {
            return;
        }

        let parent = match node.parent_in_flat_tree() {
            Some(parent) => parent,
            None => {
                // There is no parent so this is the Document node, so we
                // behave as if we were called with the document element.
                let document_element = match self.GetDocumentElement() {
                    Some(element) => element,
                    None => return,
                };
                if let Some(dirty_root) = self.dirty_root.get() {
                    // There was an existing dirty root so we mark its
                    // ancestors as dirty until the document element.
                    for ancestor in dirty_root
                        .upcast::<Node>()
                        .inclusive_ancestors_in_flat_tree()
                    {
                        if ancestor.is::<Element>() {
                            ancestor.set_flag(NodeFlags::HAS_DIRTY_DESCENDANTS, true);
                        }
                    }
                }
                self.dirty_root.set(Some(&document_element));
                return;
            },
        };

        if parent.is::<Element>() {
            if !parent.is_styled() {
                return;
            }

            if parent.is_display_none() {
                return;
            }
        }

        let element_parent: DomRoot<Element>;
        let element = match node.downcast::<Element>() {
            Some(element) => element,
            None => {
                // Current node is not an element, it's probably a text node,
                // we try to get its element parent.
                match DomRoot::downcast::<Element>(parent) {
                    Some(parent) => {
                        element_parent = parent;
                        &element_parent
                    },
                    None => {
                        // Parent is not an element so it must be a document,
                        // and this is not an element either, so there is
                        // nothing to do.
                        return;
                    },
                }
            },
        };

        let dirty_root = match self.dirty_root.get() {
            None => {
                element
                    .upcast::<Node>()
                    .set_flag(NodeFlags::HAS_DIRTY_DESCENDANTS, true);
                self.dirty_root.set(Some(element));
                return;
            },
            Some(root) => root,
        };

        for ancestor in element.upcast::<Node>().inclusive_ancestors_in_flat_tree() {
            if ancestor.get_flag(NodeFlags::HAS_DIRTY_DESCENDANTS) {
                return;
            }

            if ancestor.is::<Element>() {
                ancestor.set_flag(NodeFlags::HAS_DIRTY_DESCENDANTS, true);
            }
        }

        let new_dirty_root = element
            .upcast::<Node>()
            .common_ancestor_in_flat_tree(dirty_root.upcast())
            .expect("Couldn't find common ancestor");

        let mut has_dirty_descendants = true;
        for ancestor in dirty_root
            .upcast::<Node>()
            .inclusive_ancestors_in_flat_tree()
        {
            ancestor.set_flag(NodeFlags::HAS_DIRTY_DESCENDANTS, has_dirty_descendants);
            has_dirty_descendants &= *ancestor != *new_dirty_root;
        }

        let maybe_shadow_host = new_dirty_root
            .downcast::<ShadowRoot>()
            .map(ShadowRootMethods::Host);
        let new_dirty_root_element = new_dirty_root
            .downcast::<Element>()
            .or(maybe_shadow_host.as_deref());

        self.dirty_root.set(new_dirty_root_element);
    }

    /// Takes the dirty root of the document.
    pub(crate) fn take_dirty_root(&self) -> Option<DomRoot<Element>> {
        self.dirty_root.take()
    }

    /// Returns a reference to the document's loader.
    #[inline]
    pub(crate) fn loader(&self) -> Ref<DocumentLoader> {
        self.loader.borrow()
    }

    /// Returns a mutable reference to the document's loader.
    #[inline]
    pub(crate) fn loader_mut(&self) -> RefMut<DocumentLoader> {
        self.loader.borrow_mut()
    }

    /// Returns whether the document has a browsing context.
    #[inline]
    pub(crate) fn has_browsing_context(&self) -> bool {
        self.has_browsing_context
    }

    /// <https://html.spec.whatwg.org/multipage/#concept-document-bc>
    #[inline]
    pub(crate) fn browsing_context(&self) -> Option<DomRoot<WindowProxy>> {
        if self.has_browsing_context {
            self.window.undiscarded_window_proxy()
        } else {
            None
        }
    }

    /// Returns the webview ID of the document.
    pub(crate) fn webview_id(&self) -> WebViewId {
        self.window.webview_id()
    }

    /// Returns a reference to the document's window.
    #[inline]
    pub(crate) fn window(&self) -> &Window {
        &self.window
    }

    /// Returns whether the document is an HTML document.
    #[inline]
    pub(crate) fn is_html_document(&self) -> bool {
        self.is_html_document
    }

    /// Returns whether the document is an XHTML document.
    pub(crate) fn is_xhtml_document(&self) -> bool {
        self.content_type.matches(APPLICATION, "xhtml+xml")
    }

    /// Sets the HTTPS state of the document.
    pub(crate) fn set_https_state(&self, https_state: HttpsState) {
        self.https_state.set(https_state);
    }

    /// Returns whether the document is fully active.
    pub(crate) fn is_fully_active(&self) -> bool {
        self.activity.get() == DocumentActivity::FullyActive
    }

    /// Returns whether the document is active.
    pub(crate) fn is_active(&self) -> bool {
        self.activity.get() != DocumentActivity::Inactive
    }

    /// Sets the activity state of the document.
    pub(crate) fn set_activity(&self, activity: DocumentActivity, can_gc: CanGc) {
        // This function should only be called on documents with a browsing context
        assert!(self.has_browsing_context);
        if activity == self.activity.get() {
            return;
        }

        // Set the document's activity level, reflow if necessary, and suspend or resume timers.
        self.activity.set(activity);
        let media = ServoMedia::get();
        let pipeline_id = self.window().pipeline_id();
        let client_context_id =
            ClientContextId::build(pipeline_id.namespace_id.0, pipeline_id.index.0.get());

        if activity != DocumentActivity::FullyActive {
            self.window().suspend(can_gc);
            media.suspend(&client_context_id);
            return;
        }

        self.title_changed();
        self.dirty_all_nodes();
        self.window().resume(can_gc);
        media.resume(&client_context_id);

        if self.ready_state.get() != DocumentReadyState::Complete {
            return;
        }

        // This step used to be Step 4.6 in html.spec.whatwg.org/multipage/#history-traversal
        // But it's now Step 4 in https://html.spec.whatwg.org/multipage/#reactivate-a-document
        // TODO: See #32687 for more information.
        let document = Trusted::new(self);
        self.owner_global()
            .task_manager()
            .dom_manipulation_task_source()
            .queue(task!(fire_pageshow_event: move || {
                let document = document.root();
                let window = document.window();
                // Step 4.6.1
                if document.page_showing.get() { return; }
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
                let event = event.upcast::<Event>();
                event.set_trusted(true);
                window.dispatch_event_with_target_override(event, CanGc::note());
            }))
    }

    /// Returns the origin of the document.
    pub(crate) fn origin(&self) -> &MutableOrigin {
        &self.origin
    }

    /// <https://dom.spec.whatwg.org/#concept-document-url>
    pub(crate) fn url(&self) -> ServoUrl {
        self.url.borrow().clone()
    }

    /// Sets the URL of the document.
    pub(crate) fn set_url(&self, url: ServoUrl) {
        *self.url.borrow_mut() = url;
    }

    /// <https://html.spec.whatwg.org/multipage/#fallback-base-url>
    pub(crate) fn fallback_base_url(&self) -> ServoUrl {
        let document_url = self.url();
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

    /// <https://html.spec.whatwg.org/multipage/#document-base-url>
    pub(crate) fn base_url(&self) -> ServoUrl {
        match self.base_element() {
            // Step 1.
            None => self.fallback_base_url(),
            // Step 2.
            Some(base) => base.frozen_base_url(),
        }
    }

    /// Sets whether the document needs to be painted.
    pub(crate) fn set_needs_paint(&self, value: bool) {
        self.needs_paint.set(value)
    }

    /// Returns whether the document needs to be reflowed.
    pub(crate) fn needs_reflow(&self) -> Option<ReflowTriggerCondition> {
        // FIXME: This should check the dirty bit on the document,
        // not the document element. Needs some layout changes to make
        // that workable.
        if self.stylesheets.borrow().has_changed() {
            return Some(ReflowTriggerCondition::StylesheetsChanged);
        }

        let root = self.GetDocumentElement()?;
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

    /// Returns the first `base` element in the DOM that has an `href` attribute.
    pub(crate) fn base_element(&self) -> Option<DomRoot<HTMLBaseElement>> {
        self.base_element.get()
    }

    /// Refresh the cached first base element in the DOM.
    /// <https://github.com/w3c/web-platform-tests/issues/2122>
    pub(crate) fn refresh_base_element(&self) {
        let base = self
            .upcast::<Node>()
            .traverse_preorder(ShadowIncluding::No)
            .filter_map(DomRoot::downcast::<HTMLBaseElement>)
            .find(|element| {
                element
                    .upcast::<Element>()
                    .has_attribute(&local_name!("href"))
            });
        self.base_element.set(base.as_deref());
    }

    /// Returns the number of elements in the DOM tree.
    pub(crate) fn dom_count(&self) -> u32 {
        self.dom_count.get()
    }

    /// This is called by `bind_to_tree` when a node is added to the DOM.
    /// The internal count is used by layout to determine whether to be sequential or parallel.
    /// (it's sequential for small DOMs)
    pub(crate) fn increment_dom_count(&self) {
        self.dom_count.set(self.dom_count.get() + 1);
    }

    /// This is called by `unbind_from_tree` when a node is removed from the DOM.
    pub(crate) fn decrement_dom_count(&self) {
        self.dom_count.set(self.dom_count.get() - 1);
    }

    /// Returns the quirks mode of the document.
    pub(crate) fn quirks_mode(&self) -> QuirksMode {
        self.quirks_mode.get()
    }

    /// Sets the quirks mode of the document.
    pub(crate) fn set_quirks_mode(&self, new_mode: QuirksMode) {
        let old_mode = self.quirks_mode.replace(new_mode);

        if old_mode != new_mode {
            self.window.layout_mut().set_quirks_mode(new_mode);
        }
    }

    /// Returns the encoding of the document.
    pub(crate) fn encoding(&self) -> &'static Encoding {
        self.encoding.get()
    }

    /// Sets the encoding of the document.
    pub(crate) fn set_encoding(&self, encoding: &'static Encoding) {
        self.encoding.set(encoding);
    }

    /// Notifies the document that the content and heritage of a node have changed.
    pub(crate) fn content_and_heritage_changed(&self, node: &Node) {
        if node.is_connected() {
            node.note_dirty_descendants();
        }

        // FIXME(emilio): This is very inefficient, ideally the flag above would
        // be enough and incremental layout could figure out from there.
        node.dirty(NodeDamage::Other);
    }

    /// Remove any existing association between the provided id and any elements in this document.
    pub(crate) fn unregister_element_id(&self, to_unregister: &Element, id: Atom, can_gc: CanGc) {
        self.document_or_shadow_root
            .unregister_named_element(&self.id_map, to_unregister, &id);
        self.reset_form_owner_for_listeners(&id, can_gc);
    }

    /// Associate an element present in this document with the provided id.
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
        self.reset_form_owner_for_listeners(&id, can_gc);
    }

    /// Remove any existing association between the provided name and any elements in this document.
    pub(crate) fn unregister_element_name(&self, to_unregister: &Element, name: Atom) {
        self.document_or_shadow_root
            .unregister_named_element(&self.name_map, to_unregister, &name);
    }

    /// Associate an element present in this document with the provided name.
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

    /// Registers a form ID listener.
    pub(crate) fn register_form_id_listener<T: ?Sized + FormControl>(
        &self,
        id: DOMString,
        listener: &T,
    ) {
        let mut map = self.form_id_listener_map.borrow_mut();
        let listener = listener.to_element();
        let set = map.entry(Atom::from(id)).or_default();
        set.insert(Dom::from_ref(listener));
    }

    /// Unregisters a form ID listener.
    pub(crate) fn unregister_form_id_listener<T: ?Sized + FormControl>(
        &self,
        id: DOMString,
        listener: &T,
    ) {
        let mut map = self.form_id_listener_map.borrow_mut();
        if let Occupied(mut entry) = map.entry(Atom::from(id)) {
            entry
                .get_mut()
                .remove(&Dom::from_ref(listener.to_element()));
            if entry.get().is_empty() {
                entry.remove();
            }
        }
    }

    /// Attempt to find a named element in this page's document.
    /// <https://html.spec.whatwg.org/multipage/#the-indicated-part-of-the-document>
    pub(crate) fn find_fragment_node(&self, fragid: &str) -> Option<DomRoot<Element>> {
        // Step 1 is not handled here; the fragid is already obtained by the calling function
        // Step 2: Simply use None to indicate the top of the document.
        // Step 3 & 4
        percent_decode(fragid.as_bytes())
            .decode_utf8()
            .ok()
            // Step 5
            .and_then(|decoded_fragid| self.get_element_by_id(&Atom::from(decoded_fragid)))
            // Step 6
            .or_else(|| self.get_anchor_by_name(fragid))
        // Step 7 & 8
    }

    /// Scroll to the target element, and when we do not find a target
    /// and the fragment is empty or "top", scroll to the top.
    /// <https://html.spec.whatwg.org/multipage/#scroll-to-the-fragment-identifier>
    pub(crate) fn check_and_scroll_fragment(&self, fragment: &str, can_gc: CanGc) {
        let target = self.find_fragment_node(fragment);

        // Step 1
        self.set_target_element(target.as_deref());

        let point = target
            .as_ref()
            .map(|element| {
                // TODO: This strategy is completely wrong if the element we are scrolling to in
                // inside other scrollable containers. Ideally this should use an implementation of
                // `scrollIntoView` when that is available:
                // See https://github.com/servo/servo/issues/24059.
                let rect = element
                    .upcast::<Node>()
                    .bounding_content_box_or_zero(can_gc);

                // In order to align with element edges, we snap to unscaled pixel boundaries, since
                // the paint thread currently does the same for drawing elements. This is important
                // for pages that require pixel perfect scroll positioning for proper display
                // (like Acid2).
                let device_pixel_ratio = self.window.device_pixel_ratio().get();
                (
                    rect.origin.x.to_nearest_pixel(device_pixel_ratio),
                    rect.origin.y.to_nearest_pixel(device_pixel_ratio),
                )
            })
            .or_else(|| {
                if fragment.is_empty() || fragment.eq_ignore_ascii_case("top") {
                    // FIXME(stshine): this should be the origin of the stacking context space,
                    // which may differ under the influence of writing mode.
                    Some((0.0, 0.0))
                } else {
                    None
                }
            });

        if let Some((x, y)) = point {
            self.window
                .scroll(x as f64, y as f64, ScrollBehavior::Instant, can_gc)
        }
    }

    /// Returns the anchor element with the given name.
    fn get_anchor_by_name(&self, name: &str) -> Option<DomRoot<Element>> {
        let name = Atom::from(name);
        self.name_map.borrow().get(&name).and_then(|elements| {
            elements
                .iter()
                .find(|e| e.is::<HTMLAnchorElement>())
                .map(|e| DomRoot::from_ref(&**e))
        })
    }

    // https://html.spec.whatwg.org/multipage/#current-document-readiness
    pub(crate) fn set_ready_state(&self, state: DocumentReadyState, can_gc: CanGc) {
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
                update_with_current_instant(&self.dom_complete);
            },
            DocumentReadyState::Interactive => update_with_current_instant(&self.dom_interactive),
        };

        self.ready_state.set(state);

        self.upcast::<EventTarget>()
            .fire_event(atom!("readystatechange"), can_gc);
    }

    /// Return whether scripting is enabled or not
    pub(crate) fn is_scripting_enabled(&self) -> bool {
        self.scripting_enabled
    }

    /// Return whether scripting is enabled or not
    /// <https://html.spec.whatwg.org/multipage/#concept-n-noscript>
    pub(crate) fn scripting_enabled(&self) -> bool {
        self.has_browsing_context()
    }

    /// Return the element that currently has focus.
    // https://w3c.github.io/uievents/#events-focusevent-doc-focus
    pub(crate) fn get_focused_element(&self) -> Option<DomRoot<Element>> {
        self.focused.get()
    }

    /// Get the last sequence number sent to the constellation.
    ///
    /// Received focus-related messages with sequence numbers less than the one
    /// returned by this method must be discarded.
    pub fn get_focus_sequence(&self) -> FocusSequenceNumber {
        self.focus_sequence.get()
    }

    /// Generate the next sequence number for focus-related messages.
    fn increment_fetch_focus_sequence(&self) -> FocusSequenceNumber {
        self.focus_sequence.set(FocusSequenceNumber(
            self.focus_sequence
                .get()
                .0
                .checked_add(1)
                .expect("too many focus messages have been sent"),
        ));
        self.focus_sequence.get()
    }

    /// Initiate a new round of checking for elements requesting focus. The last element to call
    /// `request_focus` before `commit_focus_transaction` is called will receive focus.
    fn begin_focus_transaction(&self) {
        // Initialize it with the current state
        *self.focus_transaction.borrow_mut() = Some(FocusTransaction {
            element: self.focused.get().as_deref().map(Dom::from_ref),
            has_focus: self.has_focus.get(),
        });
    }

    /// <https://html.spec.whatwg.org/multipage/#focus-fixup-rule>
    pub(crate) fn perform_focus_fixup_rule(&self, not_focusable: &Element, can_gc: CanGc) {
        // Return if `not_focusable` is not the designated focused area of the
        // `Document`.
        if Some(not_focusable) != self.focused.get().as_deref() {
            return;
        }

        let implicit_transaction = self.focus_transaction.borrow().is_none();

        if implicit_transaction {
            self.begin_focus_transaction();
        }

        // Designate the viewport as the new focused area of the `Document`, but
        // do not run the focusing steps.
        {
            let mut focus_transaction = self.focus_transaction.borrow_mut();
            focus_transaction.as_mut().unwrap().element = None;
        }

        if implicit_transaction {
            self.commit_focus_transaction(FocusInitiator::Local, can_gc);
        }
    }

    /// Request that the given element receive focus once the current
    /// transaction is complete. `None` specifies to focus the document.
    ///
    /// If there's no ongoing transaction, this method automatically starts and
    /// commits an implicit transaction.
    pub(crate) fn request_focus(
        &self,
        elem: Option<&Element>,
        focus_initiator: FocusInitiator,
        can_gc: CanGc,
    ) {
        // If an element is specified, and it's non-focusable, ignore the
        // request.
        if elem.is_some_and(|e| !e.is_focusable_area()) {
            return;
        }

        let implicit_transaction = self.focus_transaction.borrow().is_none();

        if implicit_transaction {
            self.begin_focus_transaction();
        }

        {
            let mut focus_transaction = self.focus_transaction.borrow_mut();
            let focus_transaction = focus_transaction.as_mut().unwrap();
            focus_transaction.element = elem.map(Dom::from_ref);
            focus_transaction.has_focus = true;
        }

        if implicit_transaction {
            self.commit_focus_transaction(focus_initiator, can_gc);
        }
    }

    /// Update the local focus state accordingly after being notified that the
    /// document's container is removed from the top-level browsing context's
    /// focus chain (not considering system focus).
    pub(crate) fn handle_container_unfocus(&self, can_gc: CanGc) {
        assert!(
            self.window().parent_info().is_some(),
            "top-level document cannot be unfocused",
        );

        // Since this method is called from an event loop, there mustn't be
        // an in-progress focus transaction
        assert!(
            self.focus_transaction.borrow().is_none(),
            "there mustn't be an in-progress focus transaction at this point"
        );

        // Start an implicit focus transaction
        self.begin_focus_transaction();

        // Update the transaction
        {
            let mut focus_transaction = self.focus_transaction.borrow_mut();
            focus_transaction.as_mut().unwrap().has_focus = false;
        }

        // Commit the implicit focus transaction
        self.commit_focus_transaction(FocusInitiator::Remote, can_gc);
    }

    /// Reassign the focus context to the element that last requested focus during this
    /// transaction, or the document if no elements requested it.
    fn commit_focus_transaction(&self, focus_initiator: FocusInitiator, can_gc: CanGc) {
        let (mut new_focused, new_focus_state) = {
            let focus_transaction = self.focus_transaction.borrow();
            let focus_transaction = focus_transaction
                .as_ref()
                .expect("no focus transaction in progress");
            (
                focus_transaction
                    .element
                    .as_ref()
                    .map(|e| DomRoot::from_ref(&**e)),
                focus_transaction.has_focus,
            )
        };
        *self.focus_transaction.borrow_mut() = None;

        if !new_focus_state {
            // In many browsers, a document forgets its focused area when the
            // document is removed from the top-level BC's focus chain
            if new_focused.take().is_some() {
                trace!(
                    "Forgetting the document's focused area because the \ 
                    document's container was removed from the top-level BC's \ 
                    focus chain"
                );
            }
        }

        let old_focused = self.focused.get();
        let old_focus_state = self.has_focus.get();

        debug!(
            "Committing focus transaction: {:?} → {:?}",
            (&old_focused, old_focus_state),
            (&new_focused, new_focus_state),
        );

        // `*_focused_filtered` indicates the local element (if any) included in
        // the top-level BC's focus chain.
        let old_focused_filtered = old_focused.as_ref().filter(|_| old_focus_state);
        let new_focused_filtered = new_focused.as_ref().filter(|_| new_focus_state);

        let trace_focus_chain = |name, element, doc| {
            trace!(
                "{} local focus chain: {}",
                name,
                match (element, doc) {
                    (Some(e), _) => format!("[{:?}, document]", e),
                    (None, true) => "[document]".to_owned(),
                    (None, false) => "[]".to_owned(),
                }
            );
        };

        trace_focus_chain("Old", old_focused_filtered, old_focus_state);
        trace_focus_chain("New", new_focused_filtered, new_focus_state);

        if old_focused_filtered != new_focused_filtered {
            if let Some(elem) = &old_focused_filtered {
                let node = elem.upcast::<Node>();
                elem.set_focus_state(false);
                // FIXME: pass appropriate relatedTarget
                if node.is_connected() {
                    self.fire_focus_event(FocusEventType::Blur, node.upcast(), None, can_gc);
                }

                // Notify the embedder to hide the input method.
                if elem.input_method_type().is_some() {
                    self.send_to_embedder(EmbedderMsg::HideIME(self.webview_id()));
                }
            }
        }

        if old_focus_state != new_focus_state && !new_focus_state {
            self.fire_focus_event(FocusEventType::Blur, self.global().upcast(), None, can_gc);
        }

        self.focused.set(new_focused.as_deref());
        self.has_focus.set(new_focus_state);

        if old_focus_state != new_focus_state && new_focus_state {
            self.fire_focus_event(FocusEventType::Focus, self.global().upcast(), None, can_gc);
        }

        if old_focused_filtered != new_focused_filtered {
            if let Some(elem) = &new_focused_filtered {
                elem.set_focus_state(true);
                let node = elem.upcast::<Node>();
                // FIXME: pass appropriate relatedTarget
                self.fire_focus_event(FocusEventType::Focus, node.upcast(), None, can_gc);

                // Notify the embedder to display an input method.
                if let Some(kind) = elem.input_method_type() {
                    let rect = elem.upcast::<Node>().bounding_content_box_or_zero(can_gc);
                    let rect = Rect::new(
                        Point2D::new(rect.origin.x.to_px(), rect.origin.y.to_px()),
                        Size2D::new(rect.size.width.to_px(), rect.size.height.to_px()),
                    );
                    let (text, multiline) = if let Some(input) = elem.downcast::<HTMLInputElement>()
                    {
                        (
                            Some((
                                (input.Value()).to_string(),
                                input.GetSelectionEnd().unwrap_or(0) as i32,
                            )),
                            false,
                        )
                    } else if let Some(textarea) = elem.downcast::<HTMLTextAreaElement>() {
                        (
                            Some((
                                (textarea.Value()).to_string(),
                                textarea.GetSelectionEnd().unwrap_or(0) as i32,
                            )),
                            true,
                        )
                    } else {
                        (None, false)
                    };
                    self.send_to_embedder(EmbedderMsg::ShowIME(
                        self.webview_id(),
                        kind,
                        text,
                        multiline,
                        DeviceIntRect::from_untyped(&rect.to_box2d()),
                    ));
                }
            }
        }

        if focus_initiator != FocusInitiator::Local {
            return;
        }

        // We are the initiator of the focus operation, so we must broadcast
        // the change we intend to make.
        match (old_focus_state, new_focus_state) {
            (_, true) => {
                // Advertise the change in the focus chain.
                // <https://html.spec.whatwg.org/multipage/#focus-chain>
                // <https://html.spec.whatwg.org/multipage/#focusing-steps>
                //
                // If the top-level BC doesn't have system focus, this won't
                // have an immediate effect, but it will when we gain system
                // focus again. Therefore we still have to send `ScriptMsg::
                // Focus`.
                //
                // When a container with a non-null nested browsing context is
                // focused, its active document becomes the focused area of the
                // top-level browsing context instead. Therefore we need to let
                // the constellation know if such a container is focused.
                //
                // > The focusing steps for an object `new focus target` [...]
                // >
                // >  3. If `new focus target` is a browsing context container
                // >     with non-null nested browsing context, then set
                // >     `new focus target` to the nested browsing context's
                // >     active document.
                let child_browsing_context_id = new_focused
                    .as_ref()
                    .and_then(|elem| elem.downcast::<HTMLIFrameElement>())
                    .and_then(|iframe| iframe.browsing_context_id());

                let sequence = self.increment_fetch_focus_sequence();

                debug!(
                    "Advertising the focus request to the constellation \ 
                        with sequence number {} and child BC ID {}",
                    sequence,
                    child_browsing_context_id
                        .as_ref()
                        .map(|id| id as &dyn std::fmt::Display)
                        .unwrap_or(&"(none)"),
                );

                self.window()
                    .send_to_constellation(ScriptToConstellationMessage::Focus(
                        child_browsing_context_id,
                        sequence,
                    ));
            },
            (false, false) => {
                // Our `Document` doesn't have focus, and we intend to keep it
                // this way.
            },
            (true, false) => {
                unreachable!(
                    "Can't lose the document's focus without specifying \ 
                    another one to focus"
                );
            },
        }
    }

    /// Handles any updates when the document's title has changed.
    pub(crate) fn title_changed(&self) {
        if self.browsing_context().is_some() {
            self.send_title_to_embedder();
            let title = String::from(self.Title());
            self.window
                .send_to_constellation(ScriptToConstellationMessage::TitleChanged(
                    self.window.pipeline_id(),
                    title.clone(),
                ));
            if let Some(chan) = self.window.as_global_scope().devtools_chan() {
                let _ = chan.send(ScriptToDevtoolsControlMsg::TitleChanged(
                    self.window.pipeline_id(),
                    title,
                ));
            }
        }
    }

    /// Determine the title of the [`Document`] according to the specification at:
    /// <https://html.spec.whatwg.org/multipage/#document.title>. The difference
    /// here is that when the title isn't specified `None` is returned.
    fn title(&self) -> Option<DOMString> {
        let title = self.GetDocumentElement().and_then(|root| {
            if root.namespace() == &ns!(svg) && root.local_name() == &local_name!("svg") {
                // Step 1.
                root.upcast::<Node>()
                    .child_elements()
                    .find(|node| {
                        node.namespace() == &ns!(svg) && node.local_name() == &local_name!("title")
                    })
                    .map(DomRoot::upcast::<Node>)
            } else {
                // Step 2.
                root.upcast::<Node>()
                    .traverse_preorder(ShadowIncluding::No)
                    .find(|node| node.is::<HTMLTitleElement>())
            }
        });

        title.map(|title| {
            // Steps 3-4.
            let value = title.child_text_content();
            DOMString::from(str_join(split_html_space_chars(&value), " "))
        })
    }

    /// Sends this document's title to the constellation.
    pub(crate) fn send_title_to_embedder(&self) {
        let window = self.window();
        if window.is_top_level() {
            let title = self.title().map(String::from);
            self.send_to_embedder(EmbedderMsg::ChangePageTitle(self.webview_id(), title));
        }
    }

    /// Sends a message to the embedder.
    pub(crate) fn send_to_embedder(&self, msg: EmbedderMsg) {
        let window = self.window();
        window.send_to_embedder(msg);
    }

    /// Marks all nodes in the document as dirty.
    pub(crate) fn dirty_all_nodes(&self) {
        let root = match self.GetDocumentElement() {
            Some(root) => root,
            None => return,
        };
        for node in root
            .upcast::<Node>()
            .traverse_preorder(ShadowIncluding::Yes)
        {
            node.dirty(NodeDamage::Other)
        }
    }

    /// Handles a mouse button event.
    #[allow(unsafe_code)]
    pub(crate) fn handle_mouse_button_event(
        &self,
        event: MouseButtonEvent,
        hit_test_result: Option<CompositorHitTestResult>,
        pressed_mouse_buttons: u16,
        can_gc: CanGc,
    ) {
        // Ignore all incoming events without a hit test.
        let Some(hit_test_result) = hit_test_result else { return };

        debug!(
            "{:#?}: at {:?}",
            event.action,
            hit_test_result.point_in_viewport
        );

        let node = unsafe { node::from_untrusted_node_address(hit_test_result.node) };
        let Some(el) = node
            .inclusive_ancestors(ShadowIncluding::Yes)
            .filter_map(DomRoot::downcast::<Element>)
            .next()
        else { return };

        let node = el.upcast::<Node>();
        debug!("{:?} on {:?}", event.action, node.debug_str());
        // Prevent click event if form control element is disabled.
        if let MouseButtonAction::Click = event.action {
            // The click event is filtered by the disabled state.
            if el.is_actually_disabled() {
                return;
            }

            self.begin_focus_transaction();
            // Try to focus `el`. If it's not focusable, focus the document
            // instead.
            self.request_focus(None, FocusInitiator::Local, can_gc);
            self.request_focus(Some(&*el), FocusInitiator::Local, can_gc);
        }

        let dom_event = DomRoot::upcast::<Event>(MouseEvent::for_platform_mouse_event(
            event,
            pressed_mouse_buttons,
            &self.window,
            &hit_test_result,
            can_gc,
        ));

        // https://html.spec.whatwg.org/multipage/#run-authentic-click-activation-steps
        let activatable = el.as_maybe_activatable();
        match event.action {
            MouseButtonAction::Click => {
                el.set_click_in_progress(true);
                dom_event.fire(node.upcast(), can_gc);
                el.set_click_in_progress(false);
            },
            MouseButtonAction::Down => {
                if let Some(a) = activatable {
                    a.enter_formal_activation_state();
                }

                let target = node.upcast();
                dom_event.fire(target, can_gc);
            },
            MouseButtonAction::Up => {
                if let Some(a) = activatable {
                    a.exit_formal_activation_state();
                }

                let target = node.upcast();
                dom_event.fire(target, can_gc);
            },
        }

        if let MouseButtonAction::Click = event.action {
            if self.focus_transaction.borrow().is_some() {
                self.commit_focus_transaction(FocusInitiator::Local, can_gc);
            }
            self.maybe_fire_dblclick(
                hit_test_result.point_in_viewport,
                node,
                pressed_mouse_buttons,
                can_gc,
            );
        }

        // When the contextmenu event is triggered by right mouse button
        // the contextmenu event MUST be dispatched after the mousedown event.
        if let (MouseButtonAction::Down, MouseButton::Right) = (event.action, event.button) {
            self.maybe_show_context_menu(
                node.upcast(),
                pressed_mouse_buttons,
                hit_test_result.point_in_viewport,
                can_gc,
            );
        }
    }

    /// <https://www.w3.org/TR/uievents/#maybe-show-context-menu>
    fn maybe_show_context_menu(
        &self,
        target: &EventTarget,
        pressed_mouse_buttons: u16,
        client_point: Point2D<f32>,
        can_gc: CanGc,
    ) {
        let client_x = client_point.x.to_i32().unwrap_or(0);
        let client_y = client_point.y.to_i32().unwrap_or(0);

        // <https://w3c.github.io/uievents/#contextmenu>
        let menu_event = PointerEvent::new(
            &self.window,                   // window
            DOMString::from("contextmenu"), // type
            EventBubbles::Bubbles,          // can_bubble
            EventCancelable::Cancelable,    // cancelable
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
        let event = menu_event.upcast::<Event>();
        event.fire(target, can_gc);

        // if the event was not canceled, notify the embedder to show the context menu
        if event.status() == EventStatus::NotCanceled {
            let (sender, receiver) =
                ipc::channel::<ContextMenuResult>().expect("Failed to create IPC channel.");
            self.send_to_embedder(EmbedderMsg::ShowContextMenu(
                self.webview_id(),
                sender,
                None,
                vec![],
            ));
            let _ = receiver.recv().unwrap();
        };
    }

    /// Fires a dblclick event if the conditions are met.
    fn maybe_fire_dblclick(
        &self,
        click_pos: Point2D<f32>,
        target: &Node,
        pressed_mouse_buttons: u16,
        can_gc: CanGc,
    ) {
        // https://w3c.github.io/uievents/#event-type-dblclick
        let now = Instant::now();

        let opt = self.last_click_info.borrow_mut().take();

        if let Some((last_time, last_pos)) = opt {
            let DBL_CLICK_TIMEOUT =
                Duration::from_millis(pref!(dom_document_dblclick_timeout) as u64);
            let DBL_CLICK_DIST_THRESHOLD = pref!(dom_document_dblclick_dist) as u64;

            // Calculate distance between this click and the previous click.
            let line = click_pos - last_pos;
            let dist = (line.dot(line) as f64).sqrt();

            if now.duration_since(last_time) < DBL_CLICK_TIMEOUT &&
                dist < DBL_CLICK_DIST_THRESHOLD as f64
            {
                // A double click has occurred if this click is within a certain time and dist. of previous click.
                let click_count = 2;
                let client_x = click_pos.x as i32;
                let client_y = click_pos.y as i32;

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
                event.upcast::<Event>().fire(target.upcast(), can_gc);

                // When a double click occurs, self.last_click_info is left as None so that a
                // third sequential click will not cause another double click.
                return;
            }
        }

        // Update last_click_info with the time and position of the click.
        *self.last_click_info.borrow_mut() = Some((now, click_pos));
    }

    /// Fires a mouse event.
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
        let client_x = client_point.x.to_i32().unwrap_or(0);
        let client_y = client_point.y.to_i32().unwrap_or(0);

        MouseEvent::new(
            &self.window,
            DOMString::from(event_name.as_str()),
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
        .upcast::<Event>()
        .fire(target, can_gc);
    }

    /// Handles an editing action.
    pub(crate) fn handle_editing_action(&self, action: EditingActionEvent, can_gc: CanGc) -> bool {
        let clipboard_event = match action {
            EditingActionEvent::Copy => ClipboardEventType::Copy,
            EditingActionEvent::Cut => ClipboardEventType::Cut,
            EditingActionEvent::Paste => ClipboardEventType::Paste,
        };
        self.handle_clipboard_action(clipboard_event, can_gc)
    }

    /// <https://www.w3.org/TR/clipboard-apis/#clipboard-actions>
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
            DOMString::from(action.as_str()),
            EventBubbles::Bubbles,
            EventCancelable::Cancelable,
            None,
            can_gc,
        );
        self.fire_clipboard_event(&event, action, can_gc);

        // Step 3 If a script doesn't call preventDefault()
        // the event will be handled inside target's VirtualMethods::handle_event

        let e = event.upcast::<Event>();

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
                        self.write_content_to_the_clipboard(&drag_data_store);
                    }
                },
                "cut" => {
                    // Step 4.1 Call the write content to the clipboard algorithm,
                    // passing on the DataTransferItemList items, a clear-was-called flag and a types-to-clear list.
                    if let Some(clipboard_data) = event.get_clipboard_data() {
                        let drag_data_store =
                            clipboard_data.data_store().expect("This shouldn't fail");
                        self.write_content_to_the_clipboard(&drag_data_store);
                    }

                    // Step 4.2 Fire a clipboard event named clipboardchange
                    self.fire_clipboardchange_event(can_gc);
                },
                "paste" => return false,
                _ => (),
            }
        }
        //Step 5
        true
    }

    /// <https://www.w3.org/TR/clipboard-apis/#fire-a-clipboard-event>
    fn fire_clipboard_event(
        &self,
        event: &ClipboardEvent,
        action: ClipboardEventType,
        can_gc: CanGc,
    ) {
        // Step 1 Let clear_was_called be false
        // Step 2 Let types_to_clear an empty list
        let mut drag_data_store = DragDataStore::new();

        // Step 4 let clipboard-entry be the sequence number of clipboard content, null if the OS doesn't support it.

        // Step 5 let trusted be true if the event is generated by the user agent, false otherwise
        let trusted = true;

        // Step 6 if the context is editable:
        let focused = self.get_focused_element();
        let body = self.GetBody();

        let target = match (&focused, &body) {
            (Some(focused), _) => focused.upcast(),
            (&None, Some(body)) => body.upcast(),
            (&None, &None) => self.window.upcast(),
        };
        // Step 6.2 else TODO require Selection see https://github.com/w3c/clipboard-apis/issues/70

        // Step 7
        match action {
            ClipboardEventType::Copy | ClipboardEventType::Cut => {
                // Step 7.2.1
                drag_data_store.set_mode(Mode::ReadWrite);
            },
            ClipboardEventType::Paste => {
                let (sender, receiver) = ipc::channel().unwrap();
                self.window
                    .send_to_constellation(ScriptToConstellationMessage::ForwardToEmbedder(
                        EmbedderMsg::GetClipboardText(self.window.webview_id(), sender),
                    ));
                let text_contents = receiver
                    .recv()
                    .map(Result::unwrap_or_default)
                    .unwrap_or_default();

                // Step 7.1.1
                drag_data_store.set_mode(Mode::ReadOnly);
                // Step 7.1.2 If trusted or the implementation gives script-generated events access to the clipboard
                if trusted {
                    // Step 7.1.2.1 For each clipboard-part on the OS clipboard:

                    // Step 7.1.2.1.1 If clipboard-part contains plain text, then
                    let data = DOMString::from(text_contents.to_string());
                    let type_ = DOMString::from("text/plain");
                    let _ = drag_data_store.add(Kind::Text { data, type_ });

                    // Step 7.1.2.1.2 TODO If clipboard-part represents file references, then for each file reference
                    // Step 7.1.2.1.3 TODO If clipboard-part contains HTML- or XHTML-formatted text then

                    // Step 7.1.3 Update clipboard-event-data’s files to match clipboard-event-data’s items
                    // Step 7.1.4 Update clipboard-event-data’s types to match clipboard-event-data’s items
                }
            },
            ClipboardEventType::Change => (),
        }

        // Step 3
        let clipboard_event_data = DataTransfer::new(
            &self.window,
            Rc::new(RefCell::new(Some(drag_data_store))),
            can_gc,
        );

        // Step 8
        event.set_clipboard_data(Some(&clipboard_event_data));
        let event = event.upcast::<Event>();
        // Step 9
        event.set_trusted(trusted);
        // Step 10 Set event’s composed to true.
        event.set_composed(true);
        // Step 11
        event.dispatch(target, false, can_gc);
    }

    /// Fires a clipboardchange event.
    pub(crate) fn fire_clipboardchange_event(&self, can_gc: CanGc) {
        let clipboardchange_event = ClipboardEvent::new(
            &self.window,
            None,
            DOMString::from("clipboardchange"),
            EventBubbles::Bubbles,
            EventCancelable::Cancelable,
            None,
            can_gc,
        );
        self.fire_clipboard_event(&clipboardchange_event, ClipboardEventType::Change, can_gc);
    }

    /// <https://www.w3.org/TR/clipboard-apis/#write-content-to-the-clipboard>
    fn write_content_to_the_clipboard(&self, drag_data_store: &DragDataStore) {
        // Step 1
        if drag_data_store.list_len() > 0 {
            // Step 1.1 Clear the clipboard.
            self.send_to_embedder(EmbedderMsg::ClearClipboard(self.webview_id()));
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
                        ));
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
                self.send_to_embedder(EmbedderMsg::ClearClipboard(self.webview_id()));
                // Step 2.1.2 Else remove the types in the list from the clipboard
                // As of now this can't be done with Arboard, and it's possible that will be removed from the spec
            }
        }
    }

    /// Handles a mouse move event.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn handle_mouse_move_event(
        &self,
        hit_test_result: Option<CompositorHitTestResult>,
        pressed_mouse_buttons: u16,
        prev_mouse_over_target: &MutNullableDom<Element>,
        can_gc: CanGc,
    ) {
        // Ignore all incoming events without a hit test.
        let Some(hit_test_result) = hit_test_result else { return };

        let current_target =
            node::from_untrusted_node_address(hit_test_result.node).map(|node| {
                node.inclusive_ancestors(ShadowIncluding::Yes)
                    .filter_map(DomRoot::downcast::<Element>)
                    .next()
                    .map(|e| Dom::from_ref(&*e))
            });

        let prev_target = prev_mouse_over_target.get().map(|e| Dom::from_ref(&*e));
        prev_mouse_over_target.set(current_target.as_deref());

        if prev_target == current_target {
            if let Some(target) = current_target {
                self.fire_mouse_event(
                    hit_test_result.point_in_viewport.to_f32(),
                    target.upcast::<EventTarget>(),
                    FireMouseEventType::Move,
                    EventBubbles::Bubbles,
                    EventCancelable::Cancelable,
                    pressed_mouse_buttons,
                    can_gc,
                )
            }
            return;
        }

        if let Some(prev_target) = prev_target {
            let prev_target = DomRoot::from_ref(&*prev_target);
            let prev_target_node = prev_target.upcast::<Node>();
            if let Some(ref current_target) = current_target {
                let current_target = DomRoot::from_ref(&**current_target);
                let current_target_node = current_target.upcast::<Node>();

                let lca = prev_target_node.common_ancestor(current_target_node.upcast());
                let lca_target = lca.upcast::<EventTarget>();

                // Fire mouseout events.
                for target in prev_target_node
                    .inclusive_ancestors(ShadowIncluding::No)
                    .take_while(|n| *n != *lca)
                {
                    self.fire_mouse_event(
                        hit_test_result.point_in_viewport.to_f32(),
                        target.upcast(),
                        FireMouseEventType::Out,
                        EventBubbles::Bubbles,
                        EventCancelable::Cancelable,
                        pressed_mouse_buttons,
                        can_gc,
                    );
                }

                // Fire mouseover events.
                for target in current_target_node
                    .inclusive_ancestors(ShadowIncluding::No)
                    .take_while(|n| *n != *lca)
                {
                    self.fire_mouse_event(
                        hit_test_result.point_in_viewport.to_f32(),
                        target.upcast(),
                        FireMouseEventType::Over,
                        EventBubbles::Bubbles,
                        EventCancelable::Cancelable,
                        pressed_mouse_buttons,
                        can_gc,
                    );
                }

                self.fire_mouse_event(
                    hit_test_result.point_in_viewport.to_f32(),
                    prev_target.upcast(),
                    FireMouseEventType::Leave,
                    EventBubbles::DoesNotBubble,
                    EventCancelable::NotCancelable,
                    pressed_mouse_buttons,
                    can_gc,
                );
                self.fire_mouse_event(
                    hit_test_result.point_in_viewport.to_f32(),
                    current_target.upcast(),
                    FireMouseEventType::Enter,
                    EventBubbles::DoesNotBubble,
                    EventCancelable::NotCancelable,
                    pressed_mouse_buttons,
                    can_gc,
                );
                self.fire_mouse_event(
                    hit_test_result.point_in_viewport.to_f32(),
                    lca_target,
                    FireMouseEventType::Move,
                    EventBubbles::Bubbles,
                    EventCancelable::Cancelable,
                    pressed_mouse_buttons,
                    can_gc,
                );
            } else {
                for target in prev_target_node.inclusive_ancestors(ShadowIncluding::No) {
                    self.fire_mouse_event(
                        hit_test_result.point_in_viewport.to_f32(),
                        target.upcast(),
                        FireMouseEventType::Out,
                        EventBubbles::Bubbles,
                        EventCancelable::Cancelable,
                        pressed_mouse_buttons,
                        can_gc,
                    );
                }
                self.fire_mouse_event(
                    hit_test_result.point_in_viewport.to_f32(),
                    prev_target.upcast(),
                    FireMouseEventType::Leave,
                    EventBubbles::DoesNotBubble,
                    EventCancelable::NotCancelable,
                    pressed_mouse_buttons,
                    can_gc,
                );
            }
        } else if let Some(current_target) = current_target {
            let current_target = DomRoot::from_ref(&*current_target);
            for target in current_target.upcast::<Node>().inclusive_ancestors(ShadowIncluding::No) {
                self.fire_mouse_event(
                    hit_test_result.point_in_viewport.to_f32(),
                    target.upcast(),
                    FireMouseEventType::Over,
                    EventBubbles::Bubbles,
                    EventCancelable::Cancelable,
                    pressed_mouse_buttons,
                    can_gc,
                );
            }
            self.fire_mouse_event(
                hit_test_result.point_in_viewport.to_f32(),
                current_target.upcast(),
                FireMouseEventType::Enter,
                EventBubbles::DoesNotBubble,
                EventCancelable::NotCancelable,
                pressed_mouse_buttons,
                can_gc,
            );
        }
    }

    pub(crate) fn handle_mouse_leave_event(
        &self,
        _hit_test_result: Option<CompositorHitTestResult>,
        pressed_mouse_buttons: u16,
        can_gc: CanGc,
    ) {
        if let Some(target) = self.get_focused_element() {
            self.fire_mouse_event(
                Point2D::new(0., 0.),
                target.upcast(),
                FireMouseEventType::Out,
                EventBubbles::Bubbles,
                EventCancelable::Cancelable,
                pressed_mouse_buttons,
                can_gc,
            )
        }
    }

    pub(crate) fn handle_wheel_event(
        &self,
        event: WheelEvent,
        hit_test_result: Option<CompositorHitTestResult>,
        can_gc: CanGc,
    ) {
        let Some(hit_test_result) = hit_test_result else { return };

        // SAFETY: we know the node lives as long as the document.
        let Some(node) = (unsafe { node::from_untrusted_node_address(hit_test_result.node) })
        else { return };

        debug!("wheel event: node={:?}", node.debug_str());
        let dom_event = DomRoot::upcast::<Event>(DomWheelEvent::for_platform_wheel_event(
            event,
            &self.window,
            &hit_test_result,
            can_gc,
        ));
        dom_event.fire(node.upcast(), can_gc);
    }

    pub(crate) fn handle_touch_event(
        &self,
        event: TouchEvent,
        hit_test_result: Option<CompositorHitTestResult>,
        can_gc: CanGc,
    ) -> TouchEventResult {
        // FIXME: Can we filter this out earlier?
        // Note: we can't filter out events on non-touch-enabled nodes, since those
        // still need to be able to prevent default actions.
        if !self.has_touch_event_listeners() {
            return TouchEventResult::Processed(true);
        }

        let Some(hit_test_result) = hit_test_result else { return TouchEventResult::Processed(false) };

        let node = unsafe { node::from_untrusted_node_address(hit_test_result.node) };

        // FIXME: Touch events can be targeted at non-elements.
        let target = node
            .inclusive_ancestors(ShadowIncluding::Yes)
            .filter_map(DomRoot::downcast::<Element>)
            .next();
        let Some(target) = target else { return TouchEventResult::Processed(false) };
        let target = target.upcast();

        let event_type = match event.event_type {
            TouchEventType::Down => {
                atom!("touchstart")
            },
            TouchEventType::Up => {
                atom!("touchend")
            },
            TouchEventType::Move => {
                atom!("touchmove")
            },
            TouchEventType::Cancel => {
                atom!("touchcancel")
            },
        };

        let active_touch_points = self.active_touch_points.borrow();
        let touches: Vec<DomRoot<Touch>> = active_touch_points
            .iter()
            .map(|t| DomRoot::from_ref(&**t))
            .collect();
        let touches = TouchList::from(touches);

        // All changed touches have same event target according to w3c spec
        // TouchEvent.changedTouches returns a list of all the Touch objects representing individual points of contact whose states changed between the previous touch event and this one.
        // It should be safe to grab target element from first changed touch
        let changed_touches: Vec<DomRoot<Touch>> = event
            .changed_touches
            .into_iter()
            .map(|t| {
                Touch::new(
                    &self.window,
                    t.identifier as i32,
                    target,
                    t.client.x,
                    t.client.y,
                    t.page.x,
                    t.page.y,
                    t.screen.x,
                    t.screen.y,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    can_gc,
                )
            })
            .collect();
        let changed_touches = TouchList::from(changed_touches);

        // targetTouches is a list of all the Touch objects that are both currently in contact with the touch surface and were also started on the element that is the target of the event
        let target_touches: Vec<DomRoot<Touch>> = active_touch_points
            .iter()
            .filter(|t| t.get_target() == target.upcast::<EventTarget>())
            .map(|t| DomRoot::from_ref(&**t))
            .collect();
        let target_touches = TouchList::from(target_touches);

        let dom_event = DomTouchEvent::new(
            &self.window,
            &event_type,
            EventBubbles::Bubbles,
            EventCancelable::Cancelable,
            Some(&self.window),
            0,
            false,
            false,
            false,
            false,
            touches,
            target_touches,
            changed_touches,
            can_gc,
        );

        let event = dom_event.upcast::<Event>();
        let result = event.fire(target, can_gc);
        TouchEventResult::Processed(!result)
    }

    fn update_active_touch_points(&self, event: TouchEvent, can_gc: CanGc) {
        let target = self
            .get_focused_element()
            .unwrap_or(self.GetBody().unwrap())
            .upcast();
        match event.event_type {
            TouchEventType::Down => {
                for touch in event.changed_touches {
                    let new_touch = Touch::new(
                        &self.window,
                        touch.identifier as i32,
                        target,
                        touch.client.x,
                        touch.client.y,
                        touch.page.x,
                        touch.page.y,
                        touch.screen.x,
                        touch.screen.y,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        can_gc,
                    );
                    self.active_touch_points
                        .borrow_mut()
                        .push(Dom::from_ref(&*new_touch));
                }
            },
            TouchEventType::Up | TouchEventType::Cancel => {
                for touch in event.changed_touches {
                    self.active_touch_points
                        .borrow_mut()
                        .retain(|t| t.get_identifier() != touch.identifier as i32);
                }
            },
            TouchEventType::Move => {
                for touch in event.changed_touches {
                    if let Some(t) = self
                        .active_touch_points
                        .borrow_mut()
                        .iter_mut()
                        .find(|t| t.get_identifier() == touch.identifier as i32)
                    {
                        TouchMethods::update_coordinates(
                            t,
                            touch.client.x,
                            touch.client.y,
                            touch.page.x,
                            touch.page.y,
                            touch.screen.x,
                            touch.screen.y,
                        );
                    }
                }
            },
        }
    }

    fn has_touch_event_listeners(&self) -> bool {
        self.window.has_event_listener(atom!("touchstart")) ||
            self.window.has_event_listener(atom!("touchmove")) ||
            self.window.has_event_listener(atom!("touchend")) ||
            self.window.has_event_listener(atom!("touchcancel"))
    }

    /// Update the current set of active keyboard modifiers based on the most recent input event.
    /// This is not intended to be used for dispatching events directly. It is only
    /// supposed to be used for cases where we need to query the state of the
    /// keyboard modifiers at a given moment (e.g. `event.getModifierState`).
    pub(crate) fn update_active_keyboard_modifiers(&self, modifiers: Modifiers) {
        self.active_keyboard_modifiers.set(modifiers);
    }

    /// Dispatch a key event to the correct DOM node based on the focused element.
    pub(crate) fn dispatch_key_event(&self, mut event: keyboard_types::KeyEvent, can_gc: CanGc) {
        let focused_node = self.focused.get();

        if let Some(focused) = focused_node {
            event.target = focused.upcast::<Node>().stable_node_id().node_address();
        }

        let target = self
            .focused
            .get()
            .map_or(self.upcast(), |elem| elem.upcast());

        let dom_event = KeyboardEvent::for_platform_key_event(
            event,
            &self.window,
            self.active_keyboard_modifiers.get(),
            can_gc,
        );
        let event = dom_event.upcast::<Event>();
        event.fire(target, can_gc);
    }

    /// Dispatch a key event to the correct DOM node based on the focused element.
    pub(crate) fn dispatch_ime_event(&self, event: ImeEvent, can_gc: CanGc) {
        let target = self
            .focused
            .get()
            .map_or(self.upcast(), |elem| elem.upcast());

        let dom_event = CompositionEvent::for_platform_ime_event(event, &self.window, can_gc);
        let event = dom_event.upcast::<Event>();
        event.fire(target, can_gc);
    }

    /// Handles a new pending script, creating a load if needed, and possibly
    /// executing it if it's already complete.
    ///
    /// The load_type indicates if the request should be made asynchronously
    /// or not.
    pub(crate) fn new_pending_script(
        &self,
        script: &HTMLScriptElement,
        load_type: ScriptExecution,
        load: ScriptResult,
    ) {
        let can_gc = CanGc::note();
        match load_type {
            ScriptExecution::Deferred => {
                self.deferred_scripts.push(script, load, can_gc);
            },
            ScriptExecution::ASAPInOrder => {
                self.asap_in_order_scripts_list
                    .push(script, load, can_gc);
            },
            ScriptExecution::ASAP => {
                self.asap_script_loaded(script, load, can_gc);
            },
        }
    }

    pub(crate) fn asap_in_order_script_loaded(
        &self,
        script: &HTMLScriptElement,
        load: ScriptResult,
        can_gc: CanGc,
    ) {
        self.asap_in_order_scripts_list
            .note_script_loaded(script, load);
        if let Some(task) = self.asap_in_order_scripts_list.take_task() {
            self.owner_global()
                .task_manager()
                .script_event_task_source()
                .queue(task);
        }
    }

    pub(crate) fn asap_script_loaded(
        &self,
        script: &HTMLScriptElement,
        load: ScriptResult,
        can_gc: CanGc,
    ) {
        let task = self
            .asap_scripts_set
            .borrow_mut()
            .iter()
            .position(|e| *e == script)
            .map(|i| self.asap_scripts_set.borrow_mut().remove(i))
            .map(|_| script.create_task_to_execute_script(load, can_gc))
            .expect("An `asap_script_loaded` was called on a script not in the ASAP set.");

        self.owner_global()
            .task_manager()
            .script_event_task_source()
            .queue(task);
    }

    pub(crate) fn deferred_script_loaded(
        &self,
        script: &HTMLScriptElement,
        load: ScriptResult,
        can_gc: CanGc,
    ) {
        self.deferred_scripts.note_script_loaded(script, load);
        if self.deferred_scripts.is_complete() {
            let task = self.deferred_scripts.create_task(can_gc);
            self.owner_global()
                .task_manager()
                .script_event_task_source()
                .queue(task);
        }
    }

    /// Queues the task that will fire the `DOMContentLoaded` event.
    ///
    /// <https://html.spec.whatwg.org/multipage/#the-end:queue-a-task-to-fire-a-domcontentloaded-event>
    pub(crate) fn queue_domcontentloaded_event(&self, can_gc: CanGc) {
        let task_source = self.owner_global().task_manager().rendering_task_source();
        let document = Trusted::new(self);
        task_source.queue_unconditionally(task!(domcontentloaded_event: move || {
            let document = document.root();
            // Step 1
            if document.domcontentloaded_dispatched.get() { return; }
            // Step 2
            document.domcontentloaded_dispatched.set(true);
            // Step 3
            update_with_current_instant(&document.dom_content_loaded_event_start);
            // Step 4
            document.set_ready_state(DocumentReadyState::Interactive, can_gc);
            // Step 5
            document.fire_event(atom!("DOMContentLoaded"), can_gc);
            // Step 6
            update_with_current_instant(&document.dom_content_loaded_event_end);
            // Step 7
            let global = document.owner_global();
            if let Some(task) = document.deferred_scripts.take_task() {
                global.task_manager().script_event_task_source().queue(task);
            }
        }));
    }

    /// <https://html.spec.whatwg.org/multipage/#the-end:the-end-2>
    pub(crate) fn maybe_queue_document_completion(&self, can_gc: CanGc) {
        if self.ready_state.get() != DocumentReadyState::Complete {
            self.set_ready_state(DocumentReadyState::Complete, can_gc);
        }
        self.fire_event(atom!("load"), can_gc);
    }

    fn set_target_element(&self, target: Option<&Element>) {
        self.target_element.set(target);
    }

    pub(crate) fn note_pending_input_event(&self, event: ConstellationInputEvent) {
        let mut pending_compositor_events = self.pending_input_events.borrow_mut();
        if let InputEvent::MouseMove(_) = event.event {
            // First try to replace any existing mouse move event.
            if let Some(mouse_move_event) = self
                .mouse_move_event_index
                .borrow()
                .and_then(|index| pending_compositor_events.get_mut(index))
            {
                *mouse_move_event = event;
                return;
            }

            *self.mouse_move_event_index.borrow_mut() = Some(pending_compositor_events.len());
        }

        pending_compositor_events.push(event);
    }

    /// Get pending compositor events, for processing within an `update_the_rendering` task.
    pub(crate) fn take_pending_input_events(&self) -> Vec<ConstellationInputEvent> {
        // Reset the mouse event index.
        *self.mouse_move_event_index.borrow_mut() = None;
        mem::take(&mut *self.pending_input_events.borrow_mut())
    }

    pub(crate) fn set_csp_list(&self, csp_list: Option<CspList>) {
        self.policy_container.borrow_mut().set_csp_list(csp_list);
    }

    pub(crate) fn get_csp_list(&self) -> Option<CspList> {
        self.policy_container.borrow().csp_list.clone()
    }

    /// <https://www.w3.org/TR/CSP/#should-block-inline> (modified)
    pub(crate) fn should_elements_inline_type_behavior_be_blocked(
        &self,
        el: &Element,
        type_: csp::InlineCheckType,
        source: &str,
    ) -> csp::CheckResult {
        let (result, violations) = match self.get_csp_list() {
            None => {
                return csp::CheckResult::Allowed;
            },
            Some(csp_list) => {
                let element = csp::Element {
                    nonce: el.nonce_value_if_nonceable().map(Cow::Owned),
                };
                csp_list.should_elements_inline_type_behavior_be_blocked(&element, type_, source)
            },
        };

        report_csp_violations(&self.global(), violations, Some(el));

        result
    }

    /// Prevent any JS or layout from running until the corresponding call to
    /// `remove_script_and_layout_blocker`. Used to isolate periods in which
    /// the DOM is in an unstable state and should not be exposed to arbitrary
    /// web content. Any attempts to invoke content JS or query layout during
    /// that time will trigger a panic. `add_delayed_task` will cause the
    /// provided task to be executed as soon as the last blocker is removed.
    pub(crate) fn add_script_and_layout_blocker(&self) {
        self.script_and_layout_blockers
            .set(self.script_and_layout_blockers.get() + 1);
    }

    /// Terminate the period in which JS or layout is disallowed from running.
    /// If no further blockers remain, any delayed tasks in the queue will
    /// be executed in queue order until the queue is empty.
    pub(crate) fn remove_script_and_layout_blocker(&self) {
        assert!(self.script_and_layout_blockers.get() > 0);
        self.script_and_layout_blockers
            .set(self.script_and_layout_blockers.get() - 1);
        while self.script_and_layout_blockers.get() == 0 && !self.delayed_tasks.borrow().is_empty() {
            let task = self.delayed_tasks.borrow_mut().remove(0);
            task.run_box();
        }
    }

    /// Enqueue a task to run as soon as any JS and layout blockers are removed.
    pub(crate) fn add_delayed_task<T: 'static + TaskBox>(&self, task: T) {
        self.delayed_tasks.borrow_mut().push(Box::new(task));
    }

    /// Assert that the DOM is in a state that will allow running content JS or
    /// performing a layout operation.
    pub(crate) fn ensure_safe_to_run_script_or_layout(&self) {
        assert_eq!(
            self.script_and_layout_blockers.get(),
            0,
            "Attempt to use script or layout while DOM not in a stable state"
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        window: &Window,
        has_browsing_context: HasBrowsingContext,
        url: Option<ServoUrl>,
        origin: MutableOrigin,
        doctype: IsHTMLDocument,
        content_type: Option<Mime>,
        last_modified: Option<String>,
        activity: DocumentActivity,
        source: DocumentSource,
        doc_loader: DocumentLoader,
        referrer: Option<String>,
        status_code: Option<u16>,
        canceller: FetchCanceller,
        is_initial_about_blank: bool,
        allow_declarative_shadow_roots: bool,
        inherited_insecure_requests_policy: Option<InsecureRequestsPolicy>,
        has_trustworthy_ancestor_origin: bool,
        can_gc: CanGc,
    ) -> DomRoot<Document> {
        Self::new_with_proto(
            window,
            None,
            has_browsing_context,
            url,
            origin,
            doctype,
            content_type,
            last_modified,
            activity,
            source,
            doc_loader,
            referrer,
            status_code,
            canceller,
            is_initial_about_blank,
            allow_declarative_shadow_roots,
            inherited_insecure_requests_policy,
            has_trustworthy_ancestor_origin,
            can_gc,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_proto(
        window: &Window,
        proto: Option<HandleObject>,
        has_browsing_context: HasBrowsingContext,
        url: Option<ServoUrl>,
        origin: MutableOrigin,
        doctype: IsHTMLDocument,
        content_type: Option<Mime>,
        last_modified: Option<String>,
        activity: DocumentActivity,
        source: DocumentSource,
        doc_loader: DocumentLoader,
        referrer: Option<String>,
        status_code: Option<u16>,
        canceller: FetchCanceller,
        is_initial_about_blank: bool,
        allow_declarative_shadow_roots: bool,
        inherited_insecure_requests_policy: Option<InsecureRequestsPolicy>,
        has_trustworthy_ancestor_origin: bool,
        can_gc: CanGc,
    ) -> DomRoot<Document> {
        let document = reflect_dom_object_with_proto(
            Box::new(Document::new_inherited(
                window,
                has_browsing_context,
                url,
                origin,
                doctype,
                content_type,
                last_modified,
                activity,
                source,
                doc_loader,
                referrer,
                status_code,
                canceller,
                is_initial_about_blank,
                allow_declarative_shadow_roots,
                inherited_insecure_requests_policy,
                has_trustworthy_ancestor_origin,
            )),
            window,
            proto,
            can_gc,
        );
        {
            let node = document.upcast::<Node>();
            node.set_owner_doc(&document);
        }
        document
    }

    pub(crate) fn get_redirect_count(&self) -> u16 {
        self.redirect_count.get()
    }

    pub(crate) fn set_redirect_count(&self, count: u16) {
        self.redirect_count.set(count)
    }

    pub(crate) fn elements_by_name_count(&self, name: &DOMString) -> u32 {
        if name.is_empty() {
            return 0;
        }
        self.count_node_list(|n| Document::is_element_in_get_by_name(n, name))
    }

    pub(crate) fn nth_element_by_name(
        &self,
        index: u32,
        name: &DOMString,
    ) -> Option<DomRoot<Node>> {
        if name.is_empty() {
            return None;
        }
        self.nth_in_node_list(index, |n| Document::is_element_in_get_by_name(n, name))
    }

    // Note that document.getByName does not match on the same conditions
    // as the document named getter.
    fn is_element_in_get_by_name(node: &Node, name: &DOMString) -> bool {
        let element = match node.downcast::<Element>() {
            Some(element) => element,
            None => return false,
        };
        if element.namespace() != &ns!(html) {
            return false;
        }
        element.get_name().is_some_and(|n| *n == **name)
    }

    fn count_node_list<F: Fn(&Node) -> bool>(&self, callback: F) -> u32 {
        let doc = self.GetDocumentElement();
        let maybe_node = doc.as_deref().map(Castable::upcast::<Node>);
        maybe_node
            .iter()
            .flat_map(|node| node.traverse_preorder(ShadowIncluding::No))
            .filter(|node| callback(node))
            .count() as u32
    }

    fn nth_in_node_list<F: Fn(&Node) -> bool>(
        &self,
        index: u32,
        callback: F,
    ) -> Option<DomRoot<Node>> {
        let doc = self.GetDocumentElement();
        let maybe_node = doc.as_deref().map(Castable::upcast::<Node>);
        maybe_node
            .iter()
            .flat_map(|node| node.traverse_preorder(ShadowIncluding::No))
            .filter(|node| callback(node))
            .nth(index as usize)
            .map(|n| DomRoot::from_ref(&*n))
    }

    fn get_html_element(&self) -> Option<DomRoot<HTMLHtmlElement>> {
        self.GetDocumentElement().and_then(DomRoot::downcast)
    }

    /// Returns a reference to the per-document shared lock used in stylesheets.
    pub(crate) fn style_shared_lock(&self) -> &StyleSharedRwLock {
        &self.style_shared_lock
    }

    /// Flushes the stylesheet list, and returns whether any stylesheet changed.
    pub(crate) fn flush_stylesheets_for_reflow(&self) -> bool {
        // NOTE(emilio): The invalidation machinery is used on the replicated
        // list in layout.
        //
        // FIXME(emilio): This really should differentiate between CSSOM changes
        // and normal stylesheets additions / removals, because in the last case
        // layout already has that information and we could avoid dirtying the whole thing.
        let mut stylesheets = self.stylesheets.borrow_mut();
        let have_changed = stylesheets.has_changed();
        stylesheets.flush_without_invalidation();
        have_changed
    }

    pub(crate) fn salvageable(&self) -> bool {
        self.salvageable.get()
    }

    /// <https://html.spec.whatwg.org/multipage/#appropriate-template-contents-owner-document>
    pub(crate) fn appropriate_template_contents_owner_document(
        &self,
        can_gc: CanGc,
    ) -> DomRoot<Document> {
        self.appropriate_template_contents_owner_document
            .or_init(|| {
                let doctype = if self.is_html_document {
                    IsHTMLDocument::HTMLDocument
                } else {
                    IsHTMLDocument::NonHTMLDocument
                };
                let new_doc = Document::new(
                    self.window(),
                    HasBrowsingContext::No,
                    None,
                    // https://github.com/whatwg/html/issues/2109
                    MutableOrigin::new(ImmutableOrigin::new_opaque()),
                    doctype,
                    None,
                    None,
                    DocumentActivity::Inactive,
                    DocumentSource::NotFromParser,
                    DocumentLoader::new(&self.loader()),
                    None,
                    None,
                    Default::default(),
                    false,
                    self.allow_declarative_shadow_roots(),
                    Some(self.insecure_requests_policy()),
                    self.has_trustworthy_ancestor_or_current_origin(),
                    can_gc,
                );
                new_doc
                    .appropriate_template_contents_owner_document
                    .set(Some(&new_doc));
                new_doc
            })
    }

    pub(crate) fn get_element_by_id(&self, id: &Atom) -> Option<DomRoot<Element>> {
        self.id_map
            .borrow()
            .get(id)
            .map(|elements| DomRoot::from_ref(&*elements[0]))
    }

    pub(crate) fn ensure_pending_restyle(&self, el: &Element) -> RefMut<PendingRestyle> {
        let map = self.pending_restyles.borrow_mut();
        RefMut::map(map, |m| {
            &mut m
                .entry(Dom::from_ref(el))
                .or_insert_with(|| NoTrace(PendingRestyle::default()))
                .0
        })
    }

    pub(crate) fn element_state_will_change(&self, el: &Element) {
        let mut entry = self.ensure_pending_restyle(el);
        if entry.snapshot.is_none() {
            entry.snapshot = Some(Snapshot::new());
        }
        let snapshot = entry.snapshot.as_mut().unwrap();
        if snapshot.state.is_none() {
            snapshot.state = Some(el.state());
        }
    }

    pub(crate) fn element_attr_will_change(&self, el: &Element, attr: &Attr) {
        // FIXME(emilio): Kind of a shame we have to duplicate this.
        //
        // I'm getting rid of the whole hashtable soon anyway, since all it does
        // right now is populate the element restyle data in layout, and we
        // could in theory do it in the DOM I think.
        let mut entry = self.ensure_pending_restyle(el);
        if entry.snapshot.is_none() {
            entry.snapshot = Some(Snapshot::new());
        }
        if attr.local_name() == &local_name!("style") {
            entry.hint.insert(RestyleHint::RESTYLE_STYLE_ATTRIBUTE);
        }

        if vtable_for(el.upcast()).attribute_affects_presentational_hints(attr) {
            entry.hint.insert(RestyleHint::RESTYLE_SELF);
        }

        let snapshot = entry.snapshot.as_mut().unwrap();
        if attr.local_name() == &local_name!("id") {
            if snapshot.id_changed {
                return;
            }
            snapshot.id_changed = true;
        } else if attr.local_name() == &local_name!("class") {
            if snapshot.class_changed {
                return;
            }
            snapshot.class_changed = true;
        } else {
            snapshot.other_attributes_changed = true;
        }
        let local_name = style::LocalName::cast(attr.local_name());
        if !snapshot.changed_attrs.contains(local_name) {
            snapshot.changed_attrs.push(local_name.clone());
        }
        if snapshot.attrs.is_none() {
            let attrs = el
                .attrs()
                .iter()
                .map(|attr| (attr.identifier().clone(), attr.value().clone()))
                .collect();
            snapshot.attrs = Some(attrs);
        }
    }

    pub(crate) fn set_referrer_policy(&self, policy: ReferrerPolicy) {
        self.policy_container
            .borrow_mut()
            .set_referrer_policy(policy);
    }

    pub(crate) fn get_referrer_policy(&self) -> ReferrerPolicy {
        self.policy_container.borrow().get_referrer_policy()
    }

    pub(crate) fn set_target_element(&self, node: Option<&Element>) {
        if let Some(ref element) = self.target_element.get() {
            element.set_target_state(false);
        }

        self.target_element.set(node);

        if let Some(ref element) = self.target_element.get() {
            element.set_target_state(true);
        }
    }

    pub(crate) fn incr_ignore_destructive_writes_counter(&self) {
        self.ignore_destructive_writes_counter
            .set(self.ignore_destructive_writes_counter.get() + 1);
    }

    pub(crate) fn decr_ignore_destructive_writes_counter(&self) {
        self.ignore_destructive_writes_counter
            .set(self.ignore_destructive_writes_counter.get() - 1);
    }

    pub(crate) fn is_prompting_or_unloading(&self) -> bool {
        self.ignore_opens_during_unload_counter.get() > 0
    }

    fn incr_ignore_opens_during_unload_counter(&self) {
        self.ignore_opens_during_unload_counter
            .set(self.ignore_opens_during_unload_counter.get() + 1);
    }

    fn decr_ignore_opens_during_unload_counter(&self) {
        self.ignore_opens_during_unload_counter
            .set(self.ignore_opens_during_unload_counter.get() - 1);
    }

    // https://fullscreen.spec.whatwg.org/#dom-element-requestfullscreen
    pub(crate) fn enter_fullscreen(&self, pending: &Element, can_gc: CanGc) -> Rc<Promise> {
        // Step 1
        let in_realm_proof = AlreadyInRealm::assert::<crate::DomTypeHolder>();
        let promise = Promise::new_in_current_realm(InRealm::Already(&in_realm_proof), can_gc);
        let mut error = false;

        // Step 4
        // check namespace
        match *pending.namespace() {
            ns!(mathml) => {
                if pending.local_name().as_ref() != "math" {
                    error = true;
                }
            },
            ns!(svg) => {
                if pending.local_name().as_ref() != "svg" {
                    error = true;
                }
            },
            ns!(html) => (),
            _ => error = true,
        }
        // fullscreen element ready check
        if !pending.fullscreen_element_ready_check() {
            error = true;
        }

        if pref!(dom_fullscreen_test) {
            // For reftests we just take over the current window,
            // and don't try to really enter fullscreen.
            info!("Tests don't really enter fullscreen.");
        } else {
            // TODO fullscreen is supported
            // TODO This algorithm is allowed to request fullscreen.
            warn!("Fullscreen not supported yet");
        }

        // Step 5 Parallel start

        let window = self.window();
        // Step 6
        if !error {
            let event = EmbedderMsg::NotifyFullscreenStateChanged(self.webview_id(), true);
            self.send_to_embedder(event);
        }

        let pipeline_id = self.window().pipeline_id();

        // Step 7
        let trusted_pending = Trusted::new(pending);
        let trusted_promise = TrustedPromise::new(promise.clone());
        let handler = ElementPerformFullscreenEnter::new(trusted_pending, trusted_promise, error);
        // NOTE: This steps should be running in parallel
        // https://fullscreen.spec.whatwg.org/#dom-element-requestfullscreen
        let script_msg = CommonScriptMsg::Task(
            ScriptThreadEventCategory::EnterFullscreen,
            handler,
            Some(pipeline_id),
            TaskSourceName::DOMManipulation,
        );
        let msg = MainThreadScriptMsg::Common(script_msg);
        window.main_thread_script_chan().send(msg).unwrap();

        promise
    }

    // https://fullscreen.spec.whatwg.org/#exit-fullscreen
    pub(crate) fn exit_fullscreen(&self, can_gc: CanGc) -> Rc<Promise> {
        let global = self.global();
        // Step 1
        let in_realm_proof = AlreadyInRealm::assert::<crate::DomTypeHolder>();
        let promise = Promise::new_in_current_realm(InRealm::Already(&in_realm_proof), can_gc);
        // Step 2
        if self.fullscreen_element.get().is_none() {
            promise.reject_error(Error::Type(String::from("fullscreen is null")), can_gc);
            return promise;
        }
        // TODO Step 3-6
        let element = self.fullscreen_element.get().unwrap();

        // Step 7 Parallel start

        let window = self.window();
        // Step 8
        let event = EmbedderMsg::NotifyFullscreenStateChanged(self.webview_id(), false);
        self.send_to_embedder(event);

        // Step 9
        let trusted_element = Trusted::new(&*element);
        let trusted_promise = TrustedPromise::new(promise.clone());
        let handler = ElementPerformFullscreenExit::new(trusted_element, trusted_promise);
        let pipeline_id = Some(global.pipeline_id());
        // NOTE: This steps should be running in parallel
        // https://fullscreen.spec.whatwg.org/#exit-fullscreen
        let script_msg = CommonScriptMsg::Task(
            ScriptThreadEventCategory::ExitFullscreen,
            handler,
            pipeline_id,
            TaskSourceName::DOMManipulation,
        );
        let msg = MainThreadScriptMsg::Common(script_msg);
        window.main_thread_script_chan().send(msg).unwrap();

        promise
    }

    pub(crate) fn set_fullscreen_element(&self, element: Option<&Element>) {
        self.fullscreen_element.set(element);
    }

    pub(crate) fn get_allow_fullscreen(&self) -> bool {
        // https://html.spec.whatwg.org/multipage/#allowed-to-use>
        match self.browsing_context() {
            // Step 1
            None => false,
            Some(_) => {
                // Step 2
                let window = self.window();
                if window.is_top_level() {
                    true
                } else {
                    // Step 3
                    window
                        .GetFrameElement()
                        .is_some_and(|el| el.has_attribute(&local_name!("allowfullscreen")))
                }
            },
        }
    }

    fn reset_form_owner_for_listeners(&self, id: &Atom, can_gc: CanGc) {
        let map = self.form_id_listener_map.borrow();
        if let Some(listeners) = map.get(id) {
            for listener in listeners {
                listener
                    .as_maybe_form_control() // This should not fail
                    .expect("Element must be a form control")
                    .reset_form_owner(can_gc);
            }
        }
    }

    pub(crate) fn register_shadow_root(&self, shadow_root: &ShadowRoot) {
        self.shadow_roots
            .borrow_mut()
            .insert(Dom::from_ref(shadow_root));
        self.invalidate_shadow_roots_stylesheets();
    }

    pub(crate) fn unregister_shadow_root(&self, shadow_root: &ShadowRoot) {
        let mut shadow_roots = self.shadow_roots.borrow_mut();
        shadow_roots.remove(&Dom::from_ref(shadow_root));
    }

    pub(crate) fn invalidate_shadow_roots_stylesheets(&self) {
        self.shadow_roots_styles_changed.set(true);
    }

    pub(crate) fn shadow_roots_styles_changed(&self) -> bool {
        self.shadow_roots_styles_changed.get()
    }

    pub(crate) fn flush_shadow_roots_stylesheets(&self) {
        if !self.shadow_roots_styles_changed.get() {
            return;
        }
        self.shadow_roots_styles_changed.set(false);
    }

    pub(crate) fn stylesheet_count(&self) -> usize {
        self.stylesheets.borrow().len()
    }

    pub(crate) fn stylesheet_at(&self, index: usize) -> Option<DomRoot<CSSStyleSheet>> {
        let stylesheets = self.stylesheets.borrow();

        stylesheets
            .get(Origin::Author, index)
            .and_then(|s| s.owner.upcast::<Node>().get_cssom_stylesheet())
    }

    /// Add a stylesheet owned by `owner` to the list of document sheets, in the
    /// correct tree position.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))] // Owner needs to be rooted already necessarily.
    pub(crate) fn add_stylesheet(&self, owner: &Element, sheet: Arc<Stylesheet>) {
        let stylesheets = &mut *self.stylesheets.borrow_mut();
        let insertion_point = stylesheets
            .iter()
            .map(|(sheet, _origin)| sheet)
            .find(|sheet_in_doc| {
                owner
                    .upcast::<Node>()
                    .is_before(sheet_in_doc.owner.upcast())
            })
            .cloned();

        if self.has_browsing_context() {
            self.window.layout_mut().add_stylesheet(
                sheet.clone(),
                insertion_point.as_ref().map(|s| s.sheet.clone()),
            );
        }

        DocumentOrShadowRoot::add_stylesheet(
            owner,
            StylesheetSetRef::Document(stylesheets),
            sheet,
            insertion_point,
            self.style_shared_lock(),
        );
    }

    /// Given a stylesheet, load all web fonts from it in Layout.
    pub(crate) fn load_web_fonts_from_stylesheet(&self, stylesheet: Arc<Stylesheet>) {
        self.window
            .layout()
            .load_web_fonts_from_stylesheet(stylesheet);
    }

    /// Remove a stylesheet owned by `owner` from the list of document sheets.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))] // Owner needs to be rooted already necessarily.
    pub(crate) fn remove_stylesheet(&self, owner: &Element, stylesheet: &Arc<Stylesheet>) {
        if self.has_browsing_context() {
            self.window
                .layout_mut()
                .remove_stylesheet(stylesheet.clone());
        }

        DocumentOrShadowRoot::remove_stylesheet(
            owner,
            stylesheet,
            StylesheetSetRef::Document(&mut *self.stylesheets.borrow_mut())
        );
    }

    pub(crate) fn get_elements_with_id(&self, id: &Atom) -> Ref<[Dom<Element>]> {
        Ref::map(self.id_map.borrow(), |map| {
            map.get(id).map(|vec| &**vec).unwrap_or_default()
        })
    }

    pub(crate) fn get_elements_with_name(&self, name: &Atom) -> Ref<[Dom<Element>]> {
        Ref::map(self.name_map.borrow(), |map| {
            map.get(name).map(|vec| &**vec).unwrap_or_default()
        })
    }

    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn drain_pending_restyles(&self) -> Vec<(TrustedNodeAddress, PendingRestyle)> {
        self.pending_restyles
            .borrow_mut()
            .drain()
            .filter_map(|(elem, restyle)| {
                let node = elem.upcast::<Node>();
                if !node.get_flag(NodeFlags::IS_CONNECTED) {
                    return None;
                }
                node.note_dirty_descendants();
                Some((node.to_trusted_node_address(), restyle.0))
            })
            .collect()
    }

    pub(crate) fn advance_animation_timeline_for_testing(&self, delta: f64) {
        self.animation_timeline.borrow_mut().advance_specific(delta);
        let current_timeline_value = self.current_animation_timeline_value();
        self.animations
            .borrow()
            .update_for_new_timeline_value(&self.window, current_timeline_value);
    }

    pub(crate) fn maybe_mark_animating_nodes_as_dirty(&self) {
        let current_timeline_value = self.current_animation_timeline_value();
        self.animations
            .borrow()
            .mark_animating_nodes_as_dirty(current_timeline_value);
    }

    pub(crate) fn current_animation_timeline_value(&self) -> f64 {
        self.animation_timeline.borrow().current_value()
    }

    pub(crate) fn animations(&self) -> Ref<Animations> {
        self.animations.borrow()
    }

    pub(crate) fn update_animations_post_reflow(&self) {
        self.animations
            .borrow()
            .do_post_reflow_update(&self.window, self.current_animation_timeline_value());
    }

    pub(crate) fn cancel_animations_for_node(&self, node: &Node) {
        self.animations.borrow().cancel_animations_for_node(node);
    }

    /// An implementation of <https://drafts.csswg.org/web-animations-1/#update-animations-and-send-events>.
    pub(crate) fn update_animations_and_send_events(&self, can_gc: CanGc) {
        // Only update the time if it isn't being managed by a test.
        if !pref!(layout_animations_test_enabled) {
            self.animation_timeline.borrow_mut().update();
        }

        // > 1. Update the current time of all timelines associated with doc passing now
        // > as the timestamp.
        // > 2. Remove replaced animations for doc.
        //
        // We still want to update the animations, because our timeline
        // value might have been advanced previously via the TestBinding.
        let current_timeline_value = self.current_animation_timeline_value();
        self.animations
            .borrow()
            .update_for_new_timeline_value(&self.window, current_timeline_value);
        self.maybe_mark_animating_nodes_as_dirty();

        // > 3. Perform a microtask checkpoint.
        self.window()
            .as_global_scope()
            .perform_a_microtask_checkpoint(can_gc);

        // Steps 4 through 7 occur inside `send_pending_events()`.
        let _realm = enter_realm(self);
        self.animations().send_pending_events(self.window(), can_gc);
    }

    pub(crate) fn image_animation_manager(&self) -> Ref<ImageAnimationManager> {
        self.image_animation_manager.borrow()
    }
    pub(crate) fn image_animation_manager_mut(&self) -> RefMut<ImageAnimationManager> {
        self.image_animation_manager.borrow_mut()
    }

    pub(crate) fn update_animating_images(&self) {
        let mut image_animation_manager = self.image_animation_manager.borrow_mut();
        if !image_animation_manager.image_animations_present() {
            return;
        }
        image_animation_manager
            .update_active_frames(&self.window, self.current_animation_timeline_value());

        if !self.animations().animations_present() {
            let next_scheduled_time =
                image_animation_manager.next_schedule_time(self.current_animation_timeline_value());
            // TODO: Once we have refresh signal from the compositor,
            // we should get rid of timer for animated image update.
            if let Some(next_scheduled_time) = next_scheduled_time {
                self.schedule_image_animation_update(next_scheduled_time);
            }
        }
    }

    fn schedule_image_animation_update(&self, next_scheduled_time: f64) {
        let callback = ImageAnimationUpdateCallback {
            document: Trusted::new(self),
        };
        self.global().schedule_callback(
            OneshotTimerCallback::ImageAnimationUpdate(callback),
            Duration::from_secs_f64(next_scheduled_time),
        );
    }

    /// <https://html.spec.whatwg.org/multipage/#shared-declarative-refresh-steps>
    pub(crate) fn shared_declarative_refresh_steps(&self, content: &[u8]) {
        // 1. If document's will declaratively refresh is true, then return.
        if self.will_declaratively_refresh() {
            return;
        }

        // 2-11 Parsing
        static REFRESH_REGEX: LazyLock<Regex> = LazyLock::new(|| {
            // s flag is used to match . on newlines since the only places we use . in the
            // regex is to go "to end of the string"
            // (?s-u:.) is used to consume invalid unicode bytes
            Regex::new(
                r#"(?xs)
                    ^\s*
                    ((?<time>[0-9]+)|\.) # 5-6
                    [0-9.]* # 8
                    (
                        (
                            (\s*;|\s*,|\s) # 10.3
                            \s* # 10.4
                        )
                        (
                            (
                                (U|u)(R|r)(L|l) # 11.2-11.4
                                \s*=\s* # 11.5-11.7
                            )?
                        )('(?<url1>[^']*)'(?s-u:.)*|"(?<url2>[^"]*)"(?s-u:.)*|['"]?(?<url3>(?s-u:.)*))
                        |
                        (?<url4>(?s-u:.)*)
                    )
                )?
                $"
            )
            .unwrap()
        });

        // 9. Let urlRecord be document's URL.
        let mut url_record = self.url();
        let captures = if let Some(captures) = REFRESH_REGEX.captures(content) {
            captures
        } else {
            return;
        };
        let time = if let Some(time_string) = captures.name("time") {
            u64::from_str(&String::from_utf8_lossy(time_string.as_bytes())).unwrap_or(0)
        } else {
            0
        };
        let captured_url = captures.name("url1").or(captures
            .name("url2")
            .or(captures.name("url3").or(captures.name("url4"))));

        // 11.11 Parse: Set urlRecord to the result of encoding-parsing a URL given urlString, relative to document.
        if let Some(url_match) = captured_url {
            url_record = if let Ok(url) = ServoUrl::parse_with_base(
                Some(&url_record),
                &String::from_utf8_lossy(url_match.as_bytes()),
            ) {
                info!("Refresh to {}", url.debug_compact());
                url
            } else {
                // 11.12 If urlRecord is failure, then return.
                return;
            }
        }
        // 12. Set document's will declaratively refresh to true.
        if self.completely_loaded() {
            // TODO: handle active sandboxing flag
            self.window.as_global_scope().schedule_callback(
                OneshotTimerCallback::RefreshRedirectDue(RefreshRedirectDue {
                    window: DomRoot::from_ref(self.window()),
                    url: url_record,
                }),
                Duration::from_secs(time),
            );
            self.set_declarative_refresh(DeclarativeRefresh::CreatedAfterLoad);
        } else {
            self.set_declarative_refresh(DeclarativeRefresh::PendingLoad {
                url: url_record,
                time,
            });
        }
    }

    pub(crate) fn will_declaratively_refresh(&self) -> bool {
        self.declarative_refresh.borrow().is_some()
    }
    pub(crate) fn set_declarative_refresh(&self, refresh: DeclarativeRefresh) {
        *self.declarative_refresh.borrow_mut() = Some(refresh);
    }

    /// <https://html.spec.whatwg.org/multipage/#visibility-state>
    fn update_visibility_state(&self, visibility_state: DocumentVisibilityState, can_gc: CanGc) {
        // Step 1 If document's visibility state equals visibilityState, then return.
        if self.visibility_state.get() == visibility_state {
            return;
        }
        // Step 2 Set document's visibility state to visibilityState.
        self.visibility_state.set(visibility_state);
        // Step 3 Queue a new VisibilityStateEntry whose visibility state is visibilityState and whose timestamp is
        // the current high resolution time given document's relevant global object.
        let entry = VisibilityStateEntry::new(
            &self.global(),
            visibility_state,
            CrossProcessInstant::now(),
            can_gc,
        );
        self.window
            .Performance()
            .queue_entry(entry.upcast::<PerformanceEntry>(), can_gc);

        // Step 4 Run the screen orientation change steps with document.
        // TODO ScreenOrientation hasn't implemented yet

        // Step 5 Run the view transition page visibility change steps with document.
        // TODO ViewTransition hasn't implemented yet

        // Step 6 Run any page visibility change steps which may be defined in other specifications, with visibility
        // state and document. Any other specs' visibility steps will go here.

        // <https://www.w3.org/TR/gamepad/#handling-visibility-change>
        if visibility_state == DocumentVisibilityState::Hidden {
            self.window
                .Navigator()
                .GetGamepads()
                .iter_mut()
                .for_each(|gamepad| {
                    if let Some(g) = gamepad {
                        g.vibration_actuator().handle_visibility_change();
                    }
                });
        }

        // Step 7 Fire an event named visibilitychange at document, with its bubbles attribute initialized to true.
        self.upcast::<EventTarget>()
            .fire_bubbling_event(atom!("visibilitychange"), can_gc);
    }

    /// <https://html.spec.whatwg.org/multipage/#dom-document-hidden>
    fn Hidden(&self) -> bool {
        self.visibility_state.get() == DocumentVisibilityState::Hidden
    }

    /// <https://html.spec.whatwg.org/multipage/#dom-document-visibilitystate>
    fn VisibilityState(&self) -> DocumentVisibilityState {
        self.visibility_state.get()
    }

    fn CreateExpression(
        &self,
        expression: DOMString,
        resolver: Option<Rc<XPathNSResolver>>,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<super::types::XPathExpression>> {
        let global = self.global();
        let window = global.as_window();
        let evaluator = XPathEvaluator::new(window, None, can_gc);
        XPathEvaluatorMethods::<crate::DomTypeHolder>::CreateExpression(
            &*evaluator,
            expression,
            resolver,
            can_gc,
        )
    }

    fn CreateNSResolver(&self, node_resolver: &Node, can_gc: CanGc) -> DomRoot<Node> {
        let global = self.global();
        let window = global.as_window();
        let evaluator = XPathEvaluator::new(window, None, can_gc);
        XPathEvaluatorMethods::<crate::DomTypeHolder>::CreateNSResolver(&*evaluator, node_resolver)
    }

    fn Evaluate(
        &self,
        expression: DOMString,
        context_node: &Node,
        resolver: Option<Rc<XPathNSResolver>>,
        type_: u16,
        result: Option<&super::types::XPathResult>,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<super::types::XPathResult>> {
        let global = self.global();
        let window = global.as_window();
        let evaluator = XPathEvaluator::new(window, None, can_gc);
        XPathEvaluatorMethods::<crate::DomTypeHolder>::Evaluate(
            &*evaluator,
            expression,
            context_node,
            resolver,
            type_,
            result,
            can_gc,
        )
    }
}

fn update_with_current_instant(marker: &Cell<Option<CrossProcessInstant>>) {
    if marker.get().is_none() {
        marker.set(Some(CrossProcessInstant::now()))
    }
}

/// <https://w3c.github.io/webappsec-referrer-policy/#determine-policy-for-token>
pub(crate) fn determine_policy_for_token(token: &str) -> ReferrerPolicy {
    match_ignore_ascii_case! { token,
        "never" | "no-referrer" => ReferrerPolicy::NoReferrer,
        "no-referrer-when-downgrade" => ReferrerPolicy::NoReferrerWhenDowngrade,
        "origin" => ReferrerPolicy::Origin,
        "same-origin" => ReferrerPolicy::SameOrigin,
        "strict-origin" => ReferrerPolicy::StrictOrigin,
        "default" | "strict-origin-when-cross-origin" => ReferrerPolicy::StrictOriginWhenCrossOrigin,
        "origin-when-cross-origin" => ReferrerPolicy::OriginWhenCrossOrigin,
        "always" | "unsafe-url" => ReferrerPolicy::UnsafeUrl,
        _ => ReferrerPolicy::EmptyString,
    }
}

/// Specifies the type of focus event that is sent to a pipeline
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum FocusType {
    Element, // The first focus message - focus the element itself
    Parent,  // Focusing a parent element (an iframe)
}

/// Specifies the initiator of a focus operation.
#[derive(Clone, Copy, PartialEq)]
pub enum FocusInitiator {
    /// The operation is initiated by this document and to be broadcasted
    /// through the constellation.
    Local,
    /// The operation is initiated somewhere else, and we are updating our
    /// internal state accordingly.
    Remote,
}

/// Focus events
pub(crate) enum FocusEventType {
    Focus, // Element gained focus. Doesn't bubble.
    Blur,  // Element lost focus. Doesn't bubble.
}

/// This is a temporary workaround to update animated images,
/// we should get rid of this after we have refresh driver #3406
#[derive(JSTraceable, MallocSizeOf)]
pub(crate) struct ImageAnimationUpdateCallback {
    /// The document.
    #[ignore_malloc_size_of = "non-owning"]
    document: Trusted<Document>,
}

impl ImageAnimationUpdateCallback {
    pub(crate) fn invoke(self, can_gc: CanGc) {
        with_script_thread(|script_thread| script_thread.update_the_rendering(true, can_gc))
    }
}

#[derive(JSTraceable, MallocSizeOf)]
pub(crate) enum AnimationFrameCallback {
    DevtoolsFramerateTick {
        actor_name: String,
    },
    FrameRequestCallback {
        #[ignore_malloc_size_of = "Rc is hard"]
        callback: Rc<FrameRequestCallback>,
    },
}

impl AnimationFrameCallback {
    fn call(&self, document: &Document, now: f64, can_gc: CanGc) {
        match *self {
            AnimationFrameCallback::DevtoolsFramerateTick { ref actor_name } => {
                let msg = ScriptToDevtoolsControlMsg::FramerateTick(actor_name.clone(), now);
                let devtools_sender = document.window().as_global_scope().devtools_chan().unwrap();
                devtools_sender.send(msg).unwrap();
            },
            AnimationFrameCallback::FrameRequestCallback { ref callback } => {
                // TODO(jdm): The spec says that any exceptions should be suppressed:
                // https://github.com/servo/servo/issues/6928
                let _ = callback.Call__(Finite::wrap(now), ExceptionHandling::Report, can_gc);
            },
        }
    }
}

#[derive(Default, JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
struct PendingInOrderScriptVec {
    scripts: DomRefCell<VecDeque<PendingScript>>,
}

impl PendingInOrderScriptVec {
    fn is_empty(&self) -> bool {
        self.scripts.borrow().is_empty()
    }

    fn push(&self, element: &HTMLScriptElement) {
        self.scripts
            .borrow_mut()
            .push_back(PendingScript::new(element));
    }

    fn loaded(&self, element: &HTMLScriptElement, result: ScriptResult) {
        let mut scripts = self.scripts.borrow_mut();
        let entry = scripts
            .iter_mut()
            .find(|entry| &*entry.element == element)
            .unwrap();
        entry.loaded(result);
    }

    fn take_next_ready_to_be_executed(&self) -> Option<(DomRoot<HTMLScriptElement>, ScriptResult)> {
        let mut scripts = self.scripts.borrow_mut();
        let pair = scripts.front_mut()?.take_result()?;
        scripts.pop_front();
        Some(pair)
    }

    fn clear(&self) {
        *self.scripts.borrow_mut() = Default::default();
    }
}

#[derive(JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
struct PendingScript {
    element: Dom<HTMLScriptElement>,
    // TODO(sagudev): could this be all no_trace?
    load: Option<ScriptResult>,
}

impl PendingScript {
    fn new(element: &HTMLScriptElement) -> Self {
        Self {
            element: Dom::from_ref(element),
            load: None,
        }
    }

    fn new_with_load(element: &HTMLScriptElement, load: Option<ScriptResult>) -> Self {
        Self {
            element: Dom::from_ref(element),
            load,
        }
    }

    fn loaded(&mut self, result: ScriptResult) {
        assert!(self.load.is_none());
        self.load = Some(result);
    }

    fn take_result(&mut self) -> Option<(DomRoot<HTMLScriptElement>, ScriptResult)> {
        self.load
            .take()
            .map(|result| (DomRoot::from_ref(&*self.element), result))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ReflowTriggerCondition {
    StylesheetsChanged,
    DirtyDescendants,
    PendingRestyles,
    PaintPostponed,
}

fn is_named_element_with_name_attribute(elem: &Element) -> bool {
    let type_ = match elem.upcast::<Node>().type_id() {
        NodeTypeId::Element(ElementTypeId::HTMLElement(type_)) => type_,
        _ => return false,
    };
    match type_ {
        HTMLElementTypeId::HTMLFormElement |
        HTMLElementTypeId::HTMLIFrameElement |
        HTMLElementTypeId::HTMLImageElement => true,
        // TODO handle <embed> and <object>; these depend on whether the element is
        // “exposed”, a concept that doesn’t fully make sense until embed/object
        // behaviour is actually implemented
        _ => false,
    }
}

fn is_named_element_with_id_attribute(elem: &Element) -> bool {
    // TODO handle <embed> and <object>; these depend on whether the element is
    // “exposed”, a concept that doesn't fully make sense until embed/object
    // behaviour is actually implemented
    elem.is::<HTMLImageElement>() && elem.get_name().is_some_and(|name| !name.is_empty())
}

impl DocumentHelpers for Document {
    fn ensure_safe_to_run_script_or_layout(&self) {
        Document::ensure_safe_to_run_script_or_layout(self)
    }
}
