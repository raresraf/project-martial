
/**
 * @file document.rs
 * @brief Implementation of the `Document` object, the root of the DOM tree.
 *
 * This module provides the Rust implementation for the `Document` object, which is the central
 * entry point to the content of a web page. It represents the entire HTML or XML document and
 * provides the primary API for interacting with the document's content, structure, and associated
 * resources.
 *
 * ## Core Functionality:
 *
 * - **DOM Tree Management:** The `Document` is the root of the DOM tree and provides methods for
 *   creating, finding, and manipulating nodes (elements, text nodes, comments, etc.).
 * - **Event Handling:** As an `EventTarget`, it handles a wide range of document-level events,
 *   such as `DOMContentLoaded`, `load`, `click`, `keydown`, etc.
 * - **Resource Loading:** Manages the loading of sub-resources like stylesheets, scripts, and
 *   images, and orchestrates the document lifecycle based on the loading state.
 * - **Style and Layout:** Manages the document's stylesheets and is the entry point for triggering
 *   style recalculation and layout (reflow).
 * - **User Interaction:** Handles user input events (e.g., mouse clicks, keyboard input, touch
 *   events) and dispatches them to the appropriate target elements.
 * - **Scripting:** Provides the context for executing scripts and implements various DOM APIs that
 *   are accessible from JavaScript.
 *
 * This implementation is based on the WHATWG DOM and HTML specifications.
 *
 * @see https://dom.spec.whatwg.org/#interface-document
 */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::any::Any;
use std::borrow::Cow;
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;

use base::cross_process_instant::CrossProcessInstant;
use base::id::PipelineId;
use content_security_policy as csp;
use css_properties::CssEnum;
use dom_struct::{dom_struct, DomStruct};
use embedder_traits::{
    AllowOrDeny, CompositorHitTestResult, ConstellationInputEvent, EmbedderMsg, ImeEvent,
    InputEvent, TouchEvent, TouchEventResult, TouchEventType,
};
use euclid::default::Point2D;
use html5ever::{local_name, namespace_url, ns, QualName};
use ipc_channel::ipc;
use keyboard_types::{self, Code, CompositionEvent as CompositionEventType, Key, KeyState, Modifiers};
use mime::Mime;
use net_traits::cookie::Cookie;
use net_traits::origin::{Host, OpaqueOrigin, Origin};
use net_traits::policy_container::PolicyContainer;
use net_traits::request::{InsecureRequestsPolicy, RequestBuilder};
use net_traits::url::Url;
use net_traits::{SetCookiesForUrl, GetCookiesForUrl, RequestType::NonHTTP};
use servo_config::pref;
use servo_url::ServoUrl;
use style::attr::AttrValue;
use style::data::ElementRestyleHint as RestyleHint;
use style::data::Snapshot;
use style::properties::longhands::text_decoration_line::computed_value::Values;
use style::servo_arc::Arc as ServoArc;
use style::shared_lock::StyleSharedRwLock;
// ... (rest of the file remains the same)
use style::stylesheets::{DocumentStylesheetSet, OriginSet, Stylesheet, StylesheetSetRef};
use style::stylist::Stylist;
use style_traits::quirks::QuirksMode;
use stylo_atoms::{html, Atom, KnownAtom};
use time::Tm;
use url::Position;
use uuid::Uuid;
use webrender_api::Transaction;

use super::animation::{Animation, AnimationTimeline, Animations};
use super::bindings::codegen::Bindings::AttrBinding::AttrMethods;
use super::bindings::codegen::Bindings::BeforeUnloadEventBinding::BeforeUnloadEventMethods;
use super::bindings::codegen::Bindings::CDATASectionBinding::CDATASectionMethods;
use super::bindings::codegen::Bindings::CommentBinding::CommentMethods;
use super::bindings::codegen::Bindings::CompositionEventBinding::CompositionEventMethods;
use super::bindings::codegen::Bindings::CustomEventBinding::CustomEventMethods;
use super::bindings::codegen::Bindings::DocumentBinding::{DocumentMethods, DocumentReadyState};
use super::bindings::codegen::Bindings::DocumentFragmentBinding::DocumentFragmentMethods;
use super::bindings::codegen::Bindings::DocumentTypeBinding::DocumentTypeMethods;
use super::bindings::codegen::Bindings::DOMImplementationBinding::DOMImplementationMethods;
use super::bindings::codegen::Bindings::ElementBinding::ElementMethods;
use super::bindings::codegen::Bindings::EventBinding::{EventDefault, EventMethods};
use super::bindings::codegen::Bindings::EventTargetBinding::EventTargetMethods;
use super::bindings::codegen::Bindings::FocusEventBinding::FocusEventMethods;
use super::bindings::codegen::Bindings::HashChangeEventBinding::HashChangeEventMethods;
use super::bindings::codegen::Bindings::HTMLBodyElementBinding::HTMLBodyElementMethods;
use super::bindings::codegen::Bindings::HTMLCollectionBinding::HTMLCollectionMethods;
use super::bindings::codegen::Bindings::HTMLFormElementBinding::HTMLFormElementMethods;
use super::bindings::codegen::Bindings::HTMLFrameSetElementBinding::HTMLFrameSetElementMethods;
use super::bindings::codegen::Bindings::HTMLHeadElementBinding::HTMLHeadElementMethods;
use super::bindings::codegen::Bindings::HTMLHtmlElementBinding::HTMLHtmlElementMethods;
use super::bindings::codegen::Bindings::HTMLScriptElementBinding::HTMLScriptElementMethods;
use super::bindings::codegen::Bindings::HTMLTitleElementBinding::HTMLTitleElementMethods;
use super::bindings::codegen::Bindings::KeyboardEventBinding::KeyboardEventMethods;
use super::bindings::codegen::Bindings::MouseEventBinding::MouseEventMethods;
use super::bindings::codegen::Bindings::NodeBinding::{NodeFilter, NodeMethods};
use super::bindings::codegen::Bindings::NodeIteratorBinding::NodeIteratorMethods;
use super::bindings::codegen::Bindings::NodeListBinding::NodeListMethods;
use super::bindings::codegen::Bindings::PageTransitionEventBinding::PageTransitionEventMethods;
use super::bindings::codegen::Bindings::ProcessingInstructionBinding::ProcessingInstructionMethods;
use super::bindings::codegen::Bindings::RangeBinding::RangeMethods;
use super::bindings::codegen::Bindings::StyleSheetListBinding::StyleSheetListMethods;
use super::bindings::codegen::Bindings::TextBinding::TextMethods;
use super::bindings::codegen::Bindings::TouchEventBinding::TouchEventMethods;
use super::bindings::codegen::Bindings::TouchListBinding::TouchListMethods;
use super::bindings::codegen::Bindings::TreeWalkerBinding::TreeWalkerMethods;
use super::bindings::codegen::Bindings::UIEventBinding::UIEventMethods;
use super::bindings::codegen::Bindings::WheelEventBinding::WheelEventMethods;
use super::bindings::codegen::Bindings::XPathEvaluatorBinding::XPathEvaluatorMethods;
use super::bindings::codegen::Bindings::XPathExpressionBinding::XPathExpressionMethods;
use super::bindings::codegen::Bindings::XPathNSResolverBinding::XPathNSResolverMethods;
use super::bindings::codegen::Bindings::XPathResultBinding::XPathResultMethods;
use super::bindings::codegen::UnionTypes::NodeOrString;
use super::bindings::codegen::{
    Bindings, GlobalEventHandlersMethods, document_and_element_event_handlers, global_event_handlers,
};
use super::bindings::ptr::unroot;
use crate::dom::bindings::utils::vtable_for;
use crate::dom::traversal::{NodeIterator, TreeWalker};
use crate::fetch::LoadType;
use crate::image_animation::ImageAnimationManager;
use crate::loader::DocumentLoader;
use crate::media::TimeRanges;
use crate::progressive_web_metrics::{InteractiveFlag, InteractiveWindow, ProgressiveWebMetrics};
use crate::task::{TaskBox, TaskSourceName};
use crate::timers::{TimerMetadataFrameType, TimerTrait};
use crate::util::split_html_space_chars;
use crate::window_helpers::WindowHelpers;
use crate::xpath;
// ... (rest of the file remains the same)
