/*!
This module is the entry point for the Servo parser, which is responsible
for parsing HTML and XML documents. It defines the `ServoParser` struct,
which is the main entry point for the parser.

The `ServoParser` struct maintains two input streams: one for input from
script through `document.write()`, and one for input from the network. It
also maintains a tokenizer, which is responsible for breaking the input
streams into tokens.

The `ServoParser` struct provides methods for parsing complete documents,
as well as for parsing fragments of documents. It also provides methods for
handling script input and for aborting the parser.
*/

use std::borrow::Cow;
use std::cell::Cell;

use base::cross_process_instant::CrossProcessInstant;
use base::id::PipelineId;
use base64::Engine as _;
use base64::engine::general_purpose;
use content_security_policy as csp;
use devtools_traits::ScriptToDevtoolsControlMsg;
use dom_struct::dom_struct;
use embedder_traits::resources::{self, Resource};
use encoding_rs::Encoding;
use html5ever::buffer_queue::BufferQueue;
use html5ever::tendril::fmt::UTF8;
use html5ever::tendril::{ByteTendril, StrTendril, TendrilSink};
use html5ever::tree_builder::{ElementFlags, NodeOrText, QuirksMode, TreeSink};
use html5ever::{Attribute, ExpandedName, LocalName, QualName, local_name, ns};
use hyper_serde::Serde;
use markup5ever::TokenizerResult;
use mime::{self, Mime};
use net_traits::policy_container::PolicyContainer;
use net_traits::request::RequestId;
use net_traits::{
    FetchMetadata, FetchResponseListener, Metadata, NetworkError, ResourceFetchTiming,
    ResourceTimingType,
};
use profile_traits::time::{
    ProfilerCategory, ProfilerChan, TimerMetadata, TimerMetadataFrameType, TimerMetadataReflowType,
};
use profile_traits::time_profile;
use script_traits::DocumentActivity;
use servo_config::pref;
use servo_url::ServoUrl;
use style::context::QuirksMode as ServoQuirksMode;
use tendril::stream::LossyDecoder;

use crate::document_loader::{DocumentLoader, LoadType};
use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::DocumentBinding::{
    DocumentMethods, DocumentReadyState,
};
use crate::dom::bindings::codegen::Bindings::HTMLImageElementBinding::HTMLImageElementMethods;
use crate::dom::bindings::codegen::Bindings::HTMLTemplateElementBinding::HTMLTemplateElementMethods;
use crate::dom::bindings::codegen::Bindings::NodeBinding::NodeMethods;
use crate::dom::bindings::codegen::Bindings::ShadowRootBinding::{
    ShadowRootMode, SlotAssignmentMode,
};
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot, MutNullableDom};
use crate::dom::bindings::settings_stack::is_execution_stack_empty;
use crate::dom::bindings::str::{DOMString, USVString};
use crate::dom::characterdata::CharacterData;
use crate::dom::comment::Comment;
use crate::dom::csp::report_csp_violations;
use crate::dom::document::{Document, DocumentSource, HasBrowsingContext, IsHTMLDocument};
use crate::dom::documentfragment::DocumentFragment;
use crate::dom::documenttype::DocumentType;
use crate::dom::element::{CustomElementCreationMode, Element, ElementCreator};
use crate::dom::globalscope::GlobalScope;
use crate::dom::htmlformelement::{FormControlElementHelpers, HTMLFormElement};
use crate::dom::htmlimageelement::HTMLImageElement;
use crate::dom::htmlinputelement::HTMLInputElement;
use crate::dom::htmlscriptelement::{HTMLScriptElement, ScriptResult};
use crate::dom::htmltemplateelement::HTMLTemplateElement;
use crate::dom::node::{Node, ShadowIncluding};
use crate::dom::performanceentry::PerformanceEntry;
use crate::dom::performancenavigationtiming::PerformanceNavigationTiming;
use crate::dom::processinginstruction::ProcessingInstruction;
use crate::dom::shadowroot::IsUserAgentWidget;
use crate::dom::text::Text;
use crate::dom::virtualmethods::vtable_for;
use crate::network_listener::PreInvoke;
use crate::realms::enter_realm;
use crate::script_runtime::CanGc;
use crate::script_thread::ScriptThread;

mod async_html;
mod html;
mod prefetch;
mod xml;

pub(crate) use html::serialize_html_fragment;

#[dom_struct]
/// The parser maintains two input streams: one for input from script through
/// document.write(), and one for input from network.
///
/// There is no concrete representation of the insertion point, instead it
/// always points to just before the next character from the network input,
/// with all of the script input before itself.
///
/// ```text
///     ... script input ... | ... network input ...
///                          ^
///                 insertion point
/// ```
pub(crate) struct ServoParser {
    reflector: Reflector,
    /// The document associated with this parser.
    document: Dom<Document>,
    /// The BOM sniffing state.
    ///
    /// `None` means we've found the BOM, we've found there isn't one, or
    /// we're not parsing from a byte stream. `Some` contains the BOM bytes
    /// found so far.
    bom_sniff: DomRefCell<Option<Vec<u8>>>,
    /// The decoder used for the network input.
    network_decoder: DomRefCell<Option<NetworkDecoder>>,
    /// Input received from network.
    #[ignore_malloc_size_of = "Defined in html5ever"]
    #[no_trace]
    network_input: BufferQueue,
    /// Input received from script. Used only to support document.write().
    #[ignore_malloc_size_of = "Defined in html5ever"]
    #[no_trace]
    script_input: BufferQueue,
    /// The tokenizer of this parser.
    tokenizer: Tokenizer,
    /// Whether to expect any further input from the associated network request.
    last_chunk_received: Cell<bool>,
    /// Whether this parser should avoid passing any further data to the tokenizer.
    suspended: Cell<bool>,
    /// <https://html.spec.whatwg.org/multipage/#script-nesting-level>
    script_nesting_level: Cell<usize>,
    /// <https://html.spec.whatwg.org/multipage/#abort-a-parser>
    aborted: Cell<bool>,
    /// <https://html.spec.whatwg.org/multipage/#script-created-parser>
    script_created_parser: bool,
    /// We do a quick-and-dirty parse of the input looking for resources to prefetch.
    // TODO: if we had speculative parsing, we could do this when speculatively
    // building the DOM. https://github.com/servo/servo/pull/19203
    prefetch_tokenizer: prefetch::Tokenizer,
    #[ignore_malloc_size_of = "Defined in html5ever"]
    #[no_trace]
    prefetch_input: BufferQueue,
    // The whole input as a string, if needed for the devtools Sources panel.
    // TODO: use a faster type for concatenating strings?
    content_for_devtools: Option<DomRefCell<String>>,
}

/// An attribute of an element.
pub(crate) struct ElementAttribute {
    name: QualName,
    value: DOMString,
}

/// The parsing algorithm to use.
#[derive(Clone, Copy, JSTraceable, MallocSizeOf, PartialEq)]
pub(crate) enum ParsingAlgorithm {
    /// The normal parsing algorithm.
    Normal,
    /// The fragment parsing algorithm.
    Fragment,
}

impl ElementAttribute {
    /// Creates a new `ElementAttribute`.
    pub(crate) fn new(name: QualName, value: DOMString) -> ElementAttribute {
        ElementAttribute { name, value }
    }
}