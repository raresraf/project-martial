/*!
This module implements the `HTMLLinkElement` interface, which represents
a `<link>` element in the DOM. It is responsible for handling the loading
of external resources, such as stylesheets, favicons, and prefetched
resources.

The `HTMLLinkElement` struct is the main entry point for this module. It
creates a new `HTMLLinkElement` object, which can be inserted into the DOM.
The `HTMLLinkElement` object then handles the loading of the resource
specified by its `href` attribute, based on the value of its `rel`
attribute.

The `StylesheetLoader` struct is used to load external stylesheets. It
implements the `StyleStylesheetLoader` trait from the `style` crate, which
is used by the style system to request the loading of external stylesheets.

The `LinkProcessingOptions` struct holds the options for processing a
linked resource. It is created from the attributes of the `<link>` element,
and is used to create a `RequestBuilder` for fetching the resource.

The `PrefetchContext` and `PreloadContext` structs are used to handle the
loading of prefetched and preloaded resources, respectively. They implement
the `FetchResponseListener` trait, which is used to process the response
from the network layer.
*/

use std::borrow::{Borrow, ToOwned};
use std::cell::Cell;
use std::default::Default;
use std::str::FromStr;

use base::id::WebViewId;
use content_security_policy as csp;
use dom_struct::dom_struct;
use embedder_traits::EmbedderMsg;
use html5ever::{LocalName, Prefix, local_name, ns};
use js::rust::HandleObject;
use mime::Mime;
use net_traits::mime_classifier::{MediaType, MimeClassifier};
use net_traits::policy_container::PolicyContainer;
use net_traits::request::{
    CorsSettings, Destination, Initiator, InsecureRequestsPolicy, Referrer, RequestBuilder,
    RequestId,
};
use net_traits::{
    FetchMetadata, FetchResponseListener, NetworkError, ReferrerPolicy, ResourceFetchTiming,
    ResourceTimingType,
};
use servo_arc::Arc;
use servo_url::{ImmutableOrigin, ServoUrl};
use style::attr::AttrValue;
use style::stylesheets::Stylesheet;
use stylo_atoms::Atom;

use crate::dom::attr::Attr;
use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::DOMTokenListBinding::DOMTokenList_Binding::DOMTokenListMethods;
use crate::dom::bindings::codegen::Bindings::HTMLLinkElementBinding::HTMLLinkElementMethods;
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::root::{DomRoot, MutNullableDom};
use crate::dom::bindings::str::{DOMString, USVString};
use crate::dom::csp::report_csp_violations;
use crate::dom::cssstylesheet::CSSStyleSheet;
use crate::dom::document::Document;
use crate::dom::domtokenlist::DOMTokenList;
use crate::dom::element::{
    AttributeMutation, Element, ElementCreator, cors_setting_for_element,
    referrer_policy_for_element, reflect_cross_origin_attribute, reflect_referrer_policy_attribute,
    set_cross_origin_attribute,
};
use crate::dom::htmlelement::HTMLElement;
use crate::dom::medialist::MediaList;
use crate::dom::node::{BindContext, Node, NodeTraits, UnbindContext};
use crate::dom::performanceresourcetiming::InitiatorType;
use crate::dom::stylesheet::StyleSheet as DOMStyleSheet;
use crate::dom::types::{EventTarget, GlobalScope};
use crate::dom::virtualmethods::VirtualMethods;
use crate::fetch::create_a_potential_cors_request;
use crate::links::LinkRelations;
use crate::network_listener::{PreInvoke, ResourceTimingListener, submit_timing};
use crate::script_runtime::CanGc;
use crate::stylesheet_loader::{StylesheetContextSource, StylesheetLoader, StylesheetOwner};

/// A unique identifier for a request generation.
#[derive(Clone, Copy, JSTraceable, MallocSizeOf, PartialEq)]
pub(crate) struct RequestGenerationId(u32);

impl RequestGenerationId {
    /// Increments the generation ID.
    fn increment(self) -> RequestGenerationId {
        RequestGenerationId(self.0 + 1)
    }
}

/// <https://html.spec.whatwg.org/multipage/#link-processing-options>
struct LinkProcessingOptions {
    href: String,
    destination: Option<Destination>,
    integrity: String,
    link_type: String,
    cryptographic_nonce_metadata: String,
    cross_origin: Option<CorsSettings>,
    referrer_policy: ReferrerPolicy,
    policy_container: PolicyContainer,
    source_set: Option<()>,
    base_url: ServoUrl,
    origin: ImmutableOrigin,
    insecure_requests_policy: InsecureRequestsPolicy,
    has_trustworthy_ancestor_origin: bool,
    // Some fields that we don't need yet are missing
}

/// The `HTMLLinkElement` struct.
#[dom_struct]
pub(crate) struct HTMLLinkElement {
    htmlelement: HTMLElement,
    /// The relations as specified by the "rel" attribute
    rel_list: MutNullableDom<DOMTokenList>,

    /// The link relations as they are used in practice.
    ///
    /// The reason this is seperate from [HTMLLinkElement::rel_list] is that
    /// a literal list is a bit unwieldy and that there are corner cases to consider
    /// (Like `rev="made"` implying an author relationship that is not represented in rel_list)
    #[no_trace]
    relations: Cell<LinkRelations>,

    #[conditional_malloc_size_of]
    #[no_trace]
    stylesheet: DomRefCell<Option<Arc<Stylesheet>>>,
    cssom_stylesheet: MutNullableDom<CSSStyleSheet>,

    /// <https://html.spec.whatwg.org/multipage/#a-style-sheet-that-is-blocking-scripts>
    parser_inserted: Cell<bool>,
    /// The number of loads that this link element has triggered (could be more
    /// than one because of imports) and have not yet finished.
    pending_loads: Cell<u32>,
    /// Whether any of the loads have failed.
    any_failed_load: Cell<bool>,
    /// A monotonically increasing counter that keeps track of which stylesheet to apply.
    request_generation_id: Cell<RequestGenerationId>,
    /// <https://html.spec.whatwg.org/multipage/#explicitly-enabled>
    is_explicitly_enabled: Cell<bool>,
    /// Whether the previous type matched with the destination
    previous_type_matched: Cell<bool>,
    /// Whether the previous media environment matched with the media query
    previous_media_environment_matched: Cell<bool>,
}

impl HTMLLinkElement {
    /// Creates a new `HTMLLinkElement`.
    fn new_inherited(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        creator: ElementCreator,
    ) -> HTMLLinkElement {
        HTMLLinkElement {
            htmlelement: HTMLElement::new_inherited(local_name, prefix, document),
            rel_list: Default::default(),
            relations: Cell::new(LinkRelations::empty()),
            parser_inserted: Cell::new(creator.is_parser_created()),
            stylesheet: DomRefCell::new(None),
            cssom_stylesheet: MutNullableDom::new(None),
            pending_loads: Cell::new(0),
            any_failed_load: Cell::new(false),
            request_generation_id: Cell::new(RequestGenerationId(0)),
            is_explicitly_enabled: Cell::new(false),
            previous_type_matched: Cell::new(true),
            previous_media_environment_matched: Cell::new(true),
        }
    }

    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        proto: Option<HandleObject>,
        creator: ElementCreator,
        can_gc: CanGc,
    ) -> DomRoot<HTMLLinkElement> {
        Node::reflect_node_with_proto(
            Box::new(HTMLLinkElement::new_inherited(
                local_name, prefix, document, creator,
            )),
            document,
            proto,
            can_gc,
        )
    }

    /// Returns the request generation ID.
    pub(crate) fn get_request_generation_id(&self) -> RequestGenerationId {
        self.request_generation_id.get()
    }

    // FIXME(emilio): These methods are duplicated with
    // HTMLStyleElement::set_stylesheet.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn set_stylesheet(&self, s: Arc<Stylesheet>) {
        let stylesheets_owner = self.stylesheet_list_owner();
        if let Some(ref s) = *self.stylesheet.borrow() {
            stylesheets_owner.remove_stylesheet(self.upcast(), s)
        }
        *self.stylesheet.borrow_mut() = Some(s.clone());
        self.clean_stylesheet_ownership();
        stylesheets_owner.add_stylesheet(self.upcast(), s);
    }

    /// Returns the stylesheet associated with this link element.
    pub(crate) fn get_stylesheet(&self) -> Option<Arc<Stylesheet>> {
        self.stylesheet.borrow().clone()
    }

    /// Returns the CSSOM stylesheet associated with this link element.
    pub(crate) fn get_cssom_stylesheet(&self, can_gc: CanGc) -> Option<DomRoot<CSSStyleSheet>> {
        self.get_stylesheet().map(|sheet| {
            self.cssom_stylesheet.or_init(|| {
                CSSStyleSheet::new(
                    &self.owner_window(),
                    Some(self.upcast::<Element>()),
                    "text/css".into(),
                    None, // todo handle location
                    None, // todo handle title
                    sheet,
                    false, // is_constructed
                    can_gc,
                )
            })
        })
    }

    /// Returns whether this link is an alternate stylesheet.
    pub(crate) fn is_alternate(&self) -> bool {
        self.relations.get().contains(LinkRelations::ALTERNATE)
    }

    /// Returns whether this link is effectively disabled.
    pub(crate) fn is_effectively_disabled(&self) -> bool {
        (self.is_alternate() && !self.is_explicitly_enabled.get()) ||
            self.upcast::<Element>()
                .has_attribute(&local_name!("disabled"))
    }

    /// Cleans the stylesheet ownership.
    fn clean_stylesheet_ownership(&self) {
        if let Some(cssom_stylesheet) = self.cssom_stylesheet.get() {
            cssom_stylesheet.set_owner(None);
        }
        self.cssom_stylesheet.set(None);
    }
}

/// Returns the value of an attribute.
fn get_attr(element: &Element, local_name: &LocalName) -> Option<String> {
    let elem = element.get_attribute(&ns!(), local_name);
    elem.map(|e| {
        let value = e.value();
        (**value).to_owned()
    })
}

impl VirtualMethods for HTMLLinkElement {
    fn super_type(&self) -> Option<&dyn VirtualMethods> {
        Some(self.upcast::<HTMLElement>() as &dyn VirtualMethods)
    }

    fn attribute_mutated(&self, attr: &Attr, mutation: AttributeMutation, can_gc: CanGc) {
        self.super_type()
            .unwrap()
            .attribute_mutated(attr, mutation, can_gc);

        let local_name = attr.local_name();
        let is_removal = mutation.is_removal();
        if *local_name == local_name!("disabled") {
            self.handle_disabled_attribute_change(!is_removal);
            return;
        }

        if !self.upcast::<Node>().is_connected() {
            return;
        }
        match *local_name {
            local_name!("rel") | local_name!("rev") => {
                self.relations
                    .set(LinkRelations::for_element(self.upcast()));
            },
            local_name!("href") => {
                if is_removal {
                    return;
                }
                // https://html.spec.whatwg.org/multipage/#link-type-stylesheet
                // When the href attribute of the link element of an external resource link
                // that is already browsing-context connected is changed.
                if self.relations.get().contains(LinkRelations::STYLESHEET) {
                    self.handle_stylesheet_url(&attr.value());
                }

                if self.relations.get().contains(LinkRelations::ICON) {
                    let sizes = get_attr(self.upcast(), &local_name!("sizes"));
                    self.handle_favicon_url(&attr.value(), &sizes);
                }

                // https://html.spec.whatwg.org/multipage/#link-type-prefetch
                // When the href attribute of the link element of an external resource link
                // that is already browsing-context connected is changed.
                if self.relations.get().contains(LinkRelations::PREFETCH) {
                    self.fetch_and_process_prefetch_link(&attr.value());
                }

                // https://html.spec.whatwg.org/multipage/#link-type-preload
                // When the href attribute of the link element of an external resource link
                // that is already browsing-context connected is changed.
                if self.relations.get().contains(LinkRelations::PRELOAD) {
                    self.handle_preload_url();
                }
            },
            local_name!("sizes") if self.relations.get().contains(LinkRelations::ICON) => {
                if let Some(ref href) = get_attr(self.upcast(), &local_name!("href")) {
                    self.handle_favicon_url(href, &Some(attr.value().to_string()));
                }
            },
            local_name!("crossorigin") => {
                // https://html.spec.whatwg.org/multipage/#link-type-prefetch
                // When the crossorigin attribute of the link element of an external resource link
                // that is already browsing-context connected is set, changed, or removed.
                if self.relations.get().contains(LinkRelations::PREFETCH) {
                    self.fetch_and_process_prefetch_link(&attr.value());
                }

                // https://html.spec.whatwg.org/multipage/#link-type-stylesheet
                // When the crossorigin attribute of the link element of an external resource link
                // that is already browsing-context connected is set, changed, or removed.
                if self.relations.get().contains(LinkRelations::STYLESHEET) {
                    self.handle_stylesheet_url(&attr.value());
                }
            },
            local_name!("as") => {
                // https://html.spec.whatwg.org/multipage/#link-type-preload
                // When the as attribute of the link element of an external resource link
                // that is already browsing-context connected is changed.
                if self.relations.get().contains(LinkRelations::PRELOAD) {
                    if let AttributeMutation::Set(Some(_)) = mutation {
                        self.handle_preload_url();
                    }
                }
            },
            local_name!("type") => {
                // https://html.spec.whatwg.org/multipage/#link-type-stylesheet
                // When the type attribute of the link element of an external resource link that
                // is already browsing-context connected is set or changed to a value that does
                // not or no longer matches the Content-Type metadata of the previous obtained
                // external resource, if any.
                //
                // TODO: Match Content-Type metadata to check if it needs to be updated
                if self.relations.get().contains(LinkRelations::STYLESHEET) {
                    self.handle_stylesheet_url(&attr.value());
                }

                // https://html.spec.whatwg.org/multipage/#link-type-preload
                // When the type attribute of the link element of an external resource link that
                // is already browsing-context connected, but was previously not obtained due to
                // the type attribute specifying an unsupported type for the request destination,
                // is set, removed, or changed.
                if self.relations.get().contains(LinkRelations::PRELOAD) &&
                    !self.previous_type_matched.get()
                {
                    self.handle_preload_url();
                }
            },
            local_name!("media") => {
                // https://html.spec.whatwg.org/multipage/#link-type-preload
                // When the media attribute of the link element of an external resource link that
                // is already browsing-context connected, but was previously not obtained due to
                // the media attribute not matching the environment, is changed or removed.
                if self.relations.get().contains(LinkRelations::PRELOAD) &&
                    !self.previous_media_environment_matched.get()
                {
                    match mutation {
                        AttributeMutation::Removed | AttributeMutation::Set(Some(_)) => {
                            self.handle_preload_url()
                        },
                        _ => {},
                    };
                }

                let matches_media_environment =
                    self.upcast::<Element>().matches_environment(&attr.value());
                self.previous_media_environment_matched
                    .set(matches_media_environment);
            },
            _ => {},
        }
    }

    fn parse_plain_attribute(&self, name: &LocalName, value: DOMString) -> AttrValue {
        match name {
            &local_name!("rel") => AttrValue::from_serialized_tokenlist(value.into()),
            _ => self
                .super_type()
                .unwrap()
                .parse_plain_attribute(name, value),
        }
    }

    fn bind_to_tree(&self, context: &BindContext, can_gc: CanGc) {
        if let Some(s) = self.super_type() {
            s.bind_to_tree(context, can_gc);
        }

        self.relations
            .set(LinkRelations::for_element(self.upcast()));

        if context.tree_connected {
            let element = self.upcast();

            if let Some(href) = get_attr(element, &local_name!("href")) {
                let relations = self.relations.get();
                if relations.contains(LinkRelations::STYLESHEET) {
                    self.handle_stylesheet_url(&href);
                }

                if relations.contains(LinkRelations::ICON) {
                    let sizes = get_attr(self.upcast(), &local_name!("sizes"));
                    self.handle_favicon_url(&href, &sizes);
                }

                if relations.contains(LinkRelations::PREFETCH) {
                    self.fetch_and_process_prefetch_link(&href);
                }

                if relations.contains(LinkRelations::PRELOAD) {
                    self.handle_preload_url();
                }
            }
        }
    }

    fn unbind_from_tree(&self, context: &UnbindContext, can_gc: CanGc) {
        if let Some(s) = self.super_type() {
            s.unbind_from_tree(context, can_gc);
        }

        if let Some(s) = self.stylesheet.borrow_mut().take() {
            self.clean_stylesheet_ownership();
            self.stylesheet_list_owner()
                .remove_stylesheet(self.upcast(), &s);
        }
    }
}