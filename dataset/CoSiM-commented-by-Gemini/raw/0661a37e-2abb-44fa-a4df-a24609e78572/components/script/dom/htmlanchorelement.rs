
/**
 * @file htmlanchorelement.rs
 * @brief Implementation of the `HTMLAnchorElement` interface, representing `<a>` HTML elements.
 *
 * This module provides the Rust implementation for the `HTMLAnchorElement`, which corresponds to
 * the `<a>` tag in HTML. It encapsulates the properties and methods associated with hyperlinks,
 * including URL parsing, handling of `rel` and `target` attributes, and defining the element's
 * activation behavior (i.e., what happens when a user clicks on the link).
 *
 * ## Core Functionality:
 *
 * - **URL Parsing:** Implements the `HyperlinkElement` trait to parse and resolve the `href`
 *   attribute into a full URL, and provides properties like `protocol`, `host`, `pathname`, etc.
 * - **Link Relations:** Manages the `rel` attribute through a `DOMTokenList`, allowing for easy
 *   manipulation of link relations like `noopener`, `noreferrer`, etc.
 * - **Activation Behavior:** Defines the element's behavior when activated (e.g., by a mouse
 *   click), which typically involves navigating to the linked resource.
 * - **Attribute Reflection:** Reflects various HTML attributes (e.g., `target`, `download`, `ping`)
 *   as properties on the `HTMLAnchorElement` object.
 *
 * This implementation adheres to the WHATWG HTML specification for the `<a>` element.
 *
 * @see https://html.spec.whatwg.org/multipage/text-level-semantics.html#the-a-element
 */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::cell::Cell;
use std::default::Default;

use dom_struct::dom_struct;
use html5ever::{LocalName, Prefix, local_name};
use js::rust::HandleObject;
use num_traits::ToPrimitive;
use servo_url::ServoUrl;
use style::attr::AttrValue;
use stylo_atoms::Atom;

use crate::dom::activation::Activatable;
use crate::dom::attr::Attr;
use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::HTMLAnchorElementBinding::HTMLAnchorElementMethods;
use crate::dom::bindings::codegen::Bindings::MouseEventBinding::MouseEventMethods;
use crate::dom::bindings::codegen::Bindings::NodeBinding::NodeMethods;
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::root::{DomRoot, MutNullableDom};
use crate::dom::bindings::str::{DOMString, USVString};
use crate::dom::document::Document;
use crate::dom::domtokenlist::DOMTokenList;
use crate::dom::element::{AttributeMutation, Element, reflect_referrer_policy_attribute};
use crate::dom::event::Event;
use crate::dom::eventtarget::EventTarget;
use crate::dom::htmlelement::HTMLElement;
use crate::dom::htmlhyperlinkelementutils::{HyperlinkElement, HyperlinkElementTraits};
use crate::dom::htmlimageelement::HTMLImageElement;
use crate::dom::mouseevent::MouseEvent;
use crate::dom::node::{BindContext, Node};
use crate::dom::virtualmethods::VirtualMethods;
use crate::links::{LinkRelations, follow_hyperlink};
use crate::script_runtime::CanGc;

/**
 * @brief Represents an `<a>` HTML element.
 *
 * This struct holds the state for an `HTMLAnchorElement`, including its associated `HTMLElement`
 * data, its `rel` attribute as a `DOMTokenList`, the parsed link relations, and the resolved URL.
 */
#[dom_struct]
pub(crate) struct HTMLAnchorElement {
    htmlelement: HTMLElement,
    /// The `rel` attribute as a `DOMTokenList`.
    rel_list: MutNullableDom<DOMTokenList>,
    /// The parsed link relations from the `rel` and `rev` attributes.
    #[no_trace]
    relations: Cell<LinkRelations>,
    /// The resolved URL from the `href` attribute.
    #[no_trace]
    url: DomRefCell<Option<ServoUrl>>,
}

impl HTMLAnchorElement {
    fn new_inherited(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
    ) -> HTMLAnchorElement {
        HTMLAnchorElement {
            htmlelement: HTMLElement::new_inherited(local_name, prefix, document),
            rel_list: Default::default(),
            relations: Cell::new(LinkRelations::empty()),
            url: DomRefCell::new(None),
        }
    }

    /**
     * Creates a new `HTMLAnchorElement` and roots it in the JavaScript runtime.
     *
     * @param local_name The local name of the element.
     * @param prefix The namespace prefix of the element, if any.
     * @param document The document that owns this element.
     * @param proto The JavaScript prototype for this element.
     * @param can_gc A token indicating that garbage collection can be performed.
     * @return A rooted `HTMLAnchorElement`.
     */
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<HTMLAnchorElement> {
        Node::reflect_node_with_proto(
            Box::new(HTMLAnchorElement::new_inherited(
                local_name, prefix, document,
            )),
            document,
            proto,
            can_gc,
        )
    }
}

impl HyperlinkElement for HTMLAnchorElement {
    fn get_url(&self) -> &DomRefCell<Option<ServoUrl>> {
        &self.url
    }
}

impl VirtualMethods for HTMLAnchorElement {
    fn super_type(&self) -> Option<&dyn VirtualMethods> {
        Some(self.upcast::<HTMLElement>() as &dyn VirtualMethods)
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

    fn attribute_mutated(&self, attr: &Attr, mutation: AttributeMutation, can_gc: CanGc) {
        self.super_type()
            .unwrap()
            .attribute_mutated(attr, mutation, can_gc);

        match *attr.local_name() {
            local_name!("rel") | local_name!("rev") => {
                self.relations
                    .set(LinkRelations::for_element(self.upcast()));
            },
            _ => {},
        }
    }

    fn bind_to_tree(&self, context: &BindContext, can_gc: CanGc) {
        if let Some(s) = self.super_type() {
            s.bind_to_tree(context, can_gc);
        }

        self.relations
            .set(LinkRelations::for_element(self.upcast()));
    }
}

impl HTMLAnchorElementMethods<crate::DomTypeHolder> for HTMLAnchorElement {
    /**
     * Gets or sets the text content of the `<a>` element.
     * @see https://html.spec.whatwg.org/multipage/#dom-a-text
     */
    fn Text(&self) -> DOMString {
        self.upcast::<Node>().GetTextContent().unwrap()
    }

    fn SetText(&self, value: DOMString, can_gc: CanGc) {
        self.upcast::<Node>().SetTextContent(Some(value), can_gc)
    }

    /**
     * Gets or sets the `rel` attribute, specifying the relationship of the linked resource to the
     * current document.
     * @see https://html.spec.whatwg.org/multipage/#dom-a-rel
     */
    make_getter!(Rel, "rel");

    fn SetRel(&self, rel: DOMString, can_gc: CanGc) {
        self.upcast::<Element>()
            .set_tokenlist_attribute(&local_name!("rel"), rel, can_gc);
    }

    /**
     * Gets the `rel` attribute as a `DOMTokenList`.
     * @see https://html.spec.whatwg.org/multipage/#dom-a-rellist
     */
    fn RelList(&self, can_gc: CanGc) -> DomRoot<DOMTokenList> {
        self.rel_list.or_init(|| {
            DOMTokenList::new(
                self.upcast(),
                &local_name!("rel"),
                Some(vec![
                    Atom::from("noopener"),
                    Atom::from("noreferrer"),
                    Atom::from("opener"),
                ]),
                can_gc,
            )
        })
    }

    // https://html.spec.whatwg.org/multipage/#dom-a-coords
    make_getter!(Coords, "coords");

    // https://html.spec.whatwg.org/multipage/#dom-a-coords
    make_setter!(SetCoords, "coords");

    // https://html.spec.whatwg.org/multipage/#dom-a-name
    make_getter!(Name, "name");

    // https://html.spec.whatwg.org/multipage/#dom-a-name
    make_atomic_setter!(SetName, "name");

    // https://html.spec.whatwg.org/multipage/#dom-a-rev
    make_getter!(Rev, "rev");

    // https://html.spec.whatwg.org/multipage/#dom-a-rev
    make_setter!(SetRev, "rev");

    // https://html.spec.whatwg.org/multipage/#dom-a-shape
    make_getter!(Shape, "shape");

    // https://html.spec.whatwg.org/multipage/#dom-a-shape
    make_setter!(SetShape, "shape");

    /**
     * Gets or sets the `target` attribute, specifying where to display the linked resource.
     * @see https://html.spec.whatwg.org/multipage/#attr-hyperlink-target
     */
    make_getter!(Target, "target");

    make_setter!(SetTarget, "target");

    /**
     * Gets or sets the `href` attribute, containing the URL of the linked resource.
     * @see https://html.spec.whatwg.org/multipage/#dom-hyperlink-href
     */
    fn Href(&self) -> USVString {
        self.get_href()
    }

    fn SetHref(&self, value: USVString, can_gc: CanGc) {
        self.set_href(value, can_gc);
    }

    /**
     * Gets the origin of the URL.
     * @see https://html.spec.whatwg.org/multipage/#dom-hyperlink-origin
     */
    fn Origin(&self) -> USVString {
        self.get_origin()
    }

    /**
     * Gets or sets the protocol part of the URL.
     * @see https://html.spec.whatwg.org/multipage/#dom-hyperlink-protocol
     */
    fn Protocol(&self) -> USVString {
        self.get_protocol()
    }

    fn SetProtocol(&self, value: USVString, can_gc: CanGc) {
        self.set_protocol(value, can_gc);
    }

    /**
     * Gets or sets the password part of the URL.
     * @see https://html.spec.whatwg.org/multipage/#dom-hyperlink-password
     */
    fn Password(&self) -> USVString {
        self.get_password()
    }

    fn SetPassword(&self, value: USVString, can_gc: CanGc) {
        self.set_password(value, can_gc);
    }

    /**
     * Gets or sets the hash part of the URL (including the `#`).
     * @see https://html.spec.whatwg.org/multipage/#dom-hyperlink-hash
     */
    fn Hash(&self) -> USVString {
        self.get_hash()
    }

    fn SetHash(&self, value: USVString, can_gc: CanGc) {
        self.set_hash(value, can_gc);
    }

    /**
     * Gets or sets the host part of the URL (hostname and port).
     * @see https://html.spec.whatwg.org/multipage/#dom-hyperlink-host
     */
    fn Host(&self) -> USVString {
        self.get_host()
    }

    fn SetHost(&self, value: USVString, can_gc: CanGc) {
        self.set_host(value, can_gc);
    }

    /**
     * Gets or sets the hostname part of the URL.
     * @see https://html.spec.whatwg.org/multipage/#dom-hyperlink-hostname
     */
    fn Hostname(&self) -> USVString {
        self.get_hostname()
    }

    fn SetHostname(&self, value: USVString, can_gc: CanGc) {
        self.set_hostname(value, can_gc);
    }

    /**
     * Gets or sets the port part of the URL.
     * @see https://html.spec.whatwg.org/multipage/#dom-hyperlink-port
     */
    fn Port(&self) -> USVString {
        self.get_port()
    }

    fn SetPort(&self, value: USVString, can_gc: CanGc) {
        self.set_port(value, can_gc);
    }

    /**
     * Gets or sets the pathname part of the URL.
     * @see https://html.spec.whatwg.org/multipage/#dom-hyperlink-pathname
     */
    fn Pathname(&self) -> USVString {
        self.get_pathname()
    }

    fn SetPathname(&self, value: USVString, can_gc: CanGc) {
        self.set_pathname(value, can_gc);
    }

    /**
     * Gets or sets the search part of the URL (including the `?`).
     * @see https://html.spec.whatwg.org/multipage/#dom-hyperlink-search
     */
    fn Search(&self) -> USVString {
        self.get_search()
    }

    fn SetSearch(&self, value: USVString, can_gc: CanGc) {
        self.set_search(value, can_gc);
    }

    /**
     * Gets or sets the username part of the URL.
     * @see https://html.spec.whatwg.org/multipage/#dom-hyperlink-username
     */
    fn Username(&self) -> USVString {
        self.get_username()
    }

    fn SetUsername(&self, value: USVString, can_gc: CanGc) {
        self.set_username(value, can_gc);
    }

    /**
     * Gets or sets the `referrerpolicy` attribute.
     * @see https://html.spec.whatwg.org/multipage/#dom-a-referrerpolicy
     */
    fn ReferrerPolicy(&self) -> DOMString {
        reflect_referrer_policy_attribute(self.upcast::<Element>())
    }

    make_setter!(SetReferrerPolicy, "referrerpolicy");
}

impl Activatable for HTMLAnchorElement {
    fn as_element(&self) -> &Element {
        self.upcast::<Element>()
    }

    fn is_instance_activatable(&self) -> bool {
        // https://html.spec.whatwg.org/multipage/#hyperlink
        // "a [...] element[s] with an href attribute [...] must [..] create a
        // hyperlink"
        // https://html.spec.whatwg.org/multipage/#the-a-element
        // "The activation behaviour of a elements *that create hyperlinks*"
        self.as_element().has_attribute(&local_name!("href"))
    }

    /**
     * Defines the behavior when the `<a>` element is activated (e.g., clicked).
     *
     * @param event The event that triggered the activation.
     * @param target The target of the event.
     * @param can_gc A token indicating that garbage collection can be performed.
     * @see https://html.spec.whatwg.org/multipage/#the-a-element:activation-behaviour
     */
    fn activation_behavior(&self, event: &Event, target: &EventTarget, can_gc: CanGc) {
        let element = self.as_element();
        let mouse_event = event.downcast::<MouseEvent>().unwrap();
        let mut ismap_suffix = None;

        // Step 1: If the target of the click event is an img element with an ismap attribute
        // specified, then server-side image map processing must be performed.
        if let Some(element) = target.downcast::<Element>() {
            if target.is::<HTMLImageElement>() && element.has_attribute(&local_name!("ismap")) {
                let target_node = element.upcast::<Node>();
                let rect = target_node.bounding_content_box_or_zero(can_gc);
                ismap_suffix = Some(format!(
                    "?{},{}",
                    mouse_event.ClientX().to_f32().unwrap() - rect.origin.x.to_f32_px(),
                    mouse_event.ClientY().to_f32().unwrap() - rect.origin.y.to_f32_px()
                ))
            }
        }

        // Step 2.
        //TODO: Download the link is `download` attribute is set.
        follow_hyperlink(element, self.relations.get(), ismap_suffix);
    }
}
