/**
 * @ad2f5984-2cba-46ab-9239-433ee3b9f600/components/script/dom/attr.rs
 * @brief implementation of the DOM Attribute (Attr) interface for the Servo browser engine.
 * Domain: Browser Engine, DOM Specification, Script Runtime.
 * Architecture: Inherits from 'Node'; uses a reactive 'owner' pointer to link with 'Element' objects.
 * Functional Utility: Manages the lifecycle and mutation logic of DOM attributes, including namespace resolution and event dispatch (MutationObservers).
 * Synchronization: Utilizes thread-safe DOM pointers (DomRoot) and internal mutability (DomRefCell) for safe script-driven updates within the Servo task model.
 */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::borrow::ToOwned;
use std::cell::LazyCell;
use std::mem;

use devtools_traits::AttrInfo;
use dom_struct::dom_struct;
use html5ever::{LocalName, Namespace, Prefix, local_name, ns};
use style::attr::{AttrIdentifier, AttrValue};
use style::values::GenericAtomIdent;
use stylo_atoms::Atom;

use crate::dom::bindings::cell::{DomRefCell, Ref};
use crate::dom::bindings::codegen::Bindings::AttrBinding::AttrMethods;
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::root::{DomRoot, LayoutDom, MutNullableDom};
use crate::dom::bindings::str::DOMString;
use crate::dom::customelementregistry::CallbackReaction;
use crate::dom::document::Document;
use crate::dom::element::{AttributeMutation, Element};
use crate::dom::mutationobserver::{Mutation, MutationObserver};
use crate::dom::node::Node;
use crate::dom::virtualmethods::vtable_for;
use crate::script_runtime::CanGc;
use crate::script_thread::ScriptThread;

// https://dom.spec.whatwg.org/#interface-attr
/**
 * @brief Principal state container for a DOM Attribute.
 */
#[dom_struct]
pub(crate) struct Attr {
    node_: Node,
    #[no_trace]
    identifier: AttrIdentifier, # Encapsulates namespace, local name, and prefix.
    #[no_trace]
    value: DomRefCell<AttrValue>, # Reactive storage for the attribute's string or parsed value.

    /// the element that owns this attribute.
    owner: MutNullableDom<Element>, # Weak pointer to the associated element to prevent reference cycles.
}

impl Attr {
    /**
     * @brief Internal constructor for Attr objects.
     * Logic: Bootstraps the base 'Node' state and initializes identity metadata.
     */
    fn new_inherited(
        document: &Document,
        local_name: LocalName,
        value: AttrValue,
        name: LocalName,
        namespace: Namespace,
        prefix: Option<Prefix>,
        owner: Option<&Element>,
    ) -> Attr {
        Attr {
            node_: Node::new_inherited(document),
            identifier: AttrIdentifier {
                local_name: GenericAtomIdent(local_name),
                name: GenericAtomIdent(name),
                namespace: GenericAtomIdent(namespace),
                prefix: prefix.map(GenericAtomIdent),
            },
            value: DomRefCell::new(value),
            owner: MutNullableDom::new(owner),
        }
    }

    /**
     * @brief Factory method for creating a script-visible Attr instance.
     * Synchronization: Uses reflect_node to link the native Rust object with its JavaScript proxy.
     */
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        document: &Document,
        local_name: LocalName,
        value: AttrValue,
        name: LocalName,
        namespace: Namespace,
        prefix: Option<Prefix>,
        owner: Option<&Element>,
        can_gc: CanGc,
    ) -> DomRoot<Attr> {
        Node::reflect_node(
            Box::new(Attr::new_inherited(
                document, local_name, value, name, namespace, prefix, owner,
            )),
            document,
            can_gc,
        )
    }

    #[inline]
    pub(crate) fn name(&self) -> &LocalName {
        &self.identifier.name.0
    }

    #[inline]
    pub(crate) fn namespace(&self) -> &Namespace {
        &self.identifier.namespace.0
    }

    #[inline]
    pub(crate) fn prefix(&self) -> Option<&Prefix> {
        Some(&self.identifier.prefix.as_ref()?.0)
    }
}

/**
 * @brief Implementation of the WebIDL 'Attr' methods accessible from JavaScript.
 * Functional Utility: Maps standardized DOM behaviors to the internal Rust implementation.
 */
impl AttrMethods<crate::DomTypeHolder> for Attr {
    // https://dom.spec.whatwg.org/#dom-attr-localname
    fn LocalName(&self) -> DOMString {
        // FIXME(ajeffrey): convert directly from LocalName to DOMString
        DOMString::from(&**self.local_name())
    }

    // https://dom.spec.whatwg.org/#dom-attr-value
    fn Value(&self) -> DOMString {
        // FIXME(ajeffrey): convert directly from AttrValue to DOMString
        DOMString::from(&**self.value())
    }

    // https://dom.spec.whatwg.org/#dom-attr-value
    /**
     * @brief External-facing setter for attribute values.
     * Logic: Triggers specialized parsing logic if the attribute is currently owned by an element.
     */
    fn SetValue(&self, value: DOMString, can_gc: CanGc) {
        if let Some(owner) = self.owner() {
            let value = owner.parse_attribute(self.namespace(), self.local_name(), value);
            self.set_value(value, &owner, can_gc);
        } else {
            *self.value.borrow_mut() = AttrValue::String(value.into());
        }
    }

    // https://dom.spec.whatwg.org/#dom-attr-name
    fn Name(&self) -> DOMString {
        // FIXME(ajeffrey): convert directly from LocalName to DOMString
        DOMString::from(&**self.name())
    }

    // https://dom.spec.whatwg.org/#dom-attr-namespaceuri
    fn GetNamespaceURI(&self) -> Option<DOMString> {
        match *self.namespace() {
            ns!() => None,
            ref url => Some(DOMString::from(&**url)),
        }
    }

    // https://dom.spec.whatwg.org/#dom-attr-prefix
    fn GetPrefix(&self) -> Option<DOMString> {
        // FIXME(ajeffrey): convert directly from LocalName to DOMString
        self.prefix().map(|p| DOMString::from(&**p))
    }

    // https://dom.spec.whatwg.org/#dom-attr-ownerelement
    fn GetOwnerElement(&self) -> Option<DomRoot<Element>> {
        self.owner()
    }

    // https://dom.spec.whatwg.org/#dom-attr-specified
    fn Specified(&self) -> bool {
        true // Always returns true per specification for non-legacy browsers.
    }
}

impl Attr {
    /**
     * @brief Atomic state transition for the attribute's value.
     * Logic: Enqueues mutation records and triggers custom element callbacks (AttributeChanged).
     * Side Effects: Updates the owning element's style or state if the attribute is 'relevant'.
     */
    pub(crate) fn set_value(&self, mut value: AttrValue, owner: &Element, can_gc: CanGc) {
        let name = self.local_name().clone();
        let namespace = self.namespace().clone();
        let old_value = DOMString::from(&**self.value());
        let new_value = DOMString::from(&*value);
        let mutation = LazyCell::new(|| Mutation::Attribute {
            name: name.clone(),
            namespace: namespace.clone(),
            old_value: Some(old_value.clone()),
        });

        // Observability: Signals the engine's MutationObserver mechanism.
        MutationObserver::queue_a_mutation_record(owner.upcast::<Node>(), mutation);

        // Block Logic: Custom Element Integration.
        if owner.is_custom() {
            let reaction = CallbackReaction::AttributeChanged(
                name,
                Some(old_value),
                Some(new_value),
                namespace,
            );
            ScriptThread::enqueue_callback_reaction(owner, reaction, None);
        }

        assert_eq!(Some(owner), self.owner().as_deref());
        owner.will_mutate_attr(self);
        
        // Finalization: Atomic swap of the value buffer.
        self.swap_value(&mut value);

        // Functional Utility: Propagates change to style system or specialized element logic.
        if is_relevant_attribute(self.namespace(), self.local_name()) {
            vtable_for(owner.upcast()).attribute_mutated(
                self,
                AttributeMutation::Set(Some(&value)),
                can_gc,
            );
        }
    }

    /// Used to swap the attribute's value without triggering mutation events
    pub(crate) fn swap_value(&self, value: &mut AttrValue) {
        mem::swap(&mut *self.value.borrow_mut(), value);
    }

    pub(crate) fn identifier(&self) -> &AttrIdentifier {
        &self.identifier
    }

    pub(crate) fn value(&self) -> Ref<AttrValue> {
        self.value.borrow()
    }

    pub(crate) fn local_name(&self) -> &LocalName {
        &self.identifier.local_name
    }

    /// Sets the owner element. Should be called after the attribute is added
    /// or removed from its older parent.
    pub(crate) fn set_owner(&self, owner: Option<&Element>) {
        let ns = self.namespace();
        match (self.owner(), owner) {
            (Some(old), None) => {
                // Invariant: Verify detachment from old parent.
                assert!(
                    old.get_attribute(ns, &self.identifier.local_name)
                        .as_deref() !=
                        Some(self)
                )
            },
            (Some(old), Some(new)) => assert_eq!(&*old, new),
            _ => {},
        }
        self.owner.set(owner);
    }

    pub(crate) fn owner(&self) -> Option<DomRoot<Element>> {
        self.owner.get()
    }

    /**
     * @brief Generates a simplified representation for cross-process DevTools interaction.
     */
    pub(crate) fn summarize(&self) -> AttrInfo {
        AttrInfo {
            namespace: (**self.namespace()).to_owned(),
            name: String::from(self.Name()),
            value: String::from(self.Value()),
        }
    }

    pub(crate) fn qualified_name(&self) -> DOMString {
        match self.prefix() {
            Some(ref prefix) => DOMString::from(format!("{}:{}", prefix, &**self.local_name())),
            None => DOMString::from(&**self.local_name()),
        }
    }
}

/**
 * @brief High-performance accessor trait for layout-side interactions.
 * Synchronization: Uses unsafe getters to bypass script-side rooting for performance in styling threads.
 */
#[allow(unsafe_code)]
pub(crate) trait AttrHelpersForLayout<'dom> {
    fn value(self) -> &'dom AttrValue;
    fn as_str(&self) -> &'dom str;
    fn to_tokens(self) -> Option<&'dom [Atom]>;
    fn local_name(self) -> &'dom LocalName;
    fn namespace(self) -> &'dom Namespace;
}

#[allow(unsafe_code)]
impl<'dom> AttrHelpersForLayout<'dom> for LayoutDom<'dom, Attr> {
    #[inline]
    fn value(self) -> &'dom AttrValue {
        // Safety: assumes the caller has secured the DOM tree against concurrent mutations.
        unsafe { self.unsafe_get().value.borrow_for_layout() }
    }

    #[inline]
    fn as_str(&self) -> &'dom str {
        self.value()
    }

    #[inline]
    fn to_tokens(self) -> Option<&'dom [Atom]> {
        match *self.value() {
            AttrValue::TokenList(_, ref tokens) => Some(tokens),
            _ => None,
        }
    }

    #[inline]
    fn local_name(self) -> &'dom LocalName {
        &self.unsafe_get().identifier.local_name.0
    }

    #[inline]
    fn namespace(self) -> &'dom Namespace {
        &self.unsafe_get().identifier.namespace.0
    }
}

/// A helper function to check if attribute is relevant.
/**
 * Logic: Identifies attributes that impact presentation or behavior (e.g., SVG href).
 */
pub(crate) fn is_relevant_attribute(namespace: &Namespace, local_name: &LocalName) -> bool {
    // <https://svgwg.org/svg2-draft/linking.html#XLinkHrefAttribute>
    namespace == &ns!() || (namespace == &ns!(xlink) && local_name == &local_name!("href"))
}

/// A help function to check if an attribute is a boolean attribute.
/**
 * Logic: Heuristic matching against the HTML5 specification list of boolean attributes.
 * Strategy: Iterative ignore-case comparison.
 */
pub(crate) fn is_boolean_attribute(name: &str) -> bool {
    // The full list of attributes can be found in [1]. All attributes marked as "Boolean
    // attribute" in the "Value" column are boolean attributes. Note that "hidden" is effectively
    // treated as a boolean attribute, according to WPT test "test_global_boolean_attributes" in
    // webdriver/tests/classic/get_element_attribute/get.py
    //
    // [1] <https://html.spec.whatwg.org/multipage/#attributes-3>
    [
        "allowfullscreen",
        "alpha",
        "async",
        "autofocus",
        "autoplay",
        "checked",
        "controls",
        "default",
        "defer",
        "disabled",
        "formnovalidate",
        "hidden",
        "inert",
        "ismap",
        "itemscope",
        "loop",
        "multiple",
        "muted",
        "nomodule",
        "novalidate",
        "open",
        "playsinline",
        "readonly",
        "required",
        "reversed",
        "selected",
        "shadowrootclonable",
        "shadowrootcustomelementregistry",
        "shadowrootdelegatesfocus",
        "shadowrootserializable",
    ]
    .iter()
    .any(|&boolean_attr| boolean_attr.eq_ignore_ascii_case(name))
}
