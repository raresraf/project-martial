/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Implementation of the DOM `Attr` interface.
//!
//! An `Attr` object represents an attribute of an `Element`. In the modern DOM
//! specification, attributes are no longer nodes in the same sense as elements
//! or text nodes, but they still inherit from `Node` for legacy reasons in this
//! implementation.

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

/// https://dom.spec.whatwg.org/#interface-attr
/// 
/// Represents a name-value pair belonging to an Element.
/// Orchestrates the synchronization between the attribute's string representation
/// and its parsed internal state used by the style system.
#[dom_struct]
pub(crate) struct Attr {
    node_: Node,
    #[no_trace]
    identifier: AttrIdentifier,
    #[no_trace]
    value: DomRefCell<AttrValue>,

    /// The element that owns this attribute. Acts as the context for mutation events.
    owner: MutNullableDom<Element>,
}

impl Attr {
    /// Internal constructor for creating a partially initialized Attr object
    /// within an inheritance hierarchy.
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

    /// Public constructor for creating a new Attr node and reflecting it into the JS runtime.
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

    /// Functional Utility: Accessor for the attribute's name atom.
    #[inline]
    pub(crate) fn name(&self) -> &LocalName {
        &self.identifier.name.0
    }

    /// Functional Utility: Accessor for the XML namespace of the attribute.
    #[inline]
    pub(crate) fn namespace(&self) -> &Namespace {
        &self.identifier.namespace.0
    }

    /// Functional Utility: Accessor for the namespace prefix (e.g., 'xlink' in 'xlink:href').
    #[inline]
    pub(crate) fn prefix(&self) -> Option<&Prefix> {
        Some(&self.identifier.prefix.as_ref()?.0)
    }
}

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

    /// https://dom.spec.whatwg.org/#dom-attr-value
    /// 
    /// Entry point for updating the attribute's content from script.
    /// Triggers parsing logic if an owner element exists, ensuring the internal
    /// state remains consistent with the element's specific attribute rules.
    fn SetValue(&self, value: DOMString, can_gc: CanGc) {
        if let Some(owner) = self.owner() {
            // Block Logic: Context-aware parsing.
            // If the attribute is attached to an element, use the element's 
            // specific parser (e.g., for numeric or enumerated attributes).
            let value = owner.parse_attribute(self.namespace(), self.local_name(), value);
            self.set_value(value, &owner, can_gc);
        } else {
            // Invariant: Unowned attributes are stored as raw strings.
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
        true // Always returns true as per modern spec.
    }
}

impl Attr {
    /// High-level value update logic that orchestrates DOM side-effects.
    /// Handles:
    /// 1. Mutation record queuing for MutationObservers.
    /// 2. Custom Element 'attributeChangedCallback' reactions.
    /// 3. Invalidation of layout/style caches in the owner element.
    pub(crate) fn set_value(&self, mut value: AttrValue, owner: &Element, can_gc: CanGc) {
        let name = self.local_name().clone();
        let namespace = self.namespace().clone();
        let old_value = DOMString::from(&**self.value());
        let new_value = DOMString::from(&*value);
        
        // Optimization: Defer mutation record creation until needed.
        let mutation = LazyCell::new(|| Mutation::Attribute {
            name: name.clone(),
            namespace: namespace.clone(),
            old_value: Some(old_value.clone()),
        });

        // Block Logic: Observer Notification.
        // Invariant: The marketplace of observers must be notified before the physical swap.
        MutationObserver::queue_a_mutation_record(owner.upcast::<Node>(), mutation);

        // Block Logic: Custom Element Integration.
        // Logic: If the owner is a custom element, schedule its attribute change reaction.
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
        
        // Semantic Step: Invalidate owner state before updating.
        owner.will_mutate_attr(self);
        
        // Physical State Transition.
        self.swap_value(&mut value);
        
        // Block Logic: Engine Invalidation.
        // If the attribute affects rendering (is 'relevant'), trigger the owner's 
        // virtual mutation handler to schedule layout/paint updates.
        if is_relevant_attribute(self.namespace(), self.local_name()) {
            vtable_for(owner.upcast()).attribute_mutated(
                self,
                AttributeMutation::Set(Some(&value)),
                can_gc,
            );
        }
    }

    /// Low-level atomic value swap.
    /// Used to update the internal representation without triggering the DOM event cascade.
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

    /// Updates the owner reference.
    /// Maintains the invariant that an attribute cannot be shared between elements.
    pub(crate) fn set_owner(&self, owner: Option<&Element>) {
        let ns = self.namespace();
        match (self.owner(), owner) {
            (Some(old), None) => {
                // Assert that removal from element registry has already occurred.
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

    /// Provides a serialized summary for devtools or debugging.
    pub(crate) fn summarize(&self) -> AttrInfo {
        AttrInfo {
            namespace: (**self.namespace()).to_owned(),
            name: String::from(self.Name()),
            value: String::from(self.Value()),
        }
    }

    /// Returns the full name including prefix (e.g. "prefix:localName").
    pub(crate) fn qualified_name(&self) -> DOMString {
        match self.prefix() {
            Some(ref prefix) => DOMString::from(format!("{}:{}", prefix, &**self.local_name())),
            None => DOMString::from(&**self.local_name()),
        }
    }
}

/// Specialized trait for high-performance attribute access during layout traversal.
/// Bypasses DOM root management where safe.
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
        // Optimization: Raw access to value for style/layout threads.
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

/// Identifies attributes that trigger engine re-evaluations (style/layout/logic).
/// Currently filters for null-namespace attributes and XLink hrefs.
pub(crate) fn is_relevant_attribute(namespace: &Namespace, local_name: &LocalName) -> bool {
    // <https://svgwg.org/svg2-draft/linking.html#XLinkHrefAttribute>
    namespace == &ns!() || (namespace == &ns!(xlink) && local_name == &local_name!("href"))
}

/// Determines if an attribute follows the HTML boolean attribute rule
/// (presence implies true, regardless of value string).
pub(crate) fn is_boolean_attribute(name: &str) -> bool {
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
