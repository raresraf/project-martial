//! This module defines the `Attr` DOM node, which represents an attribute on an
//! `Element`. It includes the structure of the `Attr` node, its properties,
//! and its interaction with elements, mutation observers, and the script runtime,
//! as specified by the DOM Standard.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::borrow::ToOwned;
use std::cell::LazyCell;
use std::mem;
use std::sync::LazyLock;

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

/// Represents an attribute on an `Element` node.
///
/// An attribute has a name and a value. While it is a `Node` in the DOM, an `Attr`
/// node is not part of the document tree; it is owned by an `Element`.
///
/// Specification: https://dom.spec.whatwg.org/#interface-attr
#[dom_struct]
pub(crate) struct Attr {
    /// The base `Node` data, containing common properties like the owner document.
    node_: Node,
    /// A struct holding the name, namespace, and prefix of the attribute.
    #[no_trace]
    identifier: AttrIdentifier,
    /// The value of the attribute, wrapped in a `DomRefCell` to allow interior
    /// mutability with runtime borrow checking.
    #[no_trace]
    value: DomRefCell<AttrValue>,

    /// A nullable reference to the `Element` that owns this attribute.
    /// It is `None` if the attribute is not attached to any element.
    owner: MutNullableDom<Element>,
}

impl Attr {
    /// Creates a new `Attr` instance without immediately rooting it in the GC heap.
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

    /// Creates a new `Attr` instance and roots it on the GC heap.
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

    /// Returns the qualified name of the attribute.
    #[inline]
    pub(crate) fn name(&self) -> &LocalName {
        &self.identifier.name.0
    }

    /// Returns the namespace URL of the attribute.
    #[inline]
    pub(crate) fn namespace(&self) -> &Namespace {
        &self.identifier.namespace.0
    }

    /// Returns the namespace prefix of the attribute, if it has one.
    #[inline]
    pub(crate) fn prefix(&self) -> Option<&Prefix> {
        Some(&self.identifier.prefix.as_ref()?.0)
    }
}

/// Implements the public API for the `Attr` node, as exposed to JavaScript via WebIDL.
impl AttrMethods<crate::DomTypeHolder> for Attr {
    /// Gets the local name of the attribute.
    ///
    /// Specification: https://dom.spec.whatwg.org/#dom-attr-localname
    fn LocalName(&self) -> DOMString {
        // FIXME(ajeffrey): convert directly from LocalName to DOMString
        DOMString::from(&**self.local_name())
    }

    /// Gets the value of the attribute as a string.
    ///
    /// Specification: https://dom.spec.whatwg.org/#dom-attr-value
    fn Value(&self) -> DOMString {
        // FIXME(ajeffrey): convert directly from AttrValue to DOMString
        DOMString::from(&**self.value())
    }

    /// Sets the value of the attribute.
    ///
    /// If the attribute is attached to an element, this will trigger mutation
    /// observers and other related browser machinery.
    ///
    /// Specification: https://dom.spec.whatwg.org/#dom-attr-value
    fn SetValue(&self, value: DOMString, can_gc: CanGc) {
        if let Some(owner) = self.owner() {
            let value = owner.parse_attribute(self.namespace(), self.local_name(), value);
            self.set_value(value, &owner, can_gc);
        } else {
            // If not attached to an element, just update the value directly.
            *self.value.borrow_mut() = AttrValue::String(value.into());
        }
    }

    /// Gets the qualified name of the attribute (e.g., `xlink:href`).
    ///
    /// Specification: https://dom.spec.whatwg.org/#dom-attr-name
    fn Name(&self) -> DOMString {
        // FIXME(ajeffrey): convert directly from LocalName to DOMString
        DOMString::from(&**self.name())
    }

    /// Gets the namespace URI of the attribute. Returns `None` for attributes
    /// with no namespace.
    ///
    /// Specification: https://dom.spec.whatwg.org/#dom-attr-namespaceuri
    fn GetNamespaceURI(&self) -> Option<DOMString> {
        match *self.namespace() {
            ns!() => None,
            ref url => Some(DOMString::from(&**url)),
        }
    }

    /// Gets the namespace prefix of the attribute.
    ///
    /// Specification: https://dom.spec.whatwg.org/#dom-attr-prefix
    fn GetPrefix(&self) -> Option<DOMString> {
        // FIXME(ajeffrey): convert directly from LocalName to DOMString
        self.prefix().map(|p| DOMString::from(&**p))
    }

    /// Gets the element that this attribute belongs to.
    ///
    /// Specification: https://dom.spec.whatwg.org/#dom-attr-ownerelement
    fn GetOwnerElement(&self) -> Option<DomRoot<Element>> {
        self.owner()
    }

    /// Deprecated. Always returns `true`.
    ///
    /// Specification: https://dom.spec.whatwg.org/#dom-attr-specified
    fn Specified(&self) -> bool {
        true // Always returns true
    }
}

impl Attr {
    /// Sets the attribute's value and runs all associated mutation logic.
    ///
    /// This is the internal counterpart to `SetValue`, invoked when an attribute
    /// attached to an element is changed.
    pub(crate) fn set_value(&self, mut value: AttrValue, owner: &Element, can_gc: CanGc) {
        let name = self.local_name().clone();
        let namespace = self.namespace().clone();
        let old_value = DOMString::from(&**self.value());
        let new_value = DOMString::from(&*value);

        // Lazily create a mutation record. This avoids allocation if no
        // observers are registered for this node.
        let mutation = LazyCell::new(|| Mutation::Attribute {
            name: name.clone(),
            namespace: namespace.clone(),
            old_value: Some(old_value.clone()),
        });

        // Block Logic: Queue a mutation record for any active MutationObservers.
        MutationObserver::queue_a_mutation_record(owner.upcast::<Node>(), mutation);

        // Block Logic: If the owner is a custom element, enqueue a callback reaction
        // to invoke the `attributeChangedCallback`.
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
        // Notify the element that an attribute is about to change.
        owner.will_mutate_attr(self);
        // Swap the internal value.
        self.swap_value(&mut value);
        // If the attribute is relevant to styling or layout, notify the style system.
        if is_relevant_attribute(self.namespace(), self.local_name()) {
            vtable_for(owner.upcast()).attribute_mutated(
                self,
                AttributeMutation::Set(Some(&value)),
                can_gc,
            );
        }
    }

    /// Swaps the attribute's value without triggering mutation events.
    /// Used for internal state management where mutation logic is not desired.
    pub(crate) fn swap_value(&self, value: &mut AttrValue) {
        mem::swap(&mut *self.value.borrow_mut(), value);
    }

    /// Returns the internal attribute identifier.
    pub(crate) fn identifier(&self) -> &AttrIdentifier {
        &self.identifier
    }

    /// Returns a borrowed reference to the attribute's value.
    pub(crate) fn value(&self) -> Ref<AttrValue> {
        self.value.borrow()
    }

    /// Returns the local name of the attribute.
    pub(crate) fn local_name(&self) -> &LocalName {
        &self.identifier.local_name
    }

    /// Sets the owner element. Should be called after the attribute is added
    /// or removed from its older parent's attribute list.
    pub(crate) fn set_owner(&self, owner: Option<&Element>) {
        let ns = self.namespace();
        match (self.owner(), owner) {
            (Some(old), None) => {
                // Invariant: When an attribute is removed, it should no longer be
                // findable on its old owner.
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

    /// Returns the owning element, if any.
    pub(crate) fn owner(&self) -> Option<DomRoot<Element>> {
        self.owner.get()
    }

    /// Creates a serializable summary of the attribute for devtools.
    pub(crate) fn summarize(&self) -> AttrInfo {
        AttrInfo {
            namespace: (**self.namespace()).to_owned(),
            name: String::from(self.Name()),
            value: String::from(self.Value()),
        }
    }

    /// Returns the qualified name of the attribute, including prefix if present.
    pub(crate) fn qualified_name(&self) -> DOMString {
        match self.prefix() {
            Some(ref prefix) => DOMString::from(format!("{}:{}", prefix, &**self.local_name())),
            None => DOMString::from(&**self.local_name()),
        }
    }
}

/// Provides a set of performance-optimized, unsafe accessors for use by the
/// layout engine. This avoids the runtime overhead of `DomRefCell`'s borrow
/// checking during performance-critical layout calculations.
#[allow(unsafe_code)]
pub(crate) trait AttrHelpersForLayout<'dom> {
    /// Gets a direct reference to the attribute's value.
    fn value(self) -> &'dom AttrValue;
    /// Gets the attribute's value as a string slice.
    fn as_str(&self) -> &'dom str;
    /// If the attribute value is a token list, returns it as a slice of atoms.
    fn to_tokens(self) -> Option<&'dom [Atom]>;
    /// Gets a direct reference to the attribute's local name.
    fn local_name(self) -> &'dom LocalName;
    /// Gets a direct reference to the attribute's namespace.
    fn namespace(self) -> &'dom Namespace;
}

/// Implements the optimized layout helpers for `Attr`.
#[allow(unsafe_code)]
impl<'dom> AttrHelpersForLayout<'dom> for LayoutDom<'dom, Attr> {
    #[inline]
    fn value(self) -> &'dom AttrValue {
        // SAFETY: This is safe because layout has exclusive access to the DOM,
        // so there are no other threads that could be mutating this value.
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

/// A helper function to check if an attribute is relevant to styling or layout.
/// Changes to these attributes may require re-styling or re-layout.
pub(crate) fn is_relevant_attribute(namespace: &Namespace, local_name: &LocalName) -> bool {
    // For example, any attribute in the default namespace, or `xlink:href` in SVG.
    // <https://svgwg.org/svg2-draft/linking.html#XLinkHrefAttribute>
    namespace == &ns!() || (namespace == &ns!(xlink) && local_name == &local_name!("href"))
}

/// A helper function to check if an attribute is a boolean attribute according
/// to the HTML specification.
pub(crate) fn is_boolean_attribute(name: &str) -> bool {
    // The full list of attributes can be found in [1]. All attributes marked as "Boolean
    // attribute" in the "Value" column are boolean attributes. Note that "hidden" is effectively
    // treated as a boolean attribute, according to WPT test "test_global_boolean_attributes" in
    // webdriver/tests/classic/get_element_attribute/get.py
    //
    // [1] <https://html.spec.whatwg.org/multipage/#attributes-3>
    // `LazyLock` ensures this array is initialized only once, providing efficient lookups.
    static BOOLEAN_ATTRIBUTES: LazyLock<[&str; 30]> = LazyLock::new(|| {
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
    });

    BOOLEAN_ATTRIBUTES
        .iter()
        .any(|&boolean_attr| boolean_attr.eq_ignore_ascii_case(name))
}
