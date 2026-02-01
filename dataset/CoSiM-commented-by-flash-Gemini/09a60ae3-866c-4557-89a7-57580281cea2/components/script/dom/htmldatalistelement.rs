/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// @file htmldatalistelement.rs
/// @brief This file implements the `HTMLDataListElement` interface, which represents the
/// `<datalist>` HTML element. It provides functionality for managing its options and
/// integrating with the DOM.
/// Functional Utility: Enables the creation and manipulation of `<datalist>` elements,
/// used to provide a list of predefined options for input controls.

use dom_struct::dom_struct; // Macro for defining DOM structures.
use html5ever::{LocalName, Prefix}; // HTML5 parsing types (LocalName for tag, Prefix for XML namespace prefix).
use js::rust::HandleObject; // JavaScript object handle.

use crate::dom::bindings::codegen::Bindings::HTMLDataListElementBinding::HTMLDataListElementMethods; // Generated bindings for HTMLDataListElement methods.
use crate::dom::bindings::inheritance::Castable; // Trait for safe downcasting between DOM types.
use crate::dom::bindings::root::DomRoot; // Root type for DOM objects.
use crate::dom::document::Document; // Document object.
use crate::dom::htmlcollection::HTMLCollection; // HTMLCollection for live collections of elements.
use crate::dom::htmlelement::HTMLElement; // Base HTMLElement type.
use crate::dom::htmloptionelement::HTMLOptionElement; // HTMLOptionElement type.
use crate::dom::node::{Node, NodeTraits}; // Node types and traits.
use crate::script_runtime::CanGc; // Marker trait for types that can be garbage collected.

/// @struct HTMLDataListElement
/// @brief Represents the `<datalist>` HTML element.
/// Functional Utility: Provides the DOM interface for `<datalist>`, allowing access to its
/// child `<option>` elements via a live `HTMLCollection`.
///
/// <https://html.spec.whatwg.org/multipage/#htmldatalistelement>
#[dom_struct]
pub(crate) struct HTMLDataListElement {
    htmlelement: HTMLElement, //!< Inherited properties and methods from `HTMLElement`.
}

impl HTMLDataListElement {
    /// @brief Creates a new `HTMLDataListElement` instance with inherited properties.
    /// Functional Utility: Internal constructor used for setting up the basic properties
    /// of an `HTMLDataListElement` based on its tag name, prefix, and owning document.
    ///
    /// @param local_name The `LocalName` of the HTML element (e.g., "datalist").
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix, if any.
    /// @param document The owning `Document` of this element.
    /// @return A new `HTMLDataListElement` instance.
    fn new_inherited(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
    ) -> HTMLDataListElement {
        HTMLDataListElement {
            htmlelement: HTMLElement::new_inherited(local_name, prefix, document), // Initialize base HTMLElement.
        }
    }

    /// @brief Creates a new `HTMLDataListElement` and reflects it into the DOM.
    /// Functional Utility: Public constructor that builds an `HTMLDataListElement` and makes
    /// it accessible from JavaScript by integrating it into the DOM's object graph.
    ///
    /// @param local_name The `LocalName` of the HTML element.
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix.
    /// @param document The owning `Document` of this element.
    /// @param proto An `Option<HandleObject>` specifying the JavaScript prototype chain.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLDataListElement` instance wrapped in `DomRoot`.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<HTMLDataListElement> {
        Node::reflect_node_with_proto(
            Box::new(HTMLDataListElement::new_inherited(
                local_name, prefix, document,
            )),
            document,
            proto,
            can_gc,
        )
    }
}

impl HTMLDataListElementMethods<crate::DomTypeHolder> for HTMLDataListElement {
    /// @brief Returns a live `HTMLCollection` of all `<option>` elements in the datalist.
    /// Functional Utility: Implements the `options` getter for `HTMLDataListElement`,
    /// providing access to its associated `<option>` elements.
    /// <https://html.spec.whatwg.org/multipage/#dom-datalist-options>
    ///
    /// @param can_gc A `CanGc` token.
    /// @return A `DomRoot<HTMLCollection>` containing the `<option>` elements.
    fn Options(&self, can_gc: CanGc) -> DomRoot<HTMLCollection> {
        HTMLCollection::new_with_filter_fn(
            &self.owner_window(), // The owning window for the collection.
            self.upcast(),        // The datalist element itself is the root for filtering.
            // Filter function: Include only elements that are `HTMLOptionElement`s.
            |element, _| element.is::<HTMLOptionElement>(),
            can_gc,
        )
    }
}