/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// @file htmldialogelement.rs
/// @brief This file implements the `HTMLDialogElement` interface, which represents the
/// `<dialog>` HTML element. It provides functionality for controlling its open/closed state,
/// managing a return value, and interacting with its display behavior.
/// Functional Utility: Enables the creation and manipulation of modal or non-modal dialog boxes
/// in web applications.

use dom_struct::dom_struct; // Macro for defining DOM structures.
use html5ever::{LocalName, Prefix, local_name, namespace_url, ns}; // HTML5 parsing types.
use js::rust::HandleObject; // JavaScript object handle.

use crate::dom::bindings::cell::DomRefCell; // DOM-specific RefCell.
use crate::dom::bindings::codegen::Bindings::HTMLDialogElementBinding::HTMLDialogElementMethods; // Generated bindings for HTMLDialogElement methods.
use crate::dom::bindings::inheritance::Castable; // Trait for safe downcasting between DOM types.
use crate::dom::bindings::root::DomRoot; // Root type for DOM objects.
use crate::dom::bindings::str::DOMString; // DOMString representation.
use crate::dom::document::Document; // Document object.
use crate::dom::element::Element; // Element type.
use crate::dom::eventtarget::EventTarget; // Event target.
use crate::dom::htmlelement::HTMLElement; // Base HTMLElement type.
use crate::dom::node::{Node, NodeTraits}; // Node types and traits.
use crate::script_runtime::CanGc; // Marker trait for types that can be garbage collected.

/// @struct HTMLDialogElement
/// @brief Represents the `<dialog>` HTML element.
/// Functional Utility: Provides the DOM interface for `<dialog>`, allowing programmatic
/// control over its visibility, its return value when closed, and its interaction
/// with other elements in the document.
///
/// <https://html.spec.whatwg.org/multipage/#htmldialogelement>
#[dom_struct]
pub(crate) struct HTMLDialogElement {
    htmlelement: HTMLElement, //!< Inherited properties and methods from `HTMLElement`.
    return_value: DomRefCell<DOMString>, //!< Stores the return value of the dialog when it is closed.
}

impl HTMLDialogElement {
    /// @brief Creates a new `HTMLDialogElement` instance with inherited properties.
    /// Functional Utility: Internal constructor used for setting up the basic properties
    /// of an `HTMLDialogElement` based on its tag name, prefix, and owning document.
    ///
    /// @param local_name The `LocalName` of the HTML element (e.g., "dialog").
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix, if any.
    /// @param document The owning `Document` of this element.
    /// @return A new `HTMLDialogElement` instance.
    fn new_inherited(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
    ) -> HTMLDialogElement {
        HTMLDialogElement {
            htmlelement: HTMLElement::new_inherited(local_name, prefix, document), // Initialize base HTMLElement.
            return_value: DomRefCell::new(DOMString::new()), // Initialize return value to an empty DOMString.
        }
    }

    /// @brief Creates a new `HTMLDialogElement` and reflects it into the DOM.
    /// Functional Utility: Public constructor that builds an `HTMLDialogElement` and makes
    /// it accessible from JavaScript by integrating it into the DOM's object graph.
    ///
    /// @param local_name The `LocalName` of the HTML element.
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix.
    /// @param document The owning `Document` of this element.
    /// @param proto An `Option<HandleObject>` specifying the JavaScript prototype chain.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLDialogElement` instance wrapped in `DomRoot`.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<HTMLDialogElement> {
        Node::reflect_node_with_proto(
            Box::new(HTMLDialogElement::new_inherited(
                local_name, prefix, document,
            )),
            document,
            proto,
            can_gc,
        )
    }
}

impl HTMLDialogElementMethods<crate::DomTypeHolder> for HTMLDialogElement {
    // https://html.spec.whatwg.org/multipage/#dom-dialog-open
    /// @brief Returns the current open state of the `<dialog>` element.
    /// Functional Utility: Implements the `open` getter, reflecting the presence
    /// of the `open` attribute.
    /// @return `true` if the dialog is open, `false` otherwise.
    make_bool_getter!(Open, "open");

    // https://html.spec.whatwg.org/multipage/#dom-dialog-open
    /// @brief Sets the open state of the `<dialog>` element.
    /// Functional Utility: Implements the `open` setter, controlling the visibility
    /// of the dialog by adding or removing the `open` attribute.
    /// @param value `true` to open the dialog, `false` to close it.
    make_bool_setter!(SetOpen, "open");

    // https://html.spec.whatwg.org/multipage/#dom-dialog-returnvalue
    /// @brief Returns the return value of the dialog.
    /// Functional Utility: Retrieves the value that was set when the dialog was closed,
    /// typically via the `close()` method.
    /// @return A `DOMString` representing the return value.
    fn ReturnValue(&self) -> DOMString {
        let return_value = self.return_value.borrow(); // Borrow the internal return_value.
        return_value.clone() // Return a clone of the DOMString.
    }

    // https://html.spec.whatwg.org/multipage/#dom-dialog-returnvalue
    /// @brief Sets the return value of the dialog.
    /// Functional Utility: Allows the return value of the dialog to be set programmatically.
    /// @param return_value The `DOMString` to set as the return value.
    fn SetReturnValue(&self, return_value: DOMString) {
        *self.return_value.borrow_mut() = return_value; // Update the internal return_value.
    }

    /// @brief Shows the dialog element, making it visible to the user.
    /// Functional Utility: Adds the `open` attribute to the dialog,
    /// triggering its display and potentially managing focus and other popover states.
    /// <https://html.spec.whatwg.org/multipage/#dom-dialog-show>
    ///
    /// @param can_gc A `CanGc` token.
    fn Show(&self, can_gc: CanGc) {
        let element = self.upcast::<Element>(); // Get the element as a generic `Element`.

        // Step 1 TODO: Check is modal flag is false
        // Block Logic: If the dialog already has the "open" attribute, it's already shown, so return.
        if element.has_attribute(&local_name!("open")) {
            return;
        }

        // TODO: Step 2 If this has an open attribute, then throw an "InvalidStateError" DOMException.

        // Step 3: Add the "open" attribute to the element.
        element.set_bool_attribute(&local_name!("open"), true, can_gc);

        // TODO: Step 4 Set this's previously focused element to the focused element.

        // TODO: Step 5 Let hideUntil be the result of running topmost popover ancestor given this, null, and false.

        // TODO: Step 6 If hideUntil is null, then set hideUntil to this's node document.

        // TODO: Step 7 Run hide all popovers until given hideUntil, false, and true.

        // TODO(Issue #32702): Step 8 Run the dialog focusing steps given this.
    }

    /// @brief Closes the dialog element, making it hidden.
    /// Functional Utility: Removes the `open` attribute from the dialog,
    /// hides its display, sets an optional return value, and queues a `close` event.
    /// <https://html.spec.whatwg.org/multipage/#dom-dialog-close>
    ///
    /// @param return_value An `Option<DOMString>` to set as the dialog's return value.
    /// @param can_gc A `CanGc` token.
    fn Close(&self, return_value: Option<DOMString>, can_gc: CanGc) {
        let element = self.upcast::<Element>(); // Get the element as a generic `Element`.
        let target = self.upcast::<EventTarget>(); // Get the element as an `EventTarget`.

        // Step 1 & 2: Remove the "open" attribute. If it wasn't present, return.
        // Block Logic: Attempts to remove the "open" attribute. If it was already absent, return.
        if element
            .remove_attribute(&ns!(), &local_name!("open"), can_gc)
            .is_none()
        {
            return;
        }

        // Step 3: If a return value is provided, set it.
        if let Some(new_value) = return_value {
            *self.return_value.borrow_mut() = new_value; // Update the internal return_value.
        }

        // TODO: Step 4 implement pending dialog stack removal

        // Step 5: Queue a simple "close" event for the dialog.
        self.owner_global()
            .task_manager()
            .dom_manipulation_task_source()
            .queue_simple_event(target, atom!("close"));
    }
}