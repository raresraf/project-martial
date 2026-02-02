/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// @file htmlfieldsetelement.rs
/// @brief This file implements the `HTMLFieldSetElement` interface, which represents the
/// `<fieldset>` HTML element. It provides functionality for managing its disabled state,
/// associating with a form, and handling validation of its contained form controls.
/// Functional Utility: Organizes form controls into logical groups, enables/disables them
/// collectively, and participates in form submission and validation.

use std::default::Default;

use dom_struct::dom_struct;
use html5ever::{LocalName, Prefix, local_name};
use js::rust::HandleObject;
use stylo_dom::ElementState;

use crate::dom::attr::Attr;
use crate::dom::bindings::codegen::Bindings::HTMLFieldSetElementBinding::HTMLFieldSetElementMethods;
use crate::dom::bindings::inheritance::{Castable, ElementTypeId, HTMLElementTypeId, NodeTypeId};
use crate::dom::bindings::root::{DomRoot, MutNullableDom};
use crate::dom::bindings::str::DOMString;
use crate::dom::customelementregistry::CallbackReaction;
use crate::dom::document::Document;
use crate::dom::element::{AttributeMutation, Element};
use crate::dom::htmlcollection::HTMLCollection;
use crate::dom::htmlelement::HTMLElement;
use crate::dom::htmlformelement::{FormControl, HTMLFormElement};
use crate::dom::htmllegendelement::HTMLLegendElement;
use crate::dom::node::{Node, NodeTraits, ShadowIncluding};
use crate::dom::validation::Validatable;
use crate::dom::validitystate::ValidityState;
use crate::dom::virtualmethods::VirtualMethods;
use crate::script_runtime::CanGc;
use crate::script_thread::ScriptThread;

/// @struct HTMLFieldSetElement
/// @brief Represents the `<fieldset>` HTML element.
/// Functional Utility: Provides the DOM interface for `<fieldset>`, allowing control over
/// its disabled state, association with a form, and providing methods for form validation.
///
/// <https://html.spec.whatwg.org/multipage/#htmlfieldsetelement>
#[dom_struct]
pub(crate) struct HTMLFieldSetElement {
    htmlelement: HTMLElement, //!< Inherited properties and methods from `HTMLElement`.
    form_owner: MutNullableDom<HTMLFormElement>, //!< The `HTMLFormElement` this fieldset belongs to.
    validity_state: MutNullableDom<ValidityState>, //!< The `ValidityState` object for this fieldset.
}

impl HTMLFieldSetElement {
    /// @brief Creates a new `HTMLFieldSetElement` instance with inherited properties.
    /// Functional Utility: Internal constructor used for setting up the basic properties
    /// of an `HTMLFieldSetElement` and initializing its internal state, including
    /// its default enabled and valid states.
    ///
    /// @param local_name The `LocalName` of the HTML element (e.g., "fieldset").
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix, if any.
    /// @param document The owning `Document` of this element.
    /// @return A new `HTMLFieldSetElement` instance.
    fn new_inherited(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
    ) -> HTMLFieldSetElement {
        HTMLFieldSetElement {
            htmlelement: HTMLElement::new_inherited_with_state(
                ElementState::ENABLED | ElementState::VALID, // Fieldset is enabled and valid by default.
                local_name,
                prefix,
                document,
            ),
            form_owner: Default::default(), // Initialize form owner to default (None).
            validity_state: Default::default(), // Initialize validity state to default (None).
        }
    }

    /// @brief Creates a new `HTMLFieldSetElement` and reflects it into the DOM.
    /// Functional Utility: Public constructor that builds an `HTMLFieldSetElement` and makes
    /// it accessible from JavaScript by integrating it into the DOM's object graph.
    ///
    /// @param local_name The `LocalName` of the HTML element.
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix.
    /// @param document The owning `Document` of this element.
    /// @param proto An `Option<HandleObject>` specifying the JavaScript prototype chain.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLFieldSetElement` instance wrapped in `DomRoot`.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<HTMLFieldSetElement> {
        Node::reflect_node_with_proto(
            Box::new(HTMLFieldSetElement::new_inherited(
                local_name, prefix, document,
            )),
            document,
            proto,
            can_gc,
        )
    }

    /// @brief Updates the validity state of the `<fieldset>` based on its children.
    /// Functional Utility: Iterates through all descendant elements to check if any
    /// of them are invalid, and updates the fieldset's own validity state accordingly.
    ///
    /// @param can_gc A `CanGc` token.
    pub(crate) fn update_validity(&self, can_gc: CanGc) {
        // Block Logic: Check if any descendant element is in an invalid state.
        let has_invalid_child = self
            .upcast::<Node>()
            .traverse_preorder(ShadowIncluding::No) // Traverse all descendants.
            .flat_map(DomRoot::downcast::<Element>) // Filter for Element types.
            .any(|element| element.is_invalid(false, can_gc)); // Check if any element is invalid.

        // Block Logic: Update the fieldset's own valid/invalid state.
        self.upcast::<Element>()
            .set_state(ElementState::VALID, !has_invalid_child);
        self.upcast::<Element>()
            .set_state(ElementState::INVALID, has_invalid_child);
    }
}

impl HTMLFieldSetElementMethods<crate::DomTypeHolder> for HTMLFieldSetElement {
    // https://html.spec.whatwg.org/multipage/#dom-fieldset-elements
    /// @brief Returns a live `HTMLCollection` of all listed elements within the `<fieldset>`.
    /// Functional Utility: Implements the `elements` getter, providing access to form controls
    /// that are descendants of the fieldset and participate in form submission.
    ///
    /// @param can_gc A `CanGc` token.
    /// @return A `DomRoot<HTMLCollection>` containing the listed elements.
    fn Elements(&self, can_gc: CanGc) -> DomRoot<HTMLCollection> {
        HTMLCollection::new_with_filter_fn(
            &self.owner_window(), // The owning window for the collection.
            self.upcast(),        // The fieldset element itself is the root for filtering.
            // Filter function: Include only elements that are `HTMLElement` and are `listed_element`.
            |element, _| {
                element
                    .downcast::<HTMLElement>()
                    .is_some_and(HTMLElement::is_listed_element)
            },
            can_gc,
        )
    }

    // https://html.spec.whatwg.org/multipage/#dom-fieldset-disabled
    /// @brief Returns the disabled state of the `<fieldset>` element.
    /// Functional Utility: Implements the `disabled` getter, reflecting the presence
    /// of the `disabled` attribute.
    /// @return `true` if the fieldset is disabled, `false` otherwise.
    make_bool_getter!(Disabled, "disabled");

    // https://html.spec.whatwg.org/multipage/#dom-fieldset-disabled
    /// @brief Sets the disabled state of the `<fieldset>` element.
    /// Functional Utility: Implements the `disabled` setter, controlling the `disabled`
    /// attribute and consequently the interactability of its contained form controls.
    /// @param value `true` to disable the fieldset, `false` to enable it.
    make_bool_setter!(SetDisabled, "disabled");

    // https://html.spec.whatwg.org/multipage/#dom-fe-name
    /// @brief Sets the `name` attribute of the `<fieldset>` element.
    /// Functional Utility: Implements the `name` setter for the fieldset.
    make_atomic_setter!(SetName, "name");

    // https://html.spec.whatwg.org/multipage/#dom-fe-name
    /// @brief Returns the `name` attribute of the `<fieldset>` element.
    /// Functional Utility: Implements the `name` getter for the fieldset.
    make_getter!(Name, "name");

    // https://html.spec.whatwg.org/multipage/#dom-fae-form
    /// @brief Returns the `HTMLFormElement` that this `<fieldset>` belongs to.
    /// Functional Utility: Implements the `form` getter, providing access to the
    /// owning form element, which can be implicitly determined or explicitly set by `form` attribute.
    /// @return An `Option<DomRoot<HTMLFormElement>>` containing the form, or `None`.
    fn GetForm(&self) -> Option<DomRoot<HTMLFormElement>> {
        self.form_owner() // Delegates to the `FormControl` trait's `form_owner` method.
    }

    // https://html.spec.whatwg.org/multipage/#dom-cva-willvalidate
    /// @brief Returns `true` if the element will be validated.
    /// Functional Utility: Implements the `willValidate` getter, determining if
    /// the fieldset participates in form validation (which it does not directly).
    /// @return `true` if the element will be validated, `false` otherwise.
    fn WillValidate(&self) -> bool {
        self.is_instance_validatable() // Delegates to the `Validatable` trait's method.
    }

    // https://html.spec.whatwg.org/multipage/#dom-cva-validity
    /// @brief Returns the `ValidityState` object for the element.
    /// Functional Utility: Implements the `validity` getter, providing an object
    /// that represents the validation state of the fieldset.
    /// @return A `DomRoot<ValidityState>`.
    fn Validity(&self) -> DomRoot<ValidityState> {
        self.validity_state() // Delegates to the `Validatable` trait's method.
    }

    // https://html.spec.whatwg.org/multipage/#dom-cva-checkvalidity
    /// @brief Checks the validity of the element and its descendants.
    /// Functional Utility: Implements the `checkValidity()` method, triggering
    /// validation checks and returning `true` if all contained controls are valid.
    /// @param can_gc A `CanGc` token.
    /// @return `true` if the fieldset and its controls are valid, `false` otherwise.
    fn CheckValidity(&self, can_gc: CanGc) -> bool {
        self.check_validity(can_gc) // Delegates to the `Validatable` trait's method.
    }

    // https://html.spec.whatwg.org/multipage/#dom-cva-reportvalidity
    /// @brief Reports the validity of the element and its descendants.
    /// Functional Utility: Implements the `reportValidity()` method, which checks
    /// validity and informs the user about any invalid states.
    /// @param can_gc A `CanGc` token.
    /// @return `true` if the fieldset and its controls are valid, `false` otherwise.
    fn ReportValidity(&self, can_gc: CanGc) -> bool {
        self.report_validity(can_gc) // Delegates to the `Validatable` trait's method.
    }

    // https://html.spec.whatwg.org/multipage/#dom-cva-validationmessage
    /// @brief Returns the validation message for the element.
    /// Functional Utility: Implements the `validationMessage` getter, providing
    /// a human-readable message indicating why the fieldset (or its children) is invalid.
    /// @return A `DOMString` containing the validation message.
    fn ValidationMessage(&self) -> DOMString {
        self.validation_message() // Delegates to the `Validatable` trait's method.
    }

    // https://html.spec.whatwg.org/multipage/#dom-cva-setcustomvalidity
    /// @brief Sets a custom validation error message for the element.
    /// Functional Utility: Implements the `setCustomValidity()` method, allowing custom
    /// validation feedback to be provided.
    /// @param error A `DOMString` representing the custom error message.
    fn SetCustomValidity(&self, error: DOMString) {
        self.validity_state().set_custom_error_message(error); // Delegates to `ValidityState`.
    }

    /// <https://html.spec.whatwg.org/multipage/#dom-fieldset-type>
    /// @brief Returns the string "fieldset".
    /// Functional Utility: Implements the `type` getter, identifying the element type.
    fn Type(&self) -> DOMString {
        DOMString::from_string(String::from("fieldset"))
    }
}

impl VirtualMethods for HTMLFieldSetElement {
    /// @brief Returns the `VirtualMethods` implementation of the super type (`HTMLElement`).
    /// Functional Utility: Enables method overriding and calls to the superclass's implementations.
    /// @return An `Option` containing a reference to the super type's `VirtualMethods`.
    fn super_type(&self) -> Option<&dyn VirtualMethods> {
        Some(self.upcast::<HTMLElement>() as &dyn VirtualMethods) // Upcast to HTMLElement and get its VirtualMethods.
    }

    /// @brief Handles attribute mutations for the `<fieldset>` element.
    /// Functional Utility: Responds to changes in the `disabled` and `form` attributes,
    /// propagating disabled states to contained form controls and updating form associations.
    ///
    /// @param attr The `Attr` that was mutated.
    /// @param mutation The type of `AttributeMutation` that occurred.
    /// @param can_gc A `CanGc` token.
    fn attribute_mutated(&self, attr: &Attr, mutation: AttributeMutation, can_gc: CanGc) {
        self.super_type()
            .unwrap()
            .attribute_mutated(attr, mutation, can_gc); // Call super type's method.
        match *attr.local_name() {
            local_name!("disabled") => {
                // Block Logic: Determine if the fieldset is now disabled or re-enabled.
                let disabled_state = match mutation {
                    AttributeMutation::Set(None) => true, // `disabled` attribute was added without a value.
                    AttributeMutation::Set(Some(_)) => {
                        // Fieldset was already disabled before.
                        return; // No change in disabled state.
                    },
                    AttributeMutation::Removed => false, // `disabled` attribute was removed.
                };
                let node = self.upcast::<Node>(); // Get the fieldset as a Node.
                let element = self.upcast::<Element>(); // Get the fieldset as an Element.
                element.set_disabled_state(disabled_state); // Update its own disabled state.
                element.set_enabled_state(!disabled_state); // Update its own enabled state.

                let mut found_legend = false;
                // Block Logic: Filter children to exclude the `<legend>` element.
                let children = node.children().filter(|node| {
                    if found_legend {
                        true // Include all subsequent children after <legend>.
                    } else if node.is::<HTMLLegendElement>() {
                        found_legend = true;
                        false // Exclude the first <legend> element.
                    } else {
                        true // Include other children before <legend>.
                    }
                });
                // Block Logic: Collect all descendant form controls within the fieldset (excluding <legend> subtree).
                let fields = children.flat_map(|child| {
                    child
                        .traverse_preorder(ShadowIncluding::No) // Traverse descendants of children.
                        .filter(|descendant| match descendant.type_id() {
                            NodeTypeId::Element(ElementTypeId::HTMLElement(
                                HTMLElementTypeId::HTMLButtonElement |
                                HTMLElementTypeId::HTMLInputElement |
                                HTMLElementTypeId::HTMLSelectElement |
                                HTMLElementTypeId::HTMLTextAreaElement,
                            )) => true, // Standard form controls.
                            NodeTypeId::Element(ElementTypeId::HTMLElement(
                                HTMLElementTypeId::HTMLElement,
                            )) => descendant
                                .downcast::<HTMLElement>()
                                .unwrap()
                                .is_form_associated_custom_element(), // Form-associated custom elements.
                            _ => false, // Other types are not form controls for this purpose.
                        })
                });
                if disabled_state {
                    // Block Logic: If the fieldset is disabled, disable all contained form controls.
                    for field in fields {
                        let element = field.downcast::<Element>().unwrap(); // Get the form control element.
                        if element.enabled_state() {
                            element.set_disabled_state(true);
                            element.set_enabled_state(false);
                            if element
                                .downcast::<HTMLElement>()
                                .is_some_and(|h| h.is_form_associated_custom_element())
                            {
                                ScriptThread::enqueue_callback_reaction(
                                    element,
                                    CallbackReaction::FormDisabled(true), // Enqueue form disabled callback.
                                    None,
                                );
                            }
                        }
                        element.update_sequentially_focusable_status(can_gc); // Update focusable status.
                    }
                } else {
                    // Block Logic: If the fieldset is re-enabled, re-enable contained form controls,
                    // unless they are explicitly disabled or disabled by another ancestor fieldset.
                    for field in fields {
                        let element = field.downcast::<Element>().unwrap(); // Get the form control element.
                        if element.disabled_state() {
                            element.check_disabled_attribute(); // Re-check its own `disabled` attribute.
                            element.check_ancestors_disabled_state_for_form_control(); // Re-check ancestor fieldsets.
                            // Fire callback only if this has actually enabled the custom element
                            if element.enabled_state() &&
                                element
                                    .downcast::<HTMLElement>()
                                    .is_some_and(|h| h.is_form_associated_custom_element())
                            {
                                ScriptThread::enqueue_callback_reaction(
                                    element,
                                    CallbackReaction::FormDisabled(false), // Enqueue form enabled callback.
                                    None,
                                );
                            }
                        }
                        element.update_sequentially_focusable_status(can_gc); // Update focusable status.
                    }
                }
                element.update_sequentially_focusable_status(can_gc); // Update fieldset's focusable status.
            },
            local_name!("form") => {
                // Block Logic: Handle mutation of the `form` attribute.
                self.form_attribute_mutated(mutation, can_gc); // Delegates to HTMLElement's form attribute handling.
            },
            _ => {}, // Other attribute mutations are ignored.
        }
    }
}

impl FormControl for HTMLFieldSetElement {
    /// @brief Returns the form owner of the element.
    /// Functional Utility: Implements the `form_owner` method for `FormControl` trait,
    /// providing access to the `HTMLFormElement` this fieldset is associated with.
    /// @return An `Option<DomRoot<HTMLFormElement>>` containing the form owner, or `None`.
    fn form_owner(&self) -> Option<DomRoot<HTMLFormElement>> {
        self.form_owner.get() // Retrieve the cached form owner.
    }

    /// @brief Sets the form owner of the element.
    /// Functional Utility: Implements the `set_form_owner` method for `FormControl` trait,
    /// updating the association of this fieldset with a given `HTMLFormElement`.
    /// @param form An `Option<&HTMLFormElement>` for the new form owner.
    fn set_form_owner(&self, form: Option<&HTMLFormElement>) {
        self.form_owner.set(form); // Set the form owner.
    }

    /// @brief Returns a reference to the underlying `Element`.
    /// Functional Utility: Provides access to the generic `Element` methods and properties.
    /// @return A reference to the `Element`.
    fn to_element(&self) -> &Element {
        self.upcast::<Element>() // Upcast to Element.
    }
}

impl Validatable for HTMLFieldSetElement {
    /// @brief Returns a reference to the underlying `Element`.
    /// Functional Utility: Provides access to the generic `Element` methods and properties
    /// for validation purposes.
    /// @return A reference to the `Element`.
    fn as_element(&self) -> &Element {
        self.upcast() // Upcast to Element.
    }

    /// @brief Returns the `ValidityState` object for the element.
    /// Functional Utility: Lazily initializes and returns the `ValidityState` object
    /// associated with this fieldset, used to track its validation status.
    /// @return A `DomRoot<ValidityState>`.
    fn validity_state(&self) -> DomRoot<ValidityState> {
        self.validity_state
            .or_init(|| ValidityState::new(&self.owner_window(), self.upcast(), CanGc::note())) // Initialize if not already present.
    }

    /// @brief Determines if the `<fieldset>` element itself is subject to form validation.
    /// Functional Utility: Implements the `is_instance_validatable` method, specifying
    /// that fieldsets are not directly validatable (though their children are).
    /// @return `false` as a fieldset is not a submittable element.
    fn is_instance_validatable(&self) -> bool {
        // fieldset is not a submittable element (https://html.spec.whatwg.org/multipage/#category-submit)
        false
    }
}