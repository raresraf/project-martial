/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// @file htmlformelement.rs
/// @brief This file implements the `HTMLFormElement` interface, which represents the
/// `<form>` HTML element. It provides functionality for managing form submission,
/// resetting, accessing form controls, and handling validation.
/// Functional Utility: Facilitates user input collection, data submission to a server,
/// and client-side validation of form data.

use std::borrow::ToOwned;
use std::cell::Cell;

use constellation_traits::{LoadData, LoadOrigin, NavigationHistoryBehavior};
use dom_struct::dom_struct;
use encoding_rs::{Encoding, UTF_8};
use headers::{ContentType, HeaderMapExt};
use html5ever::{LocalName, Prefix, local_name, namespace_url, ns};
use http::Method;
use js::rust::HandleObject;
use mime::{self, Mime};
use net_traits::http_percent_encode;
use net_traits::request::Referrer;
use servo_rand::random;
use style::attr::AttrValue;
use style::str::split_html_space_chars;
use stylo_atoms::Atom;
use stylo_dom::ElementState;

use super::bindings::trace::{HashMapTracedValues, NoTrace};
use crate::body::Extractable;
use crate::dom::attr::Attr;
use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::AttrBinding::Attr_Binding::AttrMethods;
use crate::dom::bindings::codegen::Bindings::BlobBinding::BlobMethods;
use crate::dom::bindings::codegen::Bindings::DocumentBinding::DocumentMethods;
use crate::dom::bindings::codegen::Bindings::EventBinding::EventMethods;
use crate::dom::bindings::codegen::Bindings::HTMLButtonElementBinding::HTMLButtonElementMethods;
use crate::dom::bindings::codegen::Bindings::HTMLElementBinding::HTMLElementMethods;
use crate::dom::bindings::codegen::Bindings::HTMLFormControlsCollectionBinding::HTMLFormControlsCollectionMethods;
use crate::dom::bindings::codegen::Bindings::HTMLFormElementBinding::HTMLFormElementMethods;
use crate::dom::bindings::codegen::Bindings::HTMLInputElementBinding::HTMLInputElementMethods;
use crate::dom::bindings::codegen::Bindings::HTMLTextAreaElementBinding::HTMLTextAreaElementMethods;
use crate::dom::bindings::codegen::Bindings::NodeBinding::{NodeConstants, NodeMethods};
use crate::dom::bindings::codegen::Bindings::NodeListBinding::NodeListMethods;
use crate::dom::bindings::codegen::Bindings::RadioNodeListBinding::RadioNodeListMethods;
use crate::dom::bindings::codegen::Bindings::WindowBinding::Window_Binding::WindowMethods;
use crate::dom::bindings::codegen::UnionTypes::RadioNodeListOrElement;
use crate::dom::bindings::error::{Error, Fallible};
use crate::dom::bindings::inheritance::{Castable, ElementTypeId, HTMLElementTypeId, NodeTypeId};
use crate::dom::bindings::refcounted::Trusted;
use crate::dom::bindings::reflector::{DomGlobal, DomObject};
use crate::dom::bindings::root::{Dom, DomOnceCell, DomRoot, MutNullableDom};
use crate::dom::bindings::str::DOMString;
use crate::dom::blob::Blob;
use crate::dom::customelementregistry::CallbackReaction;
use crate::dom::document::Document;
use crate::dom::domtokenlist::DOMTokenList;
use crate::dom::element::{AttributeMutation, Element};
use crate::dom::event::{Event, EventBubbles, EventCancelable};
use crate::dom::eventtarget::EventTarget;
use crate::dom::file::File;
use crate::dom::formdata::FormData;
use crate::dom::formdataevent::FormDataEvent;
use crate::dom::htmlbuttonelement::HTMLButtonElement;
use crate::dom::htmlcollection::CollectionFilter;
use crate::dom::htmldatalistelement::HTMLDataListElement;
use crate::dom::htmlelement::HTMLElement;
use crate::dom::htmlfieldsetelement::HTMLFieldSetElement;
use crate::dom::htmlformcontrolscollection::HTMLFormControlsCollection;
use crate::dom::htmlimageelement::HTMLImageElement;
use crate::dom::htmlinputelement::{HTMLInputElement, InputType};
use crate::dom::htmllabelelement::HTMLLabelElement;
use crate::dom::htmllegendelement::HTMLLegendElement;
use crate::dom::htmlobjectelement::HTMLObjectElement;
use crate::dom::htmloutputelement::HTMLOutputElement;
use crate::dom::htmlselectelement::HTMLSelectElement;
use crate::dom::htmltextareaelement::HTMLTextAreaElement;
use crate::dom::node::{
    BindContext, Node, NodeFlags, NodeTraits, UnbindContext, VecPreOrderInsertionHelper,
};
use crate::dom::nodelist::{NodeList, RadioListMode};
use crate::dom::radionodelist::RadioNodeList;
use crate::dom::submitevent::SubmitEvent;
use crate::dom::virtualmethods::VirtualMethods;
use crate::dom::window::Window;
use crate::links::{LinkRelations, get_element_target};
use crate::script_runtime::CanGc;
use crate::script_thread::ScriptThread;

/// @struct GenerationId
/// @brief Represents a unique identifier for a form's generation, used for tracking
/// planned navigation tasks.
/// Functional Utility: Ensures that only the most recent form submission attempt triggers
/// a navigation, preventing outdated or redundant navigations.
#[derive(Clone, Copy, JSTraceable, MallocSizeOf, PartialEq)]
pub(crate) struct GenerationId(u32); //!< The unique generation ID.

/// @struct HTMLFormElement
/// @brief Represents the `<form>` HTML element.
/// Functional Utility: Provides the DOM interface for `<form>`, allowing programmatic
/// control over form submission, resetting, and access to its associated form controls.
///
/// <https://html.spec.whatwg.org/multipage/#htmlformelement>
#[dom_struct]
pub(crate) struct HTMLFormElement {
    htmlelement: HTMLElement, //!< Inherited properties and methods from `HTMLElement`.
    marked_for_reset: Cell<bool>, //!< Flag indicating if the form is currently being reset.
    /// <https://html.spec.whatwg.org/multipage/#constructing-entry-list>
    constructing_entry_list: Cell<bool>, //!< Flag indicating if the form's entry list is being constructed.
    elements: DomOnceCell<HTMLFormControlsCollection>, //!< Cached collection of form controls.
    generation_id: Cell<GenerationId>, //!< Unique ID for this form's submission generation.
    controls: DomRefCell<Vec<Dom<Element>>>, //!< List of associated form control elements.

    #[allow(clippy::type_complexity)]
    past_names_map: DomRefCell<HashMapTracedValues<Atom, (Dom<Element>, NoTrace<usize>)>>, //!< Maps past names to their elements and generation IDs.

    /// The current generation of past names, i.e., the number of name changes to the name.
    current_name_generation: Cell<usize>, //!< Counter for name changes, used in `past_names_map`.

    firing_submission_events: Cell<bool>, //!< Flag indicating if submission events are currently firing.
    rel_list: MutNullableDom<DOMTokenList>, //!< The `DOMTokenList` for the `rel` attribute.

    #[no_trace]
    relations: Cell<LinkRelations>,
}

impl HTMLFormElement {
    /// @brief Creates a new `HTMLFormElement` instance with inherited properties.
    /// Functional Utility: Internal constructor used for setting up the basic properties
    /// of an `HTMLFormElement` and initializing its internal state.
    ///
    /// @param local_name The `LocalName` of the HTML element (e.g., "form").
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix, if any.
    /// @param document The owning `Document` of this element.
    /// @return A new `HTMLFormElement` instance.
    fn new_inherited(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
    ) -> HTMLFormElement {
        HTMLFormElement {
            htmlelement: HTMLElement::new_inherited_with_state(
                ElementState::VALID, // Form is valid by default.
                local_name,
                prefix,
                document,
            ),
            marked_for_reset: Cell::new(false), // Not marked for reset initially.
            constructing_entry_list: Cell::new(false), // Not constructing entry list initially.
            elements: Default::default(), // Elements collection initialized to default.
            generation_id: Cell::new(GenerationId(0)), // Generation ID starts at 0.
            controls: DomRefCell::new(Vec::new()), // Controls list initialized empty.
            past_names_map: DomRefCell::new(HashMapTracedValues::new()), // Past names map initialized empty.
            current_name_generation: Cell::new(0), // Name generation starts at 0.
            firing_submission_events: Cell::new(false), // Not firing submission events initially.
            rel_list: Default::default(), // Rel list initialized to default.
            relations: Cell::new(LinkRelations::empty()), // Link relations initialized empty.
        }
    }

    /// @brief Creates a new `HTMLFormElement` and reflects it into the DOM.
    /// Functional Utility: Public constructor that builds an `HTMLFormElement` and makes
    /// it accessible from JavaScript by integrating it into the DOM's object graph.
    ///
    /// @param local_name The `LocalName` of the HTML element.
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix.
    /// @param document The owning `Document` of this element.
    /// @param proto An `Option<HandleObject>` specifying the JavaScript prototype chain.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLFormElement` instance wrapped in `DomRoot`.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<HTMLFormElement> {
        Node::reflect_node_with_proto(
            Box::new(HTMLFormElement::new_inherited(local_name, prefix, document)),
            document,
            proto,
            can_gc,
        )
    }

    /// @brief Filters an element for inclusion in a radio node list based on mode and name.
    /// Functional Utility: Helper function for constructing `RadioNodeList`s, determining
    /// which elements match the criteria for a specific radio button group.
    ///
    /// @param mode The `RadioListMode` (e.g., `ControlsExceptImageInputs`, `Images`).
    /// @param child The `Element` to evaluate.
    /// @param name The `Atom` representing the name or ID to match.
    /// @return `true` if the element matches the filter, `false` otherwise.
    fn filter_for_radio_list(mode: RadioListMode, child: &Element, name: &Atom) -> bool {
        if let Some(child) = child.downcast::<Element>() {
            match mode {
                RadioListMode::ControlsExceptImageInputs => {
                    // Block Logic: Check if the child is a listed element and has a matching ID or name.
                    if child
                        .downcast::<HTMLElement>()
                        .is_some_and(|c| c.is_listed_element()) &&
                        (child.get_id().is_some_and(|i| i == *name) ||
                            child.get_name().is_some_and(|n| n == *name))
                    {
                        // Block Logic: If it's an input, exclude image inputs.
                        if let Some(inp) = child.downcast::<HTMLInputElement>() {
                            return inp.input_type() != InputType::Image;
                        } else {
                            // Control, but not an input (e.g., textarea, select).
                            return true;
                        }
                    }
                    return false;
                },
                RadioListMode::Images => {
                    // Block Logic: Check if the child is an image element and has a matching ID or name.
                    return child.is::<HTMLImageElement>() &&
                        (child.get_id().is_some_and(|i| i == *name) ||
                            child.get_name().is_some_and(|n| n == *name));
                },
            }
        }
        false
    }

    /// @brief Retrieves the nth element in a radio node list for this form.
    /// Functional Utility: Provides indexed access to elements within a radio group, 
    /// based on a specific `RadioListMode` and name.
    ///
    /// @param index The zero-based index to retrieve.
    /// @param mode The `RadioListMode` for filtering.
    /// @param name The `Atom` representing the name or ID to match.
    /// @return An `Option<DomRoot<Node>>` containing the element, or `None`.
    pub(crate) fn nth_for_radio_list(
        &self,
        index: u32,
        mode: RadioListMode,
        name: &Atom,
    ) -> Option<DomRoot<Node>> {
        self.controls
            .borrow()
            .iter()
            .filter(|n| HTMLFormElement::filter_for_radio_list(mode, n, name)) // Filter controls.
            .nth(index as usize) // Get the nth matching element.
            .map(|n| DomRoot::from_ref(n.upcast::<Node>())) // Map to `DomRoot<Node>`.
    }

    /// @brief Counts the number of elements in a radio node list for this form.
    /// Functional Utility: Determines the size of a radio group based on a specific
    /// `RadioListMode` and name.
    ///
    /// @param mode The `RadioListMode` for filtering.
    /// @param name The `Atom` representing the name or ID to match.
    /// @return The count of matching elements.
    pub(crate) fn count_for_radio_list(&self, mode: RadioListMode, name: &Atom) -> u32 {
        self.controls
            .borrow()
            .iter()
            .filter(|n| HTMLFormElement::filter_for_radio_list(mode, n, name)) // Filter controls.
            .count() as u32 // Count the matching elements.
    }
}

impl HTMLFormElementMethods<crate::DomTypeHolder> for HTMLFormElement {
    // https://html.spec.whatwg.org/multipage/#dom-form-acceptcharset
    /// @brief Returns the value of the `accept-charset` attribute.
    /// Functional Utility: Implements the `acceptCharset` getter, indicating the
    /// character encodings that the server is expected to handle for form submission.
    make_getter!(AcceptCharset, "accept-charset");

    // https://html.spec.whatwg.org/multipage/#dom-form-acceptcharset
    /// @brief Sets the value of the `accept-charset` attribute.
    /// Functional Utility: Implements the `acceptCharset` setter.
    make_setter!(SetAcceptCharset, "accept-charset");

    // https://html.spec.whatwg.org/multipage/#dom-fs-action
    /// @brief Returns the absolute URL of the `action` attribute.
    /// Functional Utility: Implements the `action` getter, providing the URL to
    /// which the form data will be submitted.
    make_form_action_getter!(Action, "action");

    // https://html.spec.whatwg.org/multipage/#dom-fs-action
    /// @brief Sets the value of the `action` attribute.
    /// Functional Utility: Implements the `action` setter.
    make_setter!(SetAction, "action");

    // https://html.spec.whatwg.org/multipage/#dom-form-autocomplete
    /// @brief Returns the `autocomplete` attribute state.
    /// Functional Utility: Implements the `autocomplete` getter, indicating whether
    /// the browser should automatically complete form inputs.
    make_enumerated_getter!(
        Autocomplete,
        "autocomplete",
        "on" | "off", // Supported values.
        missing => "on", // Default value when attribute is missing.
        invalid => "on" // Default value when attribute is invalid.
    );

    // https://html.spec.whatwg.org/multipage/#dom-form-autocomplete
    /// @brief Sets the `autocomplete` attribute state.
    /// Functional Utility: Implements the `autocomplete` setter.
    make_setter!(SetAutocomplete, "autocomplete");

    // https://html.spec.whatwg.org/multipage/#dom-fs-enctype
    /// @brief Returns the `enctype` attribute state.
    /// Functional Utility: Implements the `enctype` getter, indicating the content
    /// type used for encoding the form data when submitted.
    make_enumerated_getter!(
        Enctype,
        "enctype",
        "application/x-www-form-urlencoded" | "text/plain" | "multipart/form-data", // Supported values.
        missing => "application/x-www-form-urlencoded", // Default when missing.
        invalid => "application/x-www-form-urlencoded" // Default when invalid.
    );

    // https://html.spec.whatwg.org/multipage/#dom-fs-enctype
    /// @brief Sets the `enctype` attribute state.
    /// Functional Utility: Implements the `enctype` setter.
    make_setter!(SetEnctype, "enctype");

    // https://html.spec.whatwg.org/multipage/#dom-fs-encoding
    /// @brief Returns the `enctype` attribute state (alias for `enctype`).
    /// Functional Utility: Implements the `encoding` getter for compatibility.
    fn Encoding(&self) -> DOMString {
        self.Enctype()
    }

    // https://html.spec.whatwg.org/multipage/#dom-fs-encoding
    /// @brief Sets the `enctype` attribute state (alias for `enctype`).
    /// Functional Utility: Implements the `encoding` setter for compatibility.
    fn SetEncoding(&self, value: DOMString) {
        self.SetEnctype(value)
    }

    // https://html.spec.whatwg.org/multipage/#dom-fs-method
    /// @brief Returns the `method` attribute state.
    /// Functional Utility: Implements the `method` getter, indicating the HTTP method
    /// used for submitting the form.
    make_enumerated_getter!(
        Method,
        "method",
        "get" | "post" | "dialog", // Supported values.
        missing => "get", // Default when missing.
        invalid => "get"
    );

    // https://html.spec.whatwg.org/multipage/#dom-fs-method
    /// @brief Sets the `method` attribute state.
    /// Functional Utility: Implements the `method` setter.
    make_setter!(SetMethod, "method");

    // https://html.spec.whatwg.org/multipage/#dom-form-name
    /// @brief Returns the `name` attribute of the form.
    /// Functional Utility: Implements the `name` getter.
    make_getter!(Name, "name");

    // https://html.spec.whatwg.org/multipage/#dom-form-name
    /// @brief Sets the `name` attribute of the form.
    /// Functional Utility: Implements the `name` setter.
    make_atomic_setter!(SetName, "name");

    // https://html.spec.whatwg.org/multipage/#dom-fs-novalidate
    /// @brief Returns `true` if the `novalidate` attribute is present.
    /// Functional Utility: Implements the `noValidate` getter, indicating whether
    /// the form should skip client-side validation when submitted.
    make_bool_getter!(NoValidate, "novalidate");

    // https://html.spec.whatwg.org/multipage/#dom-fs-novalidate
    /// @brief Sets the `novalidate` attribute.
    /// Functional Utility: Implements the `noValidate` setter.
    make_bool_setter!(SetNoValidate, "novalidate");

    // https://html.spec.whatwg.org/multipage/#dom-fs-target
    /// @brief Returns the `target` attribute of the form.
    /// Functional Utility: Implements the `target` getter, indicating where to display
    /// the response after form submission.
    make_getter!(Target, "target");

    // https://html.spec.whatwg.org/multipage/#dom-fs-target
    /// @brief Sets the `target` attribute of the form.
    /// Functional Utility: Implements the `target` setter.
    make_setter!(SetTarget, "target");

    // https://html.spec.whatwg.org/multipage/#dom-a-rel
    /// @brief Returns the `rel` attribute of the form.
    /// Functional Utility: Implements the `rel` getter, indicating the relationship
    /// between the current document and the linked resource.
    fn Rel(&self) -> DOMString {
        self.upcast::<Element>().get_string_attribute(&local_name!("rel"))
    }

    // https://html.spec.whatwg.org/multipage/#the-form-element:concept-form-submit
    /// @brief Submits the form.
    /// Functional Utility: Implements the `submit()` method, initiating the form
    /// submission process, including validation and navigation.
    ///
    /// @param can_gc A `CanGc` token.
    fn Submit(&self, can_gc: CanGc) {
        self.submit(
            SubmittedFrom::FromForm, // Indicates submission initiated from the form itself.
            FormSubmitterElement::Form(self), // The form itself acts as the submitter.
            can_gc,
        );
    }

    // https://html.spec.whatwg.org/multipage/#dom-form-requestsubmit
    /// @brief Requests submission of the form, optionally with a specific submitter.
    /// Functional Utility: Implements the `requestSubmit()` method, providing
    /// a programmatic way to trigger form submission as if a user clicked a submit button.
    ///
    /// @param submitter An `Option<&HTMLElement>` representing the submit button or `None`.
    /// @param can_gc A `CanGc` token.
    /// @return A `Fallible<()>` indicating success or an error.
    fn RequestSubmit(&self, submitter: Option<&HTMLElement>, can_gc: CanGc) -> Fallible<()> {
        let submitter: FormSubmitterElement = match submitter {
            Some(submitter_element) => {
                // Step 1.1: If submitter is not a submit button, throw a TypeError.
                let error_not_a_submit_button =
                    Err(Error::Type("submitter must be a submit button".to_string()));

                // Block Logic: Determine the specific type of HTML element for the submitter.
                let element = match submitter_element.upcast::<Node>().type_id() {
                    NodeTypeId::Element(ElementTypeId::HTMLElement(element)) => element,
                    _ => {
                        return error_not_a_submit_button;
                    },
                };

                // Block Logic: Downcast to the specific submitter element type (Input or Button).
                let submit_button = match element {
                    HTMLElementTypeId::HTMLInputElement => FormSubmitterElement::Input(
                        submitter_element
                            .downcast::<HTMLInputElement>()
                            .expect("Failed to downcast submitter elem to HTMLInputElement."),
                    ),
                    HTMLElementTypeId::HTMLButtonElement => FormSubmitterElement::Button(
                        submitter_element
                            .downcast::<HTMLButtonElement>()
                            .expect("Failed to downcast submitter elem to HTMLButtonElement."),
                    ),
                    _ => {
                        return error_not_a_submit_button;
                    },
                };

                // Block Logic: Check if the submitter is indeed a submit button.
                if !submit_button.is_submit_button() {
                    return error_not_a_submit_button;
                }

                // Step 1.2: If submitter's form owner is not this form, throw a NotFoundError.
                let submitters_owner = submit_button.form_owner();

                let owner = match submitters_owner {
                    Some(owner) => owner,
                    None => {
                        return Err(Error::NotFound);
                    },
                };

                if *owner != *self {
                    return Err(Error::NotFound);
                }

                submit_button
            },
            None => {
                // Step 2: If submitter is null, then let submitter be this.
                FormSubmitterElement::Form(self)
            },
        };
        // Step 3: Run the form submission algorithm, given submitter and with the submitted from
        // the requestSubmit() method flag set.
        self.submit(SubmittedFrom::NotFromForm, submitter, can_gc);
        Ok(())
    }

    // https://html.spec.whatwg.org/multipage/#dom-form-reset
    /// @brief Resets the form.
    /// Functional Utility: Implements the `reset()` method, restoring all form
    /// controls to their initial values.
    ///
    /// @param can_gc A `CanGc` token.
    fn Reset(&self, can_gc: CanGc) {
        self.reset(ResetFrom::FromForm, can_gc); // Initiates the form reset process.
    }

    // https://html.spec.whatwg.org/multipage/#dom-form-elements
    /// @brief Returns a live `HTMLFormControlsCollection` of all form controls.
    /// Functional Utility: Implements the `elements` getter, providing programmatic
    /// access to all form-associated elements within the form, filtered to exclude
    /// image inputs when appropriate.
    ///
    /// @param can_gc A `CanGc` token.
    /// @return A `DomRoot<HTMLFormControlsCollection>`.
    fn Elements(&self, can_gc: CanGc) -> DomRoot<HTMLFormControlsCollection> {
        // Block Logic: Define a filter to select form controls for the collection.
        #[derive(JSTraceable, MallocSizeOf)]
        struct ElementsFilter {
            form: DomRoot<HTMLFormElement>,
        }
        impl CollectionFilter for ElementsFilter {
            /// @brief Filters an element to check if it's a listed form control belonging to this form.
            fn filter<'a>(&self, elem: &'a Element, _root: &'a Node) -> bool {
                let form_owner = match elem.upcast::<Node>().type_id() {
                    NodeTypeId::Element(ElementTypeId::HTMLElement(t)) => match t {
                        HTMLElementTypeId::HTMLButtonElement => {
                            elem.downcast::<HTMLButtonElement>().unwrap().form_owner()
                        },
                        HTMLElementTypeId::HTMLFieldSetElement => {
                            elem.downcast::<HTMLFieldSetElement>().unwrap().form_owner()
                        },
                        HTMLElementTypeId::HTMLInputElement => {
                            let input_elem = elem.downcast::<HTMLInputElement>().unwrap();
                            if input_elem.input_type() == InputType::Image {
                                return false;
                            }
                            input_elem.form_owner()
                        },
                        HTMLElementTypeId::HTMLObjectElement => {
                            elem.downcast::<HTMLObjectElement>().unwrap().form_owner()
                        },
                        HTMLElementTypeId::HTMLOutputElement => {
                            elem.downcast::<HTMLOutputElement>().unwrap().form_owner()
                        },
                        HTMLElementTypeId::HTMLSelectElement => {
                            elem.downcast::<HTMLSelectElement>().unwrap().form_owner()
                        },
                        HTMLElementTypeId::HTMLTextAreaElement => {
                            elem.downcast::<HTMLTextAreaElement>().unwrap().form_owner()
                        },
                        HTMLElementTypeId::HTMLElement => {
                            let html_element = elem.downcast::<HTMLElement>().unwrap();
                            if html_element.is_form_associated_custom_element() {
                                html_element.form_owner()
                            } else {
                                return false;
                            }
                        },
                        _ => {
                            // Elements not explicitly handled are debug-asserted to not be listed elements.
                            debug_assert!(
                                !elem.downcast::<HTMLElement>().unwrap().is_listed_element()
                            );
                            return false;
                        },
                    },
                    _ => return false,
                };

                // Check if the element's form owner is this form.
                match form_owner {
                    Some(form_owner) => form_owner == self.form,
                    None => false,
                }
            }
        }
        DomRoot::from_ref(self.elements.init_once(|| {
            let filter = Box::new(ElementsFilter {
                form: DomRoot::from_ref(self),
            });
            let window = self.owner_window();
            HTMLFormControlsCollection::new(&window, self, filter, can_gc)
        }))
    }

    // https://html.spec.whatwg.org/multipage/#dom-form-length
    /// @brief Returns the number of controls in the form.
    /// Functional Utility: Implements the `length` getter for `HTMLFormElement`,
    /// providing the count of its associated form controls.
    /// @return The number of form controls (`u32`).
    fn Length(&self) -> u32 {
        self.Elements(CanGc::note()).Length() // Delegates to the `HTMLFormControlsCollection` length.
    }

    // https://html.spec.whatwg.org/multipage/#dom-form-item
    /// @brief Returns the element at the specified index in the form's controls.
    /// Functional Utility: Implements the indexed getter behavior for `HTMLFormElement`,
    /// allowing array-like access to form controls.
    ///
    /// @param index The zero-based index of the element to retrieve.
    /// @param can_gc A `CanGc` token.
    /// @return An `Option<DomRoot<Element>>` containing the element, or `None`.
    fn IndexedGetter(&self, index: u32, can_gc: CanGc) -> Option<DomRoot<Element>> {
        let elements = self.Elements(can_gc); // Get the form controls collection.
        elements.IndexedGetter(index) // Delegates to the collection's indexed getter.
    }

    // https://html.spec.whatwg.org/multipage/#the-form-element%3Adetermine-the-value-of-a-named-property
    /// @brief Returns the first element(s) with the given name or ID within the form.
    /// Functional Utility: Implements the named getter behavior for `HTMLFormElement`,
    /// allowing access to form controls by their `name` or `id` attributes.
    ///
    /// @param name The `DOMString` representing the name or ID to search for.
    /// @param can_gc A `CanGc` token.
    /// @return An `Option<RadioNodeListOrElement>` containing the matching element(s), or `None`.
    fn NamedGetter(&self, name: DOMString, can_gc: CanGc) -> Option<RadioNodeListOrElement> {
        let window = self.owner_window();

        let name = Atom::from(name);

        // Step 1: Let candidates be a new RadioNodeList object whose mode is "controls (excluding image inputs)",
        // whose form is the HTMLFormElement object, and whose name is name.
        let mut candidates =
            RadioNodeList::new_controls_except_image_inputs(&window, self, &name, can_gc);
        let mut candidates_length = candidates.Length();

        // Step 2: If candidates's length is zero, then let candidates be a new RadioNodeList object
        // whose mode is "image inputs", whose form is the HTMLFormElement object, and whose name is name.
        if candidates_length == 0 {
            candidates = RadioNodeList::new_images(&window, self, &name, can_gc);
            candidates_length = candidates.Length();
        }

        let mut past_names_map = self.past_names_map.borrow_mut();

        // Step 3: If candidates's length is zero, and if HTMLFormElement's past names map contains an entry
        // for name, then return the Element object in that entry.
        if candidates_length == 0 {
            if past_names_map.contains_key(&name) {
                return Some(RadioNodeListOrElement::Element(DomRoot::from_ref(
                    &*past_names_map.get(&name).unwrap().0,
                )));
            }
            return None;
        }

        // Step 4: If candidates's length is greater than one, then return candidates.
        if candidates_length > 1 {
            return Some(RadioNodeListOrElement::RadioNodeList(candidates));
        }

        // Step 5: Assert: candidates's length is one. Let element be candidates's first item.
        // candidates_length is 1, so we can unwrap item 0
        let element_node = candidates.upcast::<NodeList>().Item(0).unwrap();
        past_names_map.insert(
            name,
            (
                Dom::from_ref(element_node.downcast::<Element>().unwrap()),
                NoTrace(self.current_name_generation.get() + 1),
            ),
        );
        self.current_name_generation
            .set(self.current_name_generation.get() + 1);

        // Step 6: Return element.
        Some(RadioNodeListOrElement::Element(DomRoot::from_ref(
            element_node.downcast::<Element>().unwrap(),
        )))
    }

    // https://html.spec.whatwg.org/multipage/#dom-a-rel
    /// @brief Sets the `rel` attribute of the form.
    /// Functional Utility: Implements the `rel` setter.
    fn SetRel(&self, rel: DOMString, can_gc: CanGc) {
        self.upcast::<Element>()
            .set_tokenlist_attribute(&local_name!("rel"), rel, can_gc);
    }

    // https://html.spec.whatwg.org/multipage/#dom-a-rellist
    /// @brief Returns a live `DOMTokenList` for the `rel` attribute.
    /// Functional Utility: Implements the `relList` getter, providing programmatic
    /// access to the individual tokens in the `rel` attribute.
    ///
    /// @param can_gc A `CanGc` token.
    /// @return A `DomRoot<DOMTokenList>`.
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

    // https://html.spec.whatwg.org/multipage/#the-form-element:supported-property-names
    /// @brief Returns a list of supported property names for the form, including IDs and names.
    /// Functional Utility: Implements the `SupportedPropertyNames` method, which provides
    /// the names that can be used to access form controls via the form's named getter.
    ///
    /// @return A `Vec<DOMString>` containing all supported property names.
    #[allow(non_snake_case)]
    fn SupportedPropertyNames(&self) -> Vec<DOMString> {
        // Step 1
        #[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
        enum SourcedNameSource {
            Id,
            Name,
            Past(usize),
        }

        impl SourcedNameSource {
            fn is_past(&self) -> bool {
                matches!(self, SourcedNameSource::Past(..))
            }
        }

        struct SourcedName {
            name: Atom,
            element: DomRoot<Element>,
            source: SourcedNameSource,
        }

        let mut sourced_names_vec: Vec<SourcedName> = Vec::new();

        let controls = self.controls.borrow();

        // Step 2
        for child in controls.iter() {
            if child
                .downcast::<HTMLElement>()
                .is_some_and(|c| c.is_listed_element())
            {
                if let Some(id_atom) = child.get_id() {
                    let entry = SourcedName {
                        name: id_atom,
                        element: DomRoot::from_ref(child),
                        source: SourcedNameSource::Id,
                    };
                    sourced_names_vec.push(entry);
                }
                if let Some(name_atom) = child.get_name() {
                    let entry = SourcedName {
                        name: name_atom,
                        element: DomRoot::from_ref(child),
                        source: SourcedNameSource::Name,
                    };
                    sourced_names_vec.push(entry);
                }
            }
        }

        // Step 3
        for child in controls.iter() {
            if child.is::<HTMLImageElement>() {
                if let Some(id_atom) = child.get_id() {
                    let entry = SourcedName {
                        name: id_atom,
                        element: DomRoot::from_ref(child),
                        source: SourcedNameSource::Id,
                    };
                    sourced_names_vec.push(entry);
                }
                if let Some(name_atom) = child.get_name() {
                    let entry = SourcedName {
                        name: name_atom,
                        element: DomRoot::from_ref(child),
                        source: SourcedNameSource::Name,
                    };
                    sourced_names_vec.push(entry);
                }
            }
        }

        // Step 4
        let past_names_map = self.past_names_map.borrow();
        for (key, val) in past_names_map.iter() {
            let entry = SourcedName {
                name: key.clone(),
                element: DomRoot::from_ref(&*val.0),
                source: SourcedNameSource::Past(self.current_name_generation.get() - val.1.0),
            };
            sourced_names_vec.push(entry);
        }

        // Step 5
        // TODO need to sort as per spec.
        // if a.CompareDocumentPosition(b) returns 0 that means a=b in which case
        // the remaining part where sorting is to be done by putting entries whose source is id first,
        // then entries whose source is name, and finally entries whose source is past,
        // and sorting entries with the same element and source by their age, oldest first.

        // if a.CompareDocumentPosition(b) has set NodeConstants::DOCUMENT_POSITION_FOLLOWING
        // (this can be checked by bitwise operations) then b would follow a in tree order and
        // Ordering::Less should be returned in the closure else Ordering::Greater

        sourced_names_vec.sort_by(|a, b| {
            if a.element
                .upcast::<Node>()
                .CompareDocumentPosition(b.element.upcast::<Node>()) ==
                0
            {
                if a.source.is_past() && b.source.is_past() {
                    b.source.cmp(&a.source)
                } else {
                    a.source.cmp(&b.source)
                }
            } else if a
                .element
                .upcast::<Node>()
                .CompareDocumentPosition(b.element.upcast::<Node>()) &
                NodeConstants::DOCUMENT_POSITION_FOLLOWING ==
                NodeConstants::DOCUMENT_POSITION_FOLLOWING
            {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        // Step 6
        sourced_names_vec.retain(|sn| !sn.name.to_string().is_empty());

        // Step 7-8
        let mut names_vec: Vec<DOMString> = Vec::new();
        for elem in sourced_names_vec.iter() {
            if !names_vec.iter().any(|name| *name == *elem.name) {
                names_vec.push(DOMString::from(&*elem.name));
            }
        }

        names_vec
    }

    /// <https://html.spec.whatwg.org/multipage/#dom-form-checkvalidity>
    /// @brief Checks the validity of the form and its controls without reporting to the user.
    /// Functional Utility: Implements the `checkValidity()` method, returning `true`
    /// if all associated form controls satisfy their constraints.
    ///
    /// @param can_gc A `CanGc` token.
    /// @return `true` if the form is valid, `false` otherwise.
    fn CheckValidity(&self, can_gc: CanGc) -> bool {
        self.static_validation(can_gc).is_ok()
    }

    /// <https://html.spec.whatwg.org/multipage/#dom-form-reportvalidity>
    /// @brief Checks the validity of the form and reports any invalid states to the user.
    /// Functional Utility: Implements the `reportValidity()` method, triggering
    /// validation and potentially showing user interface feedback for invalid controls.
    ///
    /// @param can_gc A `CanGc` token.
    /// @return `true` if the form is valid, `false` otherwise.
    fn ReportValidity(&self, can_gc: CanGc) -> bool {
        self.interactive_validation(can_gc).is_ok()
    }
}

/// @enum SubmittedFrom
/// @brief Represents how a form submission was initiated.
pub(crate) enum SubmittedFrom {
    FromForm,
    NotFromForm,
}

/// @enum ResetFrom
/// @brief Represents how a form reset was initiated.
pub(crate) enum ResetFrom {
    FromForm,
    NotFromForm,
}

impl HTMLFormElement {
    // https://html.spec.whatwg.org/multipage/#picking-an-encoding-for-the-form
    /// @brief Determines the character encoding to use for form submission.
    /// Functional Utility: Selects the encoding based on the `accept-charset` attribute
    /// or falls back to the document's encoding.
    /// @return A static reference to the chosen `Encoding`.
    fn pick_encoding(&self) -> &'static Encoding {
        // Step 2: If the form element has an accept-charset attribute, then:
        if self
            .upcast::<Element>()
            .has_attribute(&local_name!("accept-charset"))
        {
            // Substep 1: Let input be the value of the form element's accept-charset attribute.
            let input = self
                .upcast::<Element>()
                .get_string_attribute(&local_name!("accept-charset"));

            // Substep 2, 3, 4: Parse input into a set of character encodings.
            let mut candidate_encodings =
                split_html_space_chars(&input).filter_map(|c| Encoding::for_label(c.as_bytes()));

            // Substep 5, 6: Return the first supported encoding, otherwise UTF-8.
            return candidate_encodings.next().unwrap_or(UTF_8);
        }

        // Step 1, 3: Otherwise, return the Document's character encoding.
        self.owner_document().encoding()
    }

    // https://html.spec.whatwg.org/multipage/#text/plain-encoding-algorithm
    /// @brief Encodes form data into a `text/plain` format.
    /// Functional Utility: Serializes form entries into a plain text string,
    /// suitable for `text/plain` form submission.
    ///
    /// @param form_data A mutable slice of `FormDatum` entries.
    /// @return A `String` containing the encoded data.
    fn encode_plaintext(&self, form_data: &mut [FormDatum]) -> String {
        // Step 1: Let result be the empty string.
        let mut result = String::new();

        // Step 2: For each entry in form_data:
        for entry in form_data.iter() {
            let value = match &entry.value {
                FormDatumValue::File(f) => f.name(),
                FormDatumValue::String(s) => s,
            };
            result.push_str(&format!(