/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// @file htmldetailselement.rs
/// @brief This file implements the `HTMLDetailsElement` interface, which represents the
/// `<details>` HTML element. It provides functionality for managing its open/closed state
/// and integrating with its shadow DOM for rendering.
/// Functional Utility: Enables the creation and manipulation of `<details>` elements,
/// offering expandable content areas in web pages.

use std::cell::{Cell, Ref}; // Cell for interior mutability, Ref for immutable borrow.

use dom_struct::dom_struct; // Macro for defining DOM structures.
use html5ever::{LocalName, Prefix, local_name}; // HTML5 parsing types.
use js::rust::HandleObject; // JavaScript object handle.

use crate::dom::attr::Attr; // DOM attribute.
use crate::dom::bindings::cell::DomRefCell; // DOM-specific RefCell.
use crate::dom::bindings::codegen::Bindings::HTMLDetailsElementBinding::HTMLDetailsElementMethods; // Generated bindings for HTMLDetailsElement methods.
use crate::dom::bindings::codegen::Bindings::HTMLSlotElementBinding::HTMLSlotElement_Binding::HTMLSlotElementMethods; // Generated bindings for HTMLSlotElement methods.
use crate::dom::bindings::codegen::Bindings::NodeBinding::Node_Binding::NodeMethods; // Generated bindings for Node methods.
use crate::dom::bindings::codegen::Bindings::ShadowRootBinding::{
    ShadowRootMode, SlotAssignmentMode,
}; // ShadowRoot related enums.
use crate::dom::bindings::codegen::UnionTypes::ElementOrText; // Union type for element or text.
use crate::dom::bindings::inheritance::Castable; // Trait for safe downcasting between DOM types.
use crate::dom::bindings::refcounted::Trusted; // Trusted reference counting for DOM objects.
use crate::dom::bindings::root::{Dom, DomRoot}; // Root DOM types.
use crate::dom::document::Document; // Document object.
use crate::dom::element::{AttributeMutation, Element}; // Element type and attribute mutation.
use crate::dom::eventtarget::EventTarget; // Event target.
use crate::dom::htmlelement::HTMLElement; // Base HTMLElement type.
use crate::dom::htmlslotelement::HTMLSlotElement; // HTMLSlotElement type.
use crate::dom::node::{BindContext, ChildrenMutation, Node, NodeDamage, NodeTraits}; // Node types and traits.
use crate::dom::shadowroot::IsUserAgentWidget; // Indicates if a shadow root is a user agent widget.
use crate::dom::text::Text; // Text node.
use crate::dom::virtualmethods::VirtualMethods; // Virtual methods for DOM objects.
use crate::script_runtime::CanGc; // Marker trait for types that can be garbage collected.

/// The summary that should be presented if no `<summary>` element is present
const DEFAULT_SUMMARY: &str = "Details";

/// @struct ShadowTree
/// @brief Holds handles to all slots and the implicit summary element in the UA shadow tree
/// for a `<details>` element.
/// Functional Utility: Manages the internal structure of the user-agent provided shadow DOM
/// for `<details>`, which handles the rendering of the summary and the expandable content.
///
/// The composition of the tree is described in
/// <https://html.spec.whatwg.org/multipage/#the-details-and-summary-elements>
#[derive(Clone, JSTraceable, MallocSizeOf)]
#[cfg_attr(crown, crown::unrooted_must_root_lint::must_root)]
struct ShadowTree {
    summary: Dom<HTMLSlotElement>,     //!< The slot for the user-provided summary or fallback.
    descendants: Dom<HTMLSlotElement>, //!< The slot for the expandable content of the `<details>` element.
    /// The summary that is displayed if no other summary exists
    implicit_summary: Dom<HTMLElement>, //!< The default `HTMLElement` used as a fallback summary.
}

/// @struct HTMLDetailsElement
/// @brief Represents the `<details>` HTML element.
/// Functional Utility: Provides the DOM interface for `<details>`, allowing control over its
/// `open` attribute and managing its shadow DOM for rendering.
///
/// <https://html.spec.whatwg.org/multipage/#htmldetailselement>
#[dom_struct]
pub(crate) struct HTMLDetailsElement {
    htmlelement: HTMLElement, //!< Inherited properties and methods from `HTMLElement`.
    toggle_counter: Cell<u32>, //!< Counter to track toggles for event debouncing.

    /// Represents the UA widget for the details element
    shadow_tree: DomRefCell<Option<ShadowTree>>, //!< The user-agent provided shadow DOM structure.
}

impl HTMLDetailsElement {
    /// @brief Creates a new `HTMLDetailsElement` instance with inherited properties.
    /// Functional Utility: Internal constructor used for setting up the basic properties
    /// of an `HTMLDetailsElement` based on its tag name, prefix, and owning document.
    ///
    /// @param local_name The `LocalName` of the HTML element (e.g., "details").
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix, if any.
    /// @param document The owning `Document` of this element.
    /// @return A new `HTMLDetailsElement` instance.
    fn new_inherited(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
    ) -> HTMLDetailsElement {
        HTMLDetailsElement {
            htmlelement: HTMLElement::new_inherited(local_name, prefix, document), // Initialize base HTMLElement.
            toggle_counter: Cell::new(0), // Initialize toggle counter to 0.
            shadow_tree: Default::default(), // Initialize shadow tree as None.
        }
    }

    /// @brief Creates a new `HTMLDetailsElement` and reflects it into the DOM.
    /// Functional Utility: Public constructor that builds an `HTMLDetailsElement` and makes
    /// it accessible from JavaScript by integrating it into the DOM's object graph.
    ///
    /// @param local_name The `LocalName` of the HTML element.
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix.
    /// @param document The owning `Document` of this element.
    /// @param proto An `Option<HandleObject>` specifying the JavaScript prototype chain.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLDetailsElement` instance wrapped in `DomRoot`.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<HTMLDetailsElement> {
        Node::reflect_node_with_proto(
            Box::new(HTMLDetailsElement::new_inherited(
                local_name, prefix, document,
            )),
            document,
            proto,
            can_gc,
        )
    }

    /// @brief Toggles the `open` state of the `<details>` element.
    /// Functional Utility: Inverts the value of the `open` attribute, causing the
    /// content to expand or collapse.
    pub(crate) fn toggle(&self) {
        self.SetOpen(!self.Open()); // Invert the `open` attribute.
    }

    /// @brief Returns a reference to the `ShadowTree` for this `<details>` element.
    /// Functional Utility: Lazily creates the user-agent shadow DOM if it doesn't already exist,
    /// ensuring the internal rendering structure is present when accessed.
    ///
    /// @param can_gc A `CanGc` token.
    /// @return A `Ref` to the `ShadowTree`.
    fn shadow_tree(&self, can_gc: CanGc) -> Ref<'_, ShadowTree> {
        // Block Logic: If the shadow tree hasn't been created yet, create it.
        if !self.upcast::<Element>().is_shadow_host() {
            self.create_shadow_tree(can_gc);
        }

        // Block Logic: Return a `Ref` to the shadow tree, unwrapping the Option.
        Ref::filter_map(self.shadow_tree.borrow(), Option::as_ref)
            .ok()
            .expect("UA shadow tree was not created") // Post-condition: Shadow tree must exist here.
    }

    /// @brief Creates the user-agent provided shadow DOM for the `<details>` element.
    /// Functional Utility: Sets up the internal shadow DOM structure consisting of
    /// `<slot>` elements for the summary and descendants, and a fallback summary element.
    ///
    /// @param can_gc A `CanGc` token.
    fn create_shadow_tree(&self, can_gc: CanGc) {
        let document = self.owner_document(); // Get the owning document.
        let root = self
            .upcast::<Element>()
            .attach_shadow(
                IsUserAgentWidget::Yes, // Mark as user-agent widget.
                ShadowRootMode::Closed, // Use a closed shadow root.
                false, // delegates focus
                false, // serializable
                false, // cloned
                SlotAssignmentMode::Manual, // Manual slot assignment.
                can_gc,
            )
            .expect("Attaching UA shadow root failed"); // Attach a shadow root to the details element.

        // Block Logic: Create a slot for the summary.
        let summary = HTMLSlotElement::new(local_name!("slot"), None, &document, None, can_gc);
        root.upcast::<Node>()
            .AppendChild(summary.upcast::<Node>(), can_gc) // Append summary slot to shadow root.
            .unwrap();

        // Block Logic: Create a fallback summary element.
        let fallback_summary =
            HTMLElement::new(local_name!("summary"), None, &document, None, can_gc);
        fallback_summary
            .upcast::<Node>()
            .SetTextContent(Some(DEFAULT_SUMMARY.into()), can_gc); // Set default summary text.
        summary
            .upcast::<Node>()
            .AppendChild(fallback_summary.upcast::<Node>(), can_gc) // Append fallback summary to summary slot.
            .unwrap();

        // Block Logic: Create a slot for the descendants (content).
        let descendants = HTMLSlotElement::new(local_name!("slot"), None, &document, None, can_gc);
        root.upcast::<Node>()
            .AppendChild(descendants.upcast::<Node>(), can_gc) // Append descendants slot to shadow root.
            .unwrap();

        // Block Logic: Store the created shadow tree structure.
        let _ = self.shadow_tree.borrow_mut().insert(ShadowTree {
            summary: summary.as_traced(), // Traceable reference to summary slot.
            descendants: descendants.as_traced(), // Traceable reference to descendants slot.
            implicit_summary: fallback_summary.as_traced(), // Traceable reference to implicit summary.
        });
        self.upcast::<Node>()
            .dirty(crate::dom::node::NodeDamage::OtherNodeDamage); // Mark node as dirty to trigger re-rendering.
    }

    /// @brief Finds the corresponding `<summary>` element within the `<details>` element's light DOM.
    /// Functional Utility: Locates the user-provided summary element which, if present,
    /// overrides the implicit summary provided by the user-agent shadow DOM.
    ///
    /// @return An `Option<DomRoot<HTMLElement>>` containing the summary element if found, otherwise `None`.
    pub(crate) fn find_corresponding_summary_element(&self) -> Option<DomRoot<HTMLElement>> {
        self.upcast::<Node>()
            .children() // Iterate over direct children of `<details>`.
            .filter_map(DomRoot::downcast::<HTMLElement>) // Filter for HTMLElement.
            .find(|html_element| {
                html_element.upcast::<Element>().local_name() == &local_name!("summary") // Find element with local name "summary".
            })
    }

    /// @brief Updates the contents of the shadow tree based on the `<details>` element's children.
    /// Functional Utility: Assigns the user-provided `<summary>` element to its slot and
    /// the remaining children to the descendants slot, ensuring correct content projection.
    ///
    /// @param can_gc A `CanGc` token.
    fn update_shadow_tree_contents(&self, can_gc: CanGc) {
        let shadow_tree = self.shadow_tree(can_gc); // Get a reference to the shadow tree.

        // Block Logic: Assign the user-provided summary to its slot.
        if let Some(summary) = self.find_corresponding_summary_element() {
            shadow_tree
                .summary
                .Assign(vec![ElementOrText::Element(DomRoot::upcast(summary))]); // Assign summary to its slot.
        }

        let mut slottable_children = vec![]; // Vector to hold children for the descendants slot.
        // Block Logic: Iterate through children and categorize them for slot assignment.
        for child in self.upcast::<Node>().children() {
            if let Some(element) = child.downcast::<Element>() {
                if element.local_name() == &local_name!("summary") {
                    continue; // Skip the summary element as it has its own slot.
                }

                slottable_children.push(ElementOrText::Element(DomRoot::from_ref(element))); // Add other elements to slottable children.
            }

            if let Some(text) = child.downcast::<Text>() {
                slottable_children.push(ElementOrText::Text(DomRoot::from_ref(text))); // Add text nodes to slottable children.
            }
        }
        shadow_tree.descendants.Assign(slottable_children); // Assign remaining children to descendants slot.

        self.upcast::<Node>().dirty(NodeDamage::OtherNodeDamage); // Mark node as dirty for re-rendering.
    }

    /// @brief Updates the styles of the shadow tree contents based on the `<details>` element's `open` state.
    /// Functional Utility: Controls the visibility of the descendants content and applies
    /// appropriate list-item styling to the implicit summary element.
    ///
    /// @param can_gc A `CanGc` token.
    fn update_shadow_tree_styles(&self, can_gc: CanGc) {
        let shadow_tree = self.shadow_tree(can_gc); // Get a reference to the shadow tree.

        // Block Logic: Set the display style for the descendants slot based on the `open` state.
        let value = if self.Open() {
            "display: block;" // If open, display descendants.
        } else {
            // TODO: This should be "display: block; content-visibility: hidden;",
            // but servo does not support content-visibility yet
            "display: none;" // If closed, hide descendants.
        };
        shadow_tree
            .descendants
            .upcast::<Element>()
            .set_string_attribute(&local_name!("style"), value.into(), can_gc); // Apply style to descendants slot.

        // Manually update the list item style of the implicit summary element.
        // Unlike the other summaries, this summary is in the shadow tree and
        // can't be styled with UA sheets
        // Block Logic: Set the list-style for the implicit summary based on the `open` state.
        let implicit_summary_list_item_style = if self.Open() {
            "disclosure-open" // Style for open state.
        } else {
            "disclosure-closed" // Style for closed state.
        };
        let implicit_summary_style = format!(
            "display: list-item;
            counter-increment: list-item 0;
            list-style: {implicit_summary_list_item_style} inside;"
        ); // Full style string for implicit summary.
        shadow_tree
            .implicit_summary
            .upcast::<Element>()
            .set_string_attribute(&local_name!("style"), implicit_summary_style.into(), can_gc); // Apply style to implicit summary.

        self.upcast::<Node>().dirty(NodeDamage::OtherNodeDamage); // Mark node as dirty for re-rendering.
    }
}

impl HTMLDetailsElementMethods<crate::DomTypeHolder> for HTMLDetailsElement {
    // https://html.spec.whatwg.org/multipage/#dom-details-open
    /// @brief Returns the current open state of the `<details>` element.
    /// Functional Utility: Implements the `open` getter, reflecting the `open` attribute.
    make_bool_getter!(Open, "open");

    // https://html.spec.whatwg.org/multipage/#dom-details-open
    /// @brief Sets the open state of the `<details>` element.
    /// Functional Utility: Implements the `open` setter, reflecting the `open` attribute.
    /// @param value The boolean value to set the `open` attribute to.
    make_bool_setter!(SetOpen, "open");
}

impl VirtualMethods for HTMLDetailsElement {
    /// @brief Returns the `VirtualMethods` implementation of the super type (`HTMLElement`).
    /// Functional Utility: Enables method overriding and calls to the superclass's implementations.
    /// @return An `Option` containing a reference to the super type's `VirtualMethods`.
    fn super_type(&self) -> Option<&dyn VirtualMethods> {
        Some(self.upcast::<HTMLElement>() as &dyn VirtualMethods)
    }

    /// @brief Handles attribute mutations for the `<details>` element.
    /// Functional Utility: Responds to changes in the `open` attribute by updating
    /// shadow tree styles, dispatching a `toggle` event, and marking for re-rendering.
    ///
    /// @param attr The `Attr` that was mutated.
    /// @param mutation The type of `AttributeMutation` that occurred.
    /// @param can_gc A `CanGc` token.
    fn attribute_mutated(&self, attr: &Attr, mutation: AttributeMutation, can_gc: CanGc) {
        self.super_type()
            .unwrap()
            .attribute_mutated(attr, mutation, can_gc); // Call super type's method.

        // Block Logic: If the 'open' attribute was mutated, update styles and dispatch a toggle event.
        if attr.local_name() == &local_name!("open") {
            self.update_shadow_tree_styles(can_gc); // Update shadow tree styles.

            let counter = self.toggle_counter.get() + 1; // Increment toggle counter.
            self.toggle_counter.set(counter); // Update counter.

            let this = Trusted::new(self); // Create trusted reference.
            self.owner_global()
                .task_manager()
                .dom_manipulation_task_source()
                .queue(task!(details_notification_task_steps: move || { // Queue a task to dispatch toggle event.
                    let this = this.root();
                    // Block Logic: Only dispatch 'toggle' if the counter hasn't changed since task creation.
                    if counter == this.toggle_counter.get() {
                        this.upcast::<EventTarget>().fire_event(atom!("toggle"), CanGc::note()); // Fire toggle event.
                    }
                }));
            self.upcast::<Node>().dirty(NodeDamage::OtherNodeDamage); // Mark node as dirty.
        }
    }

    /// @brief Handles changes in the children of the `<details>` element.
    /// Functional Utility: Updates the content projection into the shadow DOM
    /// whenever the `<details>` element's light DOM children change.
    ///
    /// @param mutation The `ChildrenMutation` that occurred.
    fn children_changed(&self, mutation: &ChildrenMutation) {
        self.super_type().unwrap().children_changed(mutation); // Call super type's method.

        self.update_shadow_tree_contents(CanGc::note()); // Update shadow tree contents.
    }

    /// @brief Binds the `<details>` element to the DOM tree.
    /// Functional Utility: Performs initialization tasks when the element is inserted
    /// into the document, including updating shadow tree contents and styles.
    ///
    /// @param context The `BindContext` for the binding operation.
    /// @param can_gc A `CanGc` token.
    fn bind_to_tree(&self, context: &BindContext, can_gc: CanGc) {
        self.super_type().unwrap().bind_to_tree(context, can_gc); // Call super type's method.

        self.update_shadow_tree_contents(CanGc::note()); // Update shadow tree contents.
        self.update_shadow_tree_styles(CanGc::note()); // Update shadow tree styles.
    }
}