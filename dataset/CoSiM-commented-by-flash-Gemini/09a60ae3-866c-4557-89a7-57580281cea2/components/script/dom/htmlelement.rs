/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// @file htmlelement.rs
/// @brief This file implements the `HTMLElement` interface, which serves as the base
/// interface for all HTML elements. It extends the generic `Element` interface with
/// properties and methods common to all HTML elements, such as `style`, `title`, `lang`,
/// and event handlers.
/// Functional Utility: Provides the foundational DOM representation and behavior for
/// all concrete HTML elements.

use std::collections::HashSet;
use std::default::Default;
use std::rc::Rc;

use dom_struct::dom_struct; // Macro for defining DOM structures.
use html5ever::{LocalName, Prefix, local_name, namespace_url, ns}; // HTML5 parsing types.
use js::rust::HandleObject; // JavaScript object handle.
use script_layout_interface::QueryMsg; // Message types for layout queries.
use style::attr::AttrValue; // Attribute value from style system.
use stylo_dom::ElementState; // Element state from Stylo's DOM.

use super::customelementregistry::CustomElementState; // Custom element state.
use crate::dom::activation::Activatable; // Trait for activatable elements.
use crate::dom::attr::Attr; // DOM attribute.
use crate::dom::bindings::codegen::Bindings::CharacterDataBinding::CharacterData_Binding::CharacterDataMethods; // CharacterData bindings.
use crate::dom::bindings::codegen::Bindings::EventHandlerBinding::{
    EventHandlerNonNull, OnErrorEventHandlerNonNull,
}; // Event handler bindings.
use crate::dom::bindings::codegen::Bindings::HTMLElementBinding::HTMLElementMethods; // Generated bindings for HTMLElement methods.
use crate::dom::bindings::codegen::Bindings::HTMLLabelElementBinding::HTMLLabelElementMethods; // HTMLLabelElement bindings.
use crate::dom::bindings::codegen::Bindings::NodeBinding::Node_Binding::NodeMethods; // Generated bindings for Node methods.
use crate::dom::bindings::codegen::Bindings::ShadowRootBinding::ShadowRoot_Binding::ShadowRootMethods; // ShadowRoot bindings.
use crate::dom::bindings::codegen::Bindings::WindowBinding::WindowMethods; // Window bindings.
use crate::dom::bindings::error::{Error, ErrorResult, Fallible}; // Error handling.
use crate::dom::bindings::inheritance::{Castable, ElementTypeId, HTMLElementTypeId, NodeTypeId}; // Inheritance and type IDs.
use crate::dom::bindings::root::{Dom, DomRoot, MutNullableDom}; // Root DOM types.
use crate::dom::bindings::str::DOMString; // DOMString representation.
use crate::dom::characterdata::CharacterData; // CharacterData type.
use crate::dom::cssstyledeclaration::{CSSModificationAccess, CSSStyleDeclaration, CSSStyleOwner}; // CSS style declaration.
use crate::dom::customelementregistry::CallbackReaction; // Callback reactions for custom elements.
use crate::dom::document::{Document, FocusType}; // Document and FocusType.
use crate::dom::documentfragment::DocumentFragment; // Document fragment.
use crate::dom::domstringmap::DOMStringMap; // DOMStringMap for `dataset`.
use crate::dom::element::{AttributeMutation, Element}; // Element and attribute mutation.
use crate::dom::elementinternals::ElementInternals; // ElementInternals for custom form elements.
use crate::dom::event::Event; // Event type.
use crate::dom::eventtarget::EventTarget; // Event target.
use crate::dom::htmlbodyelement::HTMLBodyElement; // HTMLBodyElement type.
use crate::dom::htmlbrelement::HTMLBRElement; // HTMLBRElement type.
use crate::dom::htmldetailselement::HTMLDetailsElement; // HTMLDetailsElement type.
use crate::dom::htmlformelement::{FormControl, HTMLFormElement}; // HTMLFormElement and FormControl trait.
use crate::dom::htmlframesetelement::HTMLFrameSetElement; // HTMLFrameSetElement type.
use crate::dom::htmlhtmlelement::HTMLHtmlElement; // HTMLHtmlElement type.
use crate::dom::htmlinputelement::{HTMLInputElement, InputType}; // HTMLInputElement and InputType.
use crate::dom::htmllabelelement::HTMLLabelElement; // HTMLLabelElement type.
use crate::dom::htmltextareaelement::HTMLTextAreaElement; // HTMLTextAreaElement type.
use crate::dom::node::{BindContext, Node, NodeTraits, ShadowIncluding, UnbindContext}; // Node and node-related traits.
use crate::dom::shadowroot::ShadowRoot; // ShadowRoot type.
use crate::dom::text::Text; // Text node.
use crate::dom::virtualmethods::VirtualMethods; // Virtual methods for DOM objects.
use crate::script_runtime::CanGc; // Marker trait for types that can be garbage collected.
use crate::script_thread::ScriptThread; // Script thread for enqueuing tasks.

/// @struct HTMLElement
/// @brief Represents the base HTML element interface.
/// Functional Utility: Extends the generic `Element` with common properties and methods
/// applicable to all HTML elements, such as inline styles (`style`), `dataset` attributes,
/// and various event handlers.
///
/// <https://html.spec.whatwg.org/multipage/#htmlelement>
#[dom_struct]
pub(crate) struct HTMLElement {
    element: Element, //!< Inherited properties and methods from `Element`.
    style_decl: MutNullableDom<CSSStyleDeclaration>, //!< The inline `style` attribute's `CSSStyleDeclaration`.
    dataset: MutNullableDom<DOMStringMap>, //!< The `dataset` for `data-*` attributes.
}

impl HTMLElement {
    /// @brief Creates a new `HTMLElement` instance with inherited properties.
    /// Functional Utility: Internal constructor for setting up the basic properties
    /// of an `HTMLElement` based on its tag name, prefix, and owning document.
    ///
    /// @param tag_name The `LocalName` of the HTML element.
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix.
    /// @param document The owning `Document` of this element.
    /// @return A new `HTMLElement` instance.
    pub(crate) fn new_inherited(
        tag_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
    ) -> HTMLElement {
        HTMLElement::new_inherited_with_state(ElementState::empty(), tag_name, prefix, document)
    }

    /// @brief Creates a new `HTMLElement` instance with a specified initial state.
    /// Functional Utility: Internal constructor to allow initializing an `HTMLElement`
    /// with a specific `ElementState` (e.g., for custom elements) in addition to
    /// standard properties.
    ///
    /// @param state The initial `ElementState` for the element.
    /// @param tag_name The `LocalName` of the HTML element.
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix.
    /// @param document The owning `Document` of this element.
    /// @return A new `HTMLElement` instance.
    pub(crate) fn new_inherited_with_state(
        state: ElementState,
        tag_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
    ) -> HTMLElement {
        HTMLElement {
            element: Element::new_inherited_with_state(
                state,
                tag_name,
                ns!(html), // All HTML elements are in the HTML namespace.
                prefix,
                document,
            ),
            style_decl: Default::default(), // Initialize style declaration to default (None).
            dataset: Default::default(),    // Initialize dataset to default (None).
        }
    }

    /// @brief Creates a new `HTMLElement` and reflects it into the DOM.
    /// Functional Utility: Public constructor that builds an `HTMLElement` and makes
    /// it accessible from JavaScript by integrating it into the DOM's object graph.
    ///
    /// @param local_name The `LocalName` of the HTML element.
    /// @param prefix An `Option<Prefix>` for the XML namespace prefix.
    /// @param document The owning `Document` of this element.
    /// @param proto An `Option<HandleObject>` specifying the JavaScript prototype chain.
    /// @param can_gc A `CanGc` token.
    /// @return A new `HTMLElement` instance wrapped in `DomRoot`.
    #[cfg_attr(crown, allow(crown::unrooted_must_root))]
    pub(crate) fn new(
        local_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
        proto: Option<HandleObject>,
        can_gc: CanGc,
    ) -> DomRoot<HTMLElement> {
        Node::reflect_node_with_proto(
            Box::new(HTMLElement::new_inherited(local_name, prefix, document)),
            document,
            proto,
            can_gc,
        )
    }

    /// @brief Checks if the element is either an `HTMLBodyElement` or `HTMLFrameSetElement`.
    /// Functional Utility: Used to determine special event handling behavior for these
    /// top-level document elements (e.g., event propagation to the `Window`).
    /// @return `true` if it's a `<body>` or `<frameset>` element, `false` otherwise.
    fn is_body_or_frameset(&self) -> bool {
        let eventtarget = self.upcast::<EventTarget>();
        eventtarget.is::<HTMLBodyElement>() || eventtarget.is::<HTMLFrameSetElement>()
    }

    /// @brief Generates a plain text representation of the `HTMLElement`.
    /// Functional Utility: Implements the logic for `innerText` and `outerText` JavaScript properties,
    /// by querying the layout engine for the rendered text content.
    ///
    /// <https://html.spec.whatwg.org/multipage/#get-the-text-steps>
    ///
    /// @param can_gc A `CanGc` token.
    /// @return A `DOMString` containing the plain text representation.
    fn get_inner_outer_text(&self, can_gc: CanGc) -> DOMString {
        let node = self.upcast::<Node>(); // Upcast to a generic Node.
        let window = node.owner_window(); // Get the owning window.
        let element = self.as_element(); // Get the underlying Element.

        // Step 1: If the element is not rendered (not connected or has no layout box), return its text content.
        let element_not_rendered = !node.is_connected() || !element.has_css_layout_box(can_gc);
        if element_not_rendered {
            return node.GetTextContent().unwrap_or_default(); // Return text content, or empty string if None.
        }

        // Functional Utility: Trigger a layout reflow for accurate text generation.
        window.layout_reflow(QueryMsg::ElementInnerOuterTextQuery, can_gc);
        // Functional Utility: Query the layout engine for the element's inner/outer text.
        let text = window
            .layout()
            .query_element_inner_outer_text(node.to_trusted_node_address());

        DOMString::from(text) // Convert the result to `DOMString`.
    }
}

impl HTMLElementMethods<crate::DomTypeHolder> for HTMLElement {
    // https://html.spec.whatwg.org/multipage/#the-style-attribute
    /// @brief Returns the `CSSStyleDeclaration` object for the element's inline style.
    /// Functional Utility: Implements the `style` getter, providing programmatic access
    /// to an element's inline CSS properties. Lazily initializes the style declaration.
    /// @param can_gc A `CanGc` token.
    /// @return A `DomRoot<CSSStyleDeclaration>`.
    fn Style(&self, can_gc: CanGc) -> DomRoot<CSSStyleDeclaration> {
        self.style_decl.or_init(|| {
            let global = self.owner_window(); // Get the owning window.
            CSSStyleDeclaration::new(
                &global,
                CSSStyleOwner::Element(Dom::from_ref(self.upcast())),
                None, // No parent style.
                CSSModificationAccess::ReadWrite, // Read-write access.
                can_gc,
            )
        })
    }

    // https://html.spec.whatwg.org/multipage/#attr-title
    /// @brief Returns the value of the `title` attribute.
    /// Functional Utility: Implements the `title` getter for HTML elements.
    make_getter!(Title, "title");
    // https://html.spec.whatwg.org/multipage/#attr-title
    /// @brief Sets the value of the `title` attribute.
    /// Functional Utility: Implements the `title` setter for HTML elements.
    make_setter!(SetTitle, "title");

    // https://html.spec.whatwg.org/multipage/#attr-lang
    /// @brief Returns the value of the `lang` attribute.
    /// Functional Utility: Implements the `lang` getter for HTML elements,
    /// indicating the language of the element's content.
    make_getter!(Lang, "lang");
    // https://html.spec.whatwg.org/multipage/#attr-lang
    /// @brief Sets the value of the `lang` attribute.
    /// Functional Utility: Implements the `lang` setter for HTML elements.
    make_setter!(SetLang, "lang");

    // https://html.spec.whatwg.org/multipage/#the-dir-attribute
    /// @brief Returns the value of the `dir` (directionality) attribute.
    /// Functional Utility: Implements the `dir` getter, indicating the base direction
    /// of text for the element. Handles enumerated values.
    make_enumerated_getter!(
        Dir,
        "dir",
        "ltr" | "rtl" | "auto", // Supported values.
        missing => "", // Value when attribute is missing.
        invalid => ""  // Value when attribute is invalid.
    );

    // https://html.spec.whatwg.org/multipage/#the-dir-attribute
    /// @brief Sets the value of the `dir` (directionality) attribute.
    /// Functional Utility: Implements the `dir` setter for HTML elements.
    make_setter!(SetDir, "dir");

    // https://html.spec.whatwg.org/multipage/#dom-hidden
    /// @brief Returns whether the `hidden` attribute is present.
    /// Functional Utility: Implements the `hidden` getter for HTML elements,
    /// indicating if the element is hidden from view.
    make_bool_getter!(Hidden, "hidden");
    // https://html.spec.whatwg.org/multipage/#dom-hidden
    /// @brief Sets the `hidden` attribute.
    /// Functional Utility: Implements the `hidden` setter for HTML elements,
    /// controlling its visibility.
    make_bool_setter!(SetHidden, "hidden");

    // https://html.spec.whatwg.org/multipage/#globaleventhandlers
    /// @brief Implements global event handlers, excluding `onload`.
    global_event_handlers!(NoOnload);

    // https://html.spec.whatwg.org/multipage/#documentandelementeventhandlers
    /// @brief Implements document and element event handlers.
    document_and_element_event_handlers!();

    // https://html.spec.whatwg.org/multipage/#dom-dataset
    /// @brief Returns a `DOMStringMap` for accessing `data-*` attributes.
    /// Functional Utility: Implements the `dataset` getter, providing a convenient
    /// way to access custom data attributes. Lazily initializes the `DOMStringMap`.
    /// @param can_gc A `CanGc` token.
    /// @return A `DomRoot<DOMStringMap>`.
    fn Dataset(&self, can_gc: CanGc) -> DomRoot<DOMStringMap> {
        self.dataset.or_init(|| DOMStringMap::new(self, can_gc))
    }

    // https://html.spec.whatwg.org/multipage/#handler-onerror
    /// @brief Returns the `onerror` event handler for the element.
    /// Functional Utility: Implements the `onerror` getter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param can_gc A `CanGc` token.
    /// @return An `Option<Rc<OnErrorEventHandlerNonNull>>`.
    fn GetOnerror(&self, can_gc: CanGc) -> Option<Rc<OnErrorEventHandlerNonNull>> {
        // Block Logic: If it's a body or frameset element, delegate to the window's onerror handler.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().GetOnerror() // Delegate to window.
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("error", can_gc) // Get common event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#handler-onerror
    /// @brief Sets the `onerror` event handler for the element.
    /// Functional Utility: Implements the `onerror` setter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param listener An `Option<Rc<OnErrorEventHandlerNonNull>>` for the new handler.
    fn SetOnerror(&self, listener: Option<Rc<OnErrorEventHandlerNonNull>>) {
        // Block Logic: If it's a body or frameset element, delegate to the window's onerror setter.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().SetOnerror(listener) // Delegate to window.
            }
        } else {
            // special setter for error
            self.upcast::<EventTarget>()
                .set_error_event_handler("error", listener) // Set custom error event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#handler-onload
    /// @brief Returns the `onload` event handler for the element.
    /// Functional Utility: Implements the `onload` getter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param can_gc A `CanGc` token.
    /// @return An `Option<Rc<EventHandlerNonNull>>`.
    fn GetOnload(&self, can_gc: CanGc) -> Option<Rc<EventHandlerNonNull>> {
        // Block Logic: If it's a body or frameset element, delegate to the window's onload handler.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().GetOnload() // Delegate to window.
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("load", can_gc) // Get common event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#handler-onload
    /// @brief Sets the `onload` event handler for the element.
    /// Functional Utility: Implements the `onload` setter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param listener An `Option<Rc<EventHandlerNonNull>>` for the new handler.
    fn SetOnload(&self, listener: Option<Rc<EventHandlerNonNull>>) {
        // Block Logic: If it's a body or frameset element, delegate to the window's onload setter.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().SetOnload(listener) // Delegate to window.
            }
            // Invariant: If not a body/frameset or no browsing context, no action taken on window.
        } else {
            self.upcast::<EventTarget>()
                .set_event_handler_common("load", listener) // Set common event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#handler-onblur
    /// @brief Returns the `onblur` event handler for the element.
    /// Functional Utility: Implements the `onblur` getter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param can_gc A `CanGc` token.
    /// @return An `Option<Rc<EventHandlerNonNull>>`.
    fn GetOnblur(&self, can_gc: CanGc) -> Option<Rc<EventHandlerNonNull>> {
        // Block Logic: If it's a body or frameset element, delegate to the window's onblur handler.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().GetOnblur() // Delegate to window.
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("blur", can_gc) // Get common event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#handler-onblur
    /// @brief Sets the `onblur` event handler for the element.
    /// Functional Utility: Implements the `onblur` setter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param listener An `Option<Rc<EventHandlerNonNull>>` for the new handler.
    fn SetOnblur(&self, listener: Option<Rc<EventHandlerNonNull>>) {
        // Block Logic: If it's a body or frameset element, delegate to the window's onblur setter.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().SetOnblur(listener) // Delegate to window.
            }
        } else {
            self.upcast::<EventTarget>()
                .set_event_handler_common("blur", listener) // Set common event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#handler-onfocus
    /// @brief Returns the `onfocus` event handler for the element.
    /// Functional Utility: Implements the `onfocus` getter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param can_gc A `CanGc` token.
    /// @return An `Option<Rc<EventHandlerNonNull>>`.
    fn GetOnfocus(&self, can_gc: CanGc) -> Option<Rc<EventHandlerNonNull>> {
        // Block Logic: If it's a body or frameset element, delegate to the window's onfocus handler.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().GetOnfocus() // Delegate to window.
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("focus", can_gc) // Get common event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#handler-onfocus
    /// @brief Sets the `onfocus` event handler for the element.
    /// Functional Utility: Implements the `onfocus` setter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param listener An `Option<Rc<EventHandlerNonNull>>` for the new handler.
    fn SetOnfocus(&self, listener: Option<Rc<EventHandlerNonNull>>) {
        // Block Logic: If it's a body or frameset element, delegate to the window's onfocus setter.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().SetOnfocus(listener) // Delegate to window.
            }
        } else {
            self.upcast::<EventTarget>()
                .set_event_handler_common("focus", listener) // Set common event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#handler-onresize
    /// @brief Returns the `onresize` event handler for the element.
    /// Functional Utility: Implements the `onresize` getter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param can_gc A `CanGc` token.
    /// @return An `Option<Rc<EventHandlerNonNull>>`.
    fn GetOnresize(&self, can_gc: CanGc) -> Option<Rc<EventHandlerNonNull>> {
        // Block Logic: If it's a body or frameset element, delegate to the window's onresize handler.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().GetOnresize() // Delegate to window.
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("resize", can_gc) // Get common event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#handler-onresize
    /// @brief Sets the `onresize` event handler for the element.
    /// Functional Utility: Implements the `onresize` setter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param listener An `Option<Rc<EventHandlerNonNull>>` for the new handler.
    fn SetOnresize(&self, listener: Option<Rc<EventHandlerNonNull>>) {
        // Block Logic: If it's a body or frameset element, delegate to the window's onresize setter.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().SetOnresize(listener) // Delegate to window.
            }
        } else {
            self.upcast::<EventTarget>()
                .set_event_handler_common("resize", listener) // Set common event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#handler-onscroll
    /// @brief Returns the `onscroll` event handler for the element.
    /// Functional Utility: Implements the `onscroll` getter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param can_gc A `CanGc` token.
    /// @return An `Option<Rc<EventHandlerNonNull>>`.
    fn GetOnscroll(&self, can_gc: CanGc) -> Option<Rc<EventHandlerNonNull>> {
        // Block Logic: If it's a body or frameset element, delegate to the window's onscroll handler.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().GetOnscroll() // Delegate to window.
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("scroll", can_gc) // Get common event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#handler-onscroll
    /// @brief Sets the `onscroll` event handler for the element.
    /// Functional Utility: Implements the `onscroll` setter, handling special
    /// propagation for `<body>` and `<frameset>` elements to the `Window`.
    /// @param listener An `Option<Rc<EventHandlerNonNull>>` for the new handler.
    fn SetOnscroll(&self, listener: Option<Rc<EventHandlerNonNull>>) {
        // Block Logic: If it's a body or frameset element, delegate to the window's onscroll setter.
        if self.is_body_or_frameset() {
            let document = self.owner_document(); // Get the owning document.
            if document.has_browsing_context() {
                document.window().SetOnscroll(listener) // Delegate to window.
            }
        } else {
            self.upcast::<EventTarget>()
                .set_event_handler_common("scroll", listener) // Set common event handler.
        }
    }

    // https://html.spec.whatwg.org/multipage/#attr-itemtype
    /// @brief Returns a list of `itemtype` attribute values.
    /// Functional Utility: Implements the `itemtypes` getter, extracting and
    /// normalizing the values from the `itemtype` attribute.
    /// @return An `Option<Vec<DOMString>>` containing the item type values, or `None`.
    fn Itemtypes(&self) -> Option<Vec<DOMString>> {
        let atoms = self
            .element
            .get_tokenlist_attribute(&local_name!("itemtype")); // Get token list from `itemtype` attribute.

        // Block Logic: If no tokens, return None.
        if atoms.is_empty() {
            return None;
        }

        let mut item_attr_values = HashSet::new(); // Use a HashSet to ensure unique values.
        // Block Logic: Trim whitespace and convert each token to `DOMString`.
        for attr_value in &atoms {
            item_attr_values.insert(DOMString::from(String::from(attr_value.trim())));
        }

        Some(item_attr_values.into_iter().collect()) // Convert HashSet to Vec and return.
    }

    // https://html.spec.whatwg.org/multipage/#names:-the-itemprop-attribute
    /// @brief Returns a list of `itemprop` attribute values.
    /// Functional Utility: Implements the `propertyNames` getter, extracting and
    /// normalizing the values from the `itemprop` attribute.
    /// @return An `Option<Vec<DOMString>>` containing the property names, or `None`.
    fn PropertyNames(&self) -> Option<Vec<DOMString>> {
        let atoms = self
            .element
            .get_tokenlist_attribute(&local_name!("itemprop")); // Get token list from `itemprop` attribute.

        // Block Logic: If no tokens, return None.
        if atoms.is_empty() {
            return None;
        }

        let mut item_attr_values = HashSet::new(); // Use a HashSet to ensure unique values.
        // Block Logic: Trim whitespace and convert each token to `DOMString`.
        for attr_value in &atoms {
            item_attr_values.insert(DOMString::from(String::from(attr_value.trim())));
        }

        Some(item_attr_values.into_iter().collect()) // Convert HashSet to Vec and return.
    }

    /// @brief Simulates a programmatic click on the element.
    /// Functional Utility: Implements the `click()` method, dispatching a synthetic
    /// `click` event if the element is not disabled or already processing a click.
    /// <https://html.spec.whatwg.org/multipage/#dom-click>
    ///
    /// @param can_gc A `CanGc` token.
    fn Click(&self, can_gc: CanGc) {
        let element = self.as_element(); // Get the underlying Element.
        // Block Logic: If element is disabled or already clicking, ignore the request.
        if element.disabled_state() {
            return;
        }
        if element.click_in_progress() {
            return;
        }
        element.set_click_in_progress(true); // Set click in progress flag.

        self.upcast::<Node>()
            .fire_synthetic_pointer_event_not_trusted(DOMString::from("click"), can_gc); // Fire synthetic click event.
        element.set_click_in_progress(false); // Clear click in progress flag.
    }

    // https://html.spec.whatwg.org/multipage/#dom-focus
    /// @brief Sets focus on the element.
    /// Functional Utility: Implements the `focus()` method, delegating the actual
    /// focus management to the document's focus handling mechanism.
    /// @param can_gc A `CanGc` token.
    fn Focus(&self, can_gc: CanGc) {
        // TODO: Mark the element as locked for focus and run the focusing steps.
        // https://html.spec.whatwg.org/multipage/#focusing-steps
        let document = self.owner_document(); // Get the owning document.
        document.request_focus(Some(self.upcast()), FocusType::Element, can_gc); // Request focus for this element.
    }

    // https://html.spec.whatwg.org/multipage/#dom-blur
    /// @brief Removes focus from the element.
    /// Functional Utility: Implements the `blur()` method, delegating the actual
    /// unfocusing management to the document.
    /// @param can_gc A `CanGc` token.
    fn Blur(&self, can_gc: CanGc) {
        // TODO: Run the unfocusing steps.
        // Block Logic: If the element is not currently focused, return.
        if !self.as_element().focus_state() {
            return;
        }
        // https://html.spec.whatwg.org/multipage/#unfocusing-steps
        let document = self.owner_document(); // Get the owning document.
        document.request_focus(None, FocusType::Element, can_gc); // Request to unfocus any element.
    }

    // https://drafts.csswg.org/cssom-view/#dom-htmlelement-offsetparent
    /// @brief Returns the `offsetParent` of the element.
    /// Functional Utility: Implements the `offsetParent` getter, which identifies the
    /// closest positioned ancestor element.
    /// @param can_gc A `CanGc` token.
    /// @return An `Option<DomRoot<Element>>` containing the offset parent, or `None`.
    fn GetOffsetParent(&self, can_gc: CanGc) -> Option<DomRoot<Element>> {
        // Block Logic: HTMLBodyElement and HTMLHtmlElement have no offset parent.
        if self.is::<HTMLBodyElement>() || self.is::<HTMLHtmlElement>() {
            return None;
        }

        let node = self.upcast::<Node>(); // Upcast to Node.
        let window = self.owner_window(); // Get the owning window.
        let (element, _) = window.offset_parent_query(node, can_gc); // Query layout for offset parent.

        element
    }

    // https://drafts.csswg.org/cssom-view/#dom-htmlelement-offsettop
    /// @brief Returns the `offsetTop` property of the element.
    /// Functional Utility: Implements the `offsetTop` getter, providing the distance
    /// of the element's top border edge from its `offsetParent`'s top padding edge.
    /// @param can_gc A `CanGc` token.
    /// @return The `offsetTop` value in pixels.
    fn OffsetTop(&self, can_gc: CanGc) -> i32 {
        // Block Logic: HTMLBodyElement has an offsetTop of 0.
        if self.is::<HTMLBodyElement>() {
            return 0;
        }

        let node = self.upcast::<Node>(); // Upcast to Node.
        let window = self.owner_window(); // Get the owning window.
        let (_, rect) = window.offset_parent_query(node, can_gc); // Query layout for offset rectangle.

        rect.origin.y.to_nearest_px() // Return the y-coordinate of the origin, snapped to nearest pixel.
    }

    // https://drafts.csswg.org/cssom-view/#dom-htmlelement-offsetleft
    /// @brief Returns the `offsetLeft` property of the element.
    /// Functional Utility: Implements the `offsetLeft` getter, providing the distance
    /// of the element's left border edge from its `offsetParent`'s left padding edge.
    /// @param can_gc A `CanGc` token.
    /// @return The `offsetLeft` value in pixels.
    fn OffsetLeft(&self, can_gc: CanGc) -> i32 {
        // Block Logic: HTMLBodyElement has an offsetLeft of 0.
        if self.is::<HTMLBodyElement>() {
            return 0;
        }

        let node = self.upcast::<Node>(); // Upcast to Node.
        let window = self.owner_window(); // Get the owning window.
        let (_, rect) = window.offset_parent_query(node, can_gc); // Query layout for offset rectangle.

        rect.origin.x.to_nearest_px() // Return the x-coordinate of the origin, snapped to nearest pixel.
    }

    // https://drafts.csswg.org/cssom-view/#dom-htmlelement-offsetwidth
    /// @brief Returns the `offsetWidth` property of the element.
    /// Functional Utility: Implements the `offsetWidth` getter, providing the layout
    /// width of an element, including padding and borders.
    /// @param can_gc A `CanGc` token.
    /// @return The `offsetWidth` value in pixels.
    fn OffsetWidth(&self, can_gc: CanGc) -> i32 {
        let node = self.upcast::<Node>(); // Upcast to Node.
        let window = self.owner_window(); // Get the owning window.
        let (_, rect) = window.offset_parent_query(node, can_gc); // Query layout for offset rectangle.

        rect.size.width.to_nearest_px() // Return the width, snapped to nearest pixel.
    }

    // https://drafts.csswg.org/cssom-view/#dom-htmlelement-offsetheight
    /// @brief Returns the `offsetHeight` property of the element.
    /// Functional Utility: Implements the `offsetHeight` getter, providing the layout
    /// height of an element, including padding and borders.
    /// @param can_gc A `CanGc` token.
    /// @return The `offsetHeight` value in pixels.
    fn OffsetHeight(&self, can_gc: CanGc) -> i32 {
        let node = self.upcast::<Node>(); // Upcast to Node.
        let window = self.owner_window(); // Get the owning window.
        let (_, rect) = window.offset_parent_query(node, can_gc); // Query layout for offset rectangle.

        rect.size.height.to_nearest_px() // Return the height, snapped to nearest pixel.
    }

    /// @brief Returns the `innerText` of the element.
    /// Functional Utility: Implements the `innerText` getter, providing a plain text
    /// representation of the element's rendered content.
    /// <https://html.spec.whatwg.org/multipage/#the-innertext-idl-attribute>
    /// @param can_gc A `CanGc` token.
    /// @return A `DOMString` containing the `innerText`.
    fn InnerText(&self, can_gc: CanGc) -> DOMString {
        self.get_inner_outer_text(can_gc) // Delegate to `get_inner_outer_text`.
    }

    /// @brief Sets the `innerText` of the element.
    /// Functional Utility: Implements the `innerText` setter, replacing the element's
    /// children with a text fragment derived from the input string,
    /// converting line breaks to `<br>` elements.
    /// <https://html.spec.whatwg.org/multipage/#set-the-inner-text-steps>
    ///
    /// @param input The `DOMString` to set as the new `innerText`.
    /// @param can_gc A `CanGc` token.
    fn SetInnerText(&self, input: DOMString, can_gc: CanGc) {
        // Step 1: Let fragment be the rendered text fragment for value given element's node
        // document.
        let fragment = self.rendered_text_fragment(input, can_gc);

        // Step 2: Replace all with fragment within element.
        Node::replace_all(Some(fragment.upcast()), self.upcast::<Node>(), can_gc); // Replace all children with the fragment.
    }

    /// @brief Returns the `outerText` of the element.
    /// Functional Utility: Implements the `outerText` getter, providing a plain text
    /// representation of the element itself and its rendered content.
    /// <https://html.spec.whatwg.org/multipage/#dom-outertext>
    /// @param can_gc A `CanGc` token.
    /// @return A `Fallible<DOMString>` containing the `outerText`.
    fn GetOuterText(&self, can_gc: CanGc) -> Fallible<DOMString> {
        Ok(self.get_inner_outer_text(can_gc)) // Delegate to `get_inner_outer_text`.
    }

    /// @brief Sets the `outerText` of the element.
    /// Functional Utility: Implements the `outerText` setter, replacing the element itself
    /// with a text fragment derived from the input string, and merging adjacent text nodes.
    /// <https://html.spec.whatwg.org/multipage/#the-innertext-idl-attribute:dom-outertext-2>
    ///
    /// @param input The `DOMString` to set as the new `outerText`.
    /// @param can_gc A `CanGc` token.
    /// @return A `Fallible<()>` indicating success or failure.
    fn SetOuterText(&self, input: DOMString, can_gc: CanGc) -> Fallible<()> {
        // Step 1: If this's parent is null, then throw a "NoModificationAllowedError" DOMException.
        let Some(parent) = self.upcast::<Node>().GetParentNode() else {
            return Err(Error::NoModificationAllowed); // Cannot replace if no parent.
        };

        let node = self.upcast::<Node>(); // Upcast to Node.
        let document = self.owner_document(); // Get the owning document.

        // Step 2: Let next be this's next sibling.
        let next = node.GetNextSibling();

        // Step 3: Let previous be this's previous sibling.
        let previous = node.GetPreviousSibling();

        // Step 4: Let fragment be the rendered text fragment for the given value given this's node
        // document.
        let fragment = self.rendered_text_fragment(input, can_gc); // Create text fragment from input.

        // Step 5: If fragment has no children, then append a new Text node whose data is the empty
        // string and node document is this's node document to fragment.
        if fragment.upcast::<Node>().children_count() == 0 {
            let text_node = Text::new(DOMString::from("".to_owned()), &document, can_gc); // Create empty text node.

            fragment
                .upcast::<Node>()
                .AppendChild(text_node.upcast(), can_gc)?;
        }

        // Step 6: Replace this with fragment within this's parent.
        parent.ReplaceChild(fragment.upcast(), node, can_gc)?;

        // Step 7: If next is non-null and next's previous sibling is a Text node, then merge with
        // the next text node given next's previous sibling.
        if let Some(next_sibling) = next {
            if let Some(node) = next_sibling.GetPreviousSibling() {
                Self::merge_with_the_next_text_node(node, can_gc); // Merge with previous text node.
            }
        }

        // Step 8: If previous is a Text node, then merge with the next text node given previous.
        if let Some(previous) = previous {
            Self::merge_with_the_next_text_node(previous, can_gc) // Merge with previous text node.
        }

        Ok(())
    }

    // https://html.spec.whatwg.org/multipage/#dom-translate
    /// @brief Returns whether translation is enabled for the element.
    /// Functional Utility: Implements the `translate` getter, reflecting the `translate` attribute.
    /// @return `true` if translation is enabled, `false` otherwise.
    fn Translate(&self) -> bool {
        self.as_element().is_translate_enabled()
    }

    // https://html.spec.whatwg.org/multipage/#dom-translate
    /// @brief Sets whether translation is enabled for the element.
    /// Functional Utility: Implements the `translate` setter, controlling the `translate` attribute.
    /// @param yesno `true` to enable translation, `false` to disable it.
    /// @param can_gc A `CanGc` token.
    fn SetTranslate(&self, yesno: bool, can_gc: CanGc) {
        self.as_element().set_string_attribute(
            &html5ever::local_name!("translate"), // Attribute name.
            match yesno {
                true => DOMString::from("yes"),
                false => DOMString::from("no"),
            },
            can_gc,
        );
    }

    // https://html.spec.whatwg.org/multipage/#dom-contenteditable
    /// @brief Returns the `contentEditable` state of the element.
    /// Functional Utility: Implements the `contentEditable` getter, indicating
    /// whether the content of the element is editable.
    /// @return A `DOMString` representing the `contentEditable` state.
    fn ContentEditable(&self) -> DOMString {
        // TODO: https://github.com/servo/servo/issues/12776
        self.as_element()
            .get_attribute(&ns!(), &local_name!("contenteditable")) // Get `contenteditable` attribute.
            .map(|attr| DOMString::from(&**attr.value())) // Convert to DOMString.
            .unwrap_or_else(|| DOMString::from("inherit")) // Default to "inherit".
    }

    // https://html.spec.whatwg.org/multipage/#dom-contenteditable
    /// @brief Sets the `contentEditable` state of the element.
    /// Functional Utility: Implements the `contentEditable` setter.
    /// @param _: DOMString The new `contentEditable` state.
    fn SetContentEditable(&self, _: DOMString) {
        // TODO: https://github.com/servo/servo/issues/12776
        warn!("The contentEditable attribute is not implemented yet"); // Log a warning.
    }

    // https://html.spec.whatwg.org/multipage/#dom-contenteditable
    /// @brief Returns `true` if the content of the element is editable.
    /// Functional Utility: Implements the `isContentEditable` getter.
    /// @return `true` if content is editable, `false` otherwise.
    fn IsContentEditable(&self) -> bool {
        // TODO: https://github.com/servo/servo/issues/12776
        false // Not implemented yet.
    }
    /// @brief Attaches `ElementInternals` to a custom element.
    /// Functional Utility: Implements the `attachInternals()` method, providing
    /// access to internal features of custom form-associated elements.
    /// <https://html.spec.whatwg.org/multipage#dom-attachinternals>
    ///
    /// @param can_gc A `CanGc` token.
    /// @return A `Fallible<DomRoot<ElementInternals>>` containing the attached internals.
    fn AttachInternals(&self, can_gc: CanGc) -> Fallible<DomRoot<ElementInternals>> {
        let element = self.as_element(); // Get the underlying Element.
        // Step 1: If this's is value is not null, then throw a "NotSupportedError" DOMException
        if element.get_is().is_some() {
            return Err(Error::NotSupported); // Custom element with `is` attribute is not supported.
        }

        // Step 2: Let definition be the result of looking up a custom element definition
        // Note: the element can pass this check without yet being a custom
        // element, as long as there is a registered definition
        // that could upgrade it to one later.
        let registry = self.owner_document().window().CustomElements(); // Get custom element registry.
        let definition = registry.lookup_definition(self.as_element().local_name(), None); // Look up definition.

        // Step 3: If definition is null, then throw an "NotSupportedError" DOMException
        let definition = match definition {
            Some(definition) => definition,
            None => return Err(Error::NotSupported), // No definition found.
        };

        // Step 4: If definition's disable internals is true, then throw a "NotSupportedError" DOMException
        if definition.disable_internals {
            return Err(Error::NotSupported); // Internals disabled by definition.
        }

        // Step 5: If this's attached internals is non-null, then throw an "NotSupportedError" DOMException
        let internals = element.ensure_element_internals(can_gc); // Ensure ElementInternals exist.
        if internals.attached() {
            return Err(Error::NotSupported); // Internals already attached.
        }

        // Step 6: If this's custom element state is not "precustomized" or "custom",
        // then throw a "NotSupportedError" DOMException.
        if !matches!(
            element.get_custom_element_state(),
            CustomElementState::Precustomized | CustomElementState::Custom
        ) {
            return Err(Error::NotSupported); // Invalid custom element state.
        }

        // Block Logic: Initialize state for internals if it's a form-associated custom element.
        if self.is_form_associated_custom_element() {
            element.init_state_for_internals();
        }

        // Step 6-7: Set this's attached internals to a new ElementInternals instance
        internals.set_attached(); // Mark internals as attached.
        Ok(internals) // Return the attached internals.
    }

    // FIXME: The nonce should be stored in an internal slot instead of an
    // attribute (https://html.spec.whatwg.org/multipage/#cryptographicnonce)
    // https://html.spec.whatwg.org/multipage/#dom-noncedelement-nonce
    /// @brief Returns the `nonce` attribute of the element.
    /// Functional Utility: Implements the `nonce` getter for elements that can have it.
    make_getter!(Nonce, "nonce");

    // https://html.spec.whatwg.org/multipage/#dom-noncedelement-nonce
    /// @brief Sets the `nonce` attribute of the element.
    /// Functional Utility: Implements the `nonce` setter.
    make_setter!(SetNonce, "nonce");

    // https://html.spec.whatwg.org/multipage/#dom-fe-autofocus
    /// @brief Returns whether the `autofocus` attribute is present.
    /// Functional Utility: Implements the `autofocus` getter for form-associated elements.
    /// @return `true` if `autofocus` is present, `false` otherwise.
    fn Autofocus(&self) -> bool {
        self.element.has_attribute(&local_name!("autofocus"))
    }

    // https://html.spec.whatwg.org/multipage/#dom-fe-autofocus
    /// @brief Sets the `autofocus` attribute of the element.
    /// Functional Utility: Implements the `autofocus` setter.
    /// @param autofocus `true` to add the `autofocus` attribute, `false` to remove it.
    /// @param can_gc A `CanGc` token.
    fn SetAutofocus(&self, autofocus: bool, can_gc: CanGc) {
        self.element
            .set_bool_attribute(&local_name!("autofocus"), autofocus, can_gc);
    }
}

/// @brief Appends a text node to a document fragment.
/// Functional Utility: Helper function for `rendered_text_fragment` to add text content
/// to a fragment.
///
/// @param document The owning `Document`.
/// @param fragment The `DocumentFragment` to append to.
/// @param text The text content as a `String`.
/// @param can_gc A `CanGc` token.
fn append_text_node_to_fragment(
    document: &Document,
    fragment: &DocumentFragment,
    text: String,
    can_gc: CanGc,
) {
    let text = Text::new(DOMString::from(text), document, can_gc); // Create a new Text node.
    fragment
        .upcast::<Node>()
        .AppendChild(text.upcast(), can_gc) // Append the Text node to the fragment.
        .unwrap(); // Unwrap the result (should not fail).
}

// https://html.spec.whatwg.org/multipage/#attr-data-*

static DATA_PREFIX: &str = "data-"; //!< Prefix for custom data attributes.
static DATA_HYPHEN_SEPARATOR: char = '\x2d'; //!< Hyphen separator used in data attribute names.

/// @brief Converts a camelCase `DOMString` to a kebab-case `DOMString` with a "data-" prefix.
/// Functional Utility: Used internally for mapping JavaScript `dataset` property names
/// (camelCase) to their corresponding HTML `data-*` attribute names (kebab-case).
///
/// @param name The camelCase `DOMString` to convert.
/// @return A `DOMString` in kebab-case with the "data-" prefix.
fn to_snake_case(name: DOMString) -> DOMString {
    let mut attr_name = String::with_capacity(name.len() + DATA_PREFIX.len()); // Pre-allocate string.
    attr_name.push_str(DATA_PREFIX); // Add "data-" prefix.
    // Block Logic: Iterate through characters and convert camelCase to kebab-case.
    for ch in name.chars() {
        if ch.is_ascii_uppercase() {
            attr_name.push(DATA_HYPHEN_SEPARATOR); // Add hyphen before uppercase.
            attr_name.push(ch.to_ascii_lowercase()); // Convert to lowercase.
        } else {
            attr_name.push(ch); // Append character as is.
        }
    }
    DOMString::from(attr_name) // Return the converted DOMString.
}

// https://html.spec.whatwg.org/multipage/#attr-data-*
// if this attribute is in snake case with a data- prefix,
// this function returns a name converted to camel case
// without the data prefix.

/// @brief Converts a kebab-case "data-" prefixed string to a camelCase `DOMString`.
/// Functional Utility: Used internally for mapping HTML `data-*` attribute names
/// (kebab-case) to their corresponding JavaScript `dataset` property names (camelCase).
///
/// @param name The kebab-case "data-" prefixed string to convert.
/// @return An `Option<DOMString>` in camelCase without the "data-" prefix, or `None` if invalid.
fn to_camel_case(name: &str) -> Option<DOMString> {
    // Block Logic: Check for "data-" prefix.
    if !name.starts_with(DATA_PREFIX) {
        return None;
    }
    let name = &name[5..]; // Remove "data-" prefix.
    // Block Logic: If the remaining name contains uppercase characters, it's invalid.
    let has_uppercase = name.chars().any(|curr_char| curr_char.is_ascii_uppercase());
    if has_uppercase {
        return None;
    }
    let mut result = String::with_capacity(name.len().saturating_sub(DATA_PREFIX.len())); // Pre-allocate string.
    let mut name_chars = name.chars(); // Character iterator.
    // Block Logic: Iterate through characters and convert kebab-case to camelCase.
    while let Some(curr_char) = name_chars.next() {
        //check for hyphen followed by character
        if curr_char == DATA_HYPHEN_SEPARATOR {
            if let Some(next_char) = name_chars.next() {
                if next_char.is_ascii_lowercase() {
                    result.push(next_char.to_ascii_uppercase()); // Convert next char to uppercase.
                } else {
                    result.push(curr_char); // Keep hyphen if next char is not lowercase.
                    result.push(next_char);
                }
            } else {
                result.push(curr_char); // Keep hyphen if no next char.
            }
        } else {
            result.push(curr_char); // Append character as is.
        }
    }
    Some(DOMString::from(result)) // Return the converted DOMString.
}

impl HTMLElement {
    /// @brief Sets a custom attribute with a "data-" prefix.
    /// Functional Utility: Provides the setter for `dataset` properties, converting
    /// camelCase property names to kebab-case `data-*` attribute names.
    ///
    /// @param name The camelCase `DOMString` property name.
    /// @param value The `DOMString` value to set.
    /// @param can_gc A `CanGc` token.
    /// @return An `ErrorResult` indicating success or a `Syntax` error.
    pub(crate) fn set_custom_attr(
        &self,
        name: DOMString,
        value: DOMString,
        can_gc: CanGc,
    ) -> ErrorResult {
        // Block Logic: Validate the name format (no hyphens followed by lowercase after the first char).
        if name
            .chars()
            .skip_while(|&ch| ch != '\u{2d}')
            .nth(1)
            .is_some_and(|ch| ch.is_ascii_lowercase())
        {
            return Err(Error::Syntax); // Invalid name format.
        }
        self.as_element()
            .set_custom_attribute(to_snake_case(name), value, can_gc) // Set the attribute with converted name.
    }

    /// @brief Retrieves the value of a custom attribute with a "data-" prefix.
    /// Functional Utility: Provides the getter for `dataset` properties, converting
    /// camelCase property names to kebab-case `data-*` attribute names for lookup.
    ///
    /// @param local_name The camelCase `DOMString` property name.
    /// @return An `Option<DOMString>` containing the attribute value, or `None`.
    pub(crate) fn get_custom_attr(&self, local_name: DOMString) -> Option<DOMString> {
        // FIXME(ajeffrey): Convert directly from DOMString to LocalName
        let local_name = LocalName::from(to_snake_case(local_name)); // Convert to kebab-case LocalName.
        self.as_element()
            .get_attribute(&ns!(), &local_name) // Get the attribute.
            .map(|attr| {
                DOMString::from(&**attr.value()) // FIXME(ajeffrey): Convert directly from AttrValue to DOMString
            })
    }

    /// @brief Deletes a custom attribute with a "data-" prefix.
    /// Functional Utility: Provides the deleter for `dataset` properties, converting
    /// camelCase property names to kebab-case `data-*` attribute names for deletion.
    ///
    /// @param local_name The camelCase `DOMString` property name.
    /// @param can_gc A `CanGc` token.
    pub(crate) fn delete_custom_attr(&self, local_name: DOMString, can_gc: CanGc) {
        // FIXME(ajeffrey): Convert directly from DOMString to LocalName
        let local_name = LocalName::from(to_snake_case(local_name)); // Convert to kebab-case LocalName.
        self.as_element()
            .remove_attribute(&ns!(), &local_name, can_gc); // Remove the attribute.
    }

    /// @brief Checks if the element is a "labelable" element.
    /// Functional Utility: Determines if an element can be associated with a `<label>`
    /// element, influencing accessibility and form behavior.
    /// <https://html.spec.whatwg.org/multipage/#category-label>
    ///
    /// @return `true` if the element is labelable, `false` otherwise.
    pub(crate) fn is_labelable_element(&self) -> bool {
        match self.upcast::<Node>().type_id() {
            NodeTypeId::Element(ElementTypeId::HTMLElement(type_id)) => match type_id {
                HTMLElementTypeId::HTMLInputElement => {
                    self.downcast::<HTMLInputElement>().unwrap().input_type() != InputType::Hidden // Input elements (except hidden) are labelable.
                },
                HTMLElementTypeId::HTMLButtonElement |
                HTMLElementTypeId::HTMLMeterElement |
                HTMLElementTypeId::HTMLOutputElement |
                HTMLElementTypeId::HTMLProgressElement |
                HTMLElementTypeId::HTMLSelectElement |
                HTMLElementTypeId::HTMLTextAreaElement => true, // These HTML elements are labelable.
                _ => self.is_form_associated_custom_element(), // Custom form-associated elements can be labelable.
            },
            _ => false, // Other node types are not labelable.
        }
    }

    /// @brief Checks if the element is a form-associated custom element.
    /// Functional Utility: Determines if a custom element is designed to participate
    /// in form submission, influencing its behavior within a form.
    /// <https://html.spec.whatwg.org/multipage/#form-associated-custom-element>
    ///
    /// @return `true` if the element is a form-associated custom element, `false` otherwise.
    pub(crate) fn is_form_associated_custom_element(&self) -> bool {
        // Block Logic: Check if it's a custom element and its definition allows form association.
        if let Some(definition) = self.as_element().get_custom_element_definition() {
            definition.is_autonomous() && definition.form_associated // Must be autonomous and form-associated.
        } else {
            false // Not a custom element.
        }
    }

    /// @brief Checks if the element is a "listed" element.
    /// Functional Utility: Determines if an element is part of a form's `elements`
    /// collection, influencing how it's submitted.
    /// <https://html.spec.whatwg.org/multipage/#category-listed>
    ///
    /// @return `true` if the element is listed, `false` otherwise.
    pub(crate) fn is_listed_element(&self) -> bool {
        match self.upcast::<Node>().type_id() {
            NodeTypeId::Element(ElementTypeId::HTMLElement(type_id)) => match type_id {
                HTMLElementTypeId::HTMLButtonElement |
                HTMLElementTypeId::HTMLFieldSetElement |
                HTMLElementTypeId::HTMLInputElement |
                HTMLElementTypeId::HTMLObjectElement |
                HTMLElementTypeId::HTMLOutputElement |
                HTMLElementTypeId::HTMLSelectElement |
                HTMLElementTypeId::HTMLTextAreaElement => true, // These HTML elements are listed.
                _ => self.is_form_associated_custom_element(), // Custom form-associated elements can be listed.
            },
            _ => false, // Other node types are not listed.
        }
    }

    /// @brief Checks if the element is a "submittable" element.
    /// Functional Utility: Determines if an element's value can be submitted as part
    /// of a form.
    /// <https://html.spec.whatwg.org/multipage/#category-submit>
    ///
    /// @return `true` if the element is submittable, `false` otherwise.
    pub(crate) fn is_submittable_element(&self) -> bool {
        match self.upcast::<Node>().type_id() {
            NodeTypeId::Element(ElementTypeId::HTMLElement(type_id)) => match type_id {
                HTMLElementTypeId::HTMLButtonElement |
                HTMLElementTypeId::HTMLInputElement |
                HTMLElementTypeId::HTMLSelectElement |
                HTMLElementTypeId::HTMLTextAreaElement => true, // These HTML elements are submittable.
                _ => self.is_form_associated_custom_element(), // Custom form-associated elements can be submittable.
            },
            _ => false, // Other node types are not submittable.
        }
    }

    /// @brief Returns a list of supported property names for custom attributes (dataset).
    /// Functional Utility: Provides the keys (camelCase) that can be used to access `data-*`
    /// attributes via the `dataset` property.
    ///
    /// @return A `Vec<DOMString>` containing the supported custom attribute names.
    pub(crate) fn supported_prop_names_custom_attr(&self) -> Vec<DOMString> {
        let element = self.as_element(); // Get the underlying Element.
        element
            .attrs() // Iterate over all attributes.
            .iter()
            .filter_map(|attr| {
                let raw_name = attr.local_name(); // Get the local name of the attribute.
                to_camel_case(raw_name) // Convert to camelCase (for data-* attributes).
            })
            .collect() // Collect into a vector.
    }

    // https://html.spec.whatwg.org/multipage/#dom-lfe-labels
    // This gets the nth label in tree order.
    /// @brief Retrieves the nth `<label>` element associated with this element.
    /// Functional Utility: Implements the `labels` getter (indexed access),
    /// finding `<label>` elements in tree order that are associated with this control.
    ///
    /// @param index The zero-based index of the label to retrieve.
    /// @return An `Option<DomRoot<Node>>` containing the label element, or `None`.
    pub(crate) fn label_at(&self, index: u32) -> Option<DomRoot<Node>> {
        let element = self.as_element(); // Get the underlying Element.

        // Traverse entire tree for <label> elements that have
        // this as their control.
        // There is room for performance optimization, as we don't need
        // the actual result of GetControl, only whether the result
        // would match self.
        // (Even more room for performance optimization: do what
        // nodelist ChildrenList does and keep a mutation-aware cursor
        // around; this may be hard since labels need to keep working
        // even as they get detached into a subtree and reattached to
        // a document.)
        let root_element = element.root_element(); // Get the root element of the document.
        let root_node = root_element.upcast::<Node>(); // Upcast to Node.
        root_node
            .traverse_preorder(ShadowIncluding::No) // Traverse the DOM tree.
            .filter_map(DomRoot::downcast::<HTMLLabelElement>) // Filter for HTMLLabelElement.
            .filter(|elem| match elem.GetControl() {
                Some(control) => &*control == self, // Check if the label's control is this element.
                _ => false,
            })
            .nth(index as usize) // Get the nth matching label.
            .map(|n| DomRoot::from_ref(n.upcast::<Node>())) // Map to DomRoot<Node>.
    }

    // https://html.spec.whatwg.org/multipage/#dom-lfe-labels
    // This counts the labels of the element, to support NodeList::Length
    /// @brief Counts the number of `<label>` elements associated with this element.
    /// Functional Utility: Supports the `length` property of the `labels` getter,
    /// by counting associated `<label>` elements.
    /// @return The number of associated labels.
    pub(crate) fn labels_count(&self) -> u32 {
        // see label_at comments about performance
        let element = self.as_element(); // Get the underlying Element.
        let root_element = element.root_element(); // Get the root element of the document.
        let root_node = root_element.upcast::<Node>(); // Upcast to Node.
        root_node
            .traverse_preorder(ShadowIncluding::No) // Traverse the DOM tree.
            .filter_map(DomRoot::downcast::<HTMLLabelElement>) // Filter for HTMLLabelElement.
            .filter(|elem| match elem.GetControl() {
                Some(control) => &*control == self, // Check if the label's control is this element.
                _ => false,
            })
            .count() as u32 // Count the matching labels.
    }

    // https://html.spec.whatwg.org/multipage/#the-directionality.
    // returns Some if can infer direction by itself or from child nodes
    // returns None if requires to go up to parent
    /// @brief Determines the text directionality of the element.
    /// Functional Utility: Implements the `directionality` algorithm, attempting to infer
    /// the text direction (`ltr`, `rtl`, `auto`) from the element's `dir` attribute
    /// or specific element types.
    ///
    /// @return An `Option<String>` containing the inferred directionality, or `None` if it
    ///         needs to be inferred from a parent.
    pub(crate) fn directionality(&self) -> Option<String> {
        let element_direction: &str = &self.Dir(); // Get the value of the `dir` attribute.

        // Block Logic: Handle explicit `ltr` or `rtl` directions.
        if element_direction == "ltr" {
            return Some("ltr".to_owned());
        }

        if element_direction == "rtl" {
            return Some("rtl".to_owned());
        }

        // Block Logic: Special handling for `HTMLInputElement` with `type="tel"`.
        if let Some(input) = self.downcast::<HTMLInputElement>() {
            if input.input_type() == InputType::Tel {
                return Some("ltr".to_owned()); // Telephone input is always LTR.
            }
        }

        // Block Logic: Handle `auto` directionality for input and textarea elements.
        if element_direction == "auto" {
            if let Some(directionality) = self
                .downcast::<HTMLInputElement>()
                .and_then(|input| input.auto_directionality()) // Delegate to input's auto directionality.
            {
                return Some(directionality);
            }

            if let Some(area) = self.downcast::<HTMLTextAreaElement>() {
                return Some(area.auto_directionality()); // Delegate to textarea's auto directionality.
            }
        }

        // TODO(NeverHappened): Implement condition
        // If the element's dir attribute is in the auto state OR
        // If the element is a bdi element and the dir attribute is not in a defined state
        // (i.e. it is not present or has an invalid value)
        // Requires bdi element implementation (https://html.spec.whatwg.org/multipage/#the-bdi-element)

        None // Cannot infer directionality at this level.
    }

    // https://html.spec.whatwg.org/multipage/#the-summary-element:activation-behaviour
    /// @brief Implements the activation behavior for `<summary>` elements.
    /// Functional Utility: When a `<summary>` element is activated (e.g., clicked),
    /// it toggles the `open` state of its parent `<details>` element.
    ///
    /// @pre `self` must be a `<summary>` element.
    pub(crate) fn summary_activation_behavior(&self) {
        debug_assert!(self.as_element().local_name() == &local_name!("summary")); // Pre-condition check.

        // Step 1. If this summary element is not the summary for its parent details, then return.
        if !self.is_a_summary_for_its_parent_details() {
            return;
        }

        // Step 2. Let parent be this summary element's parent.
        // Block Logic: Determine the parent `<details>` element, considering both light and shadow DOM.
        let parent = if self.is_implicit_summary_element() {
            // If it's an implicit summary, its parent is the shadow host (details element).
            DomRoot::downcast::<HTMLDetailsElement>(self.containing_shadow_root().unwrap().Host())
                .unwrap()
        } else {
            // Otherwise, get its direct parent and downcast to HTMLDetailsElement.
            self.upcast::<Node>()
                .GetParentNode()
                .and_then(DomRoot::downcast::<HTMLDetailsElement>)
                .unwrap()
        };

        // Step 3. If the open attribute is present on parent, then remove it.
        // Otherwise, set parent's open attribute to the empty string.
        parent.toggle(); // Toggle the `open` state of the parent `<details>`.
    }

    /// @brief Checks if this `<summary>` element is the one that controls its parent `<details>`.
    /// Functional Utility: Determines if a `<summary>` element is the "correct" summary
    /// for its parent `<details>` based on specification rules (first child `<summary>` or implicit).
    /// <https://html.spec.whatwg.org/multipage/#summary-for-its-parent-details>
    ///
    /// @return `true` if it's the controlling summary, `false` otherwise.
    fn is_a_summary_for_its_parent_details(&self) -> bool {
        // Block Logic: Implicit summaries are always the controlling summary.
        if self.is_implicit_summary_element() {
            return true;
        }

        // Step 1. If this summary element has no parent, then return false.
        // Step 2. Let parent be this summary element's parent.
        let Some(parent) = self.upcast::<Node>().GetParentNode() else {
            return false;
        };

        // Step 3. If parent is not a details element, then return false.
        let Some(details) = parent.downcast::<HTMLDetailsElement>() else {
            return false;
        };

        // Step 4. If parent's first summary element child is not this summary
        // element, then return false.
        // Step 5. Return true.
        // Block Logic: Check if this summary is the first summary child of the details element.
        details
            .find_corresponding_summary_element()
            .is_some_and(|summary| &*summary == self.upcast())
    }

    /// @brief Checks whether this is an implicitly generated `<summary>` element for a UA `<details>` shadow tree.
    /// Functional Utility: Differentiates between user-provided `<summary>` elements
    /// and the fallback summary that the browser automatically creates within the shadow DOM.
    ///
    /// @return `true` if it's an implicit summary, `false` otherwise.
    fn is_implicit_summary_element(&self) -> bool {
        // Note that non-implicit summary elements are not actually inside
        // the UA shadow tree, they're only assigned to a slot inside it.
        // Therefore they don't cause false positives here
        self.containing_shadow_root() // Get the shadow root containing this element.
            .as_deref()
            .map(ShadowRoot::Host) // Get the host element of the shadow root.
            .is_some_and(|host| host.is::<HTMLDetailsElement>()) // Check if the host is an HTMLDetailsElement.
    }

    /// @brief Creates a `DocumentFragment` from a `DOMString`, converting line breaks to `<br>` elements.
    /// Functional Utility: Implements the "rendered text fragment" algorithm, used by
    /// `innerText` and `outerText` setters to transform a plain string into DOM content.
    /// <https://html.spec.whatwg.org/multipage/#rendered-text-fragment>
    ///
    /// @param input The `DOMString` to convert.
    /// @param can_gc A `CanGc` token.
    /// @return A `DomRoot<DocumentFragment>` containing the rendered text fragment.
    fn rendered_text_fragment(&self, input: DOMString, can_gc: CanGc) -> DomRoot<DocumentFragment> {
        // Step 1: Let fragment be a new DocumentFragment whose node document is document.
        let document = self.owner_document(); // Get the owning document.
        let fragment = DocumentFragment::new(&document, can_gc); // Create a new DocumentFragment.

        // Step 2: Let position be a position variable for input, initially pointing at the start
        // of input.
        let mut position = input.chars().peekable(); // Peekable iterator over input characters.

        // Step 3: Let text be the empty string.
        let mut text = String::new(); // Accumulator for text content.

        // Step 4: Iterate through the input string, processing characters.
        while let Some(ch) = position.next() {
            match ch {
                // While position is not past the end of input, and the code point at position is
                // either U+000A LF or U+000D CR:
                '\u{000A}' | '\u{000D}' => { // If a line feed or carriage return is found.
                    if ch == '\u{000D}' && position.peek() == Some(&'\u{000A}') {
                        // a \r\n pair should only generate one <br>,
                        // so just skip the \r.
                        position.next(); // Consume the `\n` if it's a `\r\n` sequence.
                    }

                    // Block Logic: If there's accumulated text, append it as a Text node.
                    if !text.is_empty() {
                        append_text_node_to_fragment(&document, &fragment, text, can_gc);
                        text = String::new(); // Reset text accumulator.
                    }

                    // Block Logic: Create and append a `<br>` element for the line break.
                    let br = HTMLBRElement::new(local_name!("br"), None, &document, None, can_gc);
                    fragment
                        .upcast::<Node>()
                        .AppendChild(br.upcast(), can_gc)
                        .unwrap();
                },
                _ => {
                    // Collect a sequence of code points that are not U+000A LF or U+000D CR from
                    // input given position, and set text to the result.
                    text.push(ch); // Accumulate non-line-break characters.
                },
            }
        }

        // If text is not the empty string, then append a new Text node whose data is text and node
        // document is document to fragment.
        // Block Logic: Append any remaining accumulated text as a Text node.
        if !text.is_empty() {
            append_text_node_to_fragment(&document, &fragment, text, can_gc);
        }

        fragment // Return the constructed DocumentFragment.
    }

    /// @brief Merges a `Text` node with its next sibling if the sibling is also a `Text` node.
    /// Functional Utility: Implements the "merge with the next text node" algorithm,
    /// which consolidates adjacent `Text` nodes into a single `Text` node.
    ///
    /// <https://html.spec.whatwg.org/multipage/#merge-with-the-next-text-node>
    ///
    /// @param node The `DomRoot<Node>` representing the first `Text` node.
    /// @param can_gc A `CanGc` token.
    fn merge_with_the_next_text_node(node: DomRoot<Node>, can_gc: CanGc) {
        // Make sure node is a Text node
        if !node.is::<Text>() {
            return; // If not a Text node, return.
        }

        // Step 1: Let next be node's next sibling.
        let next = match node.GetNextSibling() {
            Some(next) => next,
            None => return, // If no next sibling, return.
        };

        // Step 2: If next is not a Text node, then return.
        if !next.is::<Text>() {
            return; // If next sibling is not a Text node, return.
        }
        // Step 3: Replace data with node, node's data's length, 0, and next's data.
        let node_chars = node.downcast::<CharacterData>().expect("Node is Text"); // Downcast first Text node to CharacterData.
        let next_chars = next.downcast::<CharacterData>().expect("Next node is Text"); // Downcast next Text node to CharacterData.
        node_chars
            .ReplaceData(node_chars.Length(), 0, next_chars.Data())
            .expect("Got chars from Text");

        // Step 4:Remove next.
        next.remove_self(can_gc); // Remove the second (merged) Text node.
    }
}

impl VirtualMethods for HTMLElement {
    /// @brief Returns the `VirtualMethods` implementation of the super type (`Element`).
    /// Functional Utility: Enables method overriding and calls to the superclass's implementations.
    /// @return An `Option` containing a reference to the super type's `VirtualMethods`.
    fn super_type(&self) -> Option<&dyn VirtualMethods> {
        Some(self.as_element() as &dyn VirtualMethods) // Upcast to Element and get its VirtualMethods.
    }

    /// @brief Handles attribute mutations for `HTMLElement`s.
    /// Functional Utility: Processes changes to specific attributes like `on*` event handlers,
    /// `form` (for custom elements), `disabled`, and `readonly`, triggering appropriate
    /// reactions for event binding, form association, and state updates.
    ///
    /// @param attr The `Attr` that was mutated.
    /// @param mutation The type of `AttributeMutation` that occurred.
    /// @param can_gc A `CanGc` token.
    fn attribute_mutated(&self, attr: &Attr, mutation: AttributeMutation, can_gc: CanGc) {
        self.super_type()
            .unwrap()
            .attribute_mutated(attr, mutation, can_gc);
        let element = self.as_element();
        match (attr.local_name(), mutation) {
            // Block Logic: Handle `on*` event handler attributes.
            (name, AttributeMutation::Set(_)) if name.starts_with("on") => {
                let evtarget = self.upcast::<EventTarget>();
                let source_line = 1; //TODO(#9604) get current JS execution line
                evtarget.set_event_handler_uncompiled(
                    self.owner_window().get_url(),
                    source_line,
                    &name[2..], // Remove "on" prefix.
                    // FIXME(ajeffrey): Convert directly from AttrValue to DOMString
                    DOMString::from(&**attr.value()), // Attribute value as DOMString.
                );
            },
            // Block Logic: Handle `form` attribute mutation for form-associated custom elements.
            (&local_name!("form"), mutation) if self.is_form_associated_custom_element() => {
                self.form_attribute_mutated(mutation, can_gc);
            },
            // Block Logic: Handle `disabled` attribute being set for form-associated custom elements.
            // Adding a "disabled" attribute disables an enabled form element.
            (&local_name!("disabled"), AttributeMutation::Set(_))
                if self.is_form_associated_custom_element() && element.enabled_state() =>
            {
                element.set_disabled_state(true);
                element.set_enabled_state(false);
                ScriptThread::enqueue_callback_reaction(
                    element,
                    CallbackReaction::FormDisabled(true), // Enqueue form disabled callback.
                    None,
                );
            },
            // Block Logic: Handle `disabled` attribute being removed for form-associated custom elements.
            // Removing the "disabled" attribute may enable a disabled
            // form element, but a fieldset ancestor may keep it disabled.
            (&local_name!("disabled"), AttributeMutation::Removed)
                if self.is_form_associated_custom_element() && element.disabled_state() =>
            {
                element.set_disabled_state(false);
                element.set_enabled_state(true);
                element.check_ancestors_disabled_state_for_form_control();
                if element.enabled_state() {
                    ScriptThread::enqueue_callback_reaction(
                        element,
                        CallbackReaction::FormDisabled(false), // Enqueue form enabled callback.
                        None,
                    );
                }
            },
            // Block Logic: Handle `readonly` attribute mutation for form-associated custom elements.
            (&local_name!("readonly"), mutation) if self.is_form_associated_custom_element() => {
                match mutation {
                    AttributeMutation::Set(_) => {
                        element.set_read_write_state(true);
                    },
                    AttributeMutation::Removed => {
                        element.set_read_write_state(false);
                    },
                }
            },
            _ => {}, // Other attribute mutations are ignored.
        }
    }

    /// @brief Binds the `HTMLElement` to the DOM tree.
    /// Functional Utility: Performs initialization tasks when the element is inserted
    /// into the document, including updating its sequentially focusable status and
    /// checking for disabled states in form controls due to ancestors.
    ///
    /// @param context The `BindContext` for the binding operation.
    /// @param can_gc A `CanGc` token.
    fn bind_to_tree(&self, context: &BindContext, can_gc: CanGc) {
        if let Some(super_type) = self.super_type() {
            super_type.bind_to_tree(context, can_gc);
        }
        let element = self.as_element();
        element.update_sequentially_focusable_status(can_gc);

        // Block Logic: If it's a form-associated custom element, check for disabled state from ancestors.
        // Binding to a tree can disable a form control if one of the new
        // ancestors is a fieldset.
        if self.is_form_associated_custom_element() && element.enabled_state() {
            element.check_ancestors_disabled_state_for_form_control();
            if element.disabled_state() {
                ScriptThread::enqueue_callback_reaction(
                    element,
                    CallbackReaction::FormDisabled(true),
                    None,
                );
            }
        }
    }

    /// @brief Unbinds the `HTMLElement` from the DOM tree.
    /// Functional Utility: Performs cleanup tasks when the element is removed
    /// from the document, including re-evaluating disabled states for form controls.
    ///
    /// @param context The `UnbindContext` for the unbinding operation.
    /// @param can_gc A `CanGc` token.
    fn unbind_from_tree(&self, context: &UnbindContext, can_gc: CanGc) {
        if let Some(super_type) = self.super_type() {
            super_type.unbind_from_tree(context, can_gc);
        }

        // Block Logic: If it's a form-associated custom element, re-evaluate disabled state.
        // Unbinding from a tree might enable a form control, if a
        // fieldset ancestor is the only reason it was disabled.
        // (The fact that it's enabled doesn't do much while it's
        // disconnected, but it is an observable fact to keep track of.)
        let element = self.as_element();
        if self.is_form_associated_custom_element() && element.disabled_state() {
            element.check_disabled_attribute();
            element.check_ancestors_disabled_state_for_form_control();
            if element.enabled_state() {
                ScriptThread::enqueue_callback_reaction(
                    element,
                    CallbackReaction::FormDisabled(false),
                    None,
                );
            }
        }
    }

    /// @brief Parses a plain attribute value based on its name.
    /// Functional Utility: Overrides the default attribute parsing for `itemprop` and `itemtype`
    /// attributes, treating their values as token lists.
    ///
    /// @param name The `LocalName` of the attribute.
    /// @param value The `DOMString` value of the attribute.
    /// @return An `AttrValue` representing the parsed attribute value.
    fn parse_plain_attribute(&self, name: &LocalName, value: DOMString) -> AttrValue {
        match *name {
            local_name!("itemprop") => AttrValue::from_serialized_tokenlist(value.into()), // Parse as token list.
            local_name!("itemtype") => AttrValue::from_serialized_tokenlist(value.into()), // Parse as token list.
            _ => self
                .super_type()
                .unwrap()
                .parse_plain_attribute(name, value), // Delegate to super type's method.
        }
    }
}

impl Activatable for HTMLElement {
    /// @brief Returns a reference to the underlying `Element`.
    /// Functional Utility: Provides access to the generic `Element` methods and properties.
    /// @return A reference to the `Element`.
    fn as_element(&self) -> &Element {
        self.upcast::<Element>()
    }

    /// @brief Checks if this specific `HTMLElement` instance is activatable.
    /// Functional Utility: Determines if the element has activation behavior
    /// (e.g., `<summary>` elements).
    /// @return `true` if the element is activatable, `false` otherwise.
    fn is_instance_activatable(&self) -> bool {
        self.as_element().local_name() == &local_name!("summary") // Only `<summary>` is activatable by default.
    }

    // Basically used to make the HTMLSummaryElement activatable (which has no IDL definition)
    /// @brief Implements the activation behavior for the element.
    /// Functional Utility: Defines what happens when an element is activated,
    /// primarily for `<summary>` elements to toggle their parent `<details>`.
    ///
    /// @param _event The `Event` that triggered activation (unused here).
    /// @param _target The `EventTarget` (unused here).
    /// @param _can_gc A `CanGc` token (unused here).
    fn activation_behavior(&self, _event: &Event, _target: &EventTarget, _can_gc: CanGc) {
        self.summary_activation_behavior(); // Delegate to summary's activation behavior.
    }
}

// Form-associated custom elements are the same interface type as
// normal HTMLElements, so HTMLElement needs to have the FormControl trait
// even though it's usually more specific trait implementations, like the
// HTMLInputElement one, that we really want. (Alternately we could put
// the FormControl trait on ElementInternals, but that raises lifetime issues.)
impl FormControl for HTMLElement {
    /// @brief Returns the form owner of the element.
    /// Functional Utility: Implements the `form` getter for form-associated custom elements,
    /// providing a reference to the `<form>` element it belongs to.
    /// @pre `self` must be a form-associated custom element.
    /// @return An `Option<DomRoot<HTMLFormElement>>` containing the form owner, or `None`.
    fn form_owner(&self) -> Option<DomRoot<HTMLFormElement>> {
        debug_assert!(self.is_form_associated_custom_element()); // Pre-condition check.
        self.as_element()
            .get_element_internals()
            .and_then(|e| e.form_owner()) // Delegate to `ElementInternals` form owner.
    }

    /// @brief Sets the form owner of the element.
    /// Functional Utility: Implements the `form` setter for form-associated custom elements.
    /// @pre `self` must be a form-associated custom element.
    /// @param form An `Option<&HTMLFormElement>` for the new form owner.
    fn set_form_owner(&self, form: Option<&HTMLFormElement>) {
        debug_assert!(self.is_form_associated_custom_element()); // Pre-condition check.
        self.as_element()
            .ensure_element_internals(CanGc::note()) // Ensure `ElementInternals` exist.
            .set_form_owner(form);
    }

    /// @brief Returns a reference to the underlying `Element`.
    /// Functional Utility: Provides access to the generic `Element` methods and properties
    /// for form control operations.
    /// @pre `self` must be a form-associated custom element.
    /// @return A reference to the `Element`.
    fn to_element(&self) -> &Element {
        debug_assert!(self.is_form_associated_custom_element()); // Pre-condition check.
        self.as_element() // Return the underlying Element.
    }

    /// @brief Checks if the element is "listed" in its form's elements collection.
    /// Functional Utility: Implements the `isListed` method for form-associated custom elements.
    /// @pre `self` must be a form-associated custom element.
    /// @return `true` as custom form-associated elements are always listed.
    fn is_listed(&self) -> bool {
        debug_assert!(self.is_form_associated_custom_element()); // Pre-condition check.
        true // Form-associated custom elements are always listed.
    }

    // TODO candidate_for_validation, satisfies_constraints traits
}