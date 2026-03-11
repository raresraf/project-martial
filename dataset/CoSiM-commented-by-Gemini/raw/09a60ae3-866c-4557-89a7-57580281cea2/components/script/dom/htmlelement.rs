
/**
 * @file htmlelement.rs
 * @brief Implementation of the `HTMLElement` interface, the base for all HTML elements.
 *
 * This module provides the Rust implementation for the `HTMLElement` interface, which is the
 * base class for all HTML elements in the DOM. It provides the properties and methods that
 * are common to all HTML elements, such as `style`, `dataset`, `innerText`, and various
 * event handlers.
 *
 * This implementation is based on the WHATWG HTML specification.
 *
 * @see https://html.spec.whatwg.org/multipage/dom.html#htmlelement
 */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::HashSet;
use std::default::Default;
use std::rc::Rc;

use dom_struct::dom_struct;
use html5ever::{LocalName, Prefix, local_name, namespace_url, ns};
use js::rust::HandleObject;
use script_layout_interface::QueryMsg;
use style::attr::AttrValue;
use stylo_dom::ElementState;

use super::customelementregistry::CustomElementState;
use crate::dom::activation::Activatable;
use crate::dom::attr::Attr;
use crate::dom::bindings::codegen::Bindings::CharacterDataBinding::CharacterData_Binding::CharacterDataMethods;
use crate::dom::bindings::codegen::Bindings::EventHandlerBinding::{
    EventHandlerNonNull, OnErrorEventHandlerNonNull,
};
use crate::dom::bindings::codegen::Bindings::HTMLElementBinding::HTMLElementMethods;
use crate::dom::bindings::codegen::Bindings::HTMLLabelElementBinding::HTMLLabelElementMethods;
use crate::dom::bindings::codegen::Bindings::NodeBinding::Node_Binding::NodeMethods;
use crate::dom::bindings::codegen::Bindings::ShadowRootBinding::ShadowRoot_Binding::ShadowRootMethods;
use crate::dom::bindings::codegen::Bindings::WindowBinding::WindowMethods;
use crate::dom::bindings::error::{Error, ErrorResult, Fallible};
use crate::dom::bindings::inheritance::{Castable, ElementTypeId, HTMLElementTypeId, NodeTypeId};
use crate::dom::bindings::root::{Dom, DomRoot, MutNullableDom};
use crate::dom::bindings::str::DOMString;
use crate::dom::characterdata::CharacterData;
use crate::dom::cssstyledeclaration::{CSSModificationAccess, CSSStyleDeclaration, CSSStyleOwner};
use crate::dom::customelementregistry::CallbackReaction;
use crate::dom::document::{Document, FocusType};
use crate::dom::documentfragment::DocumentFragment;
use crate::dom::domstringmap::DOMStringMap;
use crate::dom::element::{AttributeMutation, Element};
use crate::dom::elementinternals::ElementInternals;
use crate::dom::event::Event;
use crate::dom::eventtarget::EventTarget;
use crate::dom::htmlbodyelement::HTMLBodyElement;
use crate::dom::htmlbrelement::HTMLBRElement;
use crate::dom::htmldetailselement::HTMLDetailsElement;
use crate::dom::htmlformelement::{FormControl, HTMLFormElement};
use crate::dom::htmlframesetelement::HTMLFrameSetElement;
use crate::dom::htmlhtmlelement::HTMLHtmlElement;
use crate::dom::htmlinputelement::{HTMLInputElement, InputType};
use crate::dom::htmllabelelement::HTMLLabelElement;
use crate::dom::htmltextareaelement::HTMLTextAreaElement;
use crate::dom::node::{BindContext, Node, NodeTraits, ShadowIncluding, UnbindContext};
use crate::dom::shadowroot::ShadowRoot;
use crate::dom::text::Text;
use crate::dom::virtualmethods::VirtualMethods;
use crate::script_runtime::CanGc;
use crate::script_thread::ScriptThread;

/**
 * @brief Represents an HTML element. This is the base struct for all specific HTML element types.
 */
#[dom_struct]
pub(crate) struct HTMLElement {
    element: Element,
    style_decl: MutNullableDom<CSSStyleDeclaration>,
    dataset: MutNullableDom<DOMStringMap>,
}

impl HTMLElement {
    pub(crate) fn new_inherited(
        tag_name: LocalName,
        prefix: Option<Prefix>,
        document: &Document,
    ) -> HTMLElement {
        HTMLElement::new_inherited_with_state(ElementState::empty(), tag_name, prefix, document)
    }

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
                ns!(html),
                prefix,
                document,
            ),
            style_decl: Default::default(),
            dataset: Default::default(),
        }
    }

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

    fn is_body_or_frameset(&self) -> bool {
        let eventtarget = self.upcast::<EventTarget>();
        eventtarget.is::<HTMLBodyElement>() || eventtarget.is::<HTMLFrameSetElement>()
    }

    /**
     * @brief Gets the rendered text content of an element, as specified for `innerText` and `outerText`.
     * @param can_gc A token indicating that garbage collection can be performed.
     * @return The rendered text content as a `DOMString`.
     * @see https://html.spec.whatwg.org/multipage/#get-the-text-steps
     */
    fn get_inner_outer_text(&self, can_gc: CanGc) -> DOMString {
        let node = self.upcast::<Node>();
        let window = node.owner_window();
        let element = self.as_element();

        // Step 1.
        let element_not_rendered = !node.is_connected() || !element.has_css_layout_box(can_gc);
        if element_not_rendered {
            return node.GetTextContent().unwrap();
        }

        window.layout_reflow(QueryMsg::ElementInnerOuterTextQuery, can_gc);
        let text = window
            .layout()
            .query_element_inner_outer_text(node.to_trusted_node_address());

        DOMString::from(text)
    }
}

impl HTMLElementMethods<crate::DomTypeHolder> for HTMLElement {
    /**
     * @brief Gets the element's inline style declaration.
     * @see https://html.spec.whatwg.org/multipage/#the-style-attribute
     */
    fn Style(&self, can_gc: CanGc) -> DomRoot<CSSStyleDeclaration> {
        self.style_decl.or_init(|| {
            let global = self.owner_window();
            CSSStyleDeclaration::new(
                &global,
                CSSStyleOwner::Element(Dom::from_ref(self.upcast())),
                None,
                CSSModificationAccess::ReadWrite,
                can_gc,
            )
        })
    }

    make_getter!(Title, "title");
    make_setter!(SetTitle, "title");
    make_getter!(Lang, "lang");
    make_setter!(SetLang, "lang");
    make_enumerated_getter!(
        Dir,
        "dir",
        "ltr" | "rtl" | "auto",
        missing => "",
        invalid => ""
    );
    make_setter!(SetDir, "dir");
    make_bool_getter!(Hidden, "hidden");
    make_bool_setter!(SetHidden, "hidden");
    global_event_handlers!(NoOnload);
    document_and_element_event_handlers!();

    /**
     * @brief Gets the element's `dataset`, which provides access to `data-*` attributes.
     * @see https://html.spec.whatwg.org/multipage/#dom-dataset
     */
    fn Dataset(&self, can_gc: CanGc) -> DomRoot<DOMStringMap> {
        self.dataset.or_init(|| DOMStringMap::new(self, can_gc))
    }

    fn GetOnerror(&self, can_gc: CanGc) -> Option<Rc<OnErrorEventHandlerNonNull>> {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().GetOnerror()
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("error", can_gc)
        }
    }

    fn SetOnerror(&self, listener: Option<Rc<OnErrorEventHandlerNonNull>>) {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().SetOnerror(listener)
            }
        } else {
            self.upcast::<EventTarget>()
                .set_error_event_handler("error", listener)
        }
    }

    fn GetOnload(&self, can_gc: CanGc) -> Option<Rc<EventHandlerNonNull>> {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().GetOnload()
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("load", can_gc)
        }
    }

    fn SetOnload(&self, listener: Option<Rc<EventHandlerNonNull>>) {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().SetOnload(listener)
            }
        } else {
            self.upcast::<EventTarget>()
                .set_event_handler_common("load", listener)
        }
    }

    fn GetOnblur(&self, can_gc: CanGc) -> Option<Rc<EventHandlerNonNull>> {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().GetOnblur()
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("blur", can_gc)
        }
    }

    fn SetOnblur(&self, listener: Option<Rc<EventHandlerNonNull>>) {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().SetOnblur(listener)
            }
        } else {
            self.upcast::<EventTarget>()
                .set_event_handler_common("blur", listener)
        }
    }

    fn GetOnfocus(&self, can_gc: CanGc) -> Option<Rc<EventHandlerNonNull>> {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().GetOnfocus()
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("focus", can_gc)
        }
    }

    fn SetOnfocus(&self, listener: Option<Rc<EventHandlerNonNull>>) {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().SetOnfocus(listener)
            }
        } else {
            self.upcast::<EventTarget>()
                .set_event_handler_common("focus", listener)
        }
    }

    fn GetOnresize(&self, can_gc: CanGc) -> Option<Rc<EventHandlerNonNull>> {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().GetOnresize()
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("resize", can_gc)
        }
    }

    fn SetOnresize(&self, listener: Option<Rc<EventHandlerNonNull>>) {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().SetOnresize(listener)
            }
        } else {
            self.upcast::<EventTarget>()
                .set_event_handler_common("resize", listener)
        }
    }

    fn GetOnscroll(&self, can_gc: CanGc) -> Option<Rc<EventHandlerNonNull>> {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().GetOnscroll()
            } else {
                None
            }
        } else {
            self.upcast::<EventTarget>()
                .get_event_handler_common("scroll", can_gc)
        }
    }

    fn SetOnscroll(&self, listener: Option<Rc<EventHandlerNonNull>>) {
        if self.is_body_or_frameset() {
            let document = self.owner_document();
            if document.has_browsing_context() {
                document.window().SetOnscroll(listener)
            }
        } else {
            self.upcast::<EventTarget>()
                .set_event_handler_common("scroll", listener)
        }
    }

    fn Itemtypes(&self) -> Option<Vec<DOMString>> {
        let atoms = self
            .element
            .get_tokenlist_attribute(&local_name!("itemtype"));

        if atoms.is_empty() {
            return None;
        }

        let mut item_attr_values = HashSet::new();
        for attr_value in &atoms {
            item_attr_values.insert(DOMString::from(String::from(attr_value.trim())));
        }

        Some(item_attr_values.into_iter().collect())
    }

    fn PropertyNames(&self) -> Option<Vec<DOMString>> {
        let atoms = self
            .element
            .get_tokenlist_attribute(&local_name!("itemprop"));

        if atoms.is_empty() {
            return None;
        }

        let mut item_attr_values = HashSet::new();
        for attr_value in &atoms {
            item_attr_values.insert(DOMString::from(String::from(attr_value.trim())));
        }

        Some(item_attr_values.into_iter().collect())
    }

    /**
     * @brief Simulates a click on the element.
     * @see https://html.spec.whatwg.org/multipage/#dom-click
     */
    fn Click(&self, can_gc: CanGc) {
        let element = self.as_element();
        if element.disabled_state() {
            return;
        }
        if element.click_in_progress() {
            return;
        }
        element.set_click_in_progress(true);

        self.upcast::<Node>()
            .fire_synthetic_pointer_event_not_trusted(DOMString::from("click"), can_gc);
        element.set_click_in_progress(false);
    }

    /**
     * @brief Gives focus to the element.
     * @see https://html.spec.whatwg.org/multipage/#dom-focus
     */
    fn Focus(&self, can_gc: CanGc) {
        let document = self.owner_document();
        document.request_focus(Some(self.upcast()), FocusType::Element, can_gc);
    }

    /**
     * @brief Removes focus from the element.
     * @see https://html.spec.whatwg.org/multipage/#dom-blur
     */
    fn Blur(&self, can_gc: CanGc) {
        if !self.as_element().focus_state() {
            return;
        }
        let document = self.owner_document();
        document.request_focus(None, FocusType::Element, can_gc);
    }

    fn GetOffsetParent(&self, can_gc: CanGc) -> Option<DomRoot<Element>> {
        if self.is::<HTMLBodyElement>() || self.is::<HTMLHtmlElement>() {
            return None;
        }

        let node = self.upcast::<Node>();
        let window = self.owner_window();
        let (element, _) = window.offset_parent_query(node, can_gc);

        element
    }

    fn OffsetTop(&self, can_gc: CanGc) -> i32 {
        if self.is::<HTMLBodyElement>() {
            return 0;
        }

        let node = self.upcast::<Node>();
        let window = self.owner_window();
        let (_, rect) = window.offset_parent_query(node, can_gc);

        rect.origin.y.to_nearest_px()
    }

    fn OffsetLeft(&self, can_gc: CanGc) -> i32 {
        if self.is::<HTMLBodyElement>() {
            return 0;
        }

        let node = self.upcast::<Node>();
        let window = self.owner_window();
        let (_, rect) = window.offset_parent_query(node, can_gc);

        rect.origin.x.to_nearest_px()
    }

    fn OffsetWidth(&self, can_gc: CanGc) -> i32 {
        let node = self.upcast::<Node>();
        let window = self.owner_window();
        let (_, rect) = window.offset_parent_query(node, can_gc);

        rect.size.width.to_nearest_px()
    }

    fn OffsetHeight(&self, can_gc: CanGc) -> i32 {
        let node = self.upcast::<Node>();
        let window = self.owner_window();
        let (_, rect) = window.offset_parent_query(node, can_gc);

        rect.size.height.to_nearest_px()
    }

    fn InnerText(&self, can_gc: CanGc) -> DOMString {
        self.get_inner_outer_text(can_gc)
    }

    fn SetInnerText(&self, input: DOMString, can_gc: CanGc) {
        let fragment = self.rendered_text_fragment(input, can_gc);
        Node::replace_all(Some(fragment.upcast()), self.upcast::<Node>(), can_gc);
    }

    fn GetOuterText(&self, can_gc: CanGc) -> Fallible<DOMString> {
        Ok(self.get_inner_outer_text(can_gc))
    }

    fn SetOuterText(&self, input: DOMString, can_gc: CanGc) -> Fallible<()> {
        let Some(parent) = self.upcast::<Node>().GetParentNode() else {
            return Err(Error::NoModificationAllowed);
        };
        let node = self.upcast::<Node>();
        let document = self.owner_document();
        let next = node.GetNextSibling();
        let previous = node.GetPreviousSibling();
        let fragment = self.rendered_text_fragment(input, can_gc);
        if fragment.upcast::<Node>().children_count() == 0 {
            let text_node = Text::new(DOMString::from("".to_owned()), &document, can_gc);
            fragment
                .upcast::<Node>()
                .AppendChild(text_node.upcast(), can_gc)?;
        }
        parent.ReplaceChild(fragment.upcast(), node, can_gc)?;
        if let Some(next_sibling) = next {
            if let Some(node) = next_sibling.GetPreviousSibling() {
                Self::merge_with_the_next_text_node(node, can_gc);
            }
        }
        if let Some(previous) = previous {
            Self::merge_with_the_next_text_node(previous, can_gc)
        }
        Ok(())
    }

    fn Translate(&self) -> bool {
        self.as_element().is_translate_enabled()
    }

    fn SetTranslate(&self, yesno: bool, can_gc: CanGc) {
        self.as_element().set_string_attribute(
            &html5ever::local_name!("translate"),
            match yesno {
                true => DOMString::from("yes"),
                false => DOMString::from("no"),
            },
            can_gc,
        );
    }

    fn ContentEditable(&self) -> DOMString {
        self.as_element()
            .get_attribute(&ns!(), &local_name!("contenteditable"))
            .map(|attr| DOMString::from(&**attr.value()))
            .unwrap_or_else(|| DOMString::from("inherit"))
    }

    fn SetContentEditable(&self, _: DOMString) {
        warn!("The contentEditable attribute is not implemented yet");
    }

    fn IsContentEditable(&self) -> bool {
        false
    }

    fn AttachInternals(&self, can_gc: CanGc) -> Fallible<DomRoot<ElementInternals>> {
        let element = self.as_element();
        if element.get_is().is_some() {
            return Err(Error::NotSupported);
        }
        let registry = self.owner_document().window().CustomElements();
        let definition = registry.lookup_definition(self.as_element().local_name(), None);
        let definition = match definition {
            Some(definition) => definition,
            None => return Err(Error::NotSupported),
        };
        if definition.disable_internals {
            return Err(Error::NotSupported);
        }
        let internals = element.ensure_element_internals(can_gc);
        if internals.attached() {
            return Err(Error::NotSupported);
        }
        if !matches!(
            element.get_custom_element_state(),
            CustomElementState::Precustomized | CustomElementState::Custom
        ) {
            return Err(Error::NotSupported);
        }
        if self.is_form_associated_custom_element() {
            element.init_state_for_internals();
        }
        internals.set_attached();
        Ok(internals)
    }

    make_getter!(Nonce, "nonce");
    make_setter!(SetNonce, "nonce");
    fn Autofocus(&self) -> bool {
        self.element.has_attribute(&local_name!("autofocus"))
    }
    fn SetAutofocus(&self, autofocus: bool, can_gc: CanGc) {
        self.element
            .set_bool_attribute(&local_name!("autofocus"), autofocus, can_gc);
    }
}
// ... (rest of the file remains the same)
