
/**
 * @file devtools.rs
 * @brief Implementation of the DevTools protocol for the script thread.
 *
 * This module provides the handlers for various DevTools protocol messages that are
 * processed on the script thread. It enables functionality like inspecting the DOM,
 * evaluating JavaScript in the context of a page, and modifying styles and attributes.
 *
 * The functions in this module are typically called in response to messages received
 * from the DevTools frontend. They interact with the DOM and other parts of the script
 * thread's state to fulfill these requests and send back the results.
 */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::HashMap;
use std::rc::Rc;
use std::str;

use base::id::PipelineId;
use devtools_traits::{
    AttrModification, AutoMargins, ComputedNodeLayout, CssDatabaseProperty, EvaluateJSReply,
    NodeInfo, NodeStyle, RuleModification, TimelineMarker, TimelineMarkerType,
};
use ipc_channel::ipc::IpcSender;
use js::jsval::UndefinedValue;
use js::rust::ToString;
use servo_config::pref;
use uuid::Uuid;

use crate::document_collection::DocumentCollection;
use crate::dom::bindings::codegen::Bindings::CSSRuleListBinding::CSSRuleListMethods;
use crate::dom::bindings::codegen::Bindings::CSSStyleDeclarationBinding::CSSStyleDeclarationMethods;
use crate::dom::bindings::codegen::Bindings::CSSStyleRuleBinding::CSSStyleRuleMethods;
use crate::dom::bindings::codegen::Bindings::CSSStyleSheetBinding::CSSStyleSheetMethods;
use crate::dom::bindings::codegen::Bindings::DOMRectBinding::DOMRectMethods;
use crate::dom::bindings::codegen::Bindings::DocumentBinding::DocumentMethods;
use crate::dom::bindings::codegen::Bindings::ElementBinding::ElementMethods;
use crate::dom::bindings::codegen::Bindings::HTMLElementBinding::HTMLElementMethods;
use crate::dom::bindings::codegen::Bindings::NodeBinding::NodeConstants;
use crate::dom::bindings::codegen::Bindings::WindowBinding::WindowMethods;
use crate::dom::bindings::conversions::{ConversionResult, FromJSValConvertible, jsstring_to_str};
use crate::dom::bindings::inheritance::Castable;
use crate::dom::bindings::root::DomRoot;
use crate::dom::bindings::str::DOMString;
use crate::dom::cssstyledeclaration::ENABLED_LONGHAND_PROPERTIES;
use crate::dom::cssstylerule::CSSStyleRule;
use crate::dom::document::AnimationFrameCallback;
use crate::dom::element::Element;
use crate::dom::globalscope::GlobalScope;
use crate::dom::htmlscriptelement::SourceCode;
use crate::dom::node::{Node, NodeTraits, ShadowIncluding};
use crate::dom::types::HTMLElement;
use crate::realms::enter_realm;
use crate::script_module::ScriptFetchOptions;
use crate::script_runtime::CanGc;

/**
 * @brief Handles a request to evaluate a JavaScript expression.
 * @param global The global scope in which to evaluate the expression.
 * @param eval The JavaScript code to evaluate.
 * @param reply A sender to send the evaluation result back to the DevTools frontend.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
#[allow(unsafe_code)]
pub(crate) fn handle_evaluate_js(
    global: &GlobalScope,
    eval: String,
    reply: IpcSender<EvaluateJSReply>,
    can_gc: CanGc,
) {
    // global.get_cx() returns a valid `JSContext` pointer, so this is safe.
    let result = unsafe {
        let cx = GlobalScope::get_cx();
        let _ac = enter_realm(global);
        rooted!(in(*cx) let mut rval = UndefinedValue());
        let source_code = SourceCode::Text(Rc::new(DOMString::from_string(eval)));
        global.evaluate_script_on_global_with_result(
            &source_code,
            "<eval>",
            rval.handle_mut(),
            1,
            ScriptFetchOptions::default_classic_script(global),
            global.api_base_url(),
            can_gc,
        );

        if rval.is_undefined() {
            EvaluateJSReply::VoidValue
        } else if rval.is_boolean() {
            EvaluateJSReply::BooleanValue(rval.to_boolean())
        } else if rval.is_double() || rval.is_int32() {
            EvaluateJSReply::NumberValue(
                match FromJSValConvertible::from_jsval(*cx, rval.handle(), ()) {
                    Ok(ConversionResult::Success(v)) => v,
                    _ => unreachable!(),
                },
            )
        } else if rval.is_string() {
            let jsstr = std::ptr::NonNull::new(rval.to_string()).unwrap();
            EvaluateJSReply::StringValue(String::from(jsstring_to_str(*cx, jsstr)))
        } else if rval.is_null() {
            EvaluateJSReply::NullValue
        } else {
            assert!(rval.is_object());

            let jsstr = std::ptr::NonNull::new(ToString(*cx, rval.handle())).unwrap();
            let class_name = jsstring_to_str(*cx, jsstr);

            EvaluateJSReply::ActorValue {
                class: class_name.to_string(),
                uuid: Uuid::new_v4().to_string(),
            }
        }
    };
    reply.send(result).unwrap();
}

/**
 * @brief Handles a request to get the root node of a document.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline whose root node is being requested.
 * @param reply A sender to send the node information back.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
pub(crate) fn handle_get_root_node(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    reply: IpcSender<Option<NodeInfo>>,
    can_gc: CanGc,
) {
    let info = documents
        .find_document(pipeline)
        .map(|document| document.upcast::<Node>().summarize(can_gc));
    reply.send(info).unwrap();
}

/**
 * @brief Handles a request to get the document element of a document.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline whose document element is being requested.
 * @param reply A sender to send the node information back.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
pub(crate) fn handle_get_document_element(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    reply: IpcSender<Option<NodeInfo>>,
    can_gc: CanGc,
) {
    let info = documents
        .find_document(pipeline)
        .and_then(|document| document.GetDocumentElement())
        .map(|element| element.upcast::<Node>().summarize(can_gc));
    reply.send(info).unwrap();
}

/**
 * @brief Finds a node in the DOM tree by its unique ID.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline to search in.
 * @param node_id The unique ID of the node to find.
 * @return An `Option` containing the found node, or `None`.
 */
fn find_node_by_unique_id(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: &str,
) -> Option<DomRoot<Node>> {
    documents.find_document(pipeline).and_then(|document| {
        document
            .upcast::<Node>()
            .traverse_preorder(ShadowIncluding::Yes)
            .find(|candidate| candidate.unique_id() == node_id)
    })
}

/**
 * @brief Handles a request to get the children of a node.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline containing the node.
 * @param node_id The unique ID of the parent node.
 * @param reply A sender to send the children's node information back.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
pub(crate) fn handle_get_children(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    reply: IpcSender<Option<Vec<NodeInfo>>>,
    can_gc: CanGc,
) {
    match find_node_by_unique_id(documents, pipeline, &node_id) {
        None => reply.send(None).unwrap(),
        Some(parent) => {
            let is_whitespace = |node: &NodeInfo| {
                node.node_type == NodeConstants::TEXT_NODE &&
                    node.node_value.as_ref().is_none_or(|v| v.trim().is_empty())
            };

            let inline: Vec<_> = parent
                .children()
                .map(|child| {
                    let window = child.owner_window();
                    let Some(elem) = child.downcast::<Element>() else {
                        return false;
                    };
                    let computed_style = window.GetComputedStyle(elem, None);
                    let display = computed_style.Display();
                    display == "inline"
                })
                .collect();

            let mut children = vec![];
            if let Some(shadow_root) = parent.downcast::<Element>().and_then(Element::shadow_root) {
                if !shadow_root.is_user_agent_widget() ||
                    pref!(inspector_show_servo_internal_shadow_roots)
                {
                    children.push(shadow_root.upcast::<Node>().summarize(can_gc));
                }
            }
            let children_iter = parent.children().enumerate().filter_map(|(i, child)| {
                // Filter whitespace only text nodes that are not inline level
                // https://firefox-source-docs.mozilla.org/devtools-user/page_inspector/how_to/examine_and_edit_html/index.html#whitespace-only-text-nodes
                let prev_inline = i > 0 && inline[i - 1];
                let next_inline = i < inline.len() - 1 && inline[i + 1];

                let info = child.summarize(can_gc);
                if !is_whitespace(&info) {
                    return Some(info);
                }

                (prev_inline && next_inline).then_some(info)
            });
            children.extend(children_iter);

            reply.send(Some(children)).unwrap();
        },
    };
}

/**
 * @brief Handles a request to get the inline style of an element.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline containing the element.
 * @param node_id The unique ID of the element.
 * @param reply A sender to send the style information back.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
pub(crate) fn handle_get_attribute_style(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    reply: IpcSender<Option<Vec<NodeStyle>>>,
    can_gc: CanGc,
) {
    let node = match find_node_by_unique_id(documents, pipeline, &node_id) {
        None => return reply.send(None).unwrap(),
        Some(found_node) => found_node,
    };

    let Some(elem) = node.downcast::<HTMLElement>() else {
        // the style attribute only works on html elements
        reply.send(None).unwrap();
        return;
    };
    let style = elem.Style(can_gc);

    let msg = (0..style.Length())
        .map(|i| {
            let name = style.Item(i);
            NodeStyle {
                name: name.to_string(),
                value: style.GetPropertyValue(name.clone(), can_gc).to_string(),
                priority: style.GetPropertyPriority(name).to_string(),
            }
        })
        .collect();

    reply.send(Some(msg)).unwrap();
}

/**
 * @brief Handles a request to get the style of a rule in a stylesheet.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline containing the node.
 * @param node_id The unique ID of a node to determine the stylesheet owner.
 * @param selector The selector of the rule to get the style from.
 * @param stylesheet The index of the stylesheet to inspect.
 * @param reply A sender to send the style information back.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
#[cfg_attr(crown, allow(crown::unrooted_must_root))]
pub(crate) fn handle_get_stylesheet_style(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    selector: String,
    stylesheet: usize,
    reply: IpcSender<Option<Vec<NodeStyle>>>,
    can_gc: CanGc,
) {
    let msg = (|| {
        let node = find_node_by_unique_id(documents, pipeline, &node_id)?;

        let document = documents.find_document(pipeline)?;
        let _realm = enter_realm(document.window());
        let owner = node.stylesheet_list_owner();

        let stylesheet = owner.stylesheet_at(stylesheet)?;
        let list = stylesheet.GetCssRules(can_gc).ok()?;

        let styles = (0..list.Length())
            .filter_map(move |i| {
                let rule = list.Item(i, can_gc)?;
                let style = rule.downcast::<CSSStyleRule>()?;
                if *selector != *style.SelectorText() {
                    return None;
                };
                Some(style.Style(can_gc))
            })
            .flat_map(|style| {
                (0..style.Length()).map(move |i| {
                    let name = style.Item(i);
                    NodeStyle {
                        name: name.to_string(),
                        value: style.GetPropertyValue(name.clone(), can_gc).to_string(),
                        priority: style.GetPropertyPriority(name).to_string(),
                    }
                })
            })
            .collect();

        Some(styles)
    })();

    reply.send(msg).unwrap();
}

/**
 * @brief Handles a request to get all selectors that match a given node.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline containing the node.
 * @param node_id The unique ID of the node.
 * @param reply A sender to send the list of selectors back.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
#[cfg_attr(crown, allow(crown::unrooted_must_root))]
pub(crate) fn handle_get_selectors(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    reply: IpcSender<Option<Vec<(String, usize)>>>,
    can_gc: CanGc,
) {
    let msg = (|| {
        let node = find_node_by_unique_id(documents, pipeline, &node_id)?;

        let document = documents.find_document(pipeline)?;
        let _realm = enter_realm(document.window());
        let owner = node.stylesheet_list_owner();

        let rules = (0..owner.stylesheet_count())
            .filter_map(|i| {
                let stylesheet = owner.stylesheet_at(i)?;
                let list = stylesheet.GetCssRules(can_gc).ok()?;
                let elem = node.downcast::<Element>()?;

                Some((0..list.Length()).filter_map(move |j| {
                    let rule = list.Item(j, can_gc)?;
                    let style = rule.downcast::<CSSStyleRule>()?;
                    let selector = style.SelectorText();
                    elem.Matches(selector.clone()).ok()?.then_some(())?;
                    Some((selector.into(), i))
                }))
            })
            .flatten()
            .collect();

        Some(rules)
    })();

    reply.send(msg).unwrap();
}

/**
 * @brief Handles a request to get the computed style of an element.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline containing the element.
 * @param node_id The unique ID of the element.
 * @param reply A sender to send the computed style back.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
pub(crate) fn handle_get_computed_style(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    reply: IpcSender<Option<Vec<NodeStyle>>>,
    can_gc: CanGc,
) {
    let node = match find_node_by_unique_id(documents, pipeline, &node_id) {
        None => return reply.send(None).unwrap(),
        Some(found_node) => found_node,
    };

    let window = node.owner_window();
    let elem = node
        .downcast::<Element>()
        .expect("This should be an element");
    let computed_style = window.GetComputedStyle(elem, None);

    let msg = (0..computed_style.Length())
        .map(|i| {
            let name = computed_style.Item(i);
            NodeStyle {
                name: name.to_string(),
                value: computed_style
                    .GetPropertyValue(name.clone(), can_gc)
                    .to_string(),
                priority: computed_style.GetPropertyPriority(name).to_string(),
            }
        })
        .collect();

    reply.send(Some(msg)).unwrap();
}

/**
 * @brief Handles a request to get the layout of a node.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline containing the node.
 * @param node_id The unique ID of the node.
 * @param reply A sender to send the layout information back.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
pub(crate) fn handle_get_layout(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    reply: IpcSender<Option<ComputedNodeLayout>>,
    can_gc: CanGc,
) {
    let node = match find_node_by_unique_id(documents, pipeline, &node_id) {
        None => return reply.send(None).unwrap(),
        Some(found_node) => found_node,
    };

    let elem = node
        .downcast::<Element>()
        .expect("should be getting layout of element");
    let rect = elem.GetBoundingClientRect(can_gc);
    let width = rect.Width() as f32;
    let height = rect.Height() as f32;

    let window = node.owner_window();
    let elem = node
        .downcast::<Element>()
        .expect("should be getting layout of element");
    let computed_style = window.GetComputedStyle(elem, None);

    reply
        .send(Some(ComputedNodeLayout {
            display: String::from(computed_style.Display()),
            position: String::from(computed_style.Position()),
            z_index: String::from(computed_style.ZIndex()),
            box_sizing: String::from(computed_style.BoxSizing()),
            auto_margins: determine_auto_margins(&node, can_gc),
            margin_top: String::from(computed_style.MarginTop()),
            margin_right: String::from(computed_style.MarginRight()),
            margin_bottom: String::from(computed_style.MarginBottom()),
            margin_left: String::from(computed_style.MarginLeft()),
            border_top_width: String::from(computed_style.BorderTopWidth()),
            border_right_width: String::from(computed_style.BorderRightWidth()),
            border_bottom_width: String::from(computed_style.BorderBottomWidth()),
            border_left_width: String::from(computed_style.BorderLeftWidth()),
            padding_top: String::from(computed_style.PaddingTop()),
            padding_right: String::from(computed_style.PaddingRight()),
            padding_bottom: String::from(computed_style.PaddingBottom()),
            padding_left: String::from(computed_style.PaddingLeft()),
            width,
            height,
        }))
        .unwrap();
}

/**
 * @brief Determines which margins of a node are set to 'auto'.
 */
fn determine_auto_margins(node: &Node, can_gc: CanGc) -> AutoMargins {
    let style = node.style(can_gc).unwrap();
    let margin = style.get_margin();
    AutoMargins {
        top: margin.margin_top.is_auto(),
        right: margin.margin_right.is_auto(),
        bottom: margin.margin_bottom.is_auto(),
        left: margin.margin_left.is_auto(),
    }
}

/**
 * @brief Handles a request to modify attributes of an element.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline containing the element.
 * @param node_id The unique ID of the element.
 * @param modifications A vector of attribute modifications to apply.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
pub(crate) fn handle_modify_attribute(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    modifications: Vec<AttrModification>,
    can_gc: CanGc,
) {
    let Some(document) = documents.find_document(pipeline) else {
        return warn!("document for pipeline id {} is not found", &pipeline);
    };
    let _realm = enter_realm(document.window());

    let node = match find_node_by_unique_id(documents, pipeline, &node_id) {
        None => {
            return warn!(
                "node id {} for pipeline id {} is not found",
                &node_id, &pipeline
            );
        },
        Some(found_node) => found_node,
    };

    let elem = node
        .downcast::<Element>()
        .expect("should be getting layout of element");

    for modification in modifications {
        match modification.new_value {
            Some(string) => {
                let _ = elem.SetAttribute(
                    DOMString::from(modification.attribute_name),
                    DOMString::from(string),
                    can_gc,
                );
            },
            None => elem.RemoveAttribute(DOMString::from(modification.attribute_name), can_gc),
        }
    }
}

/**
 * @brief Handles a request to modify a CSS rule.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline containing the node.
 * @param node_id The unique ID of an element whose style is being modified.
 * @param modifications A vector of rule modifications to apply.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
pub(crate) fn handle_modify_rule(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    modifications: Vec<RuleModification>,
    can_gc: CanGc,
) {
    let Some(document) = documents.find_document(pipeline) else {
        return warn!("Document for pipeline id {} is not found", &pipeline);
    };
    let _realm = enter_realm(document.window());

    let Some(node) = find_node_by_unique_id(documents, pipeline, &node_id) else {
        return warn!(
            "Node id {} for pipeline id {} is not found",
            &node_id, &pipeline
        );
    };

    let elem = node
        .downcast::<HTMLElement>()
        .expect("This should be an HTMLElement");
    let style = elem.Style(can_gc);

    for modification in modifications {
        let _ = style.SetProperty(
            modification.name.into(),
            modification.value.into(),
            modification.priority.into(),
            can_gc,
        );
    }
}

/**
 * @brief Enables or disables live notifications from the DevTools.
 * @param global The global scope.
 * @param send_notifications True to enable notifications, false to disable.
 */
pub(crate) fn handle_wants_live_notifications(global: &GlobalScope, send_notifications: bool) {
    global.set_devtools_wants_updates(send_notifications);
}

/**
 * @brief Sets the timeline markers to be recorded.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline.
 * @param marker_types The types of timeline markers to record.
 * @param reply A sender to send the timeline markers back.
 */
pub(crate) fn handle_set_timeline_markers(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    marker_types: Vec<TimelineMarkerType>,
    reply: IpcSender<Option<TimelineMarker>>,
) {
    match documents.find_window(pipeline) {
        None => reply.send(None).unwrap(),
        Some(window) => window.set_devtools_timeline_markers(marker_types, reply),
    }
}

/**
 * @brief Drops the specified timeline markers.
 * @param documents A collection of all documents.
 * @param pipeline The ID of the pipeline.
 * @param marker_types The types of timeline markers to drop.
 */
pub(crate) fn handle_drop_timeline_markers(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    marker_types: Vec<TimelineMarkerType>,
) {
    if let Some(window) = documents.find_window(pipeline) {
        window.drop_devtools_timeline_markers(marker_types);
    }
}

/**
 * @brief Handles a request for an animation frame.
 * @param documents A collection of all documents.
 * @param id The ID of the pipeline.
 * @param actor_name The name of the actor requesting the animation frame.
 */
pub(crate) fn handle_request_animation_frame(
    documents: &DocumentCollection,
    id: PipelineId,
    actor_name: String,
) {
    if let Some(doc) = documents.find_document(id) {
        doc.request_animation_frame(AnimationFrameCallback::DevtoolsFramerateTick { actor_name });
    }
}

/**
 * @brief Handles a request to reload a document.
 * @param documents A collection of all documents.
 * @param id The ID of the pipeline to reload.
 * @param can_gc A token indicating that garbage collection can be performed.
 */
pub(crate) fn handle_reload(documents: &DocumentCollection, id: PipelineId, can_gc: CanGc) {
    if let Some(win) = documents.find_window(id) {
        win.Location().reload_without_origin_check(can_gc);
    }
}

/**
 * @brief Handles a request to get the CSS property database.
 * @param reply A sender to send the database back.
 */
pub(crate) fn handle_get_css_database(reply: IpcSender<HashMap<String, CssDatabaseProperty>>) {
    let database: HashMap<_, _> = ENABLED_LONGHAND_PROPERTIES
        .iter()
        .map(|l| {
            (
                l.name().into(),
                CssDatabaseProperty {
                    is_inherited: l.inherited(),
                    values: vec![], // TODO: Get allowed values for each property
                    supports: vec![],
                    subproperties: vec![l.name().into()],
                },
            )
        })
        .collect();
    let _ = reply.send(database);
}
