/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// @file devtools.rs
/// @brief This file provides the backend implementation for various developer tools
/// functionalities within a web browser engine. It includes mechanisms for
/// JavaScript evaluation, DOM inspection (node information, computed styles, layout),
/// CSS rule and attribute modification, and timeline event handling.
///
/// Functional Utility: Bridges the core rendering engine and scripting environment
/// with external developer tools interfaces, allowing for interactive debugging and inspection
/// of web content.

use std::collections::HashMap;
use std::rc::Rc;
use std::str;

use base::id::PipelineId; // Unique identifier for a rendering pipeline.
use devtools_traits::{
    AttrModification, AutoMargins, ComputedNodeLayout, CssDatabaseProperty, EvaluateJSReply,
    NodeInfo, NodeStyle, RuleModification, TimelineMarker, TimelineMarkerType,
}; // Traits defining the interface for devtools data structures and operations.
use ipc_channel::ipc::IpcSender; // IPC channel for sending messages between processes/threads.
use js::jsval::UndefinedValue; // Represents JavaScript 'undefined' value.
use js::rust::ToString; // Trait for converting JavaScript values to strings.
use servo_config::pref; // For accessing Servo's preference system.
use uuid::Uuid; // For generating universally unique identifiers.

use crate::document_collection::DocumentCollection; // Manages multiple documents.
use crate::dom::bindings::codegen::Bindings::CSSRuleListBinding::CSSRuleListMethods; // CSS Rule List bindings.
use crate::dom::bindings::codegen::Bindings::CSSStyleDeclarationBinding::CSSStyleDeclarationMethods; // CSS Style Declaration bindings.
use crate::dom::bindings::codegen::Bindings::CSSStyleRuleBinding::CSSStyleRuleMethods; // CSS Style Rule bindings.
use crate::dom::bindings::codegen::Bindings::CSSStyleSheetBinding::CSSStyleSheetMethods; // CSS Style Sheet bindings.
use crate::dom::bindings::codegen::Bindings::DOMRectBinding::DOMRectMethods; // DOM Rect bindings for layout.
use crate::dom::bindings::codegen::Bindings::DocumentBinding::DocumentMethods; // Document object bindings.
use crate::dom::bindings::codegen::Bindings::ElementBinding::ElementMethods; // Element object bindings.
use crate::dom::bindings::codegen::Bindings::HTMLElementBinding::HTMLElementMethods; // HTML Element specific bindings.
use crate::dom::bindings::codegen::Bindings::NodeBinding::NodeConstants; // Node type constants.
use crate::dom::bindings::codegen::Bindings::WindowBinding::WindowMethods; // Window object bindings.
use crate::dom::bindings::conversions::{ConversionResult, FromJSValConvertible, jsstring_to_str}; // JavaScript value conversion utilities.
use crate::dom::bindings::inheritance::Castable; // Trait for safe downcasting between DOM types.
use crate::dom::bindings::root::DomRoot; // Root type for DOM objects.
use crate::dom::bindings::str::DOMString; // DOMString representation.
use crate::dom::cssstyledeclaration::ENABLED_LONGHAND_PROPERTIES; // List of enabled CSS properties.
use crate::dom::cssstylerule::CSSStyleRule; // CSS Style Rule representation.
use crate::dom::document::AnimationFrameCallback; // Callback types for requestAnimationFrame.
use crate::dom::element::Element; // Element type.
use crate::dom::globalscope::GlobalScope; // Global JavaScript execution scope.
use crate::dom::htmlscriptelement::SourceCode; // Source code representation for script evaluation.
use crate::dom::node::{Node, NodeTraits, ShadowIncluding}; // Node types and traits, including shadow DOM handling.
use crate::dom::types::HTMLElement; // HTML Element specific type.
use crate::realms::enter_realm; // Utility for entering a JavaScript realm.
use crate::script_module::ScriptFetchOptions; // Options for fetching scripts.
use crate::script_runtime::CanGc; // Marker trait for types that can be garbage collected.

/// Handles JavaScript evaluation requests.
///
/// Functional Utility: Executes arbitrary JavaScript code within the context of a `GlobalScope`
/// and returns the result, handling various JavaScript value types for IPC transmission.
///
/// @param global The JavaScript global scope where the evaluation occurs.
/// @param eval The JavaScript code string to evaluate.
/// @param reply An IPC sender to send the evaluation result back to the caller.
/// @param can_gc A `CanGc` token indicating garbage collection is possible.
#[allow(unsafe_code)]
pub(crate) fn handle_evaluate_js(
    global: &GlobalScope,
    eval: String,
    reply: IpcSender<EvaluateJSReply>,
    can_gc: CanGc,
) {
    // global.get_cx() returns a valid `JSContext` pointer, so this is safe.
    let result = unsafe {
        let cx = GlobalScope::get_cx(); // Get the JavaScript context.
        let _ac = enter_realm(global); // Enter the realm of the global scope.
        rooted!(in(*cx) let mut rval = UndefinedValue()); // Create a rooted JSVal to hold the evaluation result.
        let source_code = SourceCode::Text(Rc::new(DOMString::from_string(eval))); // Wrap the evaluation string in a SourceCode object.
        // Block Logic: Evaluate the script and handle the result type for IPC serialization.
        // Invariant: `rval` will contain the result of the JavaScript evaluation.
        global.evaluate_script_on_global_with_result(
            &source_code,
            "<eval>", // Source name for debugging purposes.
            rval.handle_mut(), // Mutable handle to the JSVal for storing the result.
            1, // Line number.
            ScriptFetchOptions::default_classic_script(global), // Default script fetch options.
            global.api_base_url(), // Base URL for the script.
            can_gc, // Pass the garbage collection token.
        );

        // Block Logic: Convert the JavaScript result (`rval`) into an `EvaluateJSReply` enum variant.
        // Each `if-else if` branch handles a specific JavaScript type.
        if rval.is_undefined() {
            EvaluateJSReply::VoidValue // If undefined, return VoidValue.
        } else if rval.is_boolean() {
            EvaluateJSReply::BooleanValue(rval.to_boolean()) // If boolean, return BooleanValue.
        } else if rval.is_double() || rval.is_int32() {
            // If number (double or int32), convert and return NumberValue.
            EvaluateJSReply::NumberValue(
                match FromJSValConvertible::from_jsval(*cx, rval.handle(), ()) {
                    Ok(ConversionResult::Success(v)) => v, // Safely convert to Rust type.
                    _ => unreachable!(), // Should not happen if type check is correct.
                },
            )
        } else if rval.is_string() {
            // If string, convert to Rust String and return StringValue.
            let jsstr = std::ptr::NonNull::new(rval.to_string()).unwrap(); // Get JSString pointer.
            EvaluateJSReply::StringValue(String::from(jsstring_to_str(*cx, jsstr))) // Convert to Rust String.
        } else if rval.is_null() {
            EvaluateJSReply::NullValue // If null, return NullValue.
        } else {
            // For other object types, return ActorValue with class name and a generated UUID.
            assert!(rval.is_object()); // Assert that it must be an object at this point.

            let jsstr = std::ptr::NonNull::new(ToString(*cx, rval.handle())).unwrap(); // Get JSString representation of the object.
            let class_name = jsstring_to_str(*cx, jsstr); // Get the class name.

            EvaluateJSReply::ActorValue {
                class: class_name.to_string(), // Object's class name.
                uuid: Uuid::new_v4().to_string(), // Unique identifier for the object.
            }
        }
    };
    reply.send(result).unwrap(); // Send the result back via IPC.
}

/// Handles requests to get information about the root node of a document.
///
/// Functional Utility: Retrieves the root node (e.g., `<html>` element) of a specified
/// document within a rendering pipeline and summarizes its information for devtools.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param reply An IPC sender to send the `NodeInfo` of the root node.
/// @param can_gc A `CanGc` token.
pub(crate) fn handle_get_root_node(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    reply: IpcSender<Option<NodeInfo>>,
    can_gc: CanGc,
) {
    // Block Logic: Finds the document and then maps its root node to `NodeInfo`.
    let info = documents
        .find_document(pipeline) // Find the document by pipeline ID.
        .map(|document| document.upcast::<Node>().summarize(can_gc)); // Upcast to Node and summarize.
    reply.send(info).unwrap(); // Send the summarized node information.
}

/// Handles requests to get information about the document element (e.g., the `<html>` element).
///
/// Functional Utility: Retrieves the document element of a specified document
/// and summarizes its information for devtools.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param reply An IPC sender to send the `NodeInfo` of the document element.
/// @param can_gc A `CanGc` token.
pub(crate) fn handle_get_document_element(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    reply: IpcSender<Option<NodeInfo>>,
    can_gc: CanGc,
) {
    // Block Logic: Finds the document, gets its document element, and then maps it to `NodeInfo`.
    let info = documents
        .find_document(pipeline) // Find the document by pipeline ID.
        .and_then(|document| document.GetDocumentElement()) // Get the document element (HTMLElement).
        .map(|element| element.upcast::<Node>().summarize(can_gc)); // Upcast to Node and summarize.
    reply.send(info).unwrap(); // Send the summarized node information.
}

/// Finds a DOM Node by its unique identifier within a specific document.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param node_id The unique identifier of the node to find.
/// @return An `Option<DomRoot<Node>>` containing the found node or `None` if not found.
fn find_node_by_unique_id(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: &str,
) -> Option<DomRoot<Node>> {
    // Block Logic: Find the document, then traverse its DOM to locate the node by unique ID.
    documents.find_document(pipeline).and_then(|document| {
        document
            .upcast::<Node>() // Start traversal from the document's root node.
            .traverse_preorder(ShadowIncluding::Yes) // Traverse all nodes, including those in shadow DOMs.
            .find(|candidate| candidate.unique_id() == node_id) // Find the first node whose unique ID matches.
    })
}

/// Handles requests to get the children of a specific DOM node.
///
/// Functional Utility: Retrieves the child nodes of a given parent node, filtering out
/// certain types of whitespace-only text nodes for cleaner presentation in devtools.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param node_id The unique identifier of the parent node whose children are requested.
/// @param reply An IPC sender to send the `Vec<NodeInfo>` of the children.
/// @param can_gc A `CanGc` token.
pub(crate) fn handle_get_children(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    reply: IpcSender<Option<Vec<NodeInfo>>>,
    can_gc: CanGc,
) {
    // Block Logic: Attempt to find the parent node; if not found, send None and return.
    match find_node_by_unique_id(documents, pipeline, &node_id) {
        None => reply.send(None).unwrap(), // If parent node not found, send None.
        Some(parent) => {
            // Closure to check if a node is a whitespace-only text node.
            let is_whitespace = |node: &NodeInfo| {
                node.node_type == NodeConstants::TEXT_NODE &&
                    node.node_value.as_ref().is_none_or(|v| v.trim().is_empty())
            };

            // Block Logic: Determine if each child node is an "inline" element.
            // This is used for whitespace filtering logic later.
            let inline: Vec<_> = parent
                .children() // Iterate over direct children.
                .map(|child| {
                    let window = child.owner_window(); // Get the owner window of the child.
                    let Some(elem) = child.downcast::<Element>() else {
                        return false; // Not an Element, so not inline for this purpose.
                    };
                    let computed_style = window.GetComputedStyle(elem, None); // Get computed style.
                    let display = computed_style.Display(); // Get the 'display' property.
                    display == "inline" // Check if display is 'inline'.
                })
                .collect(); // Collect results into a vector.

            let mut children = vec![]; // Vector to store summarized child node information.

            // Block Logic: Handle shadow root children if present.
            // Invariant: If a shadow root exists and is either not a user-agent widget
            // or `inspector_show_servo_internal_shadow_roots` preference is enabled,
            // it is added to the children list.
            if let Some(shadow_root) = parent.downcast::<Element>().and_then(Element::shadow_root) {
                if !shadow_root.is_user_agent_widget() ||
                    pref!(inspector_show_servo_internal_shadow_roots)
                {
                    children.push(shadow_root.upcast::<Node>().summarize(can_gc)); // Add shadow root as a child.
                }
            }
            // Block Logic: Filter and extend children, excluding certain whitespace text nodes.
            // Invariant: Only significant nodes or whitespace nodes flanked by inline elements are included.
            let children_iter = parent.children().enumerate().filter_map(|(i, child)| {
                // Filter whitespace only text nodes that are not inline level
                // https://firefox-source-docs.mozilla.org/devtools-user/page_inspector/how_to/examine_and_edit_html/index.html#whitespace-only-text-nodes
                let prev_inline = i > 0 && inline[i - 1]; // Check if previous sibling is inline.
                let next_inline = i < inline.len() - 1 && inline[i + 1]; // Check if next sibling is inline.

                let info = child.summarize(can_gc); // Summarize the current child node.
                if !is_whitespace(&info) {
                    return Some(info); // If not whitespace-only, always include.
                }

                (prev_inline && next_inline).then_some(info) // Only include whitespace if flanked by inline elements.
            });
            children.extend(children_iter); // Add the filtered children to the list.

            reply.send(Some(children)).unwrap(); // Send the list of children.
        },
    };
}

/// Handles requests to get the inline style attributes of a specific DOM element.
///
/// Functional Utility: Extracts and summarizes the style properties explicitly defined
/// in an element's `style` attribute (inline styles).
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param node_id The unique identifier of the element.
/// @param reply An IPC sender to send the `Vec<NodeStyle>` representing the inline styles.
/// @param can_gc A `CanGc` token.
pub(crate) fn handle_get_attribute_style(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    reply: IpcSender<Option<Vec<NodeStyle>>>,
    can_gc: CanGc,
) {
    // Block Logic: Find the node by ID; if not found, send None and return.
    let node = match find_node_by_unique_id(documents, pipeline, &node_id) {
        None => return reply.send(None).unwrap(),
        Some(found_node) => found_node,
    };

    // Block Logic: Downcast the node to an HTMLElement; if not an HTMLElement, send None and return.
    let Some(elem) = node.downcast::<HTMLElement>() else {
        // the style attribute only works on html elements
        reply.send(None).unwrap();
        return;
    };
    let style = elem.Style(can_gc); // Get the CSSStyleDeclaration object for the inline style.

    // Block Logic: Iterate through all declared properties in the inline style and summarize them.
    let msg = (0..style.Length())
        .map(|i| {
            let name = style.Item(i); // Get the property name by index.
            NodeStyle {
                name: name.to_string(), // Property name.
                value: style.GetPropertyValue(name.clone(), can_gc).to_string(), // Property value.
                priority: style.GetPropertyPriority(name).to_string(), // Property priority (e.g., "important").
            }
        })
        .collect(); // Collect all `NodeStyle` entries into a vector.

    reply.send(Some(msg)).unwrap(); // Send the collected inline styles.
}

/// Handles requests to get computed styles from a stylesheet rule applied to a specific DOM element.
///
/// Functional Utility: Identifies the CSS rules that match a given selector and are applied
/// to a specific node, then extracts and summarizes their declared style properties.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param node_id The unique identifier of the node.
/// @param selector The CSS selector string to match.
/// @param stylesheet The index of the stylesheet to inspect.
/// @param reply An IPC sender to send the `Vec<NodeStyle>` from the matched rule.
/// @param can_gc A `CanGc` token.
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
    // Block Logic: Use an immediately invoked function expression (IIFE) to handle early returns.
    let msg = (|| {
        let node = find_node_by_unique_id(documents, pipeline, &node_id)?; // Find the node.

        let document = documents.find_document(pipeline)?; // Get the document.
        let _realm = enter_realm(document.window()); // Enter the window's realm.
        let owner = node.stylesheet_list_owner(); // Get the object that owns the stylesheet list (e.g., Document).

        let stylesheet = owner.stylesheet_at(stylesheet)?; // Get the stylesheet at the specified index.
        let list = stylesheet.GetCssRules(can_gc).ok()?; // Get the list of CSS rules from the stylesheet.

        // Block Logic: Filter rules by selector and then extract their style properties.
        let styles = (0..list.Length())
            .filter_map(move |i| {
                let rule = list.Item(i, can_gc)?; // Get the CSS rule by index.
                let style = rule.downcast::<CSSStyleRule>()?; // Downcast to CSSStyleRule.
                if *selector != *style.SelectorText() {
                    return None; // If selector doesn't match, skip this rule.
                };
                Some(style.Style(can_gc)) // Get the CSSStyleDeclaration of the matched rule.
            })
            .flat_map(|style| {
                // For each matched style, iterate through its properties and summarize.
                (0..style.Length()).map(move |i| {
                    let name = style.Item(i); // Property name.
                    NodeStyle {
                        name: name.to_string(), // Property name.
                        value: style.GetPropertyValue(name.clone(), can_gc).to_string(), // Property value.
                        priority: style.GetPropertyPriority(name).to_string(), // Property priority.
                    }
                })
            })
            .collect(); // Collect all `NodeStyle` entries.

        Some(styles) // Return the collected styles.
    })(); // End of IIFE.

    reply.send(msg).unwrap(); // Send the collected styles.
}

/// Handles requests to get all CSS selectors that match a specific DOM node.
///
/// Functional Utility: Traverses all available stylesheets and their rules to find
/// which CSS selectors apply to a given `Node`.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param node_id The unique identifier of the node.
/// @param reply An IPC sender to send a vector of matching selectors and their stylesheet index.
/// @param can_gc A `CanGc` token.
#[cfg_attr(crown, allow(crown::unrooted_must_root))]
pub(crate) fn handle_get_selectors(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    reply: IpcSender<Option<Vec<(String, usize)>>>,
    can_gc: CanGc,
) {
    // Block Logic: Use an IIFE to handle early returns.
    let msg = (|| {
        let node = find_node_by_unique_id(documents, pipeline, &node_id)?; // Find the node.

        let document = documents.find_document(pipeline)?; // Get the document.
        let _realm = enter_realm(document.window()); // Enter the window's realm.
        let owner = node.stylesheet_list_owner(); // Get the object that owns the stylesheet list.

        // Block Logic: Iterate through all stylesheets and their rules to find matching selectors.
        let rules = (0..owner.stylesheet_count())
            .filter_map(|i| {
                let stylesheet = owner.stylesheet_at(i)?; // Get stylesheet by index.
                let list = stylesheet.GetCssRules(can_gc).ok()?; // Get CSS rules.
                let elem = node.downcast::<Element>()?; // Downcast node to Element to use Matches method.

                Some((0..list.Length()).filter_map(move |j| {
                    let rule = list.Item(j, can_gc)?; // Get CSS rule by index.
                    let style = rule.downcast::<CSSStyleRule>()?; // Downcast to CSSStyleRule.
                    let selector = style.SelectorText(); // Get the selector text.
                    elem.Matches(selector.clone()).ok()?.then_some(())?; // Check if element matches selector.
                    Some((selector.into(), i)) // If it matches, return selector and stylesheet index.
                }))
            })
            .flatten() // Flatten the iterator of iterators into a single iterator.
            .collect(); // Collect all matching selectors.

        Some(rules) // Return the collected rules.
    })(); // End of IIFE.

    reply.send(msg).unwrap(); // Send the list of matching selectors.
}

/// Handles requests to get the computed style of a specific DOM element.
///
/// Functional Utility: Retrieves the final, resolved values of all CSS properties
/// for a given element, after applying all stylesheets and resolving inheritance.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param node_id The unique identifier of the element.
/// @param reply An IPC sender to send the `Vec<NodeStyle>` representing the computed styles.
/// @param can_gc A `CanGc` token.
pub(crate) fn handle_get_computed_style(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    reply: IpcSender<Option<Vec<NodeStyle>>>,
    can_gc: CanGc,
) {
    // Block Logic: Find the node by ID; if not found, send None and return.
    let node = match find_node_by_unique_id(documents, pipeline, &node_id) {
        None => return reply.send(None).unwrap(),
        Some(found_node) => found_node,
    };

    let window = node.owner_window(); // Get the owner window.
    let elem = node
        .downcast::<Element>()
        .expect("This should be an element"); // Downcast to Element.
    let computed_style = window.GetComputedStyle(elem, None); // Get the computed style.

    // Block Logic: Iterate through all computed style properties and summarize them.
    let msg = (0..computed_style.Length())
        .map(|i| {
            let name = computed_style.Item(i); // Property name.
            NodeStyle {
                name: name.to_string(), // Property name.
                value: computed_style
                    .GetPropertyValue(name.clone(), can_gc)
                    .to_string(), // Property value.
                priority: computed_style.GetPropertyPriority(name).to_string(), // Property priority.
            }
        })
        .collect(); // Collect all `NodeStyle` entries.

    reply.send(Some(msg)).unwrap(); // Send the collected computed styles.
}

/// Handles requests to get the layout information of a specific DOM element.
///
/// Functional Utility: Retrieves various box model metrics (width, height, margins,
/// borders, padding) and display-related CSS properties for an element.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param node_id The unique identifier of the element.
/// @param reply An IPC sender to send the `ComputedNodeLayout`.
/// @param can_gc A `CanGc` token.
pub(crate) fn handle_get_layout(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    reply: IpcSender<Option<ComputedNodeLayout>>,
    can_gc: CanGc,
) {
    // Block Logic: Find the node by ID; if not found, send None and return.
    let node = match find_node_by_unique_id(documents, pipeline, &node_id) {
        None => return reply.send(None).unwrap(),
        Some(found_node) => found_node,
    };

    let elem = node
        .downcast::<Element>()
        .expect("should be getting layout of element"); // Downcast to Element.
    let rect = elem.GetBoundingClientRect(can_gc); // Get the bounding client rectangle.
    let width = rect.Width() as f32; // Extract width.
    let height = rect.Height() as f32; // Extract height.

    let window = node.owner_window(); // Get the owner window.
    let elem = node
        .downcast::<Element>()
        .expect("should be getting layout of element"); // Downcast to Element again (redundant but safe).
    let computed_style = window.GetComputedStyle(elem, None); // Get the computed style.

    // Block Logic: Construct and send `ComputedNodeLayout` with various layout properties.
    reply
        .send(Some(ComputedNodeLayout {
            display: String::from(computed_style.Display()),
            position: String::from(computed_style.Position()),
            z_index: String::from(computed_style.ZIndex()),
            box_sizing: String::from(computed_style.BoxSizing()),
            auto_margins: determine_auto_margins(&node, can_gc), // Determine auto margins.
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
        .unwrap(); // Send the computed layout.
}

/// Determines which margins of a node are set to 'auto'.
///
/// @param node The DOM node to inspect.
/// @param can_gc A `CanGc` token.
/// @return An `AutoMargins` struct indicating which margins are 'auto'.
fn determine_auto_margins(node: &Node, can_gc: CanGc) -> AutoMargins {
    let style = node.style(can_gc).unwrap(); // Get the inline style of the node.
    let margin = style.get_margin(); // Get the computed margin properties.
    AutoMargins {
        top: margin.margin_top.is_auto(),    // Check if margin-top is auto.
        right: margin.margin_right.is_auto(), // Check if margin-right is auto.
        bottom: margin.margin_bottom.is_auto(), // Check if margin-bottom is auto.
        left: margin.margin_left.is_auto(),   // Check if margin-left is auto.
    }
}

/// Handles requests to modify attributes of a specific DOM element.
///
/// Functional Utility: Allows developer tools to change or remove attributes on an element.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param node_id The unique identifier of the element to modify.
/// @param modifications A vector of `AttrModification` structs specifying attribute changes.
/// @param can_gc A `CanGc` token.
pub(crate) fn handle_modify_attribute(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    modifications: Vec<AttrModification>,
    can_gc: CanGc,
) {
    // Block Logic: Find the document by pipeline ID; if not found, log a warning and return.
    let Some(document) = documents.find_document(pipeline) else {
        return warn!("document for pipeline id {} is not found", &pipeline);
    };
    let _realm = enter_realm(document.window()); // Enter the window's realm.

    // Block Logic: Find the node by unique ID; if not found, log a warning and return.
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
        .expect("should be getting layout of element"); // Downcast to Element.

    // Block Logic: Apply each attribute modification.
    // Invariant: For each modification, either the attribute is set to a new value or removed.
    for modification in modifications {
        match modification.new_value {
            Some(string) => {
                // If a new value is provided, set the attribute.
                let _ = elem.SetAttribute(
                    DOMString::from(modification.attribute_name), // Attribute name.
                    DOMString::from(string), // New attribute value.
                    can_gc, // CanGc token.
                );
            },
            None => elem.RemoveAttribute(DOMString::from(modification.attribute_name), can_gc), // If no new value, remove the attribute.
        }
    }
}

/// Handles requests to modify the inline style rules of a specific HTML element.
///
/// Functional Utility: Allows developer tools to dynamically change CSS properties
/// within an element's inline `style` attribute.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param node_id The unique identifier of the element.
/// @param modifications A vector of `RuleModification` structs specifying style changes.
/// @param can_gc A `CanGc` token.
#[cfg_attr(crown, allow(crown::unrooted_must_root))]
pub(crate) fn handle_modify_rule(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    node_id: String,
    modifications: Vec<RuleModification>,
    can_gc: CanGc,
) {
    // Block Logic: Find the document by pipeline ID; if not found, log a warning and return.
    let Some(document) = documents.find_document(pipeline) else {
        return warn!("Document for pipeline id {} is not found", &pipeline);
    };
    let _realm = enter_realm(document.window()); // Enter the window's realm.

    // Block Logic: Find the node by unique ID; if not found, log a warning and return.
    let Some(node) = find_node_by_unique_id(documents, pipeline, &node_id) else {
        return warn!(
            "Node id {} for pipeline id {} is not found",
            &node_id, &pipeline
        );
    };

    let elem = node
        .downcast::<HTMLElement>()
        .expect("This should be an HTMLElement"); // Downcast to HTMLElement.
    let style = elem.Style(can_gc); // Get the CSSStyleDeclaration for the inline style of the element.

    // Block Logic: Apply each style rule modification.
    // Invariant: For each modification, a CSS property is set with its value and priority.
    for modification in modifications {
        let _ = style.SetProperty(
            modification.name.into(),      // CSS property name.
            modification.value.into(),     // New CSS property value.
            modification.priority.into(),  // CSS property priority (e.g., "important").
            can_gc,                        // CanGc token.
        );
    }
}

/// Handles requests to enable or disable live notifications for developer tools.
///
/// Functional Utility: Controls whether the rendering engine should send real-time
/// updates to connected developer tools clients.
///
/// @param global The JavaScript global scope.
/// @param send_notifications A boolean indicating whether to enable (true) or disable (false) notifications.
pub(crate) fn handle_wants_live_notifications(global: &GlobalScope, send_notifications: bool) {
    global.set_devtools_wants_updates(send_notifications); // Set the flag for devtools updates.
}

/// Handles requests to set timeline markers.
///
/// Functional Utility: Instructs the rendering engine to record specific types of
/// events in a timeline for performance analysis in developer tools.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param marker_types A vector of `TimelineMarkerType` to enable.
/// @param reply An IPC sender to send the `TimelineMarker` or `None`.
pub(crate) fn handle_set_timeline_markers(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    marker_types: Vec<TimelineMarkerType>,
    reply: IpcSender<Option<TimelineMarker>>,
) {
    // Block Logic: Find the window by pipeline ID; if found, set timeline markers, otherwise send None.
    match documents.find_window(pipeline) {
        None => reply.send(None).unwrap(), // If window not found, send None.
        Some(window) => window.set_devtools_timeline_markers(marker_types, reply), // Set timeline markers.
    }
}

/// Handles requests to drop (clear) timeline markers.
///
/// Functional Utility: Instructs the rendering engine to stop recording specific types of
/// events in a timeline.
///
/// @param documents The collection of all active documents.
/// @param pipeline The `PipelineId` identifying the target document.
/// @param marker_types A vector of `TimelineMarkerType` to disable.
pub(crate) fn handle_drop_timeline_markers(
    documents: &DocumentCollection,
    pipeline: PipelineId,
    marker_types: Vec<TimelineMarkerType>,
) {
    // Block Logic: Find the window by pipeline ID; if found, drop timeline markers.
    if let Some(window) = documents.find_window(pipeline) {
        window.drop_devtools_timeline_markers(marker_types); // Drop timeline markers.
    }
}

/// Handles requests to trigger an animation frame for developer tools.
///
/// Functional Utility: Requests a single animation frame to be processed,
/// useful for stepping through animations or repaints in devtools.
///
/// @param documents The collection of all active documents.
/// @param id The `PipelineId` identifying the target document.
/// @param actor_name The name of the actor requesting the animation frame.
pub(crate) fn handle_request_animation_frame(
    documents: &DocumentCollection,
    id: PipelineId,
    actor_name: String,
) {
    // Block Logic: Find the document by pipeline ID; if found, request an animation frame.
    if let Some(doc) = documents.find_document(id) {
        doc.request_animation_frame(AnimationFrameCallback::DevtoolsFramerateTick { actor_name }); // Request animation frame.
    }
}

/// Handles requests to reload a document.
///
/// Functional Utility: Forces a refresh of the specified document,
/// bypassing the origin check for devtools purposes.
///
/// @param documents The collection of all active documents.
/// @param id The `PipelineId` identifying the target document.
/// @param can_gc A `CanGc` token.
pub(crate) fn handle_reload(documents: &DocumentCollection, id: PipelineId, can_gc: CanGc) {
    // Block Logic: Find the window by pipeline ID; if found, reload the document.
    if let Some(win) = documents.find_window(id) {
        win.Location().reload_without_origin_check(can_gc); // Reload the document.
    }
}

/// Handles requests to get a database of CSS properties.
///
/// Functional Utility: Provides developer tools with a list of all enabled longhand
/// CSS properties and their characteristics (e.g., inheritance).
///
/// @param reply An IPC sender to send the `HashMap` of CSS properties.
pub(crate) fn handle_get_css_database(reply: IpcSender<HashMap<String, CssDatabaseProperty>>) {
    // Block Logic: Construct a HashMap of CSS properties from a predefined list.
    let database: HashMap<_, _> = ENABLED_LONGHAND_PROPERTIES
        .iter() // Iterate over enabled longhand CSS properties.
        .map(|l| {
            (
                l.name().into(), // Property name as key.
                CssDatabaseProperty {
                    is_inherited: l.inherited(), // Whether the property is inherited.
                    values: vec![], // TODO: Get allowed values for each property (currently empty).
                    supports: vec![], // TODO: Get supported values (currently empty).
                    subproperties: vec![l.name().into()], // Subproperties (currently just the property itself).
                },
            )
        })
        .collect(); // Collect into a HashMap.
    let _ = reply.send(database); // Send the CSS database.
}