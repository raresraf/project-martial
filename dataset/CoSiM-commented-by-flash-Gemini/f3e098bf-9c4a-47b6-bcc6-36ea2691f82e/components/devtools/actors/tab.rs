/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! @file tab.rs
//! @brief Descriptor actor that represents a web view (browser tab) within the DevTools system.
//!
//! This actor is responsible for providing information about a browser tab and linking
//! it to other actors that enable debugging and inspection capabilities. It acts as
//! an entry point for DevTools clients to interact with a specific tab.
//!
//! Liberally derived from the [Firefox JS implementation].
//!
//! [Firefox JS implementation]: https://searchfox.org/mozilla-central/source/devtools/server/actors/descriptors/tab.js

use std::net::TcpStream; // Used for network communication with the DevTools client.

use serde::Serialize;     // Enables serialization of Rust structs into various formats (e.g., JSON).
use serde_json::{Map, Value}; // Used for handling JSON data structure (Map corresponds to JSON object).

use crate::StreamId;      // Unique identifier for a communication stream.
use crate::actor::{Actor, ActorMessageStatus, ActorRegistry}; // Core actor traits and registry.
use crate::actors::browsing_context::{BrowsingContextActor, BrowsingContextActorMsg}; // Actor representing a browsing context.
use crate::actors::root::{DescriptorTraits, RootActor}; // Root actor and descriptor traits for capabilities.
use crate::actors::watcher::{WatcherActor, WatcherActorMsg}; // Actor for describing debugging capabilities.
use crate::protocol::JsonPacketStream; // Trait for writing JSON packets to a stream.

/// Represents the serializable message format for a `TabDescriptorActor`.
/// This struct defines the structure of data sent to the DevTools client
/// when requesting tab-related information.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")] // Maps Rust's snake_case fields to camelCase for JSON.
pub struct TabDescriptorActorMsg {
    pub actor: String,          // The actor ID of this TabDescriptorActor.
    pub browser_id: u32,        // Unique ID identifying the browser window/process.
    #[serde(rename = "browsingContextID")] // Explicitly renames field for JSON.
    pub browsing_context_id: u32, // Unique ID identifying the browsing context (e.g., iframe).
    pub is_zombie_tab: bool,    // Indicates if the tab is in a 'zombie' state (not fully active).
    #[serde(rename = "outerWindowID")] // Explicitly renames field for JSON.
    pub outer_window_id: u32,   // Unique ID for the outer window.
    pub selected: bool,         // Indicates if this tab is currently selected in the browser UI.
    pub title: String,          // The current title of the web page in the tab.
    pub traits: DescriptorTraits, // Capabilities and supported features of this tab descriptor.
    pub url: String,            // The current URL loaded in the tab.
}

impl TabDescriptorActorMsg {
    /// Returns the browser ID associated with this message.
    pub fn id(&self) -> u32 {
        self.browser_id
    }
}

/// Reply structure for the `getTarget` message.
/// Contains the actor ID and the encodable `BrowsingContextActorMsg`.
#[derive(Serialize)]
struct GetTargetReply {
    from: String, // The actor ID of the sender.
    frame: BrowsingContextActorMsg, // The serialized browsing context actor message.
}

/// Reply structure for the `getFavicon` message.
/// Contains the actor ID and the favicon URL (or an empty string if not supported).
#[derive(Serialize)]
struct GetFaviconReply {
    from: String,   // The actor ID of the sender.
    favicon: String, // The URL of the tab's favicon.
}

/// Reply structure for the `getWatcher` message.
/// Contains the actor ID and the encodable `WatcherActorMsg`.
#[derive(Serialize)]
struct GetWatcherReply {
    from: String,               // The actor ID of the sender.
    #[serde(flatten)]          // Flattens the nested `WatcherActorMsg` into the parent structure.
    watcher: WatcherActorMsg,   // The serialized watcher actor message.
}

/// The `TabDescriptorActor` struct.
/// Represents the internal state of a tab descriptor actor.
pub struct TabDescriptorActor {
    name: String,                   // The unique actor ID for this tab descriptor.
    browsing_context_actor: String, // The actor ID of the associated `BrowsingContextActor`.
    is_top_level_global: bool,      // Indicates if this tab is a top-level global tab.
}

impl Actor for TabDescriptorActor {
    /// Returns the unique name (actor ID) of this actor.
    fn name(&self) -> String {
        self.name.clone()
    }

    /// @brief Handles incoming messages for the `TabDescriptorActor`.
    ///
    /// The tab actor can handle the following messages:
    ///
    /// - `getTarget`: Returns the surrounding `BrowsingContextActor`. This actor represents
    ///   the frame that contains the web content of the tab.
    ///
    /// - `getFavicon`: Should return the tab favicon, but it is not yet supported.
    ///   Currently returns an empty string for the favicon.
    ///
    /// - `getWatcher`: Returns a `WatcherActor` linked to the tab's `BrowsingContext`. It is used
    ///   to describe the debugging capabilities of this tab (e.g., script debugging, console).
    ///
    /// @param registry The `ActorRegistry` to look up other actors.
    /// @param msg_type The type of the incoming message.
    /// @param _msg The full message payload (unused directly in this handler's match arms).
    /// @param stream The `TcpStream` to write the reply to.
    /// @param _id The `StreamId` of the incoming message (unused).
    /// @return Result<ActorMessageStatus, ()>: Indicates if the message was processed or ignored.
    fn handle_message(
        &self,
        registry: &ActorRegistry,
        msg_type: &str,
        _msg: &Map<String, Value>,
        stream: &mut TcpStream,
        _id: StreamId,
    ) -> Result<ActorMessageStatus, ()> {
        Ok(match msg_type {
            "getTarget" => {
                // Retrieves the associated `BrowsingContextActor` from the registry.
                let frame = registry
                    .find::<BrowsingContextActor>(&self.browsing_context_actor)
                    .encodable(); // Converts the actor to its serializable message form.
                // Writes the reply containing the browsing context information to the stream.
                let _ = stream.write_json_packet(&GetTargetReply {
                    from: self.name(),
                    frame,
                });
                ActorMessageStatus::Processed // Indicates the message was handled.
            },
            "getFavicon" => {
                // TODO: Return a favicon when available - Placeholder for future implementation.
                // Writes a reply with an empty favicon string as it's not yet supported.
                let _ = stream.write_json_packet(&GetFaviconReply {
                    from: self.name(),
                    favicon: String::new(),
                });
                ActorMessageStatus::Processed // Indicates the message was handled.
            },
            "getWatcher" => {
                // Retrieves the `BrowsingContextActor` to get the watcher actor ID.
                let ctx_actor = registry.find::<BrowsingContextActor>(&self.browsing_context_actor);
                // Retrieves the `WatcherActor` from the registry using the ID from the browsing context.
                let watcher = registry.find::<WatcherActor>(&ctx_actor.watcher);
                // Writes the reply containing the watcher actor information to the stream.
                let _ = stream.write_json_packet(&GetWatcherReply {
                    from: self.name(),
                    watcher: watcher.encodable(), // Converts the watcher actor to its serializable message form.
                });
                ActorMessageStatus::Processed // Indicates the message was handled.
            },
            _ => ActorMessageStatus::Ignored, // Message type not recognized by this actor.
        })
    }
}

impl TabDescriptorActor {
    /// @brief Creates a new `TabDescriptorActor` instance.
    ///
    /// This function also registers the new tab actor's name with the `RootActor`
    /// to maintain a list of active tabs.
    ///
    /// @param actors The `ActorRegistry` to register the new actor and find others.
    /// @param browsing_context_actor The actor ID of the associated `BrowsingContextActor`.
    /// @param is_top_level_global A boolean indicating if this is a top-level global tab.
    /// @return TabDescriptorActor: A new instance of `TabDescriptorActor`.
    pub(crate) fn new(
        actors: &mut ActorRegistry,
        browsing_context_actor: String,
        is_top_level_global: bool,
    ) -> TabDescriptorActor {
        let name = actors.new_name("tab-description"); // Generates a unique actor ID for this tab descriptor.
        let root = actors.find_mut::<RootActor>("root"); // Retrieves the mutable `RootActor` instance.
        root.tabs.push(name.clone()); // Adds the new tab's actor ID to the `RootActor`'s list of tabs.
        TabDescriptorActor {
            name,
            browsing_context_actor,
            is_top_level_global,
        }
    }

    /// @brief Converts the `TabDescriptorActor` into its serializable message format.
    ///
    /// This function retrieves necessary information from other actors (like `BrowsingContextActor`)
    /// to construct a complete `TabDescriptorActorMsg` for the DevTools client.
    ///
    /// @param registry The `ActorRegistry` to look up other actors.
    /// @param selected A boolean indicating if this tab should be marked as selected.
    /// @return TabDescriptorActorMsg: The serializable message representation of this tab.
    pub fn encodable(&self, registry: &ActorRegistry, selected: bool) -> TabDescriptorActorMsg {
        // Retrieves the associated `BrowsingContextActor`.
        let ctx_actor = registry.find::<BrowsingContextActor>(&self.browsing_context_actor);
        // Extracts relevant IDs from the browsing context.
        let browser_id = ctx_actor.browsing_context_id.index.0.get();
        let outer_window_id = ctx_actor.active_pipeline.get().index.0.get();
        let browsing_context_id = ctx_actor.browsing_context_id.index.0.get();
        // Clones the title and URL for inclusion in the message.
        let title = ctx_actor.title.borrow().clone();
        let url = ctx_actor.url.borrow().clone();

        TabDescriptorActorMsg {
            actor: self.name(), // The actor ID of this tab descriptor.
            browsing_context_id,
            browser_id,
            is_zombie_tab: false, // Currently hardcoded to false, indicates a live tab.
            outer_window_id,
            selected,
            title,
            traits: DescriptorTraits { // Defines the capabilities of this tab descriptor.
                watcher: true, // Indicates support for watcher actor.
                supports_reload_descriptor: true, // Indicates support for reloading the descriptor.
            },
            url,
        }
    }

    /// @brief Checks if this tab is a top-level global tab.
    /// @return bool: True if it's a top-level global tab, False otherwise.
    pub(crate) fn is_top_level_global(&self) -> bool {
        self.is_top_level_global
    }
}
