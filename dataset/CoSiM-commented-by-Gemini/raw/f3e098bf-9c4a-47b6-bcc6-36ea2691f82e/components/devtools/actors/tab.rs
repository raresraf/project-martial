/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Descriptor actor that represents a web view. It can link a tab to the corresponding watcher
//! actor to enable inspection.
//!
//! Liberally derived from the [Firefox JS implementation].
//!
//! [Firefox JS implementation]: https://searchfox.org/mozilla-central/source/devtools/server/actors/descriptors/tab.js

use std::net::TcpStream;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::StreamId;
use crate::actor::{Actor, ActorMessageStatus, ActorRegistry};
use crate::actors::browsing_context::{BrowsingContextActor, BrowsingContextActorMsg};
use crate::actors::root::{DescriptorTraits, RootActor};
use crate::actors::watcher::{WatcherActor, WatcherActorMsg};
use crate::protocol::JsonPacketStream;

/// Defines the serializable message format for a tab descriptor.
/// This struct represents the state of a browser tab that is sent to the
/// DevTools client. `serde` attributes control the JSON field names.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TabDescriptorActorMsg {
    /// The unique name of this actor instance (e.g., "tab-description-1").
    actor: String,
    /// The internal browser ID associated with this tab.
    browser_id: u32,
    /// The ID of the browsing context, used for identifying the document environment.
    #[serde(rename = "browsingContextID")]
    browsing_context_id: u32,
    /// A flag indicating if the tab is a "zombie" (e.g., a crashed tab).
    is_zombie_tab: bool,
    /// The ID of the top-level window containing the tab.
    #[serde(rename = "outerWindowID")]
    outer_window_id: u32,
    /// True if this is the currently selected tab in the browser.
    selected: bool,
    /// The title of the document currently loaded in the tab.
    title: String,
    /// A struct describing the capabilities of this descriptor (e.g., if it has a watcher).
    traits: DescriptorTraits,
    /// The URL of the document currently loaded in the tab.
    url: String,
}

impl TabDescriptorActorMsg {
    /// Returns the browser ID associated with this tab message.
    pub fn id(&self) -> u32 {
        self.browser_id
    }
}

/// Defines the reply structure for a `getTarget` message.
#[derive(Serialize)]
struct GetTargetReply {
    /// The actor name from which this reply originates.
    from: String,
    /// The serializable message for the associated browsing context.
    frame: BrowsingContextActorMsg,
}

/// Defines the reply structure for a `getFavicon` message.
#[derive(Serialize)]
struct GetFaviconReply {
    from: String,
    /// The URL or data-URI of the tab's favicon. Currently unimplemented.
    favicon: String,
}

/// Defines the reply structure for a `getWatcher` message.
#[derive(Serialize)]
struct GetWatcherReply {
    from: String,
    /// The serializable message for the associated watcher actor.
    #[serde(flatten)]
    watcher: WatcherActorMsg,
}

/// An actor that represents a single browser tab for the DevTools server.
/// It holds references to other actors and state relevant to the tab.
pub struct TabDescriptorActor {
    /// The unique actor name.
    name: String,
    /// The name of the `BrowsingContextActor` that this tab descriptor represents.
    browsing_context_actor: String,
    /// A flag indicating if this actor represents a top-level context.
    is_top_level_global: bool,
}

impl Actor for TabDescriptorActor {
    /// Returns a clone of the actor's name.
    fn name(&self) -> String {
        self.name.clone()
    }

    /// The main message handler for the `TabDescriptorActor`.
    /// It processes incoming messages and sends replies back over the TCP stream.
    ///
    /// The tab actor can handle the following messages:
    ///
    /// - `getTarget`: Returns the surrounding `BrowsingContextActor`.
    ///
    /// - `getFavicon`: Should return the tab favicon, but it is not yet supported.
    ///
    /// - `getWatcher`: Returns a `WatcherActor` linked to the tab's `BrowsingContext`. It is used
    ///   to describe the debugging capabilities of this tab.
    fn handle_message(
        &self,
        registry: &ActorRegistry,
        msg_type: &str,
        _msg: &Map<String, Value>,
        stream: &mut TcpStream,
        _id: StreamId,
    ) -> Result<ActorMessageStatus, ()> {
        Ok(match msg_type {
            // Logic for the `getTarget` request.
            // This is used by the client to get a reference to the actor representing the
            // tab's actual content and browsing environment.
            "getTarget" => {
                let frame = registry
                    .find::<BrowsingContextActor>(&self.browsing_context_actor)
                    .encodable();
                let _ = stream.write_json_packet(&GetTargetReply {
                    from: self.name(),
                    frame,
                });
                ActorMessageStatus::Processed
            },
            // Logic for the `getFavicon` request.
            "getFavicon" => {
                // TODO: Return a favicon when available. Currently returns an empty string.
                let _ = stream.write_json_packet(&GetFaviconReply {
                    from: self.name(),
                    favicon: String::new(),
                });
                ActorMessageStatus::Processed
            },
            // Logic for the `getWatcher` request.
            // A "watcher" is a DevTools concept for an actor that can be used to
            // observe and control a target (like a tab or worker).
            "getWatcher" => {
                let ctx_actor = registry.find::<BrowsingContextActor>(&self.browsing_context_actor);
                let watcher = registry.find::<WatcherActor>(&ctx_actor.watcher);
                let _ = stream.write_json_packet(&GetWatcherReply {
                    from: self.name(),
                    watcher: watcher.encodable(),
                });
                ActorMessageStatus::Processed
            },
            // If the message type is not recognized, ignore it.
            _ => ActorMessageStatus::Ignored,
        })
    }
}

impl TabDescriptorActor {
    /// Creates a new `TabDescriptorActor` and registers it.
    /// This function is the designated constructor for this actor type.
    pub(crate) fn new(
        actors: &mut ActorRegistry,
        browsing_context_actor: String,
        is_top_level_global: bool,
    ) -> TabDescriptorActor {
        // Generate a new, unique name for this actor (e.g., "tab-description-2").
        let name = actors.new_name("tab-description");
        // Register this new tab actor with the root actor.
        let root = actors.find_mut::<RootActor>("root");
        root.tabs.push(name.clone());
        TabDescriptorActor {
            name,
            browsing_context_actor,
            is_top_level_global,
        }
    }

    /// Gathers the tab's current state and packages it into a serializable message.
    /// This is called when the server needs to send an update about this tab to the client.
    pub fn encodable(&self, registry: &ActorRegistry, selected: bool) -> TabDescriptorActorMsg {
        // Find the associated browsing context actor to retrieve live data.
        let ctx_actor = registry.find::<BrowsingContextActor>(&self.browsing_context_actor);
        
        // Extract all necessary data points from the context actor.
        let browser_id = ctx_actor.browsing_context_id.index.0.get();
        let outer_window_id = ctx_actor.active_pipeline.get().index.0.get();
        let browsing_context_id = ctx_actor.browsing_context_id.index.0.get();
        let title = ctx_actor.title.borrow().clone();
        let url = ctx_actor.url.borrow().clone();

        // Construct the message object to be sent.
        TabDescriptorActorMsg {
            actor: self.name(),
            browsing_context_id,
            browser_id,
            is_zombie_tab: false, // Zombie tab state not yet implemented.
            outer_window_id,
            selected,
            title,
            traits: DescriptorTraits {
                watcher: true,
                supports_reload_descriptor: true,
            },
            url,
        }
    }

    /// A getter to check if the actor represents a top-level context.
    pub(crate) fn is_top_level_global(&self) -> bool {
        self.is_top_level_global
    }
}
