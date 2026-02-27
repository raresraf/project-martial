/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! # Tab Descriptor Actor
//!
//! This actor acts as a descriptor for a web page loaded in a tab, providing a stable
//! reference that can be used by a remote DevTools client. It serves as a bridge,
//! linking the abstract concept of a "tab" to its underlying content (the `BrowsingContextActor`)
//! and its debugging capabilities (the `WatcherActor`).
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

/// Defines the serializable JSON message that represents the state of a single tab.
/// This is the "form" sent to the DevTools client.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TabDescriptorActorMsg {
    /// The unique name of this actor in the actor registry.
    actor: String,
    /// An identifier for the underlying browser associated with this tab.
    browser_id: u32,
    /// The ID of the tab's primary browsing context.
    #[serde(rename = "browsingContextID")]
    browsing_context_id: u32,
    /// Flag indicating if the tab is in a "zombie" state (e.g., crashed).
    is_zombie_tab: bool,
    /// The ID of the top-level window containing the tab.
    #[serde(rename = "outerWindowID")]
    outer_window_id: u32,
    /// True if this is the currently active/selected tab in the browser.
    selected: bool,
    /// The current title of the web page.
    title: String,
    /// A set of boolean flags indicating the capabilities of this descriptor.
    traits: DescriptorTraits,
    /// The current URL of the web page.
    url: String,
}

impl TabDescriptorActorMsg {
    /// Returns the browser ID associated with this tab message.
    pub fn id(&self) -> u32 {
        self.browser_id
    }
}

/// Defines the JSON reply for a `getTarget` message.
#[derive(Serialize)]
struct GetTargetReply {
    from: String,
    /// The serializable message representing the tab's main content frame.
    frame: BrowsingContextActorMsg,
}

/// Defines the JSON reply for a `getFavicon` message.
#[derive(Serialize)]
struct GetFaviconReply {
    from: String,
    favicon: String,
}

/// Defines the JSON reply for a `getWatcher` message.
#[derive(Serialize)]
struct GetWatcherReply {
    from: String,
    /// The message containing information about the watcher's capabilities.
    #[serde(flatten)]
    watcher: WatcherActorMsg,
}

/// The main actor struct, holding the state for a tab descriptor.
pub struct TabDescriptorActor {
    /// The unique name of this actor, e.g., "tab-description1".
    name: String,
    /// The name of the `BrowsingContextActor` that represents the tab's content.
    browsing_context_actor: String,
    /// Flag to indicate if this tab represents a top-level browsing context.
    is_top_level_global: bool,
}

impl Actor for TabDescriptorActor {
    fn name(&self) -> String {
        self.name.clone()
    }

    /// The main message handler for the `TabDescriptorActor`. It processes requests
    /// from the DevTools client related to this specific tab.
    fn handle_message(
        &self,
        registry: &ActorRegistry,
        msg_type: &str,
        _msg: &Map<String, Value>,
        stream: &mut TcpStream,
        _id: StreamId,
    ) -> Result<ActorMessageStatus, ()> {
        Ok(match msg_type {
            // The `getTarget` message is used by the client to get a reference to the
            // actor that represents the tab's main content, allowing for inspection
            // and manipulation of the DOM, console, etc.
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
            // The `getFavicon` message requests the tab's favicon URL.
            "getFavicon" => {
                // TODO: Return a favicon when available
                let _ = stream.write_json_packet(&GetFaviconReply {
                    from: self.name(),
                    favicon: String::new(),
                });
                ActorMessageStatus::Processed
            },
            // The `getWatcher` message is sent by the client to discover what
            // debugging capabilities are available for this tab (e.g., can we
            // listen for console messages, network requests, etc.). It returns
            // a `WatcherActor` which manages these capabilities.
            "getWatcher" => {
                let ctx_actor = registry.find::<BrowsingContextActor>(&self.browsing_context_actor);
                let watcher = registry.find::<WatcherActor>(&ctx_actor.watcher);
                let _ = stream.write_json_packet(&GetWatcherReply {
                    from: self.name(),
                    watcher: watcher.encodable(),
                });
                ActorMessageStatus::Processed
            },
            _ => ActorMessageStatus::Ignored,
        })
    }
}

impl TabDescriptorActor {
    /// Constructor for a new `TabDescriptorActor`.
    pub(crate) fn new(
        actors: &mut ActorRegistry,
        browsing_context_actor: String,
        is_top_level_global: bool,
    ) -> TabDescriptorActor {
        // Generate a new unique name for this actor instance.
        let name = actors.new_name("tab-description");
        // Register this new tab actor with the root actor, making it discoverable.
        let root = actors.find_mut::<RootActor>("root");
        root.tabs.push(name.clone());
        TabDescriptorActor {
            name,
            browsing_context_actor,
            is_top_level_global,
        }
    }

    /// Creates a serializable message representing the tab's current state.
    /// This is used to send information about the tab to the DevTools client.
    pub fn encodable(&self, registry: &ActorRegistry, selected: bool) -> TabDescriptorActorMsg {
        // Gather up-to-date information from the underlying browsing context actor.
        let ctx_actor = registry.find::<BrowsingContextActor>(&self.browsing_context_actor);
        let browser_id = ctx_actor.active_pipeline.get().index.0.get();
        let browsing_context_id = ctx_actor.browsing_context_id.index.0.get();
        let title = ctx_actor.title.borrow().clone();
        let url = ctx_actor.url.borrow().clone();

        // Assemble the message payload.
        TabDescriptorActorMsg {
            actor: self.name(),
            browsing_context_id,
            browser_id,
            is_zombie_tab: false, // Zombie tabs are not yet supported.
            outer_window_id: browser_id,
            selected,
            title,
            traits: DescriptorTraits {
                watcher: true,
                supports_reload_descriptor: true,
            },
            url,
        }
    }

    /// A getter to check if the tab represents a top-level context.
    pub(crate) fn is_top_level_global(&self) -> bool {
        self.is_top_level_global
    }
}
