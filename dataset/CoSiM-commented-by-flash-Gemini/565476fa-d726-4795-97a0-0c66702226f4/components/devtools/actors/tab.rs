/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Descriptor actor that represents a web view. It can link a tab to the corresponding watcher
//! actor to enable inspection.
//!
//! Liberally derived from the [Firefox JS implementation].
//!
//! [Firefox JS implementation]: https://searchfox.org/mozilla-central/source/devtools/server/actors/descriptors/tab.js
//!
//! This module implements the `TabDescriptorActor`, which serves as a crucial component
//! in a devtools architecture. It acts as a descriptor for a web view (tab), providing
//! information about the tab and enabling its inspection by linking to a `WatcherActor`.
//! It processes messages related to tab information, favicon retrieval, and debugging capabilities.

use std::net::TcpStream;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::StreamId;
use crate::actor::{Actor, ActorMessageStatus, ActorRegistry};
use crate::actors::browsing_context::{BrowsingContextActor, BrowsingContextActorMsg};
use crate::actors::root::{DescriptorTraits, RootActor};
use crate::actors::watcher::{WatcherActor, WatcherActorMsg};
use crate::protocol::JsonPacketStream;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TabDescriptorActorMsg {
    /// The unique identifier for this actor.
    actor: String,
    /// The ID of the browser instance this tab belongs to.
    browser_id: u32,
    /// The ID of the browsing context associated with this tab.
    #[serde(rename = "browsingContextID")]
    browsing_context_id: u32,
    /// Indicates if the tab is a "zombie" tab (e.g., closed but still held in memory).
    is_zombie_tab: bool,
    /// The ID of the outer window this tab belongs to.
    #[serde(rename = "outerWindowID")]
    outer_window_id: u32,
    /// Indicates if this tab is currently selected in the browser.
    selected: bool,
    /// The title of the tab.
    title: String,
    /// Traits describing the capabilities of this descriptor.
    traits: DescriptorTraits,
    /// The URL currently loaded in the tab.
    url: String,
}

impl TabDescriptorActorMsg {
    /// Returns the browser ID associated with this tab descriptor.
    pub fn id(&self) -> u32 {
        self.browser_id
    }
}

#[derive(Serialize)]
struct GetTargetReply {
    /// The 'from' field indicates the actor sending the reply.
    from: String,
    /// The frame field contains the serialized representation of the BrowsingContextActor.
    frame: BrowsingContextActorMsg,
}

#[derive(Serialize)]
struct GetFaviconReply {
    /// The 'from' field indicates the actor sending the reply.
    from: String,
    /// The favicon field contains the URL or data for the tab's favicon.
    favicon: String,
}

#[derive(Serialize)]
struct GetWatcherReply {
    /// The 'from' field indicates the actor sending the reply.
    from: String,
    /// The watcher field contains the serialized representation of the WatcherActor.
    #[serde(flatten)]
    watcher: WatcherActorMsg,
}

pub struct TabDescriptorActor {
    /// The unique name of this actor instance.
    name: String,
    /// The name of the `BrowsingContextActor` associated with this tab.
    browsing_context_actor: String,
    /// A boolean indicating if this tab represents a top-level global browsing context.
    is_top_level_global: bool,
}

impl Actor for TabDescriptorActor {
    /// Returns the unique name of this actor.
    fn name(&self) -> String {
        self.name.clone()
    }

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
            "getTarget" => {
                // Inline: Find the associated BrowsingContextActor and get its encodable representation.
                let frame = registry
                    .find::<BrowsingContextActor>(&self.browsing_context_actor)
                    .encodable();
                // Inline: Write the GetTargetReply JSON packet to the stream.
                let _ = stream.write_json_packet(&GetTargetReply {
                    from: self.name(),
                    frame,
                });
                ActorMessageStatus::Processed
            },
            "getFavicon" => {
                // TODO: Return a favicon when available
                // Inline: Write the GetFaviconReply JSON packet to the stream.
                let _ = stream.write_json_packet(&GetFaviconReply {
                    from: self.name(),
                    favicon: String::new(),
                });
                ActorMessageStatus::Processed
            },
            "getWatcher" => {
                // Inline: Find the associated BrowsingContextActor.
                let ctx_actor = registry.find::<BrowsingContextActor>(&self.browsing_context_actor);
                // Inline: Find the WatcherActor linked to the BrowsingContext and get its encodable representation.
                let watcher = registry.find::<WatcherActor>(&ctx_actor.watcher);
                // Inline: Write the GetWatcherReply JSON packet to the stream.
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
    pub(crate) fn new(
        actors: &mut ActorRegistry,
        browsing_context_actor: String,
        is_top_level_global: bool,
    ) -> TabDescriptorActor {
        // Inline: Generates a new unique name for this tab descriptor actor.
        let name = actors.new_name("tab-description");
        // Inline: Finds the root actor in the registry to update its list of tabs.
        let root = actors.find_mut::<RootActor>("root");
        // Inline: Adds the new tab's name to the root actor's list of tabs.
        root.tabs.push(name.clone());
        TabDescriptorActor {
            name,
            browsing_context_actor,
            is_top_level_global,
        }
    }

    pub fn encodable(&self, registry: &ActorRegistry, selected: bool) -> TabDescriptorActorMsg {
        // Inline: Retrieve the BrowsingContextActor associated with this tab.
        let ctx_actor = registry.find::<BrowsingContextActor>(&self.browsing_context_actor);
        // Inline: Extract the browser ID from the browsing context's active pipeline.
        let browser_id = ctx_actor.active_pipeline.get().index.0.get();
        // Inline: Extract the browsing context ID.
        let browsing_context_id = ctx_actor.browsing_context_id.index.0.get();
        // Inline: Clone the current title of the browsing context.
        let title = ctx_actor.title.borrow().clone();
        // Inline: Clone the current URL of the browsing context.
        let url = ctx_actor.url.borrow().clone();

        TabDescriptorActorMsg {
            actor: self.name(),
            browsing_context_id,
            browser_id,
            is_zombie_tab: false,
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

    pub(crate) fn is_top_level_global(&self) -> bool {
        // Inline: Returns whether this tab represents a top-level global browsing context.
        self.is_top_level_global
    }
}
