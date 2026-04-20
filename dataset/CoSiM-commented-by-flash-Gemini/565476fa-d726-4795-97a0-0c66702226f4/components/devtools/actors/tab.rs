/**
 * @565476fa-d726-4795-97a0-0c66702226f4/components/devtools/actors/tab.rs
 * @brief Descriptor actor representing a web view (tab) in the DevTools protocol.
 * 
 * Functional Intent: Acts as a bridge between the root DevTools actor and specific 
 * browsing contexts. It provides metadata about a tab (URL, title, selection state) 
 * and orchestrates the linkage to 'Watcher' actors for debugging and inspection. 
 * It maps high-level browser tab state to JSON-serializable messages for the remote 
 * debugging protocol.
 * 
 * Domain: Browser DevTools, Remote Debugging Protocol, Tab Management.
 */

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::net::TcpStream;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::StreamId;
use crate::actor::{Actor, ActorMessageStatus, ActorRegistry};
use crate::actors::browsing_context::{BrowsingContextActor, BrowsingContextActorMsg};
use crate::actors::root::{DescriptorTraits, RootActor};
use crate::actors::watcher::{WatcherActor, WatcherActorMsg};
use crate::protocol::JsonPacketStream;

/**
 * @brief Serialized representation of a tab descriptor for protocol transmission.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TabDescriptorActorMsg {
    actor: String,
    browser_id: u32,
    #[serde(rename = "browsingContextID")]
    browsing_context_id: u32,
    is_zombie_tab: bool,
    #[serde(rename = "outerWindowID")]
    outer_window_id: u32,
    selected: bool,
    title: String,
    traits: DescriptorTraits,
    url: String,
}

impl TabDescriptorActorMsg {
    pub fn id(&self) -> u32 {
        self.browser_id
    }
}

/**
 * @brief Reply containers for 'getTarget', 'getFavicon', and 'getWatcher' requests.
 */
#[derive(Serialize)]
struct GetTargetReply {
    from: String,
    frame: BrowsingContextActorMsg,
}

#[derive(Serialize)]
struct GetFaviconReply {
    from: String,
    favicon: String,
}

#[derive(Serialize)]
struct GetWatcherReply {
    from: String,
    #[serde(flatten)]
    watcher: WatcherActorMsg,
}

/**
 * @brief Primary actor implementation for browser tab descriptors.
 */
pub struct TabDescriptorActor {
    name: String,
    browsing_context_actor: String,
    is_top_level_global: bool,
}

impl Actor for TabDescriptorActor {
    fn name(&self) -> String {
        self.name.clone()
    }

    /**
     * handle_message - Dispatches protocol requests for tab-level debugging state.
     * Logic: 
     * 1. 'getTarget': Resolves the underlying BrowsingContext for the tab.
     * 2. 'getFavicon': Currently a stub; placeholder for icon resource retrieval.
     * 3. 'getWatcher': Chains through the BrowsingContext to retrieve the linked 
     *    WatcherActor responsible for event monitoring (Console, Network, etc.).
     */
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
                let frame = registry
                    .find::<BrowsingContextActor>(&self.browsing_context_actor)
                    .encodable();
                let _ = stream.write_json_packet(&GetTargetReply {
                    from: self.name(),
                    frame,
                });
                ActorMessageStatus::Processed
            },
            "getFavicon" => {
                let _ = stream.write_json_packet(&GetFaviconReply {
                    from: self.name(),
                    favicon: String::new(),
                });
                ActorMessageStatus::Processed
            },
            "getWatcher" => {
                // Block Logic: Recursive actor resolution.
                // Logic: Resolves the context actor, then its associated watcher, 
                // and serializes the watcher's capability set (WatcherActorMsg).
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
    /**
     * @brief Factory method for creating tab descriptor actors.
     * Logic: Registers the new actor name and appends it to the global 
     * tab list maintained by the RootActor.
     */
    pub(crate) fn new(
        actors: &mut ActorRegistry,
        browsing_context_actor: String,
        is_top_level_global: bool,
    ) -> TabDescriptorActor {
        let name = actors.new_name("tab-description");
        let root = actors.find_mut::<RootActor>("root");
        root.tabs.push(name.clone());
        TabDescriptorActor {
            name,
            browsing_context_actor,
            is_top_level_global,
        }
    }

    /**
     * encodable - Generates a protocol-compliant snapshot of the tab's current state.
     * Logic: Synchronizes with the associated BrowsingContextActor to extract 
     * transient state (title, URL, pipeline IDs) and defines supported 
     * reload and watcher traits.
     */
    pub fn encodable(&self, registry: &ActorRegistry, selected: bool) -> TabDescriptorActorMsg {
        let ctx_actor = registry.find::<BrowsingContextActor>(&self.browsing_context_actor);
        let browser_id = ctx_actor.active_pipeline.get().index.0.get();
        let browsing_context_id = ctx_actor.browsing_context_id.index.0.get();
        let title = ctx_actor.title.borrow().clone();
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
        self.is_top_level_global
    }
}
