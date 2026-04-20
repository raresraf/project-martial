/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @raw/127ee0ec-f41c-4698-b2c9-da8e34319580/components/devtools/network_handler.rs
 * @brief DevTools integration point for intercepting and broadcasting network events.
 * * Architectural Intent: Subscribes to lower-level network activity (requests and responses) 
 *   and bridges them into the DevTools actor system, enabling real-time network inspection.
 */

use std::net::TcpStream;
use std::sync::{Arc, Mutex};

use devtools_traits::NetworkEvent;
use serde::Serialize;

use crate::actor::ActorRegistry;
use crate::actors::browsing_context::BrowsingContextActor;
use crate::actors::network_event::NetworkEventActor;
use crate::resource::{ResourceArrayType, ResourceAvailable};

#[derive(Clone, Serialize)]
pub struct Cause {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(rename = "loadingDocumentUri")]
    pub loading_document_uri: Option<String>,
}

/**
 * @brief Routes incoming network events to the appropriate DevTools actors.
 * * Logic: Looks up the target `NetworkEventActor` in the registry, updates its internal 
 *   state with the new request or response payload, and broadcasts a `ResourceAvailable` 
 *   update through the `BrowsingContextActor` to connected DevTools clients.
 */
pub(crate) fn handle_network_event(
    actors: Arc<Mutex<ActorRegistry>>,
    netevent_actor_name: String,
    mut connections: Vec<TcpStream>,
    network_event: NetworkEvent,
    browsing_context_actor_name: String,
) {
    let mut actors = actors.lock().unwrap();
    match network_event {
        NetworkEvent::HttpRequest(httprequest) => {
            // Scope mutable borrow
            let event_actor = {
                let actor = actors.find_mut::<NetworkEventActor>(&netevent_actor_name);
                actor.add_request(httprequest);
                actor.event_actor()
            };

            let browsing_context_actor =
                actors.find::<BrowsingContextActor>(&browsing_context_actor_name);
            for stream in &mut connections {
                browsing_context_actor.resource_array(
                    event_actor.clone(),
                    "network-event".to_string(),
                    ResourceArrayType::Available,
                    stream,
                );
            }
        },
        NetworkEvent::HttpResponse(httpresponse) => {
            // Scope mutable borrow
            let resource = {
                let actor = actors.find_mut::<NetworkEventActor>(&netevent_actor_name);
                // Store the response information in the actor
                actor.add_response(httpresponse);
                actor.resource_updates()
            };

            let browsing_context_actor =
                actors.find::<BrowsingContextActor>(&browsing_context_actor_name);
            for stream in &mut connections {
                browsing_context_actor.resource_array(
                    resource.clone(),
                    "network-event".to_string(),
                    ResourceArrayType::Updated,
                    stream,
                );
            }
        },
    }
}
