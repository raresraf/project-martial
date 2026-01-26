/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file network_handler.rs
 * @brief This file provides functionality for handling network events within the devtools server.
 *        It processes HTTP requests and responses, updates the state of `NetworkEventActor` instances,
 *        and notifies connected devtools clients about network activity.
 * Algorithm: Dispatches incoming `NetworkEvent` messages (`HttpRequest` or `HttpResponse`)
 *            to the appropriate `NetworkEventActor`. It then informs the relevant
 *            `BrowsingContextActor` and clients about resource changes.
 * Time Complexity: Primarily depends on the number of active client connections and the
 *                  frequency of network events. Individual event handling is generally O(1)
 *                  or O(N_connections) where N_connections is the number of active clients.
 * Space Complexity: O(1) per event, not accounting for the state stored within actors.
 */

use std::net::TcpStream;
use std::sync::{Arc, Mutex};

use devtools_traits::NetworkEvent;
use serde::Serialize;

use crate::actor::ActorRegistry;
use crate::actors::browsing_context::BrowsingContextActor;
use crate::actors::network_event::NetworkEventActor;
use crate::resource::{ResourceArrayType, ResourceAvailable};

/**
 * @brief Represents the cause of a network event.
 *
 * This struct provides information about what triggered a network request,
 * including its type and the URI of the loading document.
 */
#[derive(Clone, Serialize)]
pub struct Cause {
    // type_: The type of the cause (e.g., "document", "script").
    #[serde(rename = "type")]
    pub type_: String,
    // loading_document_uri: The URI of the document that initiated the network request.
    #[serde(rename = "loadingDocumentUri")]
    pub loading_document_uri: Option<String>,
}

/**
 * @brief Handles incoming network events (HTTP requests and responses).
 *
 * This function processes `NetworkEvent` messages by updating the state of the
 * `NetworkEventActor` and notifying relevant `BrowsingContextActor`s and connected
 * devtools clients about the network activity.
 *
 * @param actors A shared, mutable reference to the `ActorRegistry`.
 * @param netevent_actor_name The name of the `NetworkEventActor` responsible for this event.
 * @param connections A vector of active `TcpStream`s to devtools clients.
 * @param network_event The specific `NetworkEvent` to be handled (HttpRequest or HttpResponse).
 * @param browsing_context_actor_name The name of the `BrowsingContextActor` associated with this event.
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
            // Block Logic: Handle an HTTP request event.
            // Retrieve and update the `NetworkEventActor` with the request details.
            let event_actor = {
                let actor = actors.find_mut::<NetworkEventActor>(&netevent_actor_name);
                actor.add_request(httprequest);
                actor.event_actor()
            };

            // Block Logic: Notify connected clients via the `BrowsingContextActor` about the new network event.
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
            // Block Logic: Handle an HTTP response event.
            // Retrieve and update the `NetworkEventActor` with the response details.
            let resource = {
                let actor = actors.find_mut::<NetworkEventActor>(&netevent_actor_name);
                // Store the response information in the actor
                actor.add_response(httpresponse);
                actor.resource_updates()
            };

            // Block Logic: Notify connected clients via the `BrowsingContextActor` about the updated network resource.
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
