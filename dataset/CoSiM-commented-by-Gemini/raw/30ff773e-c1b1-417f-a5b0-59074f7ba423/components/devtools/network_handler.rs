/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! This module is responsible for handling network events forwarded from the
//! main `DevtoolsInstance`. It processes HTTP requests and responses, updates
//! the state of the corresponding `NetworkEventActor`, and informs the client
//! about the new or updated network resources.

use std::net::TcpStream;
use std::sync::{Arc, Mutex};

use devtools_traits::NetworkEvent;
use serde::Serialize;

use crate::actor::ActorRegistry;
use crate::actors::browsing_context::BrowsingContextActor;
use crate::actors::network_event::NetworkEventActor;
use crate::resource::{ResourceArrayType, ResourceAvailable};

/// Represents the cause of a network request, typically initiated by a document.
#[derive(Clone, Serialize)]
pub struct Cause {
    /// The type of the cause, e.g., "document".
    #[serde(rename = "type")]
    pub type_: String,
    /// The URI of the document that initiated the request.
    #[serde(rename = "loadingDocumentUri")]
    pub loading_document_uri: Option<String>,
}

/**
 * @brief Handles a single network event (request or response).
 *
 * @param actors A shared reference to the actor registry to access and modify actors.
 * @param netevent_actor_name The name of the `NetworkEventActor` associated with this request/response pair.
 * @param connections A list of active client TCP streams to notify.
 * @param network_event The specific network event to process (either an HttpRequest or HttpResponse).
 * @param browsing_context_actor_name The name of the `BrowsingContextActor` that initiated the network activity.
 *
 * This function acts as the dispatcher for network events. It updates the state of the relevant
 * `NetworkEventActor` and then uses the `BrowsingContextActor` to send a resource notification
 * to all connected clients, informing them of the available or updated network event resource.
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
        // Case for a new HTTP request being initiated.
        NetworkEvent::HttpRequest(httprequest) => {
            // A new `NetworkEventActor` was created for this request.
            // Now, we populate it with the request data.
            let event_actor = {
                let actor = actors.find_mut::<NetworkEventActor>(&netevent_actor_name);
                actor.add_request(httprequest);
                actor.event_actor()
            };

            // Notify clients that a new "network-event" resource is available.
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
        // Case for an HTTP response being received for a request.
        NetworkEvent::HttpResponse(httpresponse) => {
            // Find the existing `NetworkEventActor` and add the response data to it.
            let resource = {
                let actor = actors.find_mut::<NetworkEventActor>(&netevent_actor_name);
                // Store the response information in the actor.
                actor.add_response(httpresponse);
                actor.resource_updates()
            };

            // Notify clients that the existing "network-event" resource has been updated
            // with response information.
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