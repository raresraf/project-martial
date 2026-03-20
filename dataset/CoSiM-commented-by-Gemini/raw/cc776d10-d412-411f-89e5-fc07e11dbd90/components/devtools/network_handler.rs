/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! # Network Event Handler
//!
//! This module is responsible for processing network events (`HttpRequest`, `HttpResponse`)
//! from the underlying web engine. It acts as a bridge between the engine's network
//! activity and the remote developer tools client.
//!
//! The primary function, `handle_network_event`, takes a network event, updates the
//! state of the corresponding `NetworkEventActor`, and broadcasts a series of
//! serialized JSON messages to all connected devtools clients to reflect the
//! state of the network request in the UI.

use std::net::TcpStream;
use std::sync::{Arc, Mutex};

use devtools_traits::NetworkEvent;
use serde::Serialize;

use crate::actor::ActorRegistry;
use crate::actors::network_event::{EventActor, NetworkEventActor, ResponseStartMsg};
use crate::protocol::JsonPacketStream;

/// Represents the initial message sent to the devtools client when a new network
/// request is detected. It informs the client about the new `event_actor` that will
/// manage the lifecycle of this specific request.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NetworkEventMsg {
    from: String,
    #[serde(rename = "type")]
    type_: String,
    event_actor: EventActor,
}

/// A generic update message for an existing network event. The `update_type` field
/// specifies which part of the network event is being updated (e.g., "requestHeaders").
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NetworkEventUpdateMsg {
    from: String,
    #[serde(rename = "type")]
    type_: String,
    update_type: String,
}

/// An update message specifically for the "responseStart" event, which includes
/// initial response data like status codes and headers.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResponseStartUpdateMsg {
    from: String,
    #[serde(rename = "type")]
    type_: String,
    update_type: String,
    response: ResponseStartMsg,
}

/// An update message for the "eventTimings" part of a network request,
/// detailing the total time taken.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EventTimingsUpdateMsg {
    total_time: u64,
}

/// An update message for the security information of a response.
#[derive(Serialize)]
struct SecurityInfoUpdateMsg {
    state: String,
}

/// Processes a single `NetworkEvent`, updates the corresponding actor, and broadcasts
/// messages to all connected developer tools clients.
///
/// # Arguments
///
/// * `actors` - A shared, mutable registry of all actors in the system.
/// * `console_actor_name` - The identifier for the parent console actor.
/// * `netevent_actor_name` - The identifier for the actor handling this specific network event.
/// * `connections` - A vector of active TCP streams to developer tools clients.
/// * `network_event` - The `NetworkEvent` enum instance to be processed.
pub fn handle_network_event(
    actors: Arc<Mutex<ActorRegistry>>,
    console_actor_name: String,
    netevent_actor_name: String,
    mut connections: Vec<TcpStream>,
    network_event: NetworkEvent,
) {
    // Lock the actor registry to gain mutable access. Panics if the mutex is poisoned.
    let mut actors = actors.lock().unwrap();
    // Find the specific actor responsible for this network event.
    let actor = actors.find_mut::<NetworkEventActor>(&netevent_actor_name);

    match network_event {
        NetworkEvent::HttpRequest(httprequest) => {
            // This arm handles the beginning of a network request.

            // Store the request information within the actor for future reference.
            actor.add_request(httprequest);

            // Create and send a `networkEvent` message to the client. This notifies the UI
            // that a new request has started and provides the actor details to track it.
            let msg = NetworkEventMsg {
                from: console_actor_name,
                type_: "networkEvent".to_owned(),
                event_actor: actor.event_actor(),
            };
            for stream in &mut connections {
                let _ = stream.write_json_packet(&msg);
            }
        },
        NetworkEvent::HttpResponse(httpresponse) => {
            // This arm handles the arrival of the response and completion of the network event.
            // It sends a series of updates to the client to populate all the details
            // in the network panel UI.

            // Store the core response information in the actor.
            actor.add_response(httpresponse);

            // --- Send Request-related Updates ---
            // Now that the response is here, we also have final details about the request
            // that we can send.

            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "requestHeaders".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &actor.request_headers());
            }

            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "requestCookies".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &actor.request_cookies());
            }

            // --- Send Response-related Updates ---

            // Send a `responseStart` update, which includes status, headers, etc.
            let msg = ResponseStartUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseStart".to_owned(),
                response: actor.response_start(),
            };
            for stream in &mut connections {
                let _ = stream.write_json_packet(&msg);
            }

            // Send timing information.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "eventTimings".to_owned(),
            };
            let extra = EventTimingsUpdateMsg {
                total_time: actor.total_time().as_millis() as u64,
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &extra);
            }

            // Send security information (e.g., TLS state). Here, it is hardcoded.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "securityInfo".to_owned(),
            };
            let extra = SecurityInfoUpdateMsg {
                state: "insecure".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &extra);
            }

            // Send the actual response body.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseContent".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &actor.response_content());
            }

            // Send cookies set by the response.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseCookies".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &actor.response_cookies());
            }

            // Send final response headers.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name,
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseHeaders".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &actor.response_headers());
            }
        },
    }
}
