/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// @file network_handler.rs
/// @brief Handles network events (HTTP requests and responses) and dispatches them to DevTools clients.
///
/// This module is part of a DevTools implementation. It processes `NetworkEvent`s,
/// updates relevant `NetworkEventActor`s, and serializes event data into
/// JSON messages according to a DevTools protocol, sending these messages
/// over TCP streams to connected clients.
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

use devtools_traits::NetworkEvent;
use serde::Serialize;

use crate::actor::ActorRegistry;
use crate::actors::network_event::{EventActor, NetworkEventActor, ResponseStartMsg};
use crate::protocol::JsonPacketStream;

/// @brief Represents a general network event message to be sent to a DevTools client.
/// @note Fields are renamed to camelCase during serialization for protocol compatibility.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NetworkEventMsg {
    from: String,          ///< @brief The sender of the message.
    #[serde(rename = "type")]
    type_: String,         ///< @brief The type of the message (e.g., "networkEvent").
    event_actor: EventActor,///< @brief The actor associated with the network event.
}

/// @brief Represents a general network event update message.
/// @note Fields are renamed to camelCase during serialization for protocol compatibility.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NetworkEventUpdateMsg {
    from: String,          ///< @brief The sender of the update.
    #[serde(rename = "type")]
    type_: String,         ///< @brief The type of the message (e.g., "networkEventUpdate").
    update_type: String,   ///< @brief The specific type of update (e.g., "requestHeaders").
}

/// @brief Represents a network event update message specifically for a response start.
/// @note Fields are renamed to camelCase during serialization for protocol compatibility.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResponseStartUpdateMsg {
    from: String,          ///< @brief The sender of the update.
    #[serde(rename = "type")]
    type_: String,         ///< @brief The type of the message (e.g., "networkEventUpdate").
    update_type: String,   ///< @brief The specific type of update (e.g., "responseStart").
    response: ResponseStartMsg,///< @brief Detailed response start information.
}

/// @brief Represents an update message containing event timing information.
#[derive(Serialize)]
struct EventTimingsUpdateMsg {
    total_time: u64,       ///< @brief The total time in milliseconds for the event.
}

/// @brief Represents an update message containing security information.
#[derive(Serialize)]
struct SecurityInfoUpdateMsg {
    state: String,         ///< @brief The security state (e.g., "insecure", "secure").
}

pub fn handle_network_event(
    actors: Arc<Mutex<ActorRegistry>>,
    console_actor_name: String,
    netevent_actor_name: String,
    mut connections: Vec<TcpStream>,
    network_event: NetworkEvent,
) {
    /// @brief Handles an incoming network event, updates the relevant actor, and dispatches messages to DevTools clients.
    ///
    /// This function acts as the central dispatcher for network events. It locks the `ActorRegistry`,
    /// finds the appropriate `NetworkEventActor`, and then processes the event based on its type.
    /// For HTTP requests, it stores the request and sends a `networkEvent` message.
    /// For HTTP responses, it stores the response and sends a series of `networkEventUpdate` messages.
    ///
    /// @param actors A thread-safe reference to the `ActorRegistry` for managing DevTools actors.
    /// @param console_actor_name The name of the console actor, used in message `from` fields.
    /// @param netevent_actor_name The name of the network event actor, used to identify and find the actor.
    /// @param connections A mutable vector of `TcpStream`s representing active client connections.
    /// @param network_event The `NetworkEvent` to be processed (either `HttpRequest` or `HttpResponse`).
    let mut actors = actors.lock().unwrap(); // Acquire a lock on the ActorRegistry for exclusive access.
    // Find the mutable NetworkEventActor corresponding to the `netevent_actor_name`.
    let actor = actors.find_mut::<NetworkEventActor>(&netevent_actor_name);

    // Match on the type of network event to handle requests and responses differently.
    match network_event {
        // Block Logic: Handles an incoming HTTP request.
        NetworkEvent::HttpRequest(httprequest) => {
            // Store the request information in the actor for later retrieval or correlation.
            actor.add_request(httprequest);

            // Send a `networkEvent` message to all connected DevTools clients.
            // This message signals the start of a new network request.
            let msg = NetworkEventMsg {
                from: console_actor_name, // The sender of this message.
                type_: "networkEvent".to_owned(), // The protocol message type.
                event_actor: actor.event_actor(), // The actor representing this specific network event.
            };
            // Iterate through all client connections and send the serialized JSON message.
            for stream in &mut connections {
                let _ = stream.write_json_packet(&msg); // Ignore potential write errors.
            }
        },
        },
        NetworkEvent::HttpResponse(httpresponse) => {
            // Block Logic: Handles an incoming HTTP response.
            // Store the response information in the actor.
            actor.add_response(httpresponse);

            // Send `networkEventUpdate` message for request headers.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "requestHeaders".to_owned(), // Specific update type.
            };
            for stream in &mut connections {
                // `write_merged_json_packet` sends the message merged with the actor's request headers.
                let _ = stream.write_merged_json_packet(&msg, &actor.request_headers());
            }

            // Send `networkEventUpdate` message for request cookies.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "requestCookies".to_owned(), // Specific update type.
            };
            for stream in &mut connections {
                // Sends the message merged with the actor's request cookies.
                let _ = stream.write_merged_json_packet(&msg, &actor.request_cookies());
            }

            // Send a `networkEventUpdate` message for `responseStart`.
            // This message signals that the response's headers have been received.
            let msg = ResponseStartUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseStart".to_owned(), // Specific update type.
                response: actor.response_start(), // Detailed response start information.
            };

            for stream in &mut connections {
                let _ = stream.write_json_packet(&msg);
            }
            
            // Send `networkEventUpdate` message for `eventTimings`.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "eventTimings".to_owned(), // Specific update type.
            };
            // Create an extra data payload for event timings.
            let extra = EventTimingsUpdateMsg {
                total_time: actor.total_time().as_millis() as u64, // Total time as u64.
            };
            for stream in &mut connections {
                // Sends the message merged with the extra timing information.
                let _ = stream.write_merged_json_packet(&msg, &extra);
            }

            // Send `networkEventUpdate` message for `securityInfo`.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "securityInfo".to_owned(), // Specific update type.
            };
            // Create a placeholder extra data payload for security info (hardcoded "insecure").
            let extra = SecurityInfoUpdateMsg {
                state: "insecure".to_owned(),
            };
            for stream in &mut connections {
                // Sends the message merged with the extra security information.
                let _ = stream.write_merged_json_packet(&msg, &extra);
            }

            // Send `networkEventUpdate` message for `responseContent`.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseContent".to_owned(), // Specific update type.
            };
            for stream in &mut connections {
                // Sends the message merged with the actor's response content.
                let _ = stream.write_merged_json_packet(&msg, &actor.response_content());
            }

            // Send `networkEventUpdate` message for `responseCookies`.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseCookies".to_owned(), // Specific update type.
            };
            for stream in &mut connections {
                // Sends the message merged with the actor's response cookies.
                let _ = stream.write_merged_json_packet(&msg, &actor.response_cookies());
            }

            // Send `networkEventUpdate` message for `responseHeaders`.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name, // No clone needed here as it's the last use.
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseHeaders".to_owned(), // Specific update type.
            };
            for stream in &mut connections {
                // Sends the message merged with the actor's response headers.
                let _ = stream.write_merged_json_packet(&msg, &actor.response_headers());
            }
        },
    }
}
