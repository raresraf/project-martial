//! # Network Event Handler for DevTools
//!
//! This module is responsible for processing network events (HTTP requests and responses)
//! and communicating them to a DevTools client. It acts as a bridge between the
//! application's network layer and a remote debugging interface, serializing
//! event data into JSON messages and sending them over TCP streams.

use crate::actor::ActorRegistry;
use crate::actors::network_event::{EventActor, NetworkEventActor, ResponseStartMsg};
use crate::protocol::JsonPacketStream;
use devtools_traits::NetworkEvent;
use serde::Serialize;
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

/// Represents the initial message sent when a new network request is detected.
/// It contains the `EventActor` which encapsulates the state of the network event.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NetworkEventMsg {
    from: String,
    #[serde(rename = "type")]
    type_: String,
    event_actor: EventActor,
}

/// A generic message used to send updates about a network event.
/// The `update_type` field specifies which part of the event is being updated
/// (e.g., request headers, response content).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NetworkEventUpdateMsg {
    from: String,
    #[serde(rename = "type")]
    type_: String,
    update_type: String,
}

/// A specialized update message for when the initial part of an HTTP response is received.
/// It includes the `ResponseStartMsg`, which contains details like status code and headers.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResponseStartUpdateMsg {
    from: String,
    #[serde(rename = "type")]
    type_: String,
    update_type: String,
    response: ResponseStartMsg,
}

/// An update message for sending event timing information.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EventTimingsUpdateMsg {
    total_time: u64,
}

/// An update message for sending security information related to the request.
#[derive(Serialize)]
struct SecurityInfoUpdateMsg {
    state: String,
}

/// Handles incoming network events, updates the corresponding actor state, and
/// notifies connected DevTools clients.
///
/// # Arguments
///
/// * `actors` - A registry of all actors in the system, used to find the `NetworkEventActor`.
/// * `console_actor_name` - The name of the console actor, used as the `from` field in some messages.
/// * `netevent_actor_name` - The name of the network event actor being updated.
/// * `connections` - A list of TCP streams connected to DevTools clients.
/// * `network_event` - The `NetworkEvent` to be processed (either a request or a response).
pub fn handle_network_event(
    actors: Arc<Mutex<ActorRegistry>>,
    console_actor_name: String,
    netevent_actor_name: String,
    mut connections: Vec<TcpStream>,
    network_event: NetworkEvent,
) {
    let mut actors = actors.lock().unwrap();
    let actor = actors.find_mut::<NetworkEventActor>(&netevent_actor_name);

    match network_event {
        NetworkEvent::HttpRequest(httprequest) => {
            // Store the request information in the actor.
            actor.add_request(httprequest);

            // Notify the client that a new network event has occurred.
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
            // Store the response information in the actor.
            actor.add_response(httpresponse);

            // Sequentially send updates to the client, detailing different parts of the response.
            
            // Update request headers.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "requestHeaders".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &actor.request_headers());
            }

            // Update request cookies.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "requestCookies".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &actor.request_cookies());
            }

            // Send the initial response information (status, headers).
            let msg = ResponseStartUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseStart".to_owned(),
                response: actor.response_start(),
            };

            for stream in &mut connections {
                let _ = stream.write_json_packet(&msg);
            }

            // Send timing information for the event.
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

            // Send security details of the connection.
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

            // Send the content of the response body.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseContent".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &actor.response_content());
            }

            // Send cookies from the response.
            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseCookies".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &actor.response_cookies());
            }

            // Send response headers.
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
