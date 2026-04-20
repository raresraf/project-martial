/**
 * @20b8a110-a360-4508-af84-77d8f0092398/components/devtools/network_handler.rs
 * @brief Network event processing and protocol serialization for DevTools.
 * 
 * Functional Intent: Orchestrates the capturing and reporting of network activity 
 * (HTTP requests/responses) to connected DevTools clients. It manages the lifecycle 
 * of NetworkEventActors and serializes internal event data into JSON packets 
 * compliant with the remote debugging protocol.
 * 
 * Domain: Browser DevTools, Remote Debugging Protocol, Network Monitoring.
 */

use crate::actor::ActorRegistry;
use crate::actors::network_event::{EventActor, NetworkEventActor, ResponseStartMsg};
use crate::protocol::JsonPacketStream;
use devtools_traits::NetworkEvent;
use serde::Serialize;
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

/**
 * @brief Container for initial network event announcements.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NetworkEventMsg {
    from: String,
    #[serde(rename = "type")]
    type_: String,
    event_actor: EventActor,
}

/**
 * @brief Generic template for incremental network event updates.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NetworkEventUpdateMsg {
    from: String,
    #[serde(rename = "type")]
    type_: String,
    update_type: String,
}

/**
 * @brief Specialized update message for the start of an HTTP response.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResponseStartUpdateMsg {
    from: String,
    #[serde(rename = "type")]
    type_: String,
    update_type: String,
    response: ResponseStartMsg,
}

/**
 * @brief Metadata update for performance profiling (timings).
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EventTimingsUpdateMsg {
    total_time: u64,
}

/**
 * @brief Metadata update for connection security status.
 */
#[derive(Serialize)]
struct SecurityInfoUpdateMsg {
    state: String,
}

/**
 * handle_network_event - Processes a single NetworkEvent and notifies all clients.
 * @actors: Thread-safe registry of active debug actors.
 * @console_actor_name: Identifier for the parent console context.
 * @netevent_actor_name: Identifier for the specific network event tracker.
 * @connections: Active TCP streams to remote DevTools clients.
 * @network_event: The raw HTTP event (Request or Response) to process.
 * 
 * Block Logic: State transition handling for HTTP transactions.
 * Logic: 
 * 1. Acquires a lock on the actor registry and retrieves the target NetworkEventActor.
 * 2. If HttpRequest: Registers the request and broadcasts the event actor metadata.
 * 3. If HttpResponse: Finalizes the transaction, calculating timings and broadcasting 
 *    comprehensive metadata updates (headers, cookies, security, content).
 */
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
            // Functional Utility: Persists request-phase metadata in the actor state.
            actor.add_request(httprequest);

            // Block Logic: Initial client notification.
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
            // Functional Utility: Finalizes the HTTP lifecycle in the actor state.
            actor.add_response(httpresponse);

            // Block Logic: Sequential broadcast of response metadata.
            // Logic: Sends multiple protocol-level updates to populate different 
            // tabs in the DevTools UI (Headers, Cookies, Timings, etc.).

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

            let msg = ResponseStartUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseStart".to_owned(),
                response: actor.response_start(),
            };
            for stream in &mut connections {
                let _ = stream.write_json_packet(&msg);
            }

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

            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseContent".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &actor.response_content());
            }

            let msg = NetworkEventUpdateMsg {
                from: netevent_actor_name.clone(),
                type_: "networkEventUpdate".to_owned(),
                update_type: "responseCookies".to_owned(),
            };
            for stream in &mut connections {
                let _ = stream.write_merged_json_packet(&msg, &actor.response_cookies());
            }

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
