/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! The watcher is the main entry point when debugging an element. Right now only web views are supported.
//! It talks to the devtools remote and lists the capabilities of the inspected target, and it serves
//! as a bridge for messages between actors.
//!
//! Liberally derived from the [Firefox JS implementation].
//!
//! [Firefox JS implementation]: https://searchfox.org/mozilla-central/source/devtools/server/actors/descriptors/watcher.js

use std::collections::HashMap;
use std::net::TcpStream;
use std::time::{SystemTime, UNIX_EPOCH};

use log::warn;
use serde::Serialize;
use serde_json::{Map, Value};

use self::network_parent::{NetworkParentActor, NetworkParentActorMsg};
use super::breakpoint::BreakpointListActor;
use super::thread::ThreadActor;
use super::worker::WorkerMsg;
use crate::actor::{Actor, ActorMessageStatus, ActorRegistry};
use crate::actors::browsing_context::{BrowsingContextActor, BrowsingContextActorMsg};
use crate::actors::root::RootActor;
use crate::actors::watcher::target_configuration::{
    TargetConfigurationActor, TargetConfigurationActorMsg,
};
use crate::actors::watcher::thread_configuration::{
    ThreadConfigurationActor, ThreadConfigurationActorMsg,
};
use crate::protocol::JsonPacketStream;
use crate::resource::{ResourceArrayType, ResourceAvailable};
use crate::{EmptyReplyMsg, StreamId, WorkerActor};

pub mod network_parent;
pub mod target_configuration;
pub mod thread_configuration;

/// Describes the debugged context. It informs the server of which objects can be debugged.
/// <https://searchfox.org/mozilla-central/source/devtools/server/actors/watcher/session-context.js>
/**
 * @brief Describes the debugged context and informs the server about debuggable objects.
 *
 * This struct represents the session context for debugging, specifying supported
 * target types (e.g., frames, workers) and resource types that can be monitored.
 * It is derived from the Firefox JS implementation.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionContext {
    // is_server_target_switching_enabled: Flag indicating if server-side target switching is enabled.
    is_server_target_switching_enabled: bool,
    // supported_targets: A map indicating which target types (e.g., "frame", "worker") are supported for debugging.
    supported_targets: HashMap<&'static str, bool>,
    // supported_resources: A map indicating which resource types (e.g., "console-message", "source") are supported.
    supported_resources: HashMap<&'static str, bool>,
    // context_type: The specific type of the session context.
    context_type: SessionContextType,
}

impl SessionContext {
    /**
     * @brief Creates a new `SessionContext` instance with default supported targets and resources.
     *
     * Initializes a new debugging session context, setting up default capabilities for
     * target debugging (currently only web views are fully supported) and resource monitoring.
     *
     * @param context_type The specific type of the session context.
     * @return A new `SessionContext` instance.
     */
    pub fn new(context_type: SessionContextType) -> Self {
        Self {
            is_server_target_switching_enabled: false,
            // Block Logic: Define supported target types. Currently, only "frame" and "worker" are truly supported.
            supported_targets: HashMap::from([
                ("frame", true),
                ("process", false),
                ("worker", true),
                ("service_worker", false),
                ("shared_worker", false),
            ]),
            // Block Logic: Define supported resource types. Most resources are initially blocked to avoid errors.
            // Support for them will be enabled gradually once the corresponding actors start working propperly.
            supported_resources: HashMap::from([
                ("console-message", true),
                ("css-change", true),
                ("css-message", false),
                ("css-registered-properties", false),
                ("document-event", false),
                ("Cache", false),
                ("cookies", false),
                ("error-message", true),
                ("extension-storage", false),
                ("indexed-db", false),
                ("local-storage", false),
                ("session-storage", false),
                ("platform-message", false),
                ("network-event", false),
                ("network-event-stacktrace", false),
                ("reflow", false),
                ("stylesheet", false),
                ("source", true),
                ("thread-state", false),
                ("server-sent-event", false),
                ("websocket", false),
                ("jstracer-trace", false),
                ("jstracer-state", false),
                ("last-private-context-exit", false),
            ]),
            context_type,
        }
    }
}

/**
 * @brief Defines the type of session context for debugging.
 */
#[derive(Serialize)]
pub enum SessionContextType {
    /// Context for a browser element.
    BrowserElement,
    /// Context for a process.
    _ContextProcess,
    /// Context for a web extension.
    _WebExtension,
    /// Context for a worker.
    _Worker,
    /// All contexts.
    _All,
}

/**
 * @brief Enum representing different types of target actors.
 */
#[derive(Serialize)]
#[serde(untagged)]
enum TargetActorMsg {
    /// Message for a BrowsingContextActor.
    BrowsingContext(BrowsingContextActorMsg),
    /// Message for a WorkerActor.
    Worker(WorkerMsg),
}

/**
 * @brief Reply message for a "watchTargets" request.
 */
#[derive(Serialize)]
struct WatchTargetsReply {
    // from: The name of the actor sending the reply.
    from: String,
    // type_: The type of the reply, indicating a target is available.
    #[serde(rename = "type")]
    type_: String,
    // target: The target actor message.
    target: TargetActorMsg,
}

/**
 * @brief Reply message containing the ID of the parent browsing context.
 */
#[derive(Serialize)]
struct GetParentBrowsingContextIDReply {
    // from: The name of the actor sending the reply.
    from: String,
    // browsing_context_id: The ID of the parent browsing context.
    #[serde(rename = "browsingContextID")]
    browsing_context_id: u32,
}

/**
 * @brief Reply message containing the network parent actor.
 */
#[derive(Serialize)]
struct GetNetworkParentActorReply {
    // from: The name of the actor sending the reply.
    from: String,
    // network: The network parent actor message.
    network: NetworkParentActorMsg,
}

/**
 * @brief Reply message containing the target configuration actor.
 */
#[derive(Serialize)]
struct GetTargetConfigurationActorReply {
    // from: The name of the actor sending the reply.
    from: String,
    // configuration: The target configuration actor message.
    configuration: TargetConfigurationActorMsg,
}

/**
 * @brief Reply message containing the thread configuration actor.
 */
#[derive(Serialize)]
struct GetThreadConfigurationActorReply {
    // from: The name of the actor sending the reply.
    from: String,
    // configuration: The thread configuration actor message.
    configuration: ThreadConfigurationActorMsg,
}

/**
 * @brief Reply message containing the breakpoint list actor.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GetBreakpointListActorReply {
    // from: The name of the actor sending the reply.
    from: String,
    // breakpoint_list: The inner breakpoint list actor reply.
    breakpoint_list: GetBreakpointListActorReplyInner,
}

/**
 * @brief Inner struct for `GetBreakpointListActorReply` containing the actor name.
 */
#[derive(Serialize)]
struct GetBreakpointListActorReplyInner {
    // actor: The actor ID of the breakpoint list.
    actor: String,
}

/**
 * @brief Represents a document event, such as loading or completion.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DocumentEvent {
    // has_native_console_api: Indicates if the native console API is available.
    #[serde(rename = "hasNativeConsoleAPI")]
    has_native_console_api: Option<bool>,
    // name: The name of the event (e.g., "dom-loading", "dom-complete").
    name: String,
    // new_uri: The new URI of the document after navigation, if applicable.
    #[serde(rename = "newURI")]
    new_uri: Option<String>,
    // time: The timestamp of the event.
    time: u64,
    // title: The title of the document, if available.
    title: Option<String>,
    // url: The URL of the document.
    url: Option<String>,
}

/**
 * @brief Defines the capabilities and supported features of the watcher.
 */
#[derive(Serialize)]
struct WatcherTraits {
    // resources: A map indicating which resource types are supported.
    resources: HashMap<&'static str, bool>,
    // targets: A map indicating which target types are supported.
    #[serde(flatten)]
    targets: HashMap<&'static str, bool>,
}

/**
 * @brief Represents an encodable message for the `WatcherActor`.
 */
#[derive(Serialize)]
pub struct WatcherActorMsg {
    // actor: The actor ID of the watcher.
    actor: String,
    // traits: The `WatcherTraits` indicating supported resources and targets.
    traits: WatcherTraits,
}

/**
 * @brief Implements the `WatcherActor` for the devtools server.
 *
 * This actor serves as the main entry point for debugging elements, currently
 * primarily supporting web views. It bridges messages between different actors
 * and communicates capabilities to the devtools remote.
 */
pub struct WatcherActor {
    // name: The unique name of this actor.
    name: String,
    // browsing_context_actor: The actor ID of the associated browsing context.
    browsing_context_actor: String,
    // network_parent: The actor ID of the network parent actor.
    network_parent: String,
    // target_configuration: The actor ID of the target configuration actor.
    target_configuration: String,
    // thread_configuration: The actor ID of the thread configuration actor.
    thread_configuration: String,
    // session_context: The session context describing debuggable capabilities.
    session_context: SessionContext,
}

impl Actor for WatcherActor {
    /// @brief Returns the unique name of this actor.
    fn name(&self) -> String {
        self.name.clone()
    }

    /**
     * @brief Handles incoming messages for the `WatcherActor`.
     *
     * This method dispatches messages based on their type to perform actions
     * such as watching targets and resources, retrieving parent browsing context IDs,
     * and obtaining various configuration actors.
     *
     * @param registry A reference to the actor registry.
     * @param msg_type The type of the incoming message.
     * @param msg The message payload as a JSON `Map`.
     * @param stream A mutable reference to the `TcpStream` of the client.
     * @param _id The `StreamId` of the client connection.
     * @return A `Result` indicating whether the message was processed or ignored.
     */
    fn handle_message(
        &self,
        registry: &ActorRegistry,
        msg_type: &str,
        msg: &Map<String, Value>,
        stream: &mut TcpStream,
        _id: StreamId,
    ) -> Result<ActorMessageStatus, ()> {
        // target: The BrowsingContextActor associated with this watcher.
        let target = registry.find::<BrowsingContextActor>(&self.browsing_context_actor);
        // root: The RootActor instance.
        let root = registry.find::<RootActor>("root");
        Ok(match msg_type {
            // Block Logic: Handles "watchTargets" requests.
            "watchTargets" => {
                // target_type: The type of target to watch (e.g., "frame", "worker").
                let target_type = msg
                    .get("targetType")
                    .and_then(Value::as_str)
                    .unwrap_or("frame"); // default to "frame"

                // Block Logic: If the target type is "frame", return the BrowsingContextActor.
                if target_type == "frame" {
                    let msg = WatchTargetsReply {
                        from: self.name(),
                        type_: "target-available-form".into(),
                        target: TargetActorMsg::BrowsingContext(target.encodable()),
                    };
                    let _ = stream.write_json_packet(&msg);

                    target.frame_update(stream);
                // Block Logic: If the target type is "worker", return all WorkerActors.
                } else if target_type == "worker" {
                    for worker_name in &root.workers {
                        let worker = registry.find::<WorkerActor>(worker_name);
                        let worker_msg = WatchTargetsReply {
                            from: self.name(),
                            type_: "target-available-form".into(),
                            target: TargetActorMsg::Worker(worker.encodable()),
                        };
                        let _ = stream.write_json_packet(&worker_msg);
                    }
                } else {
                    warn!("Unexpected target_type: {}", target_type);
                    return Ok(ActorMessageStatus::Ignored);
                }

                // Messages that contain a `type` field are used to send event callbacks, but they
                // don't count as a reply. Since every message needs to be responded, we send an
                // extra empty packet to the devtools host to inform that we successfully received
                // and processed the message so that it can continue
                // Block Logic: Send an empty reply to acknowledge message processing.
                let msg = EmptyReplyMsg { from: self.name() };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles "watchResources" requests.
            "watchResources" => {
                // resource_types: The types of resources to watch.
                let Some(resource_types) = msg.get("resourceTypes") else {
                    return Ok(ActorMessageStatus::Ignored);
                };
                let Some(resource_types) = resource_types.as_array() else {
                    return Ok(ActorMessageStatus::Ignored);
                };

                // Block Logic: Iterate through requested resource types and handle them accordingly.
                for resource in resource_types {
                    let Some(resource) = resource.as_str() else {
                        continue;
                    };
                    match resource {
                        // Block Logic: Handle "document-event" resources.
                        "document-event" => {
                            // TODO: This is a hacky way of sending the 3 messages
                            //       Figure out if there needs work to be done here, ensure the page is loaded
                            for &name in ["dom-loading", "dom-interactive", "dom-complete"].iter() {
                                let event = DocumentEvent {
                                    has_native_console_api: Some(true),
                                    name: name.into(),
                                    new_uri: None,
                                    time: SystemTime::now()
                                        .duration_since(UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_millis()
                                        as u64,
                                    title: Some(target.title.borrow().clone()),
                                    url: Some(target.url.borrow().clone()),
                                };
                                target.resource_array(
                                    event,
                                    "document-event".into(),
                                    ResourceArrayType::Available,
                                    stream,
                                );
                            }
                        },
                        // Block Logic: Handle "source" resources.
                        "source" => {
                            let thread_actor = registry.find::<ThreadActor>(&target.thread);
                            target.resources_array(
                                thread_actor.source_manager.source_forms(registry),
                                "source".into(),
                                ResourceArrayType::Available,
                                stream,
                            );

                            for worker_name in &root.workers {
                                let worker = registry.find::<WorkerActor>(worker_name);
                                let thread = registry.find::<ThreadActor>(&worker.thread);

                                worker.resources_array(
                                    thread.source_manager.source_forms(registry),
                                    "source".into(),
                                    ResourceArrayType::Available,
                                    stream,
                                );
                            }
                        },
                        // Block Logic: Handle "console-message" and "error-message" resources.
                        "console-message" | "error-message" => {},
                        // Block Logic: Log a warning for unhandled resource types.
                        _ => warn!("resource {} not handled yet", resource),
                    }

                    let msg = EmptyReplyMsg { from: self.name() };
                    let _ = stream.write_json_packet(&msg);
                }
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles "getParentBrowsingContextID" requests.
            "getParentBrowsingContextID" => {
                let msg = GetParentBrowsingContextIDReply {
                    from: self.name(),
                    browsing_context_id: target.browsing_context_id.value(),
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles "getNetworkParentActor" requests.
            "getNetworkParentActor" => {
                let network_parent = registry.find::<NetworkParentActor>(&self.network_parent);
                let msg = GetNetworkParentActorReply {
                    from: self.name(),
                    network: network_parent.encodable(),
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles "getTargetConfigurationActor" requests.
            "getTargetConfigurationActor" => {
                let target_configuration =
                    registry.find::<TargetConfigurationActor>(&self.target_configuration);
                let msg = GetTargetConfigurationActorReply {
                    from: self.name(),
                    configuration: target_configuration.encodable(),
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles "getThreadConfigurationActor" requests.
            "getThreadConfigurationActor" => {
                let thread_configuration =
                    registry.find::<ThreadConfigurationActor>(&self.thread_configuration);
                let msg = GetThreadConfigurationActorReply {
                    from: self.name(),
                    configuration: thread_configuration.encodable(),
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },
            // Block Logic: Handles "getBreakpointListActor" requests.
            "getBreakpointListActor" => {
                let breakpoint_list_name = registry.new_name("breakpoint-list");
                let breakpoint_list = BreakpointListActor::new(breakpoint_list_name.clone());
                registry.register_later(Box::new(breakpoint_list));

                let _ = stream.write_json_packet(&GetBreakpointListActorReply {
                    from: self.name(),
                    breakpoint_list: GetBreakpointListActorReplyInner {
                        actor: breakpoint_list_name,
                    },
                });
                ActorMessageStatus::Processed
            },
            // Block Logic: Ignores unhandled message types.
            _ => ActorMessageStatus::Ignored,
        })
    }
}

impl WatcherActor {
    pub fn new(
        actors: &mut ActorRegistry,
        browsing_context_actor: String,
        session_context: SessionContext,
    ) -> Self {
        let network_parent = NetworkParentActor::new(actors.new_name("network-parent"));
        let target_configuration =
            TargetConfigurationActor::new(actors.new_name("target-configuration"));
        let thread_configuration =
            ThreadConfigurationActor::new(actors.new_name("thread-configuration"));

        let watcher = Self {
            name: actors.new_name("watcher"),
            browsing_context_actor,
            network_parent: network_parent.name(),
            target_configuration: target_configuration.name(),
            thread_configuration: thread_configuration.name(),
            session_context,
        };

        actors.register(Box::new(network_parent));
        actors.register(Box::new(target_configuration));
        actors.register(Box::new(thread_configuration));

        watcher
    }

    pub fn encodable(&self) -> WatcherActorMsg {
        WatcherActorMsg {
            actor: self.name(),
            traits: WatcherTraits {
                resources: self.session_context.supported_resources.clone(),
                targets: self.session_context.supported_targets.clone(),
            },
        }
    }
}
