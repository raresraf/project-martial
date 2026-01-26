/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file console.rs
 * @brief Implements the `ConsoleActor` for the devtools server, mediating interactions
 *        between the remote web console and the application's JavaScript environment.
 *        This includes handling console logs, JavaScript evaluation, autocompletion,
 *        and caching of console events.
 * Algorithm: Processes various devtools messages related to console operations.
 *            It uses an actor-based model to manage state and communicate with
 *            other actors and the script environment via IPC. Console messages
 *            and page errors are cached and reported to clients.
 * Time Complexity: Message handling is generally O(1) for dispatch, with some
 *                  operations like caching or evaluation potentially being O(N)
 *                  where N is the number of cached events or complexity of JS evaluation.
 * Space Complexity: O(N_cached_events * S_event) for storing cached console messages.
 */


use std::cell::RefCell;
use std::collections::HashMap;
use std::net::TcpStream;
use std::time::{SystemTime, UNIX_EPOCH};

use base::id::TEST_PIPELINE_ID;
use devtools_traits::EvaluateJSReply::{
    ActorValue, BooleanValue, NullValue, NumberValue, StringValue, VoidValue,
};
use devtools_traits::{
    CachedConsoleMessage, CachedConsoleMessageTypes, ConsoleLog, ConsoleMessage,
    DevtoolScriptControlMsg, PageError,
};
use ipc_channel::ipc::{self, IpcSender};
use log::debug;
use serde::Serialize;
use serde_json::{self, Map, Number, Value};
use uuid::Uuid;

use crate::actor::{Actor, ActorMessageStatus, ActorRegistry};
use crate::actors::browsing_context::BrowsingContextActor;
use crate::actors::object::ObjectActor;
use crate::actors::worker::WorkerActor;
use crate::protocol::JsonPacketStream;
use crate::resource::{ResourceArrayType, ResourceAvailable};
use crate::{StreamId, UniqueId};

/**
 * @brief Trait for encoding cached console messages into JSON strings.
 */
trait EncodableConsoleMessage {
    /// @brief Encodes the cached console message into a JSON string.
    fn encode(&self) -> serde_json::Result<String>;
}

impl EncodableConsoleMessage for CachedConsoleMessage {
    // Block Logic: Implements encoding logic for different types of CachedConsoleMessage.
    fn encode(&self) -> serde_json::Result<String> {
        match *self {
            CachedConsoleMessage::PageError(ref a) => serde_json::to_string(a),
            CachedConsoleMessage::ConsoleLog(ref a) => serde_json::to_string(a),
        }
    }
}

/**
 * @brief Marker struct for traits related to started listeners.
 */
#[derive(Serialize)]
struct StartedListenersTraits;

/**
 * @brief Reply message for a "startListeners" request.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StartedListenersReply {
    // from: The name of the actor sending the reply.
    from: String,
    // native_console_api: Indicates if the native console API is supported.
    native_console_api: bool,
    // started_listeners: A list of listeners that were successfully started.
    started_listeners: Vec<String>,
    // traits: Additional traits related to started listeners.
    traits: StartedListenersTraits,
}

/**
 * @brief Reply message for a "getCachedMessages" request.
 */
#[derive(Serialize)]
struct GetCachedMessagesReply {
    // from: The name of the actor sending the reply.
    from: String,
    // messages: A list of cached console messages.
    messages: Vec<Map<String, Value>>,
}

/**
 * @brief Reply message for a "stopListeners" request.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StopListenersReply {
    // from: The name of the actor sending the reply.
    from: String,
    // stopped_listeners: A list of listeners that were successfully stopped.
    stopped_listeners: Vec<String>,
}

/**
 * @brief Reply message for an "autocomplete" request.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AutocompleteReply {
    // from: The name of the actor sending the reply.
    from: String,
    // matches: A list of autocompletion matches.
    matches: Vec<String>,
    // match_prop: The property that was matched for autocompletion.
    match_prop: String,
}

/**
 * @brief Reply message for a synchronous "evaluateJS" request.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EvaluateJSReply {
    // from: The name of the actor sending the reply.
    from: String,
    // input: The JavaScript code that was evaluated.
    input: String,
    // result: The result of the JavaScript evaluation.
    result: Value,
    // timestamp: The timestamp of when the evaluation occurred.
    timestamp: u64,
    // exception: Any exception that occurred during evaluation.
    exception: Value,
    // exception_message: The message of any exception that occurred.
    exception_message: Value,
    // helper_result: Any helper result from the evaluation.
    helper_result: Value,
}

/**
 * @brief Event message for an asynchronous JavaScript evaluation result.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EvaluateJSEvent {
    // from: The name of the actor sending the event.
    from: String,
    // type_: The type of the event (e.g., "evaluationResult").
    #[serde(rename = "type")]
    type_: String,
    // input: The JavaScript code that was evaluated.
    input: String,
    // result: The result of the JavaScript evaluation.
    result: Value,
    // timestamp: The timestamp of when the evaluation occurred.
    timestamp: u64,
    // result_id: A unique ID linking this event to a previous asynchronous request.
    #[serde(rename = "resultID")]
    result_id: String,
    // exception: Any exception that occurred during evaluation.
    exception: Value,
    // exception_message: The message of any exception that occurred.
    exception_message: Value,
    // helper_result: Any helper result from the evaluation.
    helper_result: Value,
}

/**
 * @brief Early reply message for an asynchronous "evaluateJSAsync" request.
 */
#[derive(Serialize)]
struct EvaluateJSAsyncReply {
    // from: The name of the actor sending the reply.
    from: String,
    // result_id: A unique ID that will be used in a subsequent `EvaluateJSEvent`.
    #[serde(rename = "resultID")]
    result_id: String,
}

/**
 * @brief Reply message for a "setPreferences" request.
 */
#[derive(Serialize)]
struct SetPreferencesReply {
    // from: The name of the actor sending the reply.
    from: String,
    // updated: A list of preferences that were updated.
    updated: Vec<String>,
}

/**
 * @brief Wrapper struct for a `PageError` to be serialized.
 */
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PageErrorWrapper {
    // page_error: The `PageError` details.
    page_error: PageError,
}

/**
 * @brief Enum representing the root context for a console actor.
 *
 * A console actor can be associated with a browsing context (tab) or a dedicated worker.
 */
pub(crate) enum Root {
    /// The console actor is associated with a BrowsingContextActor.
    BrowsingContext(String),
    /// The console actor is associated with a DedicatedWorkerActor.
    DedicatedWorker(String),
}

/**
 * @brief Implements the `ConsoleActor` for the devtools server.
 *
 * This actor handles all messages related to the web console, including
 * JavaScript evaluation, autocompletion, and caching of console events.
 */
pub(crate) struct ConsoleActor {
    /// The unique name of this actor.
    pub name: String,
    /// The root context this console actor is associated with (BrowsingContext or Worker).
    pub root: Root,
    /// A mutable reference to a HashMap caching console events for different unique IDs.
    pub cached_events: RefCell<HashMap<UniqueId, Vec<CachedConsoleMessage>>>,
}

impl ConsoleActor {
    /**
     * @brief Retrieves the IPC sender for communicating with the script environment.
     * @param registry A reference to the actor registry.
     * @return A reference to the `IpcSender` for `DevtoolScriptControlMsg`.
     */
    fn script_chan<'a>(
        &self,
        registry: &'a ActorRegistry,
    ) -> &'a IpcSender<DevtoolScriptControlMsg> {
        match &self.root {
            Root::BrowsingContext(bc) => &registry.find::<BrowsingContextActor>(bc).script_chan,
            Root::DedicatedWorker(worker) => &registry.find::<WorkerActor>(worker).script_chan,
        }
    }

    /**
     * @brief Determines the unique ID for the current context (pipeline or worker).
     * @param registry A reference to the actor registry.
     * @return The `UniqueId` representing the current context.
     */
    fn current_unique_id(&self, registry: &ActorRegistry) -> UniqueId {
        match &self.root {
            Root::BrowsingContext(bc) => UniqueId::Pipeline(
                registry
                    .find::<BrowsingContextActor>(bc)
                    .active_pipeline_id
                    .get(),
            ),
            Root::DedicatedWorker(w) => UniqueId::Worker(registry.find::<WorkerActor>(w).worker_id),
        }
    }

    /**
     * @brief Evaluates JavaScript code in the associated script environment.
     *
     * Sends the JavaScript input to the script environment via IPC and
     * processes the result, converting it into a serializable JSON `Value`.
     *
     * @param registry A reference to the actor registry.
     * @param msg The message containing the JavaScript code to evaluate.
     * @return A `Result` containing `EvaluateJSReply` on success, or `()` on error.
     */
    fn evaluate_js(
        &self,
        registry: &ActorRegistry,
        msg: &Map<String, Value>,
    ) -> Result<EvaluateJSReply, ()> {
        // input: The JavaScript code string to be evaluated.
        let input = msg.get("text").unwrap().as_str().unwrap().to_owned();
        // chan, port: IPC channel for sending the evaluation request and receiving the result.
        let (chan, port) = ipc::channel().unwrap();
        // FIXME: Redesign messages so we don't have to fake pipeline ids when
        //        communicating with workers.
        // pipeline: The ID of the pipeline associated with the evaluation context.
        let pipeline = match self.current_unique_id(registry) {
            UniqueId::Pipeline(p) => p,
            UniqueId::Worker(_) => TEST_PIPELINE_ID,
        };
        // Block Logic: Send the JavaScript code to the script environment for evaluation.
        self.script_chan(registry)
            .send(DevtoolScriptControlMsg::EvaluateJS(
                pipeline,
                input.clone(),
                chan,
            ))
            .unwrap();

        // TODO: Extract conversion into protocol module or some other useful place
        // Block Logic: Process the evaluation result received from the script environment.
        let result = match port.recv().map_err(|_| ())? {
            VoidValue => {
                // Undefined value.
                let mut m = Map::new();
                m.insert("type".to_owned(), Value::String("undefined".to_owned()));
                Value::Object(m)
            },
            NullValue => {
                // Null value.
                let mut m = Map::new();
                m.insert("type".to_owned(), Value::String("null".to_owned()));
                Value::Object(m)
            },
            BooleanValue(val) => Value::Bool(val),
            NumberValue(val) => {
                // Block Logic: Handle special numeric values (NaN, Infinity, -0).
                if val.is_nan() {
                    let mut m = Map::new();
                    m.insert("type".to_owned(), Value::String("NaN".to_owned()));
                    Value::Object(m)
                } else if val.is_infinite() {
                    let mut m = Map::new();
                    if val < 0. {
                        m.insert("type".to_owned(), Value::String("-Infinity".to_owned()));
                    } else {
                        m.insert("type".to_owned(), Value::String("Infinity".to_owned()));
                    }
                    Value::Object(m)
                } else if val == 0. && val.is_sign_negative() {
                    let mut m = Map::new();
                    m.insert("type".to_owned(), Value::String("-0".to_owned()));
                    Value::Object(m)
                } else {
                    Value::Number(Number::from_f64(val).unwrap())
                }
            },
            StringValue(s) => Value::String(s),
            ActorValue { class, uuid } => {
                // Block Logic: Handle actor values, registering an `ObjectActor` for inspection.
                // TODO: Make initial ActorValue message include these properties?
                let mut m = Map::new();
                let actor = ObjectActor::register(registry, uuid);

                m.insert("type".to_owned(), Value::String("object".to_owned()));
                m.insert("class".to_owned(), Value::String(class));
                m.insert("actor".to_owned(), Value::String(actor));
                m.insert("extensible".to_owned(), Value::Bool(true));
                m.insert("frozen".to_owned(), Value::Bool(false));
                m.insert("sealed".to_owned(), Value::Bool(false));
                Value::Object(m)
            },
        };

        // TODO: Catch and return exception values from JS evaluation
        // Block Logic: Construct the `EvaluateJSReply` message.
        let reply = EvaluateJSReply {
            from: self.name(),
            input,
            result,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            exception: Value::Null,
            exception_message: Value::Null,
            helper_result: Value::Null,
        };
        std::result::Result::Ok(reply)
    }

    /**
     * @brief Handles a page error message, caches it, and reports it to the client.
     * @param page_error The `PageError` details.
     * @param id The unique ID of the context where the error occurred.
     * @param registry A reference to the actor registry.
     * @param stream A mutable reference to the `TcpStream` of the client.
     */
    pub(crate) fn handle_page_error(
        &self,
        page_error: PageError,
        id: UniqueId,
        registry: &ActorRegistry,
        stream: &mut TcpStream,
    ) {
        // Block Logic: Cache the page error.
        self.cached_events
            .borrow_mut()
            .entry(id.clone())
            .or_default()
            .push(CachedConsoleMessage::PageError(page_error.clone()));
        // Block Logic: If the error is from the current active context, report it to the client.
        if id == self.current_unique_id(registry) {
            if let Root::BrowsingContext(bc) = &self.root {
                registry.find::<BrowsingContextActor>(bc).resource_array(
                    PageErrorWrapper { page_error },
                    "error-message".into(),
                    ResourceArrayType::Available,
                    stream,
                )
            };
        }
    }

    /**
     * @brief Handles a console API message, caches it, and reports it to the client.
     * @param console_message The `ConsoleMessage` details.
     * @param id The unique ID of the context where the message originated.
     * @param registry A reference to the actor registry.
     * @param stream A mutable reference to the `TcpStream` of the client.
     */
    pub(crate) fn handle_console_api(
        &self,
        console_message: ConsoleMessage,
        id: UniqueId,
        registry: &ActorRegistry,
        stream: &mut TcpStream,
    ) {
        // Block Logic: Convert `ConsoleMessage` to `ConsoleLog` and cache it.
        let log_message: ConsoleLog = console_message.into();
        self.cached_events
            .borrow_mut()
            .entry(id.clone())
            .or_default()
            .push(CachedConsoleMessage::ConsoleLog(log_message.clone()));
        // Block Logic: If the message is from the current active context, report it to the client.
        if id == self.current_unique_id(registry) {
            if let Root::BrowsingContext(bc) = &self.root {
                registry.find::<BrowsingContextActor>(bc).resource_array(
                    log_message,
                    "console-message".into(),
                    ResourceArrayType::Available,
                    stream,
                )
            };
        }
    }
}

impl Actor for ConsoleActor {
    /// @brief Returns the unique name of this actor.
    fn name(&self) -> String {
        self.name.clone()
    }

    /**
     * @brief Handles incoming messages for the `ConsoleActor`.
     *
     * Dispatches messages based on their type to perform actions like clearing caches,
     * retrieving cached messages, starting/stopping listeners, autocompletion,
     * and JavaScript evaluation.
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
        Ok(match msg_type {
            // Block Logic: Handles clearing the cached console messages for the current unique ID.
            "clearMessagesCache" => {
                self.cached_events
                    .borrow_mut()
                    .remove(&self.current_unique_id(registry));
                ActorMessageStatus::Processed
            },

            // Block Logic: Handles retrieving cached console messages based on specified types.
            "getCachedMessages" => {
                // str_types: Iterates over requested message types (e.g., "PageError", "ConsoleAPI").
                let str_types = msg
                    .get("messageTypes")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|json_type| json_type.as_str().unwrap());
                // message_types: Bitmask to filter cached messages.
                let mut message_types = CachedConsoleMessageTypes::empty();
                // Block Logic: Populate `message_types` based on requested message type strings.
                for str_type in str_types {
                    match str_type {
                        "PageError" => message_types.insert(CachedConsoleMessageTypes::PAGE_ERROR),
                        "ConsoleAPI" => {
                            message_types.insert(CachedConsoleMessageTypes::CONSOLE_API)
                        },
                        s => debug!("unrecognized message type requested: \"{}\"", s),
                    };
                }
                // messages: Vector to store filtered and encoded cached messages.
                let mut messages = vec![];
                // Block Logic: Filter cached messages and add to `messages` vector.
                for event in self
                    .cached_events
                    .borrow()
                    .get(&self.current_unique_id(registry))
                    .unwrap_or(&vec![])
                    .iter()
                {
                    // include: Flag to determine if the current event should be included in the reply.
                    let include = match event {
                        CachedConsoleMessage::PageError(_)
                            if message_types.contains(CachedConsoleMessageTypes::PAGE_ERROR) =>
                        {
                            true
                        },
                        CachedConsoleMessage::ConsoleLog(_)
                            if message_types.contains(CachedConsoleMessageTypes::CONSOLE_API) =>
                        {
                            true
                        },
                        _ => false,
                    };
                    if include {
                        let json_string = event.encode().unwrap();
                        let json = serde_json::from_str::<Value>(&json_string).unwrap();
                        messages.push(json.as_object().unwrap().to_owned())
                    }
                }

                // Block Logic: Construct and send the reply with cached messages.
                let msg = GetCachedMessagesReply {
                    from: self.name(),
                    messages,
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },

            // Block Logic: Handles starting console listeners. (TODO: Implement actual listener filters)
            "startListeners" => {
                // listeners: List of listener types to start.
                let listeners = msg.get("listeners").unwrap().as_array().unwrap().to_owned();
                let msg = StartedListenersReply {
                    from: self.name(),
                    native_console_api: true,
                    started_listeners: listeners
                        .into_iter()
                        .map(|s| s.as_str().unwrap().to_owned())
                        .collect(),
                    traits: StartedListenersTraits,
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },

            // Block Logic: Handles stopping console listeners. (TODO: Implement actual listener filters)
            "stopListeners" => {
                let msg = StopListenersReply {
                    from: self.name(),
                    stopped_listeners: msg
                        .get("listeners")
                        .unwrap()
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .map(|listener| listener.as_str().unwrap().to_owned())
                        .collect(),
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },

            // Block Logic: Handles autocompletion requests. (TODO: Implement actual autocompletion logic)
            //TODO: implement autocompletion like onAutocomplete in
            //      http://mxr.mozilla.org/mozilla-central/source/toolkit/devtools/server/actors/webconsole.js
            "autocomplete" => {
                let msg = AutocompleteReply {
                    from: self.name(),
                    matches: vec![],
                    match_prop: "".to_owned(),
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },

            // Block Logic: Handles synchronous JavaScript evaluation requests.
            "evaluateJS" => {
                let msg = self.evaluate_js(registry, msg);
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },

            // Block Logic: Handles asynchronous JavaScript evaluation requests.
            "evaluateJSAsync" => {
                // result_id: Unique ID for the asynchronous evaluation result.
                let result_id = Uuid::new_v4().to_string();
                // early_reply: Initial reply to the client for async evaluation.
                let early_reply = EvaluateJSAsyncReply {
                    from: self.name(),
                    result_id: result_id.clone(),
                };
                // Emit an eager reply so that the client starts listening
                // for an async event with the resultID
                if stream.write_json_packet(&early_reply).is_err() {
                    return Ok(ActorMessageStatus::Processed);
                }

                // Block Logic: If eager evaluation is requested, process it (currently not fully supported for side-effects).
                if msg.get("eager").and_then(|v| v.as_bool()).unwrap_or(false) {
                    // We don't support the side-effect free evaluation that eager evalaution
                    // really needs.
                    return Ok(ActorMessageStatus::Processed);
                }

                // Block Logic: Evaluate JavaScript and send the result as an asynchronous event.
                let reply = self.evaluate_js(registry, msg).unwrap();
                let msg = EvaluateJSEvent {
                    from: self.name(),
                    type_: "evaluationResult".to_owned(),
                    input: reply.input,
                    result: reply.result,
                    timestamp: reply.timestamp,
                    result_id,
                    exception: reply.exception,
                    exception_message: reply.exception_message,
                    helper_result: reply.helper_result,
                };
                // Send the data from evaluateJS along with a resultID
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },

            // Block Logic: Handles setting console preferences. (TODO: Implement actual preference setting)
            "setPreferences" => {
                let msg = SetPreferencesReply {
                    from: self.name(),
                    updated: vec![],
                };
                let _ = stream.write_json_packet(&msg);
                ActorMessageStatus::Processed
            },

            _ => ActorMessageStatus::Ignored,
        })
    }
}
