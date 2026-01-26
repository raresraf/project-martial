/**
 * @file lib.rs
 * @brief This file implements an actor-based remote devtools server. It orchestrates
 *        TCP communication with devtools clients, manages various actor instances
 *        for different devtools domains (e.g., console, network), and handles
 *        inter-process communication with the embedder and script environments.
 *        The server is designed to facilitate debugging and inspection of web content.
 * Algorithm: Utilizes an event-driven, actor-based model where different devtools
 *            functionalities are encapsulated in actors. It uses message passing
 *            (crossbeam channels, IPC) for communication between threads and processes.
 *            TCP listeners accept client connections, and dedicated threads handle
 *            client-specific communication.
 * Time Complexity: The overall time complexity depends heavily on the number of connected
 *                  clients, the volume of devtools messages, and the complexity of actor
 *                  processing. Individual message handling is generally O(1) or O(N)
 *                  where N is the size of the message or affected data.
 * Space Complexity: O(N_actors * S_actor + N_clients * S_client_buffer) where N_actors
 *                   is the number of active actors, S_actor is memory per actor, N_clients
 *                   is number of clients, and S_client_buffer is memory per client connection.
 */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */


#![crate_name = "devtools"]
#![crate_type = "rlib"]
#![deny(unsafe_code)]

use std::borrow::ToOwned;
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::io::Read;
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use base::id::{BrowsingContextId, PipelineId, WebViewId};
use crossbeam_channel::{Receiver, Sender, unbounded};
use devtools_traits::{
    ChromeToDevtoolsControlMsg, ConsoleMessage, ConsoleMessageBuilder, DevtoolScriptControlMsg,
    DevtoolsControlMsg, DevtoolsPageInfo, LogLevel, NavigationState, NetworkEvent, PageError,
    ScriptToDevtoolsControlMsg, SourceInfo, WorkerId,
};
use embedder_traits::{AllowOrDeny, EmbedderMsg, EmbedderProxy};
use ipc_channel::ipc::{self, IpcSender};
use log::trace;
use resource::{ResourceArrayType, ResourceAvailable};
use serde::Serialize;
use servo_rand::RngCore;

use crate::actor::{Actor, ActorRegistry};
use crate::actors::browsing_context::BrowsingContextActor;
use crate::actors::console::{ConsoleActor, Root};
use crate::actors::device::DeviceActor;
use crate::actors::framerate::FramerateActor;
use crate::actors::network_event::NetworkEventActor;
use crate::actors::performance::PerformanceActor;
use crate::actors::preference::PreferenceActor;
use crate::actors::process::ProcessActor;
use crate::actors::root::RootActor;
use crate::actors::source::SourceActor;
use crate::actors::thread::ThreadActor;
use crate::actors::worker::{WorkerActor, WorkerType};
use crate::id::IdMap;
use crate::network_handler::handle_network_event;
use crate::protocol::JsonPacketStream;

mod actor;
/// <https://searchfox.org/mozilla-central/source/devtools/server/actors>
mod actors {
    pub mod breakpoint;
    pub mod browsing_context;
    pub mod console;
    pub mod device;
    pub mod framerate;
    pub mod inspector;
    pub mod memory;
    pub mod network_event;
    pub mod object;
    pub mod performance;
    pub mod preference;
    pub mod process;
    pub mod reflow;
    pub mod root;
    pub mod source;
    pub mod stylesheets;
    pub mod tab;
    pub mod thread;
    pub mod timeline;
    pub mod watcher;
    pub mod worker;
}
mod id;
mod network_handler;
mod protocol;
mod resource;

/**
 * @brief Represents a unique identifier for a pipeline or worker, used for routing messages.
 */
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum UniqueId {
    Pipeline(PipelineId),
    Worker(WorkerId),
}

/**
 * @brief A simple empty reply message used for acknowledgements or where no data is needed.
 */
#[derive(Serialize)]
pub struct EmptyReplyMsg {
    pub from: String,
}

/**
 * @brief Spins up a devtools server that listens for connections on the specified port.
 *
 * This function initializes the main devtools server instance, spawns a new thread
 * to run the server's event loop, and returns a sender for control messages.
 *
 * @param port The TCP port on which the devtools server should listen for incoming connections.
 * @param embedder A proxy to communicate with the embedder environment.
 * @return A `Sender<DevtoolsControlMsg>` channel for sending control messages to the running server.
 */
pub fn start_server(port: u16, embedder: EmbedderProxy) -> Sender<DevtoolsControlMsg> {
    // sender, receiver: MPSC channel for sending and receiving DevtoolsControlMsg.
    let (sender, receiver) = unbounded();
    {
        let sender = sender.clone();
        // Block Logic: Spawn a new thread for the main devtools instance to run asynchronously.
        thread::Builder::new()
            .name("Devtools".to_owned())
            .spawn(move || {
                if let Some(instance) = DevtoolsInstance::create(sender, receiver, port, embedder) {
                    instance.run()
                }
            })
            .expect("Thread spawning failed");
    }
    sender
}

/// @brief Represents a unique identifier for a TCP stream connection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct StreamId(u32);

/**
 * @brief The main structure representing a running Devtools server instance.
 *
 * Manages actor registries, ID mappings, browsing contexts, pipelines, worker actors,
 * pending actor requests, and active client connections.
 */
struct DevtoolsInstance {
    // actors: A shared, mutable registry of all active actors in the devtools server.
    actors: Arc<Mutex<ActorRegistry>>,
    // id_map: A shared, mutable map for translating various IDs (e.g., WebViewId) to devtools-specific IDs.
    id_map: Arc<Mutex<IdMap>>,
    // browsing_contexts: Maps BrowsingContextId to the name of its associated BrowsingContextActor.
    browsing_contexts: HashMap<BrowsingContextId, String>,
    // receiver: The receiving end of the MPSC channel for DevtoolsControlMsg.
    receiver: Receiver<DevtoolsControlMsg>,
    // pipelines: Maps PipelineId to its associated BrowsingContextId.
    pipelines: HashMap<PipelineId, BrowsingContextId>,
    // actor_workers: Maps WorkerId to the name of its associated WorkerActor.
    actor_workers: HashMap<WorkerId, String>,
    // actor_requests: Maps request IDs to actor names for pending requests.
    actor_requests: HashMap<String, String>,
    // connections: Maps StreamId to the TcpStream for active client connections.
    connections: HashMap<StreamId, TcpStream>,
}

impl DevtoolsInstance {
    /**
     * @brief Creates and initializes a new `DevtoolsInstance`.
     *
     * Sets up the TCP listener, registers initial core actors, and spawns a thread
     * to accept new client connections. It also communicates the listening port
     * and a security token to the embedder.
     *
     * @param sender The sending end of the MPSC channel for `DevtoolsControlMsg`.
     * @param receiver The receiving end of the MPSC channel for `DevtoolsControlMsg`.
     * @param port The desired TCP port to listen on.
     * @param embedder A proxy to communicate with the embedder environment.
     * @return An `Option<Self>` containing the initialized `DevtoolsInstance` if successful, `None` otherwise.
     */
    fn create(
        sender: Sender<DevtoolsControlMsg>,
        receiver: Receiver<DevtoolsControlMsg>,
        port: u16,
        embedder: EmbedderProxy,
    ) -> Option<Self> {
        // Block Logic: Bind to the specified TCP port and get the assigned port if successful.
        let bound = TcpListener::bind(("0.0.0.0", port)).ok().and_then(|l| {
            l.local_addr()
                .map(|addr| addr.port())
                .ok()
                .map(|port| (l, port))
        });

        // Block Logic: Generate a security token and inform the embedder about the server's status and token.
        let port = if bound.is_some() { Ok(port) } else { Err(()) };
        let token = format!("{:X}", servo_rand::ServoRng::default().next_u32());
        embedder.send(EmbedderMsg::OnDevtoolsStarted(port, token.clone()));

        // Block Logic: Extract the TcpListener or return None if binding failed.
        let listener = match bound {
            Some((l, _)) => l,
            None => {
                return None;
            },
        };

        // Block Logic: Create and register fundamental devtools actors (Performance, Device, Preference, Process, Root).
        let mut registry = ActorRegistry::new();
        let performance = PerformanceActor::new(registry.new_name("performance"));
        let device = DeviceActor::new(registry.new_name("device"));
        let preference = PreferenceActor::new(registry.new_name("preference"));
        let process = ProcessActor::new(registry.new_name("process"));
        let root = Box::new(RootActor {
            tabs: vec![],
            workers: vec![],
            device: device.name(),
            performance: performance.name(),
            preference: preference.name(),
            process: process.name(),
            active_tab: None.into(),
        });

        registry.register(root);
        registry.register(Box::new(performance));
        registry.register(Box::new(device));
        registry.register(Box::new(preference));
        registry.register(Box::new(process));
        registry.find::<RootActor>("root");

        let actors = registry.create_shareable();

        // Block Logic: Initialize the DevtoolsInstance struct.
        let instance = Self {
            actors,
            id_map: Arc::new(Mutex::new(IdMap::default())),
            browsing_contexts: HashMap::new(),
            pipelines: HashMap::new(),
            receiver,
            actor_requests: HashMap::new(),
            actor_workers: HashMap::new(),
            connections: HashMap::new(),
        };

        // Block Logic: Spawn a new thread to continuously accept incoming client connections.
        thread::Builder::new()
            .name("DevtoolsCliAcceptor".to_owned())
            .spawn(move || {
                // accept connections and process them, spawning a new thread for each one
                for stream in listener.incoming() {
                    let mut stream = stream.expect("Can't retrieve stream");
                    // Block Logic: Check if the devtools client is allowed to connect.
                    if !allow_devtools_client(&mut stream, &embedder, &token) {
                        continue;
                    };
                    // connection succeeded and accepted
                    // Block Logic: Send a message to the main devtools instance to add the new client.
                    sender
                        .send(DevtoolsControlMsg::FromChrome(
                            ChromeToDevtoolsControlMsg::AddClient(stream),
                        ))
                        .unwrap();
                }
            })
            .expect("Thread spawning failed");

        Some(instance)
    }

    /**
     * @brief The main event loop for the `DevtoolsInstance`.
     *
     * This function continuously receives and processes `DevtoolsControlMsg` messages,
     * dispatching them to appropriate handler methods. It also manages the lifecycle
     * of client connections.
     */
    fn run(mut self) {
        // next_id: Counter for assigning unique IDs to new stream connections.
        let mut next_id = StreamId(0);
        // Block Logic: Main event loop for processing incoming control messages.
        while let Ok(msg) = self.receiver.recv() {
            trace!("{:?}", msg);
            match msg {
                // Block Logic: Handle a new client connection from Chrome.
                DevtoolsControlMsg::FromChrome(ChromeToDevtoolsControlMsg::AddClient(stream)) => {
                    let actors = self.actors.clone();
                    let id = next_id;
                    next_id = StreamId(id.0 + 1);
                    self.connections.insert(id, stream.try_clone().unwrap());

                    // Block Logic: Inform existing browsing contexts about the new stream.
                    for name in self.browsing_contexts.values() {
                        let actors = actors.lock().unwrap();
                        let browsing_context = actors.find::<BrowsingContextActor>(name);
                        let mut streams = browsing_context.streams.borrow_mut();
                        streams.insert(id, stream.try_clone().unwrap());
                    }

                    // Block Logic: Spawn a new thread to handle communication with this specific client.
                    thread::Builder::new()
                        .name("DevtoolsClientHandler".to_owned())
                        .spawn(move || handle_client(actors, stream.try_clone().unwrap(), id))
                        .expect("Thread spawning failed");
                },
                // Block Logic: Handle a framerate tick message from a script.
                DevtoolsControlMsg::FromScript(ScriptToDevtoolsControlMsg::FramerateTick(
                    actor_name,
                    tick,
                )) => self.handle_framerate_tick(actor_name, tick),
                // Block Logic: Handle a title changed message from a script.
                DevtoolsControlMsg::FromScript(ScriptToDevtoolsControlMsg::TitleChanged(
                    pipeline,
                    title,
                )) => self.handle_title_changed(pipeline, title),
                // Block Logic: Handle a new global script context being created.
                DevtoolsControlMsg::FromScript(ScriptToDevtoolsControlMsg::NewGlobal(
                    ids,
                    script_sender,
                    pageinfo,
                )) => self.handle_new_global(ids, script_sender, pageinfo),
                // Block Logic: Handle a navigation event.
                DevtoolsControlMsg::FromScript(ScriptToDevtoolsControlMsg::Navigate(
                    browsing_context,
                    state,
                )) => self.handle_navigate(browsing_context, state),
                // Block Logic: Handle a console API message.
                DevtoolsControlMsg::FromScript(ScriptToDevtoolsControlMsg::ConsoleAPI(
                    pipeline_id,
                    console_message,
                    worker_id,
                )) => self.handle_console_message(pipeline_id, worker_id, console_message),
                // Block Logic: Handle a script source being loaded.
                DevtoolsControlMsg::FromScript(ScriptToDevtoolsControlMsg::ScriptSourceLoaded(
                    pipeline_id,
                    source_info,
                )) => self.handle_script_source_info(pipeline_id, source_info),
                // Block Logic: Handle a page error reported from a script.
                DevtoolsControlMsg::FromScript(ScriptToDevtoolsControlMsg::ReportPageError(
                    pipeline_id,
                    page_error,
                )) => self.handle_page_error(pipeline_id, None, page_error),
                // Block Logic: Handle a CSS error reported from a script, converting it to a console message.
                DevtoolsControlMsg::FromScript(ScriptToDevtoolsControlMsg::ReportCSSError(
                    pipeline_id,
                    css_error,
                )) => {
                    let mut console_message = ConsoleMessageBuilder::new(
                        LogLevel::Warn,
                        css_error.filename,
                        css_error.line,
                        css_error.column,
                    );
                    console_message.add_argument(css_error.msg.into());

                    self.handle_console_message(pipeline_id, None, console_message.finish())
                },
                // Block Logic: Handle a network event reported from Chrome.
                DevtoolsControlMsg::FromChrome(ChromeToDevtoolsControlMsg::NetworkEvent(
                    request_id,
                    network_event,
                )) => {
                    // copy the connections vector
                    let mut connections = Vec::<TcpStream>::new();
                    for stream in self.connections.values() {
                        connections.push(stream.try_clone().unwrap());
                    }

                    let pipeline_id = match network_event {
                        NetworkEvent::HttpResponse(ref response) => response.pipeline_id,
                        NetworkEvent::HttpRequest(ref request) => request.pipeline_id,
                    };
                    self.handle_network_event(connections, pipeline_id, request_id, network_event);
                },
                // Block Logic: Handle a server exit message, breaking the main loop.
                DevtoolsControlMsg::FromChrome(ChromeToDevtoolsControlMsg::ServerExitMsg) => break,
            }
        }

        // Block Logic: Shut down all active client connections when the server exits.
        for connection in self.connections.values_mut() {
            let _ = connection.shutdown(Shutdown::Both);
        }
    }

    /**
     * @brief Handles a framerate tick message by updating the corresponding framerate actor.
     * @param actor_name The name of the `FramerateActor` to update.
     * @param tick The framerate tick value.
     */
    fn handle_framerate_tick(&self, actor_name: String, tick: f64) {
        let mut actors = self.actors.lock().unwrap();
        let framerate_actor = actors.find_mut::<FramerateActor>(&actor_name);
        framerate_actor.add_tick(tick);
    }

    /**
     * @brief Handles a navigation event by informing the relevant `BrowsingContextActor`.
     * @param browsing_context_id The ID of the browsing context that navigated.
     * @param state The new navigation state.
     */
    fn handle_navigate(&self, browsing_context_id: BrowsingContextId, state: NavigationState) {
        let actor_name = self.browsing_contexts.get(&browsing_context_id).unwrap();
        self.actors
            .lock()
            .unwrap()
            .find::<BrowsingContextActor>(actor_name)
            .navigate(state, &mut self.id_map.lock().expect("Mutex poisoned"));
    }

    /**
     * @brief Handles a new global script context being created.
     *
     * This method initializes and registers new actors (e.g., `WorkerActor`, `ConsoleActor`,
     * `BrowsingContextActor`) associated with the new global context.
     *
     * @param ids A tuple containing `BrowsingContextId`, `PipelineId`, optional `WorkerId`, and `WebViewId`.
     * @param script_sender An IPC sender for communicating with the script environment.
     * @param page_info Information about the new page.
     */
    fn handle_new_global(
        &mut self,
        ids: (BrowsingContextId, PipelineId, Option<WorkerId>, WebViewId),
        script_sender: IpcSender<DevtoolScriptControlMsg>,
        page_info: DevtoolsPageInfo,
    ) {
        let mut actors = self.actors.lock().unwrap();

        let (browsing_context_id, pipeline_id, worker_id, webview_id) = ids;
        let id_map = &mut self.id_map.lock().expect("Mutex poisoned");
        let devtools_browser_id = id_map.browser_id(webview_id);
        let devtools_browsing_context_id = id_map.browsing_context_id(browsing_context_id);
        let devtools_outer_window_id = id_map.outer_window_id(pipeline_id);

        let console_name = actors.new_name("console");

        let parent_actor = if let Some(id) = worker_id {
            // Block Logic: Handle initialization for a new Worker global.
            assert!(self.pipelines.contains_key(&pipeline_id));
            assert!(self.browsing_contexts.contains_key(&browsing_context_id));

            let thread = ThreadActor::new(actors.new_name("thread"));
            let thread_name = thread.name();
            actors.register(Box::new(thread));

            let worker_name = actors.new_name("worker");
            let worker = WorkerActor {
                name: worker_name.clone(),
                console: console_name.clone(),
                thread: thread_name,
                worker_id: id,
                url: page_info.url.clone(),
                type_: WorkerType::Dedicated,
                script_chan: script_sender,
                streams: Default::default(),
            };
            let root = actors.find_mut::<RootActor>("root");
            root.workers.push(worker.name.clone());

            self.actor_workers.insert(id, worker_name.clone());
            actors.register(Box::new(worker));

            Root::DedicatedWorker(worker_name)
        } else {
            // Block Logic: Handle initialization for a new BrowsingContext global.
            self.pipelines.insert(pipeline_id, browsing_context_id);
            let name = self
                .browsing_contexts
                .entry(browsing_context_id)
                .or_insert_with(|| {
                    let browsing_context_actor = BrowsingContextActor::new(
                        console_name.clone(),
                        devtools_browser_id,
                        devtools_browsing_context_id,
                        page_info,
                        pipeline_id,
                        devtools_outer_window_id,
                        script_sender,
                        &mut actors,
                    );
                    let name = browsing_context_actor.name();
                    actors.register(Box::new(browsing_context_actor));
                    name
                });

            // Block Logic: Add existing client streams to the newly created browsing context.
            let browsing_context = actors.find::<BrowsingContextActor>(name);
            let mut streams = browsing_context.streams.borrow_mut();
            for (id, stream) in &self.connections {
                streams.insert(*id, stream.try_clone().unwrap());
            }

            Root::BrowsingContext(name.clone())
        };

        // Block Logic: Register a new `ConsoleActor` for the global context.
        let console = ConsoleActor {
            name: console_name,
            cached_events: Default::default(),
            root: parent_actor,
        };

        actors.register(Box::new(console));
    }

    /**
     * @brief Handles a title changed message from a pipeline.
     * @param pipeline_id The ID of the pipeline whose title changed.
     * @param title The new title of the page.
     */
    fn handle_title_changed(&self, pipeline_id: PipelineId, title: String) {
        let bc = match self.pipelines.get(&pipeline_id) {
            Some(bc) => bc,
            None => return,
        };
        let name = match self.browsing_contexts.get(bc) {
            Some(name) => name,
            None => return,
        };
        let actors = self.actors.lock().unwrap();
        let browsing_context = actors.find::<BrowsingContextActor>(name);
        browsing_context.title_changed(pipeline_id, title);
    }

    /**
     * @brief Handles a page error reported from a pipeline or worker.
     * @param pipeline_id The ID of the pipeline where the error occurred.
     * @param worker_id An optional ID of the worker where the error occurred.
     * @param page_error The `PageError` details.
     */
    fn handle_page_error(
        &mut self,
        pipeline_id: PipelineId,
        worker_id: Option<WorkerId>,
        page_error: PageError,
    ) {
        let console_actor_name = match self.find_console_actor(pipeline_id, worker_id) {
            Some(name) => name,
            None => return,
        };
        let actors = self.actors.lock().unwrap();
        let console_actor = actors.find::<ConsoleActor>(&console_actor_name);
        let id = worker_id.map_or(UniqueId::Pipeline(pipeline_id), UniqueId::Worker);
        for stream in self.connections.values_mut() {
            console_actor.handle_page_error(page_error.clone(), id.clone(), &actors, stream);
        }
    }

    /**
     * @brief Handles a console API message from a pipeline or worker.
     * @param pipeline_id The ID of the pipeline where the message originated.
     * @param worker_id An optional ID of the worker where the message originated.
     * @param console_message The `ConsoleMessage` details.
     */
    fn handle_console_message(
        &mut self,
        pipeline_id: PipelineId,
        worker_id: Option<WorkerId>,
        console_message: ConsoleMessage,
    ) {
        let console_actor_name = match self.find_console_actor(pipeline_id, worker_id) {
            Some(name) => name,
            None => return,
        };
        let actors = self.actors.lock().unwrap();
        let console_actor = actors.find::<ConsoleActor>(&console_actor_name);
        let id = worker_id.map_or(UniqueId::Pipeline(pipeline_id), UniqueId::Worker);
        for stream in self.connections.values_mut() {
            console_actor.handle_console_api(console_message.clone(), id.clone(), &actors, stream);
        }
    }

    /**
     * @brief Finds the name of the console actor associated with a given pipeline or worker.
     * @param pipeline_id The ID of the pipeline.
     * @param worker_id An optional ID of the worker.
     * @return An `Option<String>` containing the console actor's name if found, `None` otherwise.
     */
    fn find_console_actor(
        &self,
        pipeline_id: PipelineId,
        worker_id: Option<WorkerId>,
    ) -> Option<String> {
        let actors = self.actors.lock().unwrap();
        if let Some(worker_id) = worker_id {
            let actor_name = self.actor_workers.get(&worker_id)?;
            Some(actors.find::<WorkerActor>(actor_name).console.clone())
        } else {
            let id = self.pipelines.get(&pipeline_id)?;
            let actor_name = self.browsing_contexts.get(id)?;
            Some(
                actors
                    .find::<BrowsingContextActor>(actor_name)
                    .console
                    .clone(),
            )
        }
    }

    /**
     * @brief Handles a network event, dispatching it to the appropriate `NetworkEventActor`.
     * @param connections A vector of active TCP streams to devtools clients.
     * @param pipeline_id The ID of the pipeline where the network event occurred.
     * @param request_id The ID of the network request.
     * @param network_event The `NetworkEvent` details.
     */
    fn handle_network_event(
        &mut self,
        connections: Vec<TcpStream>,
        pipeline_id: PipelineId,
        request_id: String,
        network_event: NetworkEvent,
    ) {
        let netevent_actor_name = self.find_network_event_actor(request_id);

        let Some(id) = self.pipelines.get(&pipeline_id) else {
            return;
        };
        let Some(browsing_context_actor_name) = self.browsing_contexts.get(id) else {
            return;
        };

        // Block Logic: Call the network handler to process the event.
        handle_network_event(
            Arc::clone(&self.actors),
            netevent_actor_name,
            connections,
            network_event,
            browsing_context_actor_name.to_string(),
        )
    }

    /**
     * @brief Finds the name of the `NetworkEventActor` corresponding to a given request ID.
     *
     * If an actor for the `request_id` already exists, its name is returned. Otherwise,
     * a new `NetworkEventActor` is created, registered, and its name is returned.
     *
     * @param request_id The ID of the network request.
     * @return The name of the `NetworkEventActor` handling the request.
     */
    fn find_network_event_actor(&mut self, request_id: String) -> String {
        let mut actors = self.actors.lock().unwrap();
        match self.actor_requests.entry(request_id) {
            Occupied(name) => {
                //TODO: Delete from map like Firefox does?
                name.into_mut().clone()
            },
            Vacant(entry) => {
                let actor_name = actors.new_name("netevent");
                let actor = NetworkEventActor::new(actor_name.clone());
                entry.insert(actor_name.clone());
                actors.register(Box::new(actor));
                actor_name
            },
        }
    }

    /**
     * @brief Handles a script source being loaded, informing relevant actors.
     * @param pipeline_id The ID of the pipeline where the script source was loaded.
     * @param source_info Information about the loaded script source.
     */
    fn handle_script_source_info(&mut self, pipeline_id: PipelineId, source_info: SourceInfo) {
        let mut actors = self.actors.lock().unwrap();

        // Block Logic: Create and register a new `SourceActor` for the loaded script.
        let source_actor = SourceActor::new_registered(
            &mut actors,
            source_info.url,
            source_info.content.clone(),
            source_info.content_type.unwrap(),
        );
        let source_actor_name = source_actor.name.clone();
        let source_form = source_actor.source_form();

        if let Some(worker_id) = source_info.worker_id {
            // Block Logic: If it's a worker script, update the associated worker's thread actor.
            let Some(worker_actor_name) = self.actor_workers.get(&worker_id) else {
                return;
            };

            let thread_actor_name = actors.find::<WorkerActor>(worker_actor_name).thread.clone();
            let thread_actor = actors.find_mut::<ThreadActor>(&thread_actor_name);

            thread_actor.source_manager.add_source(&source_actor_name);

            let worker_actor = actors.find::<WorkerActor>(worker_actor_name);

            for stream in self.connections.values_mut() {
                worker_actor.resource_array(
                    &source_form,
                    "source".into(),
                    ResourceArrayType::Available,
                    stream,
                );
            }
        } else {
            // Block Logic: If it's a main thread script, update the associated browsing context's thread actor.
            let Some(browsing_context_id) = self.pipelines.get(&pipeline_id) else {
                return;
            };
            let Some(actor_name) = self.browsing_contexts.get(browsing_context_id) else {
                return;
            };

            let thread_actor_name = {
                let browsing_context = actors.find::<BrowsingContextActor>(actor_name);
                browsing_context.thread.clone()
            };

            let thread_actor = actors.find_mut::<ThreadActor>(&thread_actor_name);

            thread_actor.source_manager.add_source(&source_actor_name);

            // Block Logic: Notify browsing context about the new source.
            let browsing_context = actors.find::<BrowsingContextActor>(actor_name);

            for stream in self.connections.values_mut() {
                browsing_context.resource_array(
                    &source_form,
                    "source".into(),
                    ResourceArrayType::Available,
                    stream,
                );
            }
        }
    }
}

/**
 * @brief Authenticates a devtools client connection.
 *
 * Checks for a valid authentication token provided by the client. If no token
 * is provided or it's invalid, it prompts the embedder for permission to connect.
 *
 * @param stream A mutable reference to the `TcpStream` of the client connection.
 * @param embedder A proxy to communicate with the embedder environment.
 * @param token The expected authentication token.
 * @return `true` if the client is allowed to connect, `false` otherwise.
 */
fn allow_devtools_client(stream: &mut TcpStream, embedder: &EmbedderProxy, token: &str) -> bool {
    // Block Logic: Construct the expected token message format.
    let token = format!("25:{{\"auth_token\":\"{}\"}}", token);
    // buf: Buffer to read the incoming client message.
    let mut buf = [0; 28];
    // timeout: Set a read timeout to prevent blocking indefinitely.
    let timeout = std::time::Duration::from_millis(500);
    // This will read but not consume the bytes from the stream.
    stream.set_read_timeout(Some(timeout)).unwrap();
    let peek = stream.peek(&mut buf);
    stream.set_read_timeout(None).unwrap();
    if let Ok(len) = peek {
        if len == buf.len() {
            if let Ok(s) = std::str::from_utf8(&buf) {
                if s == token {
                    // Consume the message as it was relevant to us.
                    let _ = stream.read_exact(&mut buf);
                    return true;
                }
            }
        }
    };

    // Block Logic: No token found or invalid token. Prompt user via embedder for permission.
    let (request_sender, request_receiver) = ipc::channel().expect("Failed to create IPC channel!");
    embedder.send(EmbedderMsg::RequestDevtoolsConnection(request_sender));
    request_receiver.recv().unwrap() == AllowOrDeny::Allow
}

/**
 * @brief Processes messages from a single devtools client until EOF.
 *
 * This function continuously reads JSON packets from the client stream,
 * dispatches them to the actor registry for processing, and handles
 * connection termination.
 *
 * @param actors A shared, mutable reference to the `ActorRegistry`.
 * @param stream The `TcpStream` for communication with the client.
 * @param stream_id The unique identifier for this client stream.
 */
fn handle_client(actors: Arc<Mutex<ActorRegistry>>, mut stream: TcpStream, stream_id: StreamId) {
    log::info!("Connection established to {}", stream.peer_addr().unwrap());
    // Block Logic: Send an initial message to the client indicating successful connection.
    let msg = actors.lock().unwrap().find::<RootActor>("root").encodable();
    if let Err(e) = stream.write_json_packet(&msg) {
        log::warn!("Error writing response: {:?}", e);
        return;
    }

    // Block Logic: Loop indefinitely to read and process messages from the client.
    loop {
        match stream.read_json_packet() {
            // Block Logic: Successfully read a JSON packet, dispatch it to the actors.
            Ok(Some(json_packet)) => {
                if let Err(()) = actors.lock().unwrap().handle_message(
                    json_packet.as_object().unwrap(),
                    &mut stream,
                    stream_id,
                ) {
                    log::error!("Devtools actor stopped responding");
                    let _ = stream.shutdown(Shutdown::Both);
                    break;
                }
            },
            // Block Logic: Client closed the connection (EOF).
            Ok(None) => {
                log::info!("Devtools connection closed");
                break;
            },
            // Block Logic: An error occurred while reading from the client stream.
            Err(err_msg) => {
                log::error!("Failed to read message from devtools client: {}", err_msg);
                break;
            },
        }
    }

    // Block Logic: Clean up resources associated with the disconnected client.
    actors.lock().unwrap().cleanup(stream_id);
}
