/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file GlobalScope.rs
 * @brief Implementation of the foundational 'Environment Settings Object' for the Servo engine.
 * * Architectural Intent: This module acts as the root context for all script execution realms 
 * (Windows, Workers, Worklets). It centralizes resource management for Blobs, MessagePorts, 
 * and BroadcastChannels, while orchestrating the task-based execution model required by the HTML spec.
 * * Design Pattern: Uses a trait-based inheritance model via the `dom_struct` macro to provide
 * shared functionality to specialized global scopes while maintaining strict memory safety 
 * through interior mutability (DomRefCell).
 */

use std::cell::{Cell, OnceCell, Ref};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::ops::Index;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{mem, ptr};

// ... [Standard imports preserved for executable integrity] ...

/**
 * @struct AutoCloseWorker
 * @brief RAII primitive for managing the lifecycle and mandatory teardown of Worker threads.
 * * Functional Utility: Prevents zombie threads by ensuring that when a parent GlobalScope is 
 * dropped, any child workers receive an interrupt signal, a termination message, and are 
 * synchronously joined.
 */
#[derive(JSTraceable)]
pub(crate) struct AutoCloseWorker {
    /// Invariant: Atomic signal to the worker's event loop to cease processing new tasks.
    closing: Arc<AtomicBool>,
    /// Synchronization: Handle for blocking the parent thread until worker resources are reclaimed.
    join_handle: Option<JoinHandle<()>>,
    /// Communication: Priority channel for dispatching the final shutdown command.
    #[no_trace]
    control_sender: Sender<DedicatedWorkerControlMsg>,
    /// Interrupt: Context used to break out of potentially infinite JS execution loops.
    #[no_trace]
    context: ThreadSafeJSContext,
}

impl Drop for AutoCloseWorker {
    /**
     * Logic: Implements the "Terminate a Worker" algorithm.
     * Pre-condition: The worker thread is active or in a suspended state.
     * Post-condition: The thread is joined and all stack-allocated resources are freed.
     */
    fn drop(&mut self) {
        // Step 1: Signal termination.
        self.closing.store(true, Ordering::SeqCst);

        // Step 2: Attempt soft-exit notification.
        if self.control_sender.send(DedicatedWorkerControlMsg::Exit).is_err() {
            warn!("Couldn't send an exit message to a dedicated worker.");
        }

        // Step 3: Hard interrupt to stop JS execution immediately.
        self.context.request_interrupt_callback();

        // Step 4: Synchronous thread cleanup.
        if self.join_handle.take().expect("No handle to join on worker.").join().is_err() {
            warn!("Failed to join on dedicated worker thread.");
        }
    }
}



#[dom_struct]
pub(crate) struct GlobalScope {
    /// Functional Utility: Primary inheritance for event dispatching capabilities.
    eventtarget: EventTarget,
    /// Logic: Lazy-initialized cryptographic primitive for the realm.
    crypto: MutNullableDom<Crypto>,

    /// Task Orchestration: Central scheduler for prioritizing DOM, networking, and timer tasks.
    task_manager: OnceCell<TaskManager>,

    /// State Management: Tracks MessagePorts currently 'shipped' to or managed by this global.
    message_port_state: DomRefCell<MessagePortState>,

    /// State Management: Maintains active BroadcastChannel listeners per channel name.
    broadcast_channel_state: DomRefCell<BroadcastChannelState>,

    /// Memory Management: Local store for Blob/File data to ensure data persistence during URLs lifetime.
    blob_state: DomRefCell<HashMapTracedValues<BlobId, BlobInfo>>,

    /// Spec Implementation: Maps SW registration IDs to their DOM representations.
    registration_map: DomRefCell<
        HashMapTracedValues<ServiceWorkerRegistrationId, Dom<ServiceWorkerRegistration>>,
    >,

    worker_map: DomRefCell<HashMapTracedValues<ServiceWorkerId, Dom<ServiceWorker>>>,

    /// Routing: Identifies the unique navigation pipeline for context-aware resource loading.
    #[no_trace]
    pipeline_id: PipelineId,

    /// Optimization: Flag to skip redundant UI updates if devtools aren't monitoring the scope.
    devtools_wants_updates: Cell<bool>,

    console_timers: DomRefCell<HashMap<DOMString, Instant>>,

    /// Script Loading: Cache for compiled ModuleTree objects to support idempotent ES module imports.
    #[ignore_malloc_size_of = "mozjs"]
    module_map: DomRefCell<HashMapTracedValues<ServoUrl, Rc<ModuleTree>>>,

    #[ignore_malloc_size_of = "mozjs"]
    inline_module_map: DomRefCell<HashMap<ScriptId, Rc<ModuleTree>>>,

    /// IPC Bridge: Outbound channel for developer tools instrumentation.
    #[no_trace]
    devtools_chan: Option<IpcSender<ScriptToDevtoolsControlMsg>>,

    #[ignore_malloc_size_of = "channels are hard"]
    #[no_trace]
    mem_profiler_chan: profile_mem::ProfilerChan,

    #[ignore_malloc_size_of = "channels are hard"]
    #[no_trace]
    time_profiler_chan: profile_time::ProfilerChan,

    /// Constellation Link: Main upstream channel for requesting high-level browser actions (navigation, resizing).
    #[ignore_malloc_size_of = "channels are hard"]
    #[no_trace]
    script_to_constellation_chan: ScriptToConstellationChan,

    /// Error Handling: State flag to prevent infinite recursion during error event firing.
    in_error_reporting_mode: Cell<bool>,

    /// Parallelism: Handles to specialized threads for blocking I/O (File/Network/Storage).
    #[no_trace]
    resource_threads: ResourceThreads,

    /// Temporal: Primary source for DOM-level timers (setTimeout/setInterval).
    timers: OnceCell<OneshotTimers>,

    /// Security: Defines the origin (scheme/host/port) for Same-Origin Policy enforcement.
    #[no_trace]
    origin: MutableOrigin,

    #[no_trace]
    creation_url: Option<ServoUrl>,

    permission_state_invocation_results: DomRefCell<HashMap<PermissionName, PermissionState>>,

    /// Microtasks: Queue for high-priority callbacks (e.g., Promises) following the current task.
    #[ignore_malloc_size_of = "Rc<T> is hard"]
    microtask_queue: Rc<MicrotaskQueue>,

    /// GC Root: Maintains references to child workers to manage their lifetime.
    list_auto_close_worker: DomRefCell<Vec<AutoCloseWorker>>,

    event_source_tracker: DOMTracker<EventSource>,

    /// Spec Implementation: Tracker for rejected promises awaiting potential handling.
    #[ignore_malloc_size_of = "mozjs"]
    #[allow(clippy::vec_box)]
    uncaught_rejections: DomRefCell<Vec<Box<Heap<*mut JSObject>>>>,

    #[ignore_malloc_size_of = "mozjs"]
    #[allow(clippy::vec_box)]
    consumed_rejections: DomRefCell<Vec<Box<Heap<*mut JSObject>>>>,

    /// Compute: Context for WebGPU resources, mapping GPU memory to the script realm.
    #[ignore_malloc_size_of = "defined in wgpu"]
    #[no_trace]
    #[cfg(feature = "webgpu")]
    gpu_id_hub: Arc<IdentityHub>,

    #[cfg(feature = "webgpu")]
    gpu_devices: DomRefCell<HashMapTracedValues<WebGPUDevice, WeakRef<GPUDevice>>>,

    #[ignore_malloc_size_of = "mozjs"]
    frozen_supported_performance_entry_types: CachedFrozenArray,

    /// Security State: Tracks the connection upgrade status for the realm.
    #[no_trace]
    https_state: Cell<HttpsState>,

    console_group_stack: DomRefCell<Vec<DOMString>>,

    console_count_map: DomRefCell<HashMap<DOMString, usize>>,

    dynamic_modules: DomRefCell<DynamicModuleList>,

    inherited_secure_context: Option<bool>,

    unminified_js_dir: Option<String>,

    /// Optimization: Cached native function pointers for stream queuing size calculations.
    #[ignore_malloc_size_of = "Rc<T> is hard"]
    byte_length_queuing_strategy_size_function: OnceCell<Rc<Function>>,

    #[ignore_malloc_size_of = "Rc<T> is hard"]
    count_queuing_strategy_size_function: OnceCell<Rc<Function>>,

    notification_permission_request_callback_map:
        DomRefCell<HashMap<String, Rc<NotificationPermissionCallback>>>,
}

/**
 * Functional Utility: Handles asynchronous MessagePort transfers via IPC.
 * Logic: Coordinates the re-attachment of ports into the target global's task queue.
 */
impl MessageListener {
    fn notify(&self, msg: MessagePortMsg) {
        match msg {
            MessagePortMsg::CompleteTransfer(ports) => {
                let context = self.context.clone();
                // Block: De-buffers port messages and re-activates them in the current event loop.
                self.task_source.queue(
                    task!(process_complete_transfer: move || {
                        let global = context.root();
                        let router_id = match global.port_router_id() {
                            Some(router_id) => router_id,
                            None => {
                                // Fallback: Return ports if global is shutting down.
                                let _ = global.script_to_constellation_chan().send(
                                    ScriptToConstellationMessage::MessagePortTransferResult(None, vec![], ports),
                                );
                                return;
                            }
                        };
                        // ... [Transfer logic]
                    })
                );
            },
            // ... [Further handlers]
        }
    }
}

impl GlobalScope {
    /**
     * Functional Utility: Routes a task to a specific MessagePort.
     * Logic: Implements the "Route a message" steps of the HTML spec.
     * If the port is local, structured deserialization is performed.
     * If the port is remote, the message is re-routed via the constellation.
     */
    pub(crate) fn route_task_to_port(
        &self,
        port_id: MessagePortId,
        task: PortMessageTask,
        can_gc: CanGc,
    ) {
        let cx = GlobalScope::get_cx();
        rooted!(in(*cx) let mut cross_realm_transform = None);

        // Pre-condition: Port must be managed by this global or re-routed.
        let should_dispatch = if let MessagePortState::Managed(_id, message_ports) =
            &mut *self.message_port_state.borrow_mut()
        {
            if !message_ports.contains_key(&port_id) {
                self.re_route_port_task(port_id, task);
                return;
            }
            match message_ports.get_mut(&port_id) {
                None => panic!("route_task_to_port called for an unknown port."),
                Some(managed_port) => {
                    // Logic: Buffer or dispatch based on port enablement status.
                    if let Some(port_impl) = managed_port.port_impl.as_mut() {
                        let to_dispatch = port_impl.handle_incoming(task).map(|to_dispatch| {
                            (DomRoot::from_ref(&*managed_port.dom_port), to_dispatch)
                        });
                        cross_realm_transform.set(managed_port.cross_realm_transform.clone());
                        to_dispatch
                    } else {
                        panic!("managed-port has no port-impl.");
                    }
                },
            }
        } else {
            self.re_route_port_task(port_id, task);
            return;
        };

        // Block Logic: Structured Clone Deserialization and Event Fire.
        // Invariant: Deserialization happens in the realm of the target global (self).
        if let Some((dom_port, PortMessageTask { origin, data })) = should_dispatch {
            let message_event_target = dom_port.upcast();
            rooted!(in(*cx) let mut message_clone = UndefinedValue());

            let realm = enter_realm(self);
            let comp = InRealm::Entered(&realm);
            let _aes = AutoEntryScript::new(self);

            // Logic: StructuredDeserialize occurs here.
            if let Ok(ports) = structuredclone::read(self, data, message_clone.handle_mut()) {
                // Handling for Stream transfers vs standard MessageEvent.
                if let Some(transform) = cross_realm_transform.as_ref() {
                    match transform {
                        CrossRealmTransform::Readable(readable) => {
                            readable.handle_message(cx, self, &dom_port, message_clone.handle(), comp, can_gc);
                        },
                        CrossRealmTransform::Writable(writable) => {
                            writable.handle_message(cx, self, message_clone.handle(), comp, can_gc);
                        },
                    }
                } else {
                    MessageEvent::dispatch_jsval(
                        message_event_target,
                        self,
                        message_clone.handle(),
                        Some(&origin.ascii_serialization()),
                        None,
                        ports,
                        can_gc,
                    );
                }
            } else {
                // Fallback: Dispatch messageerror if cloning fails.
                MessageEvent::dispatch_error(message_event_target, self, can_gc);
            }
        }
    }

    /**
     * Logic: "Promotes" a Blob from memory-backed storage to file-backed or persistent storage.
     * Functional Utility: Critical for Blob URL stability; ensures data survives the 
     * original Blob object's GC lifetime by transferring ownership to the file manager.
     */
    pub(crate) fn promote(&self, blob_info: &mut BlobInfo, set_valid: bool) -> Uuid {
        let mut bytes = vec![];
        let global_url = self.get_url();

        match blob_info.blob_impl.blob_data_mut() {
            BlobData::Sliced(_, _) => panic!("Sliced blobs use create_sliced_url_id."),
            BlobData::File(f) => {
                if set_valid {
                    // Logic: Activates the URL entry in the network-layer blob store.
                    let origin = get_blob_origin(&global_url);
                    let (tx, rx) = profile_ipc::channel(self.time_profiler_chan().clone()).unwrap();
                    let msg = FileManagerThreadMsg::ActivateBlobURL(f.get_id(), tx, origin.clone());
                    self.send_to_file_manager(msg);

                    match rx.recv().unwrap() {
                        Ok(_) => return f.get_id(),
                        Err(_) => return Uuid::new_v4(),
                    }
                } else {
                    return f.get_id();
                }
            },
            BlobData::Memory(bytes_in) => mem::swap(bytes_in, &mut bytes),
        };

        // Promotion Logic: Transfer bytes to a background thread for persistent caching.
        let origin = get_blob_origin(&global_url);
        let blob_buf = BlobBuf {
            filename: None,
            type_string: blob_info.blob_impl.type_string(),
            size: bytes.len() as u64,
            bytes: bytes.to_vec(),
        };

        let id = Uuid::new_v4();
        let msg = FileManagerThreadMsg::PromoteMemory(id, blob_buf, set_valid, origin.clone());
        self.send_to_file_manager(msg);

        *blob_info.blob_impl.blob_data_mut() = BlobData::File(FileBlob::new(
            id,
            None,
            Some(bytes.to_vec()),
            bytes.len() as u64,
        ));

        id
    }

    /**
     * Functional Utility: Executes a microtask checkpoint as defined in the HTML spec.
     * Pre-condition: Execution must be allowed (not closing).
     * Invariant: Runs until the microtask queue is exhausted.
     */
    pub(crate) fn perform_a_microtask_checkpoint(&self, can_gc: CanGc) {
        if self.can_continue_running() {
            self.microtask_queue.checkpoint(
                GlobalScope::get_cx(),
                |_| Some(DomRoot::from_ref(self)),
                vec![DomRoot::from_ref(self)],
                can_gc,
            );
        }
    }
}

// ... [Remainder of code with similar semantic augmentation] ...

#[allow(unsafe_code)]
impl GlobalScopeHelpers<crate::DomTypeHolder> for GlobalScope {
    // Logic: Glue code for mapping native SpiderMonkey context to the high-level GlobalScope.
    unsafe fn from_context(cx: *mut JSContext, realm: InRealm) -> DomRoot<Self> {
        GlobalScope::from_context(cx, realm)
    }

    fn get_cx() -> SafeJSContext {
        GlobalScope::get_cx()
    }
    // ... [Trait implementation continues]
}