"""
A simulation framework for a network of communicating devices using a thread pool.

This module provides a complex simulation of a device network where script
execution is managed by a thread pool. It features a "lead device" (ID 0) that
is responsible for initializing shared resources and coordinating synchronization
across all devices.
"""


from threading import Event, Thread, Lock, Condition, Semaphore


class Device(object):
    """
    Represents a discrete computational node within a simulated sensor network.
    
    Each device serves as an execution context for scripts that process distributed 
    sensor data. Coordination is achieved through a hierarchical threading model
    where a dedicated DeviceThread manages a pool of SimpleWorkers. 
    Device 0 acts as the system orchestrator (leader) for initialization and 
    temporal barrier synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device node with its local data and network context.

        Args:
            device_id (int): Unique identifier. ID 0 assumes leadership responsibilities.
            sensor_data (dict): Local sensor mappings (location -> value).
            supervisor: Interface for network topology discovery.
        """
        
        self.devices = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []

        # Synchronization: Signal for the completion of a global time step.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.lead_device_index = -1

        # Resource Management: Shared locks for atomic access to specific data locations.
        self.location_locks = []
        if device_id == 0:
            # Leader Role: State for implementing a global barrier.
            self.threads_that_finished_no = 0
            self.next_time_point_cond = Condition()
            
            # Initialization Signal: Ensures all devices wait for the leader to setup shared locks.
            self.can_start = Event()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Orchestrates the initialization of shared synchronization primitives across the network.

        Logic: Device 0 scans the entire network's data footprint to create a global set 
        of location-specific Locks. This enables fine-grained concurrency control 
        where multiple workers can access different data locations simultaneously.
        """
        self.devices = devices

        # Discovery: Locate the leader in the global device list.
        for i in xrange(len(self.devices)):
            if devices[i].device_id == 0:
                self.lead_device_index = i
                break

        if self.device_id == 0:
            self.can_start.clear()

            # Pre-condition: Determine the superset of all sensor locations to size the lock array.
            max_lock = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > max_lock:
                        max_lock = location

            # Action: Create a Lock for every possible data location in the simulation.
            for _ in range(0, max_lock + 1):
                self.location_locks.append(Lock())

            # Distribution: Broadcast the shared lock array to all peer devices.
            for device in devices:
                device.location_locks = self.location_locks

            self.can_start.set()
        else:
            # Wait Condition: Peers must block until the leader completes shared resource allocation.
            devices[self.lead_device_index].can_start.wait()

    def assign_script(self, script, location):
        """
        Enqueues a processing task for the current simulation step.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Termination/Skip: Signals that no more scripts are coming for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Atomic Access: Retrieves value from the sensor map.
        Note: Caller is responsible for holding the appropriate location_lock.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Atomic Update: Updates the sensor state.
        Note: Caller is responsible for holding the appropriate location_lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Graceful termination of the device execution context."""
        self.thread.join()

    def notify_finish(self):
        """
        Implements a global software barrier using Device 0 as the synchronization point.

        Algorithm: 
        1. Acquire the leader's condition variable.
        2. Increment the arrival counter.
        3. If last to arrive, reset counter and release all waiting threads.
        4. Otherwise, block until notified.
        """
        self.devices[self.lead_device_index].next_time_point_cond.acquire()
        self.devices[self.lead_device_index].threads_that_finished_no += 1

        if self.devices[self.lead_device_index].threads_that_finished_no == len(self.devices):
            self.devices[self.lead_device_index].threads_that_finished_no = 0
            # Barrier release: All devices may proceed to the next simulation timepoint.
            self.devices[self.lead_device_index].next_time_point_cond.notifyAll()
        else:
            self.devices[self.lead_device_index].next_time_point_cond.wait()

        self.devices[self.lead_device_index].next_time_point_cond.release()

class DeviceThread(Thread):
    """
    Control Plane Thread: Manages the lifecycle of a device's computation.
    Coordinates between the simulation supervisor and the local worker pool.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = WorkerThreadPool(device)

    def run(self):
        """
        Main Event Loop: Iteratively processes simulation timepoints.
        In each step, it triggers parallel script execution and waits for network-wide consensus.
        """
        while True:
            # Discovery: Fetch topological neighbors for data exchange.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Terminal State: Network simulation has concluded.
                self.thread_pool.shutdown()
                break

            # Block Logic: Waits for the supervisor to signal the start of a task batch.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Dispatch: Offload scripts to the worker pool for parallel processing.
            for (script, location) in self.device.scripts:
                self.thread_pool.do_work(script, location, neighbours)

            # Join: Ensure all local workers finished before participating in the global barrier.
            self.thread_pool.wait_to_finish_work()

            # Synchronize: Align with the rest of the network nodes.
            self.device.notify_finish()

class WorkerThreadPool(object):
    """
    Custom Concurrency Manager: Maintains a fixed-size pool of workers.
    Uses Semaphores and Events to manage task distribution and completion signaling.
    """
    
    def __init__(self, device):
        self.device = device
        self.work_finished_event = Event()
        self.work_finished_event.set() # Initially idle.
        self.worker_pool = []

        self.ready_for_work_queue = []
        # Resource management: Limits the number of concurrent tasks to the pool size (8).
        self.read_to_work_thread_sem = Semaphore(8)
        self.queue_lock = Lock()
        
        # Scaffolding: Pre-spawn worker threads to minimize latency during task dispatch.
        for _ in xrange(8):
            thread = SimpleWorker(self, self.device)
            self.worker_pool.append(thread)
            self.ready_for_work_queue.append(thread)
            thread.start()

    def do_work(self, script, location, neighbours):
        """
        Task Distribution: Assigns a computational task to the next available worker.
        """
        if self.work_finished_event.isSet():
            self.work_finished_event.clear()
        
        # Throttling: Blocks if all workers are currently busy.
        self.read_to_work_thread_sem.acquire()
        
        self.queue_lock.acquire()
        worker = self.ready_for_work_queue.pop(0)
        self.queue_lock.release()
        
        worker.do_work(script, location, neighbours)

    def shutdown(self):
        """System Cleanup: Signals all workers to exit and joins their threads."""
        for worker in self.worker_pool:
            worker.should_i_stop = True
            worker.data_for_work_ready.release() # Wake up to check exit flag.
        for worker in self.worker_pool:
            worker.join()

    def worker_finished(self, worker):
        """
        Callback Mechanism: Invoked by SimpleWorker upon task completion.
        Updates pool state and signals if the entire batch is finished.
        """
        self.queue_lock.acquire()
        self.ready_for_work_queue.append(worker)

        # Invariant Check: If all workers are idle, the current timepoint's work is done.
        if len(self.ready_for_work_queue) == 8 and not self.work_finished_event.isSet():
            self.work_finished_event.set()

        self.queue_lock.release()
        self.read_to_work_thread_sem.release() # Release slot for the next task.

    def wait_to_finish_work(self):
        """Join Logic: Blocks until the work_finished_event is signaled."""
        self.work_finished_event.wait()



class SimpleWorker(Thread):
    """
    Data Processing Worker: Executes a single script within the device context.
    Manages atomic data acquisition from neighbors and local state updates.
    """
    
    def __init__(self, worker_pool, device):
        Thread.__init__(self)
        self.worker_pool = worker_pool
        self.should_i_stop = False
        self.data_for_work_ready = Semaphore(0)
        self.device = device
        self.script = None
        self.location = None
        self.neighbours = None

    def do_work(self, script, location, neighbours):
        """Assigns new workload and triggers the execution loop."""
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.data_for_work_ready.release()


    def run(self):
        """
        Execution Engine: Waits for a work signal, then performs atomic data processing.
        """
        while True:
            self.data_for_work_ready.acquire()

            if self.should_i_stop is True:
                break
            
            # Critical Section: Protect access to specific sensor location across the entire network.
            self.device.location_locks[self.location].acquire()
            script_data = []
            
            # Block Logic: Data Aggregation.
            # Collects current sensor readings from all reachable peer nodes.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Include local node data in the aggregation.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Computation and Propagation.
            if script_data != []:
                # Transform data using the assigned algorithm.
                result = self.script.run(script_data)

                # Update the distributed state for this location across the neighborhood.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)
            
            self.device.location_locks[self.location].release()
            self.worker_pool.worker_finished(self)
