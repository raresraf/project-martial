"""
@e609cfff-06a5-4cd5-b8e6-f1c08a3d23af/device.py
@brief Distributed sensor processing simulation using a persistent worker pool and centralized leader-based coordination.
* Algorithm: Round-robin task distribution to a pool of 20 persistent `DeviceOwnThread` workers with leader-managed global barriers and location mutexes.
* Functional Utility: Orchestrates simulation timepoints through a multi-stage synchronization pipeline (Discovery -> Assignment -> Execution -> Consensus) ensuring cluster-wide data consistency.
"""

from threading import Event, Thread, Lock
from barrier import Barrier
from device_thread import DeviceOwnThread

class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, coordination state, and persistent worker workforce.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the persistent worker pool (20 threads).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        self.new_sensor_data = {}
        self.new_sensor_data_lock = Lock()

        self.other_devices = []

        # Intent: Signals that the device is ready to accept script assignments for the current timepoint.
        self.ready_to_get_scripts = Event()

        # Intent: Signals that all scripts for the current phase have been delivered by the supervisor.
        self.got_all_scripts = Event()

        self.assign_script_lock = Lock() # Intent: Serializes the round-robin task assignment.
        self.set_data_lock = Lock()      # Intent: Serializes local sensor data updates.

        # Block Logic: Leader-managed Synchronization Barriers.
        # Strategy: These are configured on the 'main' device and accessed via proxy methods.
        self.start_loop_barrier = Barrier()
        self.got_scripts_barrier = Barrier()
        self.everyone_done = Barrier()

        self.location_mutex = [] # Intent: Shared registry of mutexes for sensor locations.
        self.get_neighbours_lock = Lock() # Intent: Leader lock for serializing neighbor discovery calls.

        self.data_ready = Event()
        self.data_ready.set()

        # Logic: Spawns the persistent worker pool.
        self.own_threads = []
        self.power = 20 # Domain: Concurrency Scaling - fixed workers per node.
        for _ in range(0, self.power): 
            new_thread = DeviceOwnThread(self)
            self.own_threads.append(new_thread)
            new_thread.start()

        self.own_threads_rr = 0 # Intent: Index for round-robin script distribution.
        self.initialized = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

    def get_main_device(self):
        """
        @brief Elects the node with the minimum ID as the cluster leader.
        Invariant: All global synchronization resources reside on the elected leader.
        """
        min_device = self
        min_id = self.device_id
        for device in self.other_devices:
            if device.device_id < min_id:
                min_device = device
                min_id = device.device_id
        return min_device

    def get_start_loop_barrier(self):
        """
        @brief Proxy for the leader's start-of-cycle barrier.
        """
        return self.get_main_device().start_loop_barrier

    def get_got_scripts_barrier(self):
        """
        @brief Proxy for the leader's post-assignment barrier.
        """
        return self.get_main_device().got_scripts_barrier

    def get_get_neighbours_lock(self):
        """
        @brief Proxy for the leader's discovery serialization lock.
        """
        return self.get_main_device().get_neighbours_lock

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collective resource initialization and distribution.
        Logic: Leader node calculates the global location count and initializes a mutex pool.
        """
        for device in devices:
            self.other_devices.append(device)

        if self.get_main_device() == self:
            # Logic: Configuring barriers with total cluster size.
            self.start_loop_barrier.set_n(len(devices))
            self.got_scripts_barrier.set_n(len(devices))
            self.everyone_done.set_n(len(devices))

            # Logic: Global lock pool creation.
            number_of_locations = 0
            for device in devices:
                current_number = max(device.sensor_data)
                if current_number > number_of_locations:
                    number_of_locations = current_number

            for _ in range(0, number_of_locations + 1):
                self.location_mutex.append(Lock())

        self.ready_to_get_scripts.set()
        self.initialized.set()

    def assign_script(self, script, location):
        """
        @brief Top-level interface for script arrival.
        Algorithm: Greedy Round-Robin task distribution to the persistent local worker pool.
        """
        self.ready_to_get_scripts.wait()
        self.assign_script_lock.acquire()
        if script is not None:
            self.own_threads[self.own_threads_rr].assign_script(script, location)
            self.own_threads_rr = (self.own_threads_rr + 1) % len(self.own_threads)
        else:
            # Logic: Signals completion of assignment phase and prepares for execution.
            self.ready_to_get_scripts.clear()
            self.data_ready.clear()
            self.got_all_scripts.set()
        self.assign_script_lock.release()

    def get_data(self, location):
        """
        @brief Standard data retrieval interface for local sensor readings.
        Pre-condition: Blocks until data is flagged as 'ready' (consensus reached).
        """
        self.data_ready.wait()
        self.set_data_lock.acquire()
        result = self.sensor_data[location] if location in self.sensor_data else None
        self.set_data_lock.release()
        return result

    def get_temp_data(self, location):
        """
        @brief Lock-free retrieval used during internal execution phase.
        """
        result = self.sensor_data[location] if location in self.sensor_data else None
        return result

    def set_data(self, location, data):
        """
        @brief Synchronized update for local sensor readings.
        """
        self.set_data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_data_lock.release()

    def shutdown(self):
        """
        @brief Gracefully terminates all device threads.
        """
        for dot in self.own_threads:
            dot.join()
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief coordination thread implementing the simulation phase state machine.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator for a specific device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core lifecycle loop of the device node coordination.
        Algorithm: Multi-barrier synchronization pipeline (Start -> Discover -> Deliver -> Execute -> Done).
        """
        self.device.initialized.wait()
        self.device.ready_to_get_scripts.set()
        self.device.data_ready.clear()
        
        while True:
            # Sync Phase 1: Cluster Alignment before neighbor discovery.
            self.device.get_start_loop_barrier().wait()

            # Logic: Serialized discovery across the neighborhood via leader lock.
            self.device.get_get_neighbours_lock().acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.get_get_neighbours_lock().release()

            # Coordination Logic: Wakes local workers and propagates new neighborhood info.
            for dot in self.device.own_threads:
                dot.waiting_for_permission.wait()
            for dot in self.device.own_threads:
                dot.neighbours = neighbours
                dot.waiting_for_permission.clear()
                dot.start_loop_condition.acquire()
                dot.start_loop_condition.notify_all()
                dot.start_loop_condition.release()

            if neighbours is None:
                # Logic: Shutdown signal.
                break

            # Block Logic: Waits for script delivery completion for the current timepoint.
            self.device.got_all_scripts.wait()
            self.device.got_all_scripts.clear()
            self.device.data_ready.clear()

            # Sync Phase 2: Cluster Alignment before parallel execution.
            self.device.get_got_scripts_barrier().wait()

            # Execution Dispatch: Triggers worker threads to process their local task lists.
            for dot in self.device.own_threads:
                dot.execute_scripts_event.set()

            # Logic: Waits for all local workers to signal completion.
            for dot in self.device.own_threads:
                dot.done.wait()

            # Sync Phase 3: Cluster Consensus.
            self.device.get_main_device().everyone_done.wait()

            # Phase Finalization: Unlocks data access and resets worker state.
            self.device.data_ready.set()
            for dot in self.device.own_threads:
                dot.done.clear()
                dot.execute_scripts_event.clear()

            self.device.ready_to_get_scripts.set()


from threading import Event, Thread, Condition

class DeviceOwnThread(Thread):
    """
    @brief Persistent worker thread implementing the computational component of the simulation.
    """

    def __init__(self, device):
        """
        @brief Initializes the worker with its task synchronization primitives.
        """
        Thread.__init__(self)
        self.device = device
        self.done = Event() # Intent: Signals completion of the current task batch.
        self.scripts = []   # Intent: Local partition of tasks assigned via round-robin.
        self.execute_scripts_event = Event() # Intent: Phase trigger from coordinator.
        self.start_loop_condition = Condition() # Intent: Alignment monitor for phase start.
        self.waiting_for_permission = Event()   # Intent: Readiness signal to coordinator.
        self.neighbours = []

    def assign_script(self, script, location):
        """
        @brief Appends a task unit to the worker's private list.
        """
        self.scripts.append((script, location))

    def execute_scripts(self):
        """
        @brief Main computational logic for the worker's assigned tasks.
        Algorithm: Iterative data aggregation from peers followed by synchronized propagation.
        """
        for (script, location) in self.scripts:
            # Pre-condition: Must acquire leader-managed location mutex for atomic distributed update.
            self.device.get_main_device().location_mutex[location].acquire()
            script_data = []
            
            # Distributed Aggregation: Collect readings from neighborhood.
            for device in self.neighbours:
                data = device.get_temp_data(location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_temp_data(location)
            if data is not None:
                script_data.append(data)
                
            if script_data != []:
                # Execution and Propagation Phase.
                result = script.run(script_data)
                for device in self.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            
            # Post-condition: Release global location mutex.
            self.device.get_main_device().location_mutex[location].release()

    def run(self):
        """
        @brief Main worker execution loop.
        Algorithm: Monitor-based synchronization with multi-stage event triggers.
        """
        while True:
            # Stage 1: Wait for cycle permission.
            self.start_loop_condition.acquire()
            self.waiting_for_permission.set()
            self.start_loop_condition.wait()
            self.start_loop_condition.release()

            if self.neighbours is None:
                # Logic: Exit worker loop.
                break

            # Stage 2: Wait for execution signal.
            self.execute_scripts_event.wait()
            self.execute_scripts()

            # Stage 3: Signal task completion and reset for next cycle.
            self.done.set()
