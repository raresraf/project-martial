"""
@e609cfff-06a5-4cd5-b8e6-f1c08a3d23af/device.py
@brief Distributed sensor network simulation with multi-phase synchronized worker pools.
This module implements a sophisticated parallel processing architecture where 
computational scripts are executed by a pool of persistent worker threads 
(DeviceOwnThread). Each simulation timepoint is orchestrated through multiple 
synchronization phases (Topology Discovery -> Task Distribution -> Execution -> 
Consensus) using a hierarchy of barriers and events. A designated leader node 
(minimum ID) acts as the global resource factory for network-wide synchronization.

Domain: Multi-Phase Synchronization, Leader-Based Orchestration, Persistent Worker Pools.
"""

from threading import Event, Thread, Lock
from barrier import Barrier
from device_thread import DeviceOwnThread

class Device(object):
    """
    Core network node representation.
    Functional Utility: Manages local data, coordinates leader-based resource 
    discovery, and distributes tasks to an internal parallel worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        self.new_sensor_data = {}
        self.new_sensor_data_lock = Lock()
        self.other_devices = []

        # Simulation Phase Events.
        self.ready_to_get_scripts = Event()
        self.got_all_scripts = Event()
        self.assign_script_lock = Lock()
        # Mutex for protecting local data state.
        self.set_data_lock = Lock()

        # Phase Barriers: sized during setup_devices.
        self.start_loop_barrier = Barrier()
        self.got_scripts_barrier = Barrier()
        self.everyone_done = Barrier()

        # Spatial lock pool for network-wide mutual exclusion.
        self.location_mutex = []
        self.get_neighbours_lock = Lock()
        self.data_ready = Event()
        self.data_ready.set()

        # Worker Pool Logic.
        self.own_threads = []
        self.power = 20 # degree of parallelism.
        for _ in range(0, self.power): 
            new_thread = DeviceOwnThread(self)
            self.own_threads.append(new_thread)
            new_thread.start()

        # Round-robin assignment pointer.
        self.own_threads_rr = 0
        self.initialized = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

    def get_main_device(self):
        """
        Leader Election Logic.
        Logic: Identifies the node with the minimum ID to act as the coordinator.
        @return: The coordinator Device instance.
        """
        min_device = self
        min_id = self.device_id
        for device in self.other_devices:
            if device.device_id < min_id:
                min_device = device
                min_id = device.device_id
        return min_device

    def get_start_loop_barrier(self):
        """Accesses the shared arrival barrier from the coordinator."""
        return self.get_main_device().start_loop_barrier

    def get_got_scripts_barrier(self):
        """Accesses the shared task-ready barrier from the coordinator."""
        return self.get_main_device().got_scripts_barrier

    def get_get_neighbours_lock(self):
        """Accesses the shared topology lock."""
        return self.get_main_device().get_neighbours_lock

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global resource discovery and initialization.
        Logic: If this node is the coordinator, it configures all shared barriers 
        and pre-allocates the global spatial lock pool.
        """
        for device in devices:
            self.other_devices.append(device)

        if self.get_main_device() == self:
            # Atomic setup of shared rendezvous points.
            self.start_loop_barrier.set_n(len(devices))
            self.got_scripts_barrier.set_n(len(devices))
            self.everyone_done.set_n(len(devices))

            # Block Logic: Spatial Lock Pool setup.
            number_of_locations = 0
            for device in devices:
                current_number = max(device.sensor_data) if device.sensor_data else 0
                if current_number > number_of_locations:
                    number_of_locations = current_number
            # Allocate dedicated mutex for every spatial location index.
            for _ in range(0, number_of_locations + 1):
                self.location_mutex.append(Lock())

        self.ready_to_get_scripts.set()
        self.initialized.set()

    def assign_script(self, script, location):
        """
        Workload Distribution.
        Logic: Uses a round-robin strategy to assign tasks to the persistent pool.
        """
        self.ready_to_get_scripts.wait()
        self.assign_script_lock.acquire()
        if script is not None:
            self.own_threads[self.own_threads_rr].assign_script(script, location)
            self.own_threads_rr = (self.own_threads_rr + 1) % len(self.own_threads)
        else:
            # Assignments complete: signal end of step.
            self.ready_to_get_scripts.clear()
            self.data_ready.clear()
            self.got_all_scripts.set()
        self.assign_script_lock.release()

    def get_data(self, location):
        """Atomic retrieval of local sensor data."""
        self.data_ready.wait()
        self.set_data_lock.acquire()
        result = self.sensor_data[location] if location in self.sensor_data else None
        self.set_data_lock.release()
        return result

    def get_temp_data(self, location):
        """Unsynchronized retrieval for internal pipeline use."""
        result = self.sensor_data[location] if location in self.sensor_data else None
        return result

    def set_data(self, location, data):
        """Atomic update of local sensor state."""
        self.set_data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_data_lock.release()

    def shutdown(self):
        """Gracefully joins all management and worker threads."""
        for dot in self.own_threads:
            dot.join()
        self.thread.join()


class DeviceThread(Thread):
    """
    Main orchestration thread for node simulation steps.
    Functional Utility: coordinates the sequence of events and barriers to align 
    local worker pool activities with the rest of the network.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Simulation loop.
        Algorithm: Iterative multi-phase synchronization sequence.
        """
        self.device.initialized.wait()
        self.device.ready_to_get_scripts.set()
        self.device.data_ready.clear()
        while True:
            # Phase 1: Arrival Rendezvous.
            self.device.get_start_loop_barrier().wait()

            # Phase 2: Topology Discovery (Synchronized).
            self.device.get_get_neighbours_lock().acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.get_get_neighbours_lock().release()

            # Signal workers to prepare for the new topology.
            for dot in self.device.own_threads:
                dot.waiting_for_permission.wait()
            for dot in self.device.own_threads:
                dot.neighbours = neighbours
                dot.waiting_for_permission.clear()
                dot.start_loop_condition.acquire()
                dot.start_loop_condition.notify_all()
                dot.start_loop_condition.release()

            if neighbours is None:
                break

            # Phase 3: Wait for Work Assignment.
            self.device.got_all_scripts.wait()
            self.device.got_all_scripts.clear()
            self.device.data_ready.clear()

            # Phase 4: Task Readiness Rendezvous.
            self.device.get_got_scripts_barrier().wait()

            # Phase 5: Trigger Parallel Execution.
            for dot in self.device.own_threads:
                dot.execute_scripts_event.set()

            # Phase 6: Wait for local pool completion.
            for dot in self.device.own_threads:
                dot.done.wait()

            # Phase 7: Global consensus point.
            self.device.get_main_device().everyone_done.wait()

            # Phase 8: Data Commit and Reset for next cycle.
            self.device.data_ready.set()
            for dot in self.device.own_threads:
                dot.done.clear()
                dot.execute_scripts_event.clear()

            self.device.ready_to_get_scripts.set()



from threading import Event, Thread, Condition

class DeviceOwnThread(Thread):
    """
    Persistent worker thread implementation.
    Functional Utility: Participates in coordinated simulation phases, 
    executing computational scripts using conditions and execution events.
    """

    def __init__(self, device):
        Thread.__init__(self)
        self.device = device
        self.done = Event()
        self.scripts = []
        self.execute_scripts_event = Event()
        # Condition variable used for phase arrival and permission flow.
        self.start_loop_condition = Condition()
        self.waiting_for_permission = Event()
        self.neighbours = []

    def assign_script(self, script, location):
        """Queues a task in the worker-local buffer."""
        self.scripts.append((script, location))

    def execute_scripts(self):
        """
        Main computational execution logic.
        Algorithm: Iterates through assigned scripts using global spatial locking.
        """
        for (script, location) in self.scripts:
            # Critical Section: Network-wide spatial lock for the location.
            self.device.get_main_device().location_mutex[location].acquire()
            script_data = []
            
            # Neighborhood aggregation.
            for device in self.neighbours:
                data = device.get_temp_data(location)
                if data is not None:
                    script_data.append(data)

            # Local state integration.
            data = self.device.get_temp_data(location)
            if data is not None:
                script_data.append(data)
                
            if script_data != []:
                # Process and propagate results.
                result = script.run(script_data)
                for device in self.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
                
            # Release spatial mutex.
            self.device.get_main_device().location_mutex[location].release()

    def run(self):
        """
        Worker lifecycle loop.
        Logic: Coordinates between Condition-based phase starts and Event-based 
        execution triggers.
        """
        while True:
            # Gate 1: Synchronized phase entry.
            self.start_loop_condition.acquire()
            self.waiting_for_permission.set()
            # Wait for DeviceThread to signal topology readiness.
            self.start_loop_condition.wait()
            self.start_loop_condition.release()

            if self.neighbours is None:
                break

            # Gate 2: Execution signal.
            self.execute_scripts_event.wait()
            self.execute_scripts()

            # Signal completion of local work.
            self.done.set()
