"""
@d29be76a-403a-4bc2-9df0-78b81ae5de03/device.py
@brief Distributed sensor network simulation with tightly-coupled parallel execution.
This module implements a highly synchronized multi-threaded architecture where all 
worker threads across the entire network are temporally aligned via a global 
barrier. Each node designates a primary thread (ID 0) for topology coordination, 
while the remaining pool participates in a shared-list task distribution model. 
Consistency is guaranteed through a network-wide spatial lock pool and fine-grained 
mutexes for individual device state updates.

Domain: Tightly-Coupled Parallelism, Global Barrier Synchronization, Spatial Locking.
"""

from threading import Event, Thread, Lock
from barrier import Barrier


class Device(object):
    """
    Representation of a network node with an internal worker pool.
    Functional Utility: Manages local data, coordinates the discovery of global 
    locks/barriers, and provides a container for assigned computational tasks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and scaffolds its internal thread pool.
        @param device_id: Unique integer identifier.
        @param sensor_data: Initial state of local sensors.
        @param supervisor: Topology and control hub.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbours = []
        self.scripts = []
        # Transient task buffer for the current timepoint.
        self.temp_scripts = []

        self._thread_list = []
        self.timepoint_done = Event()
        # Mutex for protecting local sensor data state.
        self.device_lock = Lock()
        # Mutex for protecting the shared task list (temp_scripts).
        self.script_list_lock = Lock()
        self.locations_locks = {}
        self.device_thread_barrier = None
        self.thread_number = 8

        # Spawns a pool of persistent worker threads.
        for thread_id in xrange(self.thread_number):
            thread = DeviceThread(self, thread_id)
            self._thread_list.append(thread)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global resource discovery and network-wide initialization.
        Logic: Configures a global barrier encompassing every thread in the network 
        and populates a shared pool of spatial locks for all sensor locations.
        """
        # Block Logic: Global Barrier Setup.
        # Logic: Sizes the barrier to match (Number of Devices * Threads per Device).
        if self.device_thread_barrier == None:
            self.device_thread_barrier = Barrier(len(devices) * self.thread_number)
            for dev in devices:
                dev.device_thread_barrier = self.device_thread_barrier

        # Block Logic: Global Spatial Lock Pool Discovery.
        # Logic: Discovers the maximum location index across all devices to size the pool.
        max_location = -1
        if not self.locations_locks:
            for dev in devices:
                for key in dev.sensor_data:
                    if key > max_location:
                        max_location = key
            # Allocate a dedicated mutex for every possible spatial location.
            for i in xrange(max_location + 1):
                self.locations_locks[i] = Lock()
            # Propagate the lock map to all peer devices.
            for dev in devices:
                dev.locations_locks = self.locations_locks

        # Activate the local thread pool.
        for thread_id in xrange(self.thread_number):
            self._thread_list[thread_id].start()

    def assign_script(self, script, location):
        """Registers a task and signals completion of the simulation step assignment."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for the given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all local worker threads."""
        for i in xrange(len(self._thread_list)):
            self._thread_list[i].join()


class DeviceThread(Thread):
    """
    Worker thread implementation with role-based coordination.
    Functional Utility: Executes scripts from a shared device-level buffer while 
    using multiple global barrier points to synchronize simulation phases.
    """

    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        Main execution loop for the worker.
        Algorithm: Multi-stage synchronization sequence: 
        Wait for Step -> Node 0 Fetches Topology -> Wait for Tasks -> Parallel Process.
        """
        while True:
            # Barrier Point 1: Wait for start of simulation step.
            self.device.device_thread_barrier.wait()
            
            # Role-Based Logic: only the coordinator thread handles topology refresh.
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # Barrier Point 2: Ensure topology is fetched before any thread proceeds.
            self.device.device_thread_barrier.wait()
            
            # Exit Logic.
            if self.device.neighbours is None:
                break

            # Role-Based Logic: coordinator waits for task arrival and initializes the buffer.
            if self.thread_id == 0:
                self.device.timepoint_done.wait()
                self.device.timepoint_done.clear()
                self.device.temp_scripts = list(self.device.scripts)

            # Barrier Point 3: Ensure task buffer is ready for the worker pool.
            self.device.device_thread_barrier.wait()

            done_iter = False
            # Block Logic: Workload Consumption.
            # Threads pull tasks from the shared 'temp_scripts' list until empty.
            while True:
                item = ()
                
                # Atomic Task Acquisition: uses node-level mutex to protect the list.
                self.device.script_list_lock.acquire()
                if len(self.device.temp_scripts) > 0:
                    item = self.device.temp_scripts.pop(0)
                else:
                    done_iter = True
                self.device.script_list_lock.release()

                if done_iter:
                    break

                script = item[0]
                location = item[1]

                # Global Critical Section: ensures atomic access to the spatial location.
                self.device.locations_locks[location].acquire()

                script_data = []
                # Aggregate neighborhood state.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Include local state.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Run logic and propagate results to all nodes in the neighborhood.
                    result = script.run(script_data)

                    for device in self.device.neighbours:
                        # Cross-device update protection.
                        device.device_lock.acquire()
                        device.set_data(location, result)
                        device.device_lock.release()

                    # Self-update.
                    self.device.device_lock.acquire()
                    self.device.set_data(location, result)
                    self.device.device_lock.release()

                # Release the spatial lock.
                self.device.locations_locks[location].release()
