"""
@e7987aac-868a-4829-aed7-10cbf4c909fd/device.py
@brief Distributed sensor processing simulation using a persistent worker pool and tiered barrier synchronization.
* Algorithm: Static task partitioning (Stride-8) across a pool of persistent worker threads with local and global barriers and per-location semaphore locking.
* Functional Utility: Orchestrates simulation timepoints across a network of devices by managing a local pool of worker threads that collaboratively process scripts and synchronize state updates through shared global resources.
"""

from threading import Event, Thread, Semaphore
from reusable_barrier import TimePointsBarrier, ClassicBarrier

class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, internal worker pool, and shared synchronization state.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and prepares its internal thread infrastructure.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # Intent: Signals completion of task delivery for the current phase.

        self.threads = []
        self.neighbours = []
        self.num_threads = 8 # Domain: Concurrency Scaling - fixed workers per node.

        self.locations_semaphore = None # Intent: Shared list of semaphores (binary locks) for sensor locations.
        self.devices_barrier = None     # Intent: Global barrier for cluster-wide alignment.
        self.neighbours_barrier = None  # Intent: Local barrier for aligning the 8 internal workers.
        self.all_devices = None

    def set_neighbours(self, new_neighbours):
        """
        @brief Updates the local cache of neighbor node references.
        """
        self.neighbours = new_neighbours

    def set_devices_barrier(self, barrier):
        """
        @brief Links the device to the cluster-wide synchronization point.
        """
        self.devices_barrier = barrier

    def set_locations_semaphore(self, locations_semaphore):
        """
        @brief Assigns the shared global lock set for location-level synchronization.
        """
        self.locations_semaphore = locations_semaphore

    def get_locations(self, location_list):
        """
        @brief Aggregates all local sensor location IDs into a provided list.
        """
        for location in self.sensor_data:
            location_list.append(location)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization and distribution of shared simulation resources.
        Invariant: Root node (ID 0) initializes the cluster-wide barrier and location locks.
        """
        if self.device_id == 0:
            # Logic: Barrier shared across all participating devices.
            barrier = ClassicBarrier(len(devices))
            locations = []
            locations_semaphore = []
            
            # Logic: Discover all unique sensor locations across the cluster to create granular locks.
            for device in devices:
                device.get_locations(locations)
            locations = sorted(list(set(locations)))

            # Logic: Pre-allocates a lock (semaphore) for every discovered global location.
            for i in range(0, len(locations)):
                locations_semaphore.append(Semaphore(value=1))

            for device in devices:
                device.set_devices_barrier(barrier)
                device.set_locations_semaphore(locations_semaphore)

        self.all_devices = devices
        
        # Intent: Local barrier implementation specifically tailored for device threads to coordinate timepoints.
        self.neighbours_barrier = TimePointsBarrier(self.num_threads, self)

        # Logic: Spawns the persistent local execution workforce.
        for i in range(self.num_threads):
            current_thread = DeviceThread(self, i)
            current_thread.start()
            self.threads.append(current_thread)

    def assign_script(self, script, location):
        """
        @brief Receives a processing task for the current simulation cycle.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: Signals completion of task delivery for this node.
            self.timepoint_done.set()

    def has_data(self, location):
        """
        @brief Diagnostic check for data presence at a specific location.
        """
        return location in self.sensor_data

    def get_data(self, location):
        """
        @brief Standard data retrieval interface for local sensor readings.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Standard data update interface for local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device's persistent worker threads.
        """
        for i in range(self.num_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    @brief Persistent worker thread implementing the execution lifecycle for partitioned sensor scripts.
    """

    def __init__(self, device, thread_id):
        """
        @brief Initializes the worker with its device context and local sequence ID.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id # Intent: Target for static work partitioning.

    def run(self):
        """
        @brief Core execution loop for simulation phases.
        Algorithm: Multi-stage synchronization with static stride-based task partitioning.
        """
        while True:
            # Synchronization Phase 1: Local alignment before neighbor refresh.
            # Invariant: The TimePointsBarrier likely handles global alignment internally as well.
            self.device.neighbours_barrier.wait()

            if self.device.neighbours is None:
                # Logic: Shutdown path.
                break

            # Block Logic: Waits for script delivery completion for the current timepoint.
            self.device.timepoint_done.wait()

            if len(self.device.neighbours) != 0:
                devices_with_date = []
                
                # Task Execution Phase: Static Partitioning.
                # Strategy: Each thread processes every 8th script starting from its ID (Round-Robin style).
                for index in range(self.thread_id, len(self.device.scripts), self.device.num_threads):
                    (script, location) = self.device.scripts[index]
                    
                    # Pre-condition: Acquire shared global location lock (semaphore) for atomic distributed update.
                    self.device.locations_semaphore[location].acquire()
                    script_data = []

                    # Distributed Aggregation: Collect readings from neighbors and self.
                    for device in self.device.neighbours:
                        if device.has_data(location):
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                                devices_with_date.append(device)

                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                        devices_with_date.append(self.device)

                    if script_data != []:
                        # Execution and Propagation Phase: Updates all nodes that contributed to the dataset.
                        result = script.run(script_data)
                        for device in devices_with_date:
                            device.set_data(location, result)
                        devices_with_date = []

                    # Post-condition: Release global location lock.
                    self.device.locations_semaphore[location].release()
