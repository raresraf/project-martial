"""
@e167713d-ad87-46f4-aa9e-1b9679494d30/device.py
@brief Distributed sensor processing simulation with a collective thread pool and global synchronization.
* Algorithm: Atomic task consumption from a shared per-device list using a pool of 8 persistent worker threads, coordinated via multi-stage global barriers.
* Functional Utility: Orchestrates simulation timepoints by managing concurrent script execution, using fine-grained location locks and coarse-grained device locks to ensure cluster-wide data consistency.
"""

from threading import Event, Thread, Lock
from barrier import Barrier

class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, coordination state, and internal worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the internal worker thread pool.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbours = []
        self.scripts = []
        self.temp_scripts = [] # Intent: Active task list for the current simulation phase.

        self._thread_list = []
        self.timepoint_done = Event()
        self.device_lock = Lock()      # Intent: Serializes global updates to this device's sensor data.
        self.script_list_lock = Lock() # Intent: Protects the shared task list during worker consumption.
        self.locations_locks = {}      # Intent: Map of global sensor location locks.
        self.device_thread_barrier = None
        self.thread_number = 8 # Domain: Concurrency Scaling - fixed workers per device.

        # Logic: Spawns the persistent worker thread pool.
        for thread_id in xrange(self.thread_number):
            thread = DeviceThread(self, thread_id)
            self._thread_list.append(thread)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization and distribution of shared simulation resources.
        Invariant: Establishes a cluster-wide barrier and a global set of location locks.
        """
        # Logic: Collective barrier creation - shared across ALL threads of ALL devices.
        if self.device_thread_barrier == None:
            self.device_thread_barrier = Barrier(len(devices) * self.thread_number)
            for dev in devices:
                dev.device_thread_barrier = self.device_thread_barrier

        # Logic: Global lock pool creation for unique sensor locations.
        max_location = -1
        if not self.locations_locks:
            for dev in devices:
                for key in dev.sensor_data:
                    if key > max_location:
                        max_location = key
            for i in xrange(max_location + 1):
                self.locations_locks[i] = Lock()
            for dev in devices:
                dev.locations_locks = self.locations_locks

        # Logic: Starts the pre-initialized worker pool.
        for thread_id in xrange(self.thread_number):
            self._thread_list[thread_id].start()

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing task for the current simulation cycle.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signals completion of task arrival for this node.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Basic data retrieval for sensor locations.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Basic data update for sensor locations.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device's worker threads.
        """
        for i in xrange(len(self._thread_list)):
            self._thread_list[i].join()


class DeviceThread(Thread):
    """
    @brief Persistent worker thread implementing the execution lifecycle for assigned scripts.
    """

    def __init__(self, device, thread_id):
        """
        @brief Initializes the worker with its device context and local sequence ID.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id # Intent: Used to designate a coordinator thread (ID 0).

    def run(self):
        """
        @brief Core execution loop for the worker thread.
        Algorithm: Multi-stage barrier synchronization with role-based coordination.
        """
        while True:
            # Synchronization Phase 1: Collective alignment before neighbor discovery.
            self.device.device_thread_barrier.wait()
            if self.thread_id == 0:
                # Logic: Designated coordinator thread refreshes the neighbor set.
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # Synchronization Phase 2: Ensures all threads have seen the new neighborhood.
            self.device.device_thread_barrier.wait()
            if self.device.neighbours is None:
                # Logic: Shutdown signal.
                break

            if self.thread_id == 0:
                # Logic: Coordinator thread manages script batch arrival and resets the active task list.
                self.device.timepoint_done.wait()
                self.device.timepoint_done.clear()
                self.device.temp_scripts = list(self.device.scripts)

            # Synchronization Phase 3: Collective alignment before task execution.
            self.device.device_thread_barrier.wait()

            done_iter = False
            while True:
                # Task Consumption Phase.
                item = ()
                # Pre-condition: Must acquire list lock for atomic task removal.
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

                # Pre-condition: Acquire global location lock for atomic distributed update.
                self.device.locations_locks[location].acquire()

                script_data = []
                # Distributed Aggregation: Collect readings from neighbors and local node.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Execution Phase: Computes new state.
                    result = script.run(script_data)

                    # Propagation Phase: Broadcasts results under coarse-grained device locks.
                    for device in self.device.neighbours:
                        device.device_lock.acquire()
                        device.set_data(location, result)
                        device.device_lock.release()

                    self.device.device_lock.acquire()
                    self.device.set_data(location, result)
                    self.device.device_lock.release()

                # Post-condition: Release global location lock.
                self.device.locations_locks[location].release()
