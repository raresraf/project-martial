"""
@file device.py
@brief Distributed sensor network simulation with synchronized script execution.
@details Implements a peer-to-peer device architecture where entities execute data 
aggregation scripts. Shared state synchronization is managed via class-level barriers 
and location-specific locks.
"""

from threading import Event, Thread, Lock, Condition

class Device(object):
    """
    @brief Represents an autonomous sensing unit in a distributed cluster.
    Functional Utility: Manages local sensor buffers and coordinates cluster-wide 
    concurrency using shared class-level synchronization primitives.
    """
    
    # Static Class Attributes: Shared across all device instances for global synchronization.
    dev_barrier = None
    dev_locks = []

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier for the instance.
        @param sensor_data Local key-value store of sensor readings.
        @param supervisor Reference to the central network topology manager.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.device_lock = Lock()
        self.devices = []
        # Lifecycle: Spawns the main management thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of shared synchronization resources.
        Logic: The first device in the list (rank 0) is responsible for initializing 
        the global lock registry and the cyclic barrier.
        """
        self.devices = devices

        if self.devices[0].device_id == self.device_id:
            list_loc = []
            # Discovery: Collects all unique sensor locations across the entire cluster.
            for device_ in self.devices:
                for location in list(device_.sensor_data.viewkeys()):
                    if location not in list_loc:
                        list_loc.append(location)

            # Allocation: Creates a dedicated mutual exclusion lock for each unique location.
            for index in range(len(list_loc)):
                Device.dev_locks.append(Lock())

            # Barrier Initialization: Creates a reusable barrier for temporal synchronization.
            if Device.dev_barrier is None:
                Device.dev_barrier = ReusableBarrierCond(len(self.devices))

    def assign_script(self, script, location):
        """
        @brief Schedules a processing script for execution on specific location data.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Termination: Signals the end of script assignment for the current cycle.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of local sensor state.
        """
        with self.device_lock:
            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        """
        @brief Thread-safe update of local sensor state.
        """
        self.device_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.device_lock.release()

    def shutdown(self):
        """
        @brief Gracefully terminates the device's management lifecycle.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Management thread responsible for the high-level control loop of a device.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Continuous cycle of neighbor discovery, script execution, and synchronization.
        """
        while True:
            # Topology Discovery: Fetches the current topological neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until the device is notified that all scripts for the cycle are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            threads_script = []

            /**
             * Block Logic: Parallel execution of assigned scripts.
             * Logic: Spawns a dedicated ScriptThread for each (script, location) pair.
             */
            for (script, location) in self.device.scripts:
                thread = ScriptThread(self.device, script, neighbours, location)
                threads_script.append(thread)

            for thread in threads_script:
                thread.start()

            # Sync: Wait for all local script executions to finalize.
            for thread in threads_script:
                thread.join()

            # Global Synchronization: Aligns all devices at the end of the timepoint cycle.
            Device.dev_barrier.wait()


class ScriptThread(Thread):
    """
    @brief Worker thread for executing a specific data aggregation script.
    Functional Utility: Implements a distributed Map-Reduce operation over a shared sensor location.
    """

    def __init__(self, device, script, neighbours, location):
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.location = location

    def run(self):
        """
        @brief Execution logic for a single script run.
        Critical Section: Uses the global location-specific lock to ensure atomicity.
        """
        script_data = []

        # Lock Acquisition: Prevents concurrent modifications to data for this specific location.
        Device.dev_locks[self.location].acquire()

        # Map Phase: Aggregates data from neighbors.
        for device_ in self.neighbours:
            data = device_.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Local Data Collection.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Computation Phase.
            result = self.script.run(script_data)
            
            # Reduce/Update Phase: Writes the resultant state back to all participants.
            for device_ in self.neighbours:
                device_.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        # Release the location lock for other devices/threads.
        Device.dev_locks[self.location].release()


class ReusableBarrierCond(object):
    """
    @brief Implementation of a cyclic barrier using Condition variables.
    Functional Utility: Enables recurring synchronization points across a fixed set of threads.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        @brief Blocks the calling thread until the threshold number of threads have arrived.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread triggers release of all blocked participants.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Block until notified.
            self.cond.wait()
        self.cond.release()
