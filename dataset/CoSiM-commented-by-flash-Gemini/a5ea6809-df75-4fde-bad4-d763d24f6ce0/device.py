"""
@file device.py
@brief Distributed sensing and data processing system with multi-threaded execution.
@details Implements a peer-to-peer network of devices that perform synchronized data 
aggregation using worker threads and a reusable barrier for temporal alignment.
"""

from threading import Event, Thread, Condition, Lock

class Device(object):
    """
    @brief Core logic controller for a distributed sensing unit.
    Functional Utility: Manages local sensor data, organizes execution of processing 
    scripts, and shares synchronization resources with peer devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier for the device instance.
        @param sensor_data Dictionary of local sensor location-value pairs.
        @param supervisor Coordinator for network topology management.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Sync: Event to signal when all scripts for a timepoint have been assigned.
        self.scripts_received = Event()
        self.scripts_dict = {}
        self.locations_locks = {}
        self.timepoint_done = None
        self.neighbours = None
        # Background Loop: Spawns the main management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of shared synchronization primitives.
        Logic: Ensures all devices in the cluster share the same barrier and 
        location-specific mutual exclusion locks.
        """
        nr_devices = len(devices)
        # Shared Barrier: All devices synchronize on a single ReusableBarrierCond.
        if self.timepoint_done is None:
            self.timepoint_done = ReusableBarrierCond(nr_devices)
            for device in devices:
                if device.timepoint_done is None and device != self:
                    device.timepoint_done = self.timepoint_done

        /**
         * Block Logic: Dynamic lock allocation for concurrent data access.
         * Logic: Maps each unique sensor location to a global Lock shared across the cluster.
         */
        for location in self.sensor_data.keys():
            if location not in self.locations_locks:
                self.locations_locks[location] = Lock()
                for device in devices:
                    if location not in device.locations_locks and \
                        device != self:
                        device.locations_locks[location] = \
                            self.locations_locks[location]

    def assign_script(self, script, location):
        """
        @brief Registers a computation task for a specific sensor location.
        """
        if script is not None:
            if location in self.scripts_dict:
                self.scripts_dict[location].append(script)
            else:
                self.scripts_dict[location] = [script]
        else:
            # Protocol: Signals the end of script assignment phase.
            self.scripts_received.set()

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of sensor state.
        """
        return self.sensor_data[location] \
            if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates local sensor state.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Terminates the device lifecycle by joining the management thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Orchestrator thread that manages execution cycles (timepoints).
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main control loop: discovery -> execution -> synchronization.
        """
        while True:
            # Discovery: Queries the supervisor for the current neighborhood.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                break

            # Block until assignment phase is complete.
            self.device.scripts_received.wait()

            /**
             * Block Logic: Parallel script execution.
             * Logic: Spawns one worker thread per active sensor location.
             */
            threads = []
            for location in self.device.scripts_dict.keys():
                thread = DeviceWorkerThread(self.device, location)
                thread.start()
                threads.append(thread)

            # Sync: Wait for all local computation to complete.
            for thread in threads:
                thread.join()

            # Reset synchronization flags for next cycle.
            self.device.scripts_received.clear()

            # Global Sync: Ensure cluster alignment at the timepoint barrier.
            self.device.timepoint_done.wait()

class DeviceWorkerThread(Thread):
    """
    @brief Specialized worker responsible for executing scripts on a specific location.
    """

    def __init__(self, device, location):
        Thread.__init__(self, name="Worker %d-%s" % (device.device_id, location))
        self.device = device
        self.location = location

    def run(self):
        """
        @brief Execution logic for all scripts assigned to a single location.
        Critical Section: Uses the shared location lock to prevent race conditions during aggregation.
        """
        for script in self.device.scripts_dict[self.location]:
            # Lock: Ensures atomic read-modify-write across the distributed state.
            self.device.locations_locks[self.location].acquire()

            script_data = []
            
            # Map Phase: Aggregates data from neighbors.
            for device in self.device.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            # Inclusion: Adds the device's own data to the processing set.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Computation: Executes the user-defined aggregation script.
                result = script.run(script_data)
                
                # Reduce/Sync Phase: Updates all participating devices with the new state.
                for device in self.device.neighbours:
                    device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)

            self.device.locations_locks[self.location].release()


class ReusableBarrierCond(object):
    """
    @brief Cyclic barrier implementation using Condition variables.
    Functional Utility: Provides a synchronization point for a fixed number of threads.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        @brief Blocks the caller until the N-th thread arrives.
        """
        self.cond.acquire()
        self.count_threads -= 1

        if self.count_threads == 0:
            # Turnstile release: Wakes all waiting threads and resets for the next cycle.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Passive wait.
            self.cond.wait()

        self.cond.release()
