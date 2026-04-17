"""
@bc6ab4a2-27ba-4a35-9bbe-1507685da9e5/device.py
@brief Concurrent sensor network simulation with dynamic thread spawning and multi-stage barrier synchronization.
* Algorithm: Event-driven worker thread management with location-based locking and shared barrier coordination.
* Functional Utility: Manages individual device lifecycles, neighbor interactions, and parallel execution of data processing scripts.
"""

from threading import Thread, Lock, Event, Condition, Semaphore

class ReusableBarrier():
    """
    @brief Synchronizes a fixed number of threads, allowing them to wait until all have arrived at the barrier.
    * Functional Utility: Facilitates phased execution in a multi-threaded environment.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with the target thread count.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        @brief Blocks the calling thread until the specified number of threads have called wait().
        Algorithm: Monitor-based synchronization using condition variables.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Logic: Last thread to arrive releases all waiting threads.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    @brief Represents a sensor node that executes scripts and coordinates with neighbors.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and starts its management thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_event = Event()

        self.lock_location = []
        self.lock_n = Lock()
        self.barrier = None

        self.thread_script = []
        self.num_thread = 0
        # Resource Constraint: Limits the number of concurrent scripts to 8 per device.
        self.sem = Semaphore(value=8)

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collective initialization of shared resources for all devices.
        Invariant: Initialized only by the device with ID 0 to ensure consistent global state.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # Logic: Pre-allocates locks for potential sensor locations.
            for _ in xrange(25):
                self.lock_location.append(Lock())

            # Logic: Distributes shared synchronization objects to all peers.
            for dev in devices:
                dev.barrier = barrier
                dev.lock_location = self.lock_location
                dev.setup_event.set()

    def assign_script(self, script, location):
        """
        @brief Appends a new script task or signals completion of the current timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves local sensor data for a specific location.
        """
        return self.sensor_data[location] if location in \
            self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates local sensor data for a specific location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device management thread.
        """
        self.thread.join()

    def shutdown_script(self):
        """
        @brief Cleans up script execution threads after a simulation phase.
        """
        for i in xrange(self.num_thread):
            self.thread_script[i].join()

        for i in xrange(self.num_thread):
            del self.thread_script[-1]

        self.num_thread = 0

class NewThreadScript(Thread):
    """
    @brief Individual thread spawned for a single script execution task.
    """
    
    def __init__(self, parent, neighbours, location, script):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.parent = parent
        self.location = location
        self.script = script

    def run(self):
        """
        @brief Executes the script logic while ensuring data consistency.
        Pre-condition: Must acquire the location-specific lock.
        """
        with self.parent.lock_location[self.location]:
            # Distributed Data Aggregation: Collects data from peers and self.
            script_data = []
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.parent.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Execution Logic: Processes data and propagates results back to neighbors and self.
            if script_data != []:
                result = self.script.run(script_data)

                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.parent.set_data(self.location, result)
            
            # Post-condition: Signal task completion to allow more scripts to start.
            self.parent.sem.release()

class DeviceThread(Thread):
    """
    @brief Main management thread coordinating timepoints and script dispatching.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core device lifecycle loop.
        Algorithm: Alternating between script dispatch and barrier synchronization.
        """
        # Block Logic: Ensures initial setup is complete before starting the simulation.
        self.device.setup_event.wait()

        while True:
            # Logic: Neighbor discovery phase.
            with self.device.lock_n:
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    break

            # Block Logic: Waits for the signal that all scripts for the current timepoint are assigned.
            self.device.timepoint_done.wait()

            # Dispatch Phase: Spawns a thread for each assigned script, constrained by the device semaphore.
            for (script, location) in self.device.scripts:
                self.device.sem.acquire()
                self.device.thread_script.append(NewThreadScript \
                    (self.device, neighbours, location, script))

                self.device.num_thread = self.device.num_thread + 1
                self.device.thread_script[-1].start()

            # Synchronization Phase: Multi-barrier coordination to ensure consistency across the network.
            self.device.barrier.wait()
            self.device.shutdown_script()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
