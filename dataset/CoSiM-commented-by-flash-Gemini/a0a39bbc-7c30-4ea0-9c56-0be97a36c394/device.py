"""
@file device.py
@brief Distributed sensor network simulation with multi-threaded data processing and synchronization.
@details Implements a system where autonomous devices collect sensor data, execute scripts via 
worker threads, and synchronize their state using a reusable barrier mechanism.
"""

from threading import Event, Thread, Lock, Semaphore

class Device(object):
    """
    @brief Core entity representing a hardware unit in the sensor network.
    Functional Utility: Manages local sensor state, active scripts, and coordinate system-wide 
    synchronization via shared barriers and locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier for the device instance.
        @param sensor_data Dictionary mapping locations to local sensor readings.
        @param supervisor Reference to the network coordinator for topology discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Sync Primitives: Events to signal script availability and timepoint completion.
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.lock = {}
        self.barrier = None
        self.devices = []
        # Background Execution: Starts the primary device management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared synchronization resources across a group of peer devices.
        @param devices List of peer device instances in the local cluster.
        Logic: Allocates a common barrier and a global lock registry for all sensor locations.
        """
        self.devices = devices
        self.barrier = ReusableBarrierSem(len(self.devices))

        # Functional Utility: Initialization of mutual exclusion locks per unique sensor location.
        for location in self.sensor_data:
            self.lock[location] = Lock()
        for device in devices:
            for location in device.sensor_data:
                self.lock[location] = Lock()

        # Distribution: Propagates shared sync primitives to all cluster members.
        for i in xrange(len(self.devices)):
            self.devices[i].barrier = self.barrier
            self.devices[i].lock = self.lock

    def assign_script(self, script, location):
        """
        @brief Schedules a processing task (script) for a specific physical location.
        Logic: If script is None, signals that the current temporal window (timepoint) is closed.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of sensor readings (assumes external locking).
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Thread-safe update of sensor readings (assumes external locking).
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device's management lifecycle.
        """
        self.thread.join()


class MyThread(Thread):
    """
    @brief Worker thread responsible for executing a specific script on distributed data.
    Functional Utility: Implements a Map-Reduce style local computation on a shared location.
    """

    def __init__(self, my_id, device, neighbours, lock, script, location):
        Thread.__init__(self, name="Thread %d from device %d" % (my_id, device.device_id))
        self.device = device
        self.my_id = my_id
        self.neighbours = neighbours
        self.lock = lock
        self.script = script
        self.location = location

    def run(self):
        """
        @brief Execution logic for a single script run.
        Critical Section: Protected by the per-location lock to prevent race conditions during data aggregation.
        """
        with self.lock[self.location]:
            script_data = []
            
            # Map Phase: Collects data from the device and its immediate topological neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Computation: Executes the opaque logic defined in the script object.
                result = self.script.run(script_data)

                # Reduce/Sync Phase: Propagates the computed result back to all participating devices.
                for device in self.neighbours:
                    device.set_data(self.location, result)

                self.device.set_data(self.location, result)

    def shutdown(self):
        self.join()


class DeviceThread(Thread):
    """
    @brief Management thread that orchestrates the lifecycle of worker threads for a device.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.numThreads = 0
        self.listThreads = []

    def run(self):
        """
        @brief Continuous loop that handles script assignment and execution cycles.
        Synchronization: Uses barriers to ensure all devices in the cluster progress through timepoints together.
        """
        while True:
            # Topology Discovery: Fetches the current set of network neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until scripts are assigned for the current cycle.
            self.device.script_received.wait()

            /**
             * Block Logic: Manages a pool of worker threads with a fixed capacity (8).
             * Logic: Reuses thread slots by joining dead threads when the pool is full.
             */
            for (script, location) in self.device.scripts:
                if len(self.listThreads) < 8:
                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.append(thread)
                    thread.start()
                    self.numThreads += 1
                else:
                    index = -1
                    # Optimization: Finds an inactive thread to replace, minimizing overhead.
                    for i in xrange(len(self.listThreads)):
                        if not self.listThreads[i].is_alive():
                            self.listThreads[i].join()
                            index = i

                    self.listThreads.remove(self.listThreads[index])

                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.insert(index,thread)
                    self.listThreads[index].start()
                    self.numThreads += 1

            # Functional Utility: Ensures all workers finish before proceeding to the next synchronization phase.
            for i in xrange(len(self.listThreads)):
                self.listThreads[i].join()

            # Signals that the device has finished its local processing for the timepoint.
            self.device.timepoint_done.wait()
            
            # Reset state for the next temporal window.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            
            # Global Sync Barrier: Prevents any device from starting the next timepoint until all are ready.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    @brief Custom implementation of a two-phase cyclic barrier using semaphores.
    Functional Utility: Enables multiple threads to synchronize in cycles without re-allocation.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        @brief Primary wait method implementing the two-phase turnstile protocol.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief First turnstile: Collects all threads before releasing them.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread to arrive releases the others.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second turnstile: Ensures all threads have exited phase 1 before the barrier resets.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()
