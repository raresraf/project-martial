"""
@file device.py
@brief Distributed simulation framework with asynchronous script execution and barrier synchronization.

Functional Intent: Implements a multi-threaded environment where individual devices 
coordinate task execution using a shared memory model. It leverages a reusable 
barrier to align timepoints across the network and fine-grained locking to 
ensure data consistency during localized state transitions.

Domain: Production Systems, Distributed Computing, Parallel Processing.
"""

from threading import Condition, Event, RLock, Thread

class ReusableBarrier(object):
    """
    @brief Reusable thread synchronization barrier using a condition variable.
    
    Functional Utility: Orchestrates a rendezvous point for a fixed number of 
    threads. Unlike standard barriers, it supports dynamic reconfiguration 
    (via reinit) and can be immediately reused for subsequent execution phases.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with a target thread count.
        @param num_threads The total number of threads that must arrive before release.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        # Synchronization: Protects the arrival counter and coordinates thread suspension/awakening.
        self.cond = Condition()

    def reinit(self):
        """
        @brief Dynamically reduces the participant count and enters a wait state.
        
        Logic: Used when a device leaves the network, ensuring the remaining 
        threads are not permanently blocked at the barrier.
        """
        self.cond.acquire()
        self.num_threads -= 1
        self.cond.release()
        self.wait()

    def wait(self):
        """
        @brief Blocks the calling thread until the collective rendezvous is reached.
        
        Algorithm: Last-in-releases-all pattern.
        Invariant: At the moment of release, the internal counter is reset to 
        the full thread quota for the next cycle.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Block Logic: The final thread to arrive awakens the entire swarm.
            self.cond.notify_all() 
            self.count_threads = self.num_threads 
        else:
            # Block Logic: Arrival threads suspend here until notify_all is called.
            self.cond.wait() 
        self.cond.release()

class Device(object):
    """
    @brief Represents a physical or logical computational node in the network.
    
    Functional Utility: Manages device-local sensor state and orchestrates 
    parallel task execution. Coordinates with global peer devices through 
    shared synchronization primitives established during the setup phase.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device's operational state and starts its manager thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Synchronization: Event-based signaling for asynchronous message and task delivery.
        self.script_received = Event()
        self.start = Event()
        self.timepoint_done = Event()

        self.scripts = [] 
        self.scripts_to_process = [] 
        self.nr_script_threats = 0 

        # Control: Bootstraps the primary management loop in a separate thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        
        self.script_threats = [] 
        self.barrier_devices = None 
        self.neighbours = None 
        
        # Optimization: Throttling parameter for controlling local CPU core utilization.
        self.cors = 8 
        
        # Synchronization: Global locks for protecting shared data planes across the device cluster.
        self.lock = None 
        self.lock_self = None 
        self.results = {} 
        self.results_lock = None 

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Orchestrates the cluster-wide initialization of shared sync resources.
        
        Algorithm: Lazy initialization with double-check locking logic.
        Logic: Ensures that all devices in the simulation converge on a single 
        set of RLock and Barrier instances to maintain global consistency.
        """
        # Block Logic: Pre-execution task queueing.
        for script in self.scripts:
            self.lock.acquire()
            self.scripts_to_process.append(script)
            self.lock.release()

        # Invariant: Each device in the 'devices' list is updated to reference the same sync objects.
        if not self.lock_self:
            lock = RLock()
            for device in devices:
                device.lock_self = lock

        self.lock_self.acquire()
        if not self.lock:
            rlock = RLock()
            for device in devices:
                device.lock = rlock
        self.lock_self.release()

        self.lock_self.acquire()
        if not self.results_lock:
            results_lock = RLock()
            for device in devices:
                device.results_lock = results_lock
        self.lock_self.release()

        self.lock_self.acquire()
        if not self.barrier_devices:
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier_devices = barrier
                # Synchronization: Triggers all management threads to transition to 'active' state.
                device.start.set() 
        self.lock_self.release()

    def assign_script(self, script, location):
        """
        @brief Registers a new processing task and triggers the availability signal.
        """
        if script is not None:
            self.lock.acquire()
            self.scripts.append((script, location))
            self.scripts_to_process.append((script, location))
            self.script_received.set() 
            self.lock.release()
        else:
            # Logic: Signaling 'None' indicates the end of task distribution for the current epoch.
            self.lock.acquire()
            self.timepoint_done.set() 
            self.script_received.set() 
            self.lock.release()

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of sensor data at a specific logical address.
        """
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else:
            data = None
        return data

    def set_data(self, location, data):
        """
        @brief Atomically updates sensor data for a specific logical address.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
        
    def shutdown(self):
        """
        @brief Gracefully terminates the device by waiting for the management loop to settle.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief Management thread that orchestrates task scheduling and inter-device communication.
    
    Logic: Implements the high-level workflow for a simulation timepoint: 
    neighborhood discovery, data gathering, parallel script dispatch, 
    and result dissemination.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.device.neighbours = None

    def run(self):
        """
        @brief Core operational cycle for the device node.
        
        Algorithm: Iterative epoch-based execution.
        Logic: 
        1. Waits for global network readiness.
        2. Discovers neighborhood topology.
        3. Executes localized parallel data transformations.
        4. Reaches global consensus at the barrier.
        """
        self.device.start.wait()
        while True:
            # Block Logic: Timepoint initialization.
            self.device.scripts_to_process = []
            for script in self.device.scripts:
                self.device.scripts_to_process.append(script)

            # Block Logic: Dynamic topology resolution via supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                # Synchronization: Signals departure to prevent remaining peers from hanging.
                self.device.barrier_devices.reinit()
                break

            self.device.results = {}
            
            /**
             * Block Logic: Script processing batch loop.
             * Invariant: Continuously spawns worker threads until all scripts for 
             * the current timepoint have been executed and joined.
             */
            while True:
                if not self.device.timepoint_done.is_set():
                    self.device.script_received.wait()
                    self.device.script_received.clear()

                if len(self.device.scripts_to_process) == 0:
                    if self.device.timepoint_done.is_set():
                        break

                while len(self.device.scripts_to_process):
                    list_threats = []
                    self.device.script_threats = []
                    self.device.nr_script_threats = 0
                    
                    # Logic: Throttles worker spawning based on configured core count.
                    while len(self.device.scripts_to_process) and self.device.nr_script_threats < self.device.cors:
                        script, location = self.device.scripts_to_process.pop(0)
                        list_threats.append((script, location))
                        self.nr_script_threats += 1

                    for script, location in list_threats:
                        script_data = []
                        
                        # Block Logic: Neighborhood data aggregation phase.
                        neighbours = self.device.neighbours
                        for device in neighbours:
                            device.lock_self.acquire()
                            data = device.get_data(location)
                            device.lock_self.release()
                            if data is not None:
                                script_data.append(data)
                        
                        self.device.lock_self.acquire()
                        data = self.device.get_data(location)
                        self.device.lock_self.release()
                        if data is not None:
                            script_data.append(data)

                        # Parallel Execution: Offloads computational task to a dedicated worker.
                        thread_script_d = ScriptThread(self.device, script, location, script_data)
                        self.device.script_threats.append(thread_script_d)
                        thread_script_d.start()

                    # Synchronization: Waits for all local workers to complete their sub-tasks.
                    for thread in self.device.script_threats:
                        thread.join()

            # Block Logic: State dissemination phase.
            for location, result in self.device.results.iteritems():
                for device in self.device.neighbours:
                    device.lock_self.acquire()
                    device.set_data(location, result)
                    device.lock_self.release()
                
                self.device.lock_self.acquire()
                self.device.set_data(location, result)
                self.device.lock_self.release()

            # Synchronization: Finalizes the timepoint and waits for all network peers.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            self.device.barrier_devices.wait()

class ScriptThread(Thread):
    """
    @brief Worker thread specialized for executing a single data-processing script.
    
    Functional Utility: Isolates script execution logic to prevent blocking 
    the device manager thread. Ensures thread-safe updates to the device-local 
    results map upon completion.
    """

    def __init__(self, device, script, location, script_data):
        Thread.__init__(self, name="Device Script Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.script_data = script_data

    def run(self):
        """
        @brief Executes the transformation script and registers the result.
        """
        if self.script_data != []:
            # Functional Intent: Executes user-defined script logic on aggregated neighbor data.
            result = self.script.run(self.script_data)
            
            # Synchronization: Protects the result map from concurrent write hazards.
            self.device.results_lock.acquire()
            self.device.results[self.location] = result
            self.device.results_lock.release()
        
        # Logic: Decrements the device's active thread counter to allow further scheduling.
        self.device.nr_script_threats -= 1
