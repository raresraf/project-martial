"""
@file device.py
@brief Distributed sensor unit simulation with cyclic synchronization and parallel processing.
@details Implements a peer-to-peer network of sensing devices that execute aggregation 
scripts. Utilizes a two-phase semaphore-based barrier for tight temporal coordination.
"""

from threading import Event, Semaphore, Lock, Thread

class ReusableBarrierSem(object):
    """
    @brief Two-phase cyclic barrier implemented using semaphores.
    Functional Utility: Synchronizes a fixed number of threads in repetitive execution cycles.
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
        @brief Primary wait protocol: executes Phase 1 (arrival) followed by Phase 2 (departure).
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief First turnstile: Ensures all threads have arrived before any are released.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last arrival releases the entire group.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second turnstile: Prevents threads from entering the next cycle prematurely.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    @brief Controller for a sensing entity in a distributed network.
    Functional Utility: Manages local data, thread lifecycle, and synchronization 
    references shared with peer devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Dictionary of local sensor readings.
        @param supervisor Coordination entity for topology discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.my_lock = Lock()
        self.barrier = ReusableBarrierSem(0)
        self.timepoint_done = Event()
        # Lifecycle: Spawns the main orchestrator thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of shared synchronization resources.
        Logic: Designates device 0 as the barrier allocator.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            # Sync: Peer devices adopt the barrier instance created by device 0.
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """
        @brief Schedules a processing script for the current cycle.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Protocol: Signals end of task assignment phase.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieval of local sensor readings.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        @brief Updates local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device lifecycle.
        """
        self.thread.join()


class MyScriptThread(Thread):
    """
    @brief Worker thread for executing a single data aggregation script.
    Architecture: Implements a distributed Map-Reduce operation.
    """

    def __init__(self, script, location, device, neighbours):
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        @brief Execution logic for aggregation.
        Critical Section: Uses per-device locks to ensure atomic state updates across the cluster.
        """
        script_data = []

        # Map Phase: Collect readings from neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Inclusion: Adds local data.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Computation Phase.
            result = self.script.run(script_data)

            # Reduce/Sync Phase: Propagates result to all participants.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()

class DeviceThread(Thread):
    """
    @brief Management thread coordinating cycle execution.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main loop: topology -> sync -> parallel execution -> sync.
        """
        while True:
            # Discovery: Fetches neighbors from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Initial Barrier: Aligns devices before starting the timepoint.
            self.device.barrier.wait()
            
            # Wait for task assignment.
            self.device.script_received.wait()
            
            /**
             * Block Logic: Parallel script execution.
             * Logic: Spawns one thread per assigned script and waits for total completion.
             */
            script_threads = []
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            
            for thread in script_threads:
                thread.start()
            
            for thread in script_threads:
                thread.join()
            
            # Final Barrier: Synchronizes devices at the end of the unit of time.
            self.device.timepoint_done.wait()
            self.device.barrier.wait()
            
            # State Reset for next cycle.
            self.device.script_received.clear()
