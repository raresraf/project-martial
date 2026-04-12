"""
@file device.py
@brief Distributed sensor unit simulation with cyclic coordination and multi-threaded script execution.
@details Implements a peer-to-peer network of sensing devices that perform synchronized 
data aggregation. Utilizes a two-phase semaphore-based barrier for cluster-wide temporal 
alignment and individual thread management for script processing.
"""

from threading import Event, Semaphore, Lock, Thread

class ReusableBarrierSem(object):
    """
    @brief Implementation of a two-phase cyclic barrier using semaphores.
    Functional Utility: Synchronizes a fixed group of threads in recurring execution windows.
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
        @brief Barrier protocol: Phase 1 (Collect) -> Phase 2 (Release/Reset).
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Turnstile phase: Blocks all threads until the last one arrives.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Release all blocked threads.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Reset phase: Ensures all threads have exited Phase 1 before the barrier is reused.
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
    @brief Core logic controller for a distributed sensing entity.
    Functional Utility: Manages local data buffers and coordinates lifecycle threads 
    shared with peer devices in the network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of sensor location readings.
        @param supervisor Coordination entity for topology management.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.my_lock = Lock()
        self.barrier = ReusableBarrierSem(0)
        self.timepoint_done = Event()
        # Lifecycle: Spawns the main management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Distributed initialization of shared synchronization resources.
        Logic: Designates device 0 as the barrier allocator for the entire cluster.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            # Sync: Peers adopt the shared barrier reference from device 0.
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """
        @brief Registers a computation script for execution in the current cycle.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Protocol: Signals end of script assignment phase.
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
        @brief Update of local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device management thread.
        """
        self.thread.join()


class MyScriptThread(Thread):
    """
    @brief Worker thread for executing data aggregation logic.
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
        Critical Section: Uses per-device locks to ensure atomic state updates across peers.
        """
        script_data = []

        # Map Phase: Aggregates state from all neighborhood members.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Inclusion: Adds the local device state.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Computation Phase.
            result = self.script.run(script_data)

            # Reduce/Sync Phase: Propagates the new state to all participating devices.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()

class DeviceThread(Thread):
    """
    @brief Management thread coordinating discrete units of execution (timepoints).
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main orchestration loop: discovery -> synchronization -> parallel execution -> reset.
        """
        while True:
            # Discovery: Fetches the current set of neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Synchronization: Ensures all devices are aligned before starting the cycle.
            self.device.barrier.wait()
            
            # Wait for script assignment phase to complete.
            self.device.script_received.wait()
            
            /**
             * Block Logic: Parallel processing of assigned scripts.
             * Logic: Spawns one worker thread per (script, location) pair.
             */
            script_threads = []
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            
            for thread in script_threads:
                thread.start()
            
            for thread in script_threads:
                thread.join()
            
            # Final Sync: Cluster-wide alignment at the end of the timepoint.
            self.device.timepoint_done.wait()
            self.device.barrier.wait()
            
            # Reset state for the next temporal unit.
            self.device.script_received.clear()
