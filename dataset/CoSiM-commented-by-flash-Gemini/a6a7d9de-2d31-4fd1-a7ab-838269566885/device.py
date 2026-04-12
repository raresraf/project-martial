"""
@file device.py
@brief Distributed sensor data processing system with resource-constrained concurrency.
@details Implements a peer-to-peer network of sensing units that execute aggregation scripts. 
Uses semaphores to bound concurrent execution and class-level locks for data consistency.
"""

from threading import Event, Thread, BoundedSemaphore, Lock
from cond_barrier import ReusableBarrier

class Device(object):
    """
    @brief Controller for a sensing unit in a distributed network.
    Functional Utility: Manages local data state, schedules processing tasks, and coordinates 
    cluster-wide synchronization.
    """
    
    # Class-level Synchronization: Shared across all instances.
    barrier = None
    barrier_event = Event()

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Dictionary of local sensor readings.
        @param supervisor Entity responsible for neighborhood discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Concurrency Control: Limits active script threads to 8 to prevent resource exhaustion.
        self.scripts = []
        self.scripts_semaphore = BoundedSemaphore(8)
        self.scripts_lock = Lock()

        # Phase Coordination Events.
        self.script_received = Event()
        self.timepoint_done = Event()

        # Background Loop: Spawns the main management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of the shared cyclic barrier.
        Logic: Designates device 0 as the barrier allocator.
        """
        if Device.barrier is None and self.device_id == 0:
            Device.barrier = ReusableBarrier(len(devices))
            # Signal peers that the barrier is ready for use.
            Device.barrier_event.set()

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing script for the current cycle.
        """
        with self.scripts_lock:
            self.script_received.set()
            if script is not None:
                self.scripts.append((script, location))
            else:
                # Termination: Signals end of task assignment phase.
                self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of local sensor readings.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Terminates the device lifecycle threads.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Orchestrator thread managing execution phases (discovery -> execution -> sync).
    """
    
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main loop coordinating timepoint cycles.
        """
        # Protocol: Wait for global barrier initialization.
        Device.barrier_event.wait()
        
        while True:
            # Discovery: Fetches current neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break

            script_index = 0
            script_threads = []
            
            /**
             * Block Logic: Script execution engine with throttling.
             * Invariant: At most 8 script threads per device are active at any time.
             */
            while True:
                self.device.scripts_lock.acquire()
                if script_index < len(self.device.scripts):
                    self.device.scripts_lock.release()
                    
                    # Backpressure: Blocks if the worker pool is saturated.
                    self.device.scripts_semaphore.acquire()

                    # Task Spawning: Launches a worker for the next script.
                    script_threads.append(ScriptThread(self.device, self.device.scripts[script_index][0],
                                                       self.device.scripts[script_index][1], neighbours))
                    script_threads[-1].start()
                    script_index += 1
                else:
                    # Check for completion of the assignment phase.
                    if self.device.timepoint_done.is_set() and script_index == len(self.device.scripts):
                        self.device.timepoint_done.clear()
                        self.device.scripts_lock.release()
                        break
                    else:
                        # Wait for new assignments or end-of-timepoint signal.
                        self.device.scripts_lock.release()
                        self.device.script_received.wait()
                        self.device.scripts_lock.acquire()
                        self.device.script_received.clear()
                        self.device.scripts_lock.release()

            # Sync: Wait for all local script executions to finalize.
            for script_thread in script_threads:
                script_thread.join()

            # Global Sync: Ensure all devices in the cluster are aligned.
            Device.barrier.wait()


class ScriptThread(Thread):
    """
    @brief Execution unit for a single data aggregation script.
    Architecture: Implements distributed Map-Reduce on a shared location.
    """
    
    # Static Lock Registry: Shared across all instances to synchronize access to sensor locations.
    locations_locks = {}

    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.location = location
        self.script = script
        self.device = device
        self.neighbours = neighbours

        # Initialization: Dynamic lock allocation per unique sensor location.
        if location not in ScriptThread.locations_locks:
            ScriptThread.locations_locks[location] = Lock()

    def run(self):
        """
        @brief Execution logic for aggregation.
        Critical Section: Uses the location-specific lock to ensure cluster-wide atomicity.
        """
        with ScriptThread.locations_locks[self.location]:
            script_data = []

            # Map Phase: Collects data from neighbors and self.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Computation Phase.
                result = self.script.run(script_data)
                
                # Reduce/Update Phase: Writes back the computed result to all participants.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)

        # Semaphore Release: Frees a slot in the device's worker pool.
        self.device.scripts_semaphore.release()
