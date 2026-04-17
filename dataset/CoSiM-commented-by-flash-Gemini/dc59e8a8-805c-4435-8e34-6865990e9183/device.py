"""
@dc59e8a8-805c-4435-8e34-6865990e9183/device.py
@brief Distributed sensor processing simulation using serialized worker threads and a shared location-lock map.
* Algorithm: Sequential execution of individual `MyThread` workers managed by a `DeviceThread` coordinator, using multi-phase barriers for global alignment.
* Functional Utility: Orchestrates simulation timepoints across a device cluster by aggregating neighbor data and propagating state transitions under location-specific locks.
"""

from threading import Event, Thread, Semaphore, Lock


class ReusableBarrier(object):
    """
    @brief Two-phase synchronization barrier implementation using counting semaphores.
    * Algorithm: Dual-stage arrival/release logic to prevent thread overruns between consecutive simulation phases.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with a target thread threshold.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Intent: Shared mutable counter for phase 1.
        self.count_threads2 = [self.num_threads] # Intent: Shared mutable counter for phase 2.
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread through both stages of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single synchronization stage.
        Invariant: The last thread to arrive releases the entire group and resets the counter.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, shared barrier, and management thread.
    """
    barrier = None # Domain: Global class-level barrier for cluster-wide coordination.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and bootstraps the main coordinator thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization setup for the device cluster.
        """
        Device.barrier = ReusableBarrier(len(devices))

    def assign_script(self, script, location):
        """
        @brief Enqueues a task for the current simulation cycle and signals its arrival.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signals completion of task delivery for this node.
            self.timepoint_done.set()
        self.script_received.set()

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
        @brief Terminates the device management thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Main coordination thread implementing sequential worker task dispatching.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator thread for a specific device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core lifecycle loop of the device node.
        Algorithm: Iterative script execution with explicit joining, effectively serializing sub-tasks.
        """
        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            script_index = 0
            script_threads = []
            length_scripts_threads = 0
            
            while True:
                # Dispatch Phase.
                if script_index < len(self.device.scripts):
                    # Logic: Concurrency gating (threshold of 8 active workers).
                    if length_scripts_threads < 8:
                        # Functional Utility: Triggers serialized execution via start-then-join.
                        thread = self.call_threads(neighbours, script_index)
                        if thread.is_alive():
                            script_threads.append((thread, True))
                            length_scripts_threads += 1
                        script_index += 1
                    else:
                        # Post-condition Check: Prunes completed threads from the tracking list.
                        local_index = 0
                        while local_index < len(script_threads):
                            if (script_threads[local_index][0].isAlive()
                                    and script_threads[local_index][1] is True):
                                script_threads[local_index][1] = False
                                length_scripts_threads -= 1
                            local_index += 1
                elif self.device.timepoint_done.is_set():
                    # Phase Completion.
                    self.device.timepoint_done.clear()
                    self.device.script_received.clear()
                    break
                else:
                    # Logic: Wait for next batch of scripts to arrive.
                    self.device.script_received.wait()
                    self.device.script_received.clear()

            # Synchronization Phase: Align all devices across the cluster.
            Device.barrier.wait()

    def call_threads(self, neighbours, index):
        """
        @brief Instantiates and executes a single worker thread.
        Invariant: Performs an immediate join, ensuring sequential processing of scripts within this device.
        """
        thread = MyThread(self.device, neighbours, self.device.scripts[index])
        thread.start()
        thread.join()
        return thread


class MyThread(Thread):
    """
    @brief Worker thread implementing the execution of a single sensor script unit.
    """
    # Domain: Shared class-level registry for location locks to ensure atomic distributed updates.
    locations_locks = {}

    def __init__(self, device, neighbours, (script, location)):
        """
        @brief Initializes worker with task parameters and ensures target location is locked.
        """
        Thread.__init__(self)
        self.location, self.script = location, script
        self.device, self.neighbours = device, neighbours

        # Logic: On-demand initialization of the shared lock for this location.
        if location not in MyThread.locations_locks:
            MyThread.locations_locks[location] = Lock()

    def run(self):
        """
        @brief Main execution logic for a single script unit.
        Algorithm: Resource-locked execution with distributed data aggregation and propagation.
        """
        MyThread.locations_locks[self.location].acquire()
        
        script_data = []
        # Distributed Aggregation Phase: Collect readings from neighbors and self.
        for device in self.neighbours:
            if device.get_data(self.location) is not None:
                script_data.append(device.get_data(self.location))
        
        if self.device.get_data(self.location) is not None:
            script_data.append(self.device.get_data(self.location))

        if script_data != []:
            # Execution and Propagation Phase.
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
            
        # Post-condition: Release global location lock.
        MyThread.locations_locks[self.location].release()
