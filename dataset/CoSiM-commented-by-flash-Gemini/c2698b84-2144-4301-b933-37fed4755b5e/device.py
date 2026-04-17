"""
@c2698b84-2144-4301-b933-37fed4755b5e/device.py
@brief Distributed sensor processing simulation using dynamic thread instantiation and phased barrier synchronization.
* Algorithm: Event-based coordination with internal worker thread aggregation and global cluster-wide barriers.
* Functional Utility: Orchestrates the parallel execution of sensor scripts across a network of devices, ensuring synchronized timepoint progression.
"""

from threading import Event, Thread, Condition

class ReusableBarrierCond():
    """
    @brief Synchronizes a dynamic number of threads using condition variables.
    * Functional Utility: Facilitates collective waiting points in a multi-threaded execution path.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with a specific thread threshold.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        @brief Blocks calling thread until the barrier threshold is met.
        Algorithm: Monitor pattern using wait/notify_all on a shared condition variable.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Logic: Release all waiters and reset for potential re-use in subsequent phases.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    @brief Encapsulates a sensor node with its local data and execution management thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the main control thread.
        """
        self.device_id = device_id
        self.devices = None
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs cluster-wide resource distribution.
        Invariant: Root device (ID 0) initializes and propagates the shared global barrier.
        """
        self.devices = devices
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(self.devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Adds a script to the processing queue or signals end-of-batch.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: Triggers the execution phase for the current timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves local sensor data for a specific location.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

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

class ScriptsThread(Thread):
    """
    @brief Individual worker thread responsible for executing a subset of scripts.
    """

    def __init__(self, device, scripts, neighbours):
        """
        @brief Initializes the worker with its target tasks and neighborhood context.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        """
        @brief Core execution logic for a batch of scripts.
        Algorithm: Iterative script processing with distributed data aggregation.
        """
        for (script, location) in self.scripts:
            script_data = []
            
            # Distributed Aggregation Phase: Collect readings from neighbors.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    # Logic: Avoids duplicate local data if neighbors include self.
                    if data != self.device.get_data(location):
                        script_data.append(data)
                
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Execution Phase: Processes collected data and propagates state changes.
            if script_data != []:
                result = script.run(script_data)

                for device in self.neighbours:
                    # Domain: Conditional Update - Updates neighbor state if result is 'greater'.
                    if result > device.get_data(result):
                        device.set_data(location, result)
                    
                if result > self.device.get_data(result):
                    self.device.set_data(location, result)
            
        # Synchronization: Wait on internal device barrier before thread termination.
        self.device.thread.barrier.wait()

class DeviceThread(Thread):
    """
    @brief Coordinator thread managing the device lifecycle and worker dispatching.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = None
        self.list_of_threads = []

    def run(self):
        """
        @brief Main coordination loop.
        Algorithm: Dynamic thread spawning based on assigned script volume.
        """
        while True:
            # Logic: Neighbor discovery and exit condition.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Ensures all scripts are assigned before dispatching workers.
            self.device.timepoint_done.wait()
            now_thread = 0
            now_script = 0
            
            # Dispatch Phase: Maps scripts to internal worker threads.
            for script in self.device.scripts:
                # Logic: Distributes tasks among a pool of up to 8 threads.
                if now_script == 8:
                    now_script = 0
                else:
                    if now_script < 8:
                        self.list_of_threads.append(ScriptsThread(self.device, [script], neighbours))
                    else:
                        # Logic: Aggregates additional scripts into existing threads.
                        self.list_of_threads[now_thread].scripts.add(script)
                now_thread += 1
                now_script += 1
            
            # Synchronization Phase: Intra-device worker alignment.
            self.barrier = ReusableBarrierCond(len(self.list_of_threads))
            for thread in self.list_of_threads:
                thread.start()

            for thread in self.list_of_threads:
                thread.join()
            
            # Post-condition Phase Cleanup and Cluster alignment.
            self.list_of_threads = []
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
            self.list_of_threads = []
