"""
@df05e4e2-b1c8-460d-a2f7-ef11c1a84653/device.py
@brief Distributed sensor processing simulation using transient worker threads and multi-phase semaphore barriers.
* Algorithm: Dynamic task spawning where each script is executed in a dedicated `Node` thread, followed by collective state propagation and barrier alignment.
* Functional Utility: Orchestrates simulation timepoints across a network of devices, managing distributed data aggregation and synchronized state updates.
"""

from threading import enumerate, Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    @brief Two-phase synchronization barrier implementation using counting semaphores.
    * Algorithm: Dual-stage arrival/release logic to prevent thread overruns between consecutive simulation steps.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with a target thread count and dual phase primitives.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()       
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """
        @brief Blocks the calling thread through both stages of the barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Stage 1: Collects all threads and releases them simultaneously.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Logic: Collective release of all threads at the synchronization point.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Stage 2: Secondary synchronization to ensure consistent state across repeated cycles.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            self.count_threads1 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings and coordination thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and prepares the management thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization setup.
        Invariant: Root device (ID 0) initializes and shares the collective barrier.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    self.barrier = device.barrier

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing task for the current simulation phase.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signals completion of task arrival.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Standard data retrieval interface.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Standard data update interface.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Terminates the device management thread.
        """
        self.thread.join()

class Node(Thread):
    """
    @brief Transient worker thread dedicated to executing a single sensor script.
    """

    def __init__(self, script, script_data):
        """
        @brief Initializes the worker with its target script and input data batch.
        """
        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        self.result = None
         
    def run(self):
        """
        @brief Executes the script logic.
        """
        self.result = self.script.run(self.script_data)

    def join(self):
        """
        @brief Waits for completion and returns the execution artifacts.
        """
        Thread.join(self)
        return (self.script, self.result)


class DeviceThread(Thread):
    """
    @brief Main coordinator thread managing the lifecycle of the device and its workers.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator thread.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core execution loop for simulation phases.
        Algorithm: Phased execution involving data aggregation, parallel task spawning, and result propagation.
        """
        while True:
            # Logic: Neighbor discovery and exit condition.
            neighbours = self.device.supervisor.get_neighbours()
            thread_list = []
            scripts_result = {}
            scripts_data = {}
            if neighbours is None:
                break

            # Block Logic: Waits for script delivery start.
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            # Distributed Aggregation Phase: Collect readings from neighbors and self.
            for (script, location) in self.device.scripts:
                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                scripts_data[script] = script_data
                
                if script_data != []:
                    # Dispatch Phase: Offloads each script to a dedicated Node thread.
                    nod = Node(script, script_data)
                    thread_list.append(nod)
            
            # Execution Phase: Start all local workers.
            for nod in thread_list:
                nod.start()
            
            # Logic: Collect results from all local workers.
            for nod in thread_list:
                key, value = nod.join()
                scripts_result[key] = value
            
            # Propagation Phase: Broadcast computed results back to the neighborhood.
            for (script, location) in self.device.scripts:
                if scripts_data[script] != []:
                    for device in neighbours:
                        device.set_data(location, scripts_result[script])
                    self.device.set_data(location, scripts_result[script])
            
            # Synchronization Phase: Align all devices across the cluster.
            self.device.barrier.wait()
