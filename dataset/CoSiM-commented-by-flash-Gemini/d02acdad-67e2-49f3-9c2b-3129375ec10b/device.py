"""
@d02acdad-67e2-49f3-9c2b-3129375ec10b/device.py
@brief Distributed sensor processing simulation using static task chunking and a barrier singleton pattern.
* Algorithm: Fair task partitioning (Round-Robin) across 8 worker threads with two-phase semaphore synchronization and max-value state propagation.
* Functional Utility: Manages simulation timepoints across a network of devices by dividing scripts into chunks and synchronizing results through neighborhood-wide state updates.
"""

from threading import *

# Domain: Global state management - acts as a registry for reusable barriers to ensure consistent counts across devices.
dictionary = {}


class ReusableBarrier(object):
    """
    @brief Implementation of a two-phase synchronization barrier using semaphores.
    * Algorithm: Dual-stage arrival/release logic to prevent thread overruns in consecutive simulation steps.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier state and its internal phase semaphores.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Intent: Shared mutable counter for phase 1.
        self.count_threads2 = [self.num_threads] # Intent: Shared mutable counter for phase 2.
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Synchronizes the calling thread through both stages of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single synchronization stage.
        Invariant: The last thread to arrive releases the entire group.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

def pseudo_singleton(count):
    """
    @brief Ensures a single ReusableBarrier instance exists for a given thread count.
    Functional Utility: Facilitates global synchronization without explicit shared object passing.
    """
    global dictionary
    if not dictionary.has_key(count):
        dictionary[count] = ReusableBarrier(count)
    return dictionary[count]

class Device(object):
    """
    @brief Encapsulates a sensor node with its local data and coordination thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and bootstraps the main coordinator thread.
        """
        self.lock = Lock() # Intent: Serializes local sensor data updates.
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self, 0)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization setup.
        """
        self.barrier = pseudo_singleton(len(devices))

    def assign_script(self, script, location):
        """
        @brief Receives a processing task for the current simulation phase.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: Signals completion of task delivery for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Standard data retrieval for sensor locations.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Standard data update for sensor locations.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Terminates the device coordination thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Coordinator thread managing task chunking and worker execution phases.
    """

    def listoflists(self, list, number):
        """
        @brief Partitions a list into a fixed number of roughly equal-sized chunks.
        Algorithm: Dynamic slicing with round-robin tail distribution.
        """
        size = int(len(list) / number)
        chunks = []
        for i in xrange(number):
            chunks.append(list[0 + size * i: size * (i + 1)])
        # Logic: Distributes remaining items across chunks to maintain balance.
        for i in xrange(len(list) - size * number):
            chunks[i % number].append(list[(size * number) + i])
        return chunks

    def __init__(self, device, id):
        """
        @brief Initializes the coordinator thread.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = id

    class Instance(Thread):
        """
        @brief worker thread implementing the execution of a script chunk.
        """
        
        def __init__(self, device, listfromlist, neighbours):
            Thread.__init__(self, name="Instance")
            self.device = device
            self.listfromlist = listfromlist
            self.neighbours = neighbours

        def set_data_for_all_devices(self, location, result):
            """
            @brief Propagates a result to all neighbors, keeping the maximum value found.
            Invariant: Uses the parent device lock to ensure atomic read-update cycles.
            """
            for device in self.neighbours:
                self.device.lock.acquire()
                device.set_data(location, max(result, device.get_data(location)))
                self.device.lock.release()
            
            self.device.lock.acquire()
            self.device.set_data(location, max(result, self.device.get_data(location)))
            self.device.lock.release()

        def run(self):
            """
            @brief main loop for script chunk processing.
            """
            script_data = []
            for (script, location) in self.listfromlist:
                # Distributed Aggregation Phase.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                # Execution and Propagation Phase.
                if script_data != []:
                    result = script.run(script_data)
                    self.set_data_for_all_devices(location, result)


    def run(self):
        """
        @brief Main coordination loop for the device node.
        Algorithm: Iterative chunked execution with barrier alignment.
        """
        while True:
            # Logic: Neighbor discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Waits for script delivery completion.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            
            # Partitioning Phase: Divides tasks into exactly 8 chunks.
            list_of_scripts = self.listoflists(self.device.scripts, 8)
            instances = []
            
            # Dispatch Phase: Spawns a worker thread for each non-empty chunk.
            for i in range(8):
                if len(list_of_scripts):
                    instances.append(self.Instance(self.device, list_of_scripts[i], neighbours))
            
            for index in range(len(instances)):
                instances[index].start()
            
            # Logic: Wait for all local workers to complete.
            for index in range(len(instances)):
                instances[index].join()
            
            # Synchronization Phase: Align all devices across the cluster.
            self.device.barrier.wait()
