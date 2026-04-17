"""
@d02acdad-67e2-49f3-9c2b-3129375ec10b/device.py
@brief Distributed sensor network simulation with balanced chunk task distribution.
This module implements a parallel processing architecture where computational 
scripts are partitioned into balanced chunks and assigned to a fixed pool of 8 
worker threads (Instance). The system utilizes a monotonic update strategy 
(maintaining the maximum value) for state propagation and ensures temporal 
consistency through a singleton-like reusable barrier.

Domain: Parallel Task Partitioning, Load Balancing, Monotonic State Updates.
"""

from threading import *

# Global Registry: used to provide singleton access to synchronization primitives.
dictionary = {}


class ReusableBarrier(object):
    """
    Two-phase reusable barrier implementation using semaphores.
    Functional Utility: Implements a double-gate rendezvous to coordinate simulation 
    timepoints across all participating nodes.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: total count of participants.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Executes the two-phase arrival and exit protocol."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Threshold reached: release all threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

def pseudo_singleton(count):
    """
    Functional Utility: Provides a shared barrier instance based on the 
    required thread count, acting as a network-wide coordination hub.
    """
    global dictionary
    if not dictionary.has_key(count):
        dictionary[count] = ReusableBarrier(count)
    return dictionary[count]

class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data state and coordinates simulation phases 
    through a centralized management thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.lock = Lock()
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Main lifecycle thread.
        self.thread = DeviceThread(self, 0)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global resource acquisition.
        Logic: Fetches the shared barrier from the global singleton registry.
        """
        self.barrier = pseudo_singleton(len(devices))

    def assign_script(self, script, location):
        """Registers a task and signals the orchestration thread to begin processing."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully joins the device management thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Simulated node manager.
    Functional Utility: Implements a chunked workload partitioning strategy to 
    distribute tasks among transient worker instances.
    """

    def listoflists(self, list, number):
        """
        Chunking Heuristic.
        Algorithm: Balanced partitioning of a list into N roughly equal sub-lists.
        @return: A list of task chunks.
        """
        size = int(len(list) / number)
        chunks = []
        for i in xrange(number):
            chunks.append(list[0 + size * i: size * (i + 1)])
        # Block Logic: distributes remaining tasks across chunks.
        for i in xrange(len(list) - size * number):
            chunks[i % number].append(list[(size * number) + i])
        return chunks

    def __init__(self, device, id):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = id

    class Instance(Thread):
        """
        Transient worker thread for batch processing.
        Functional Utility: Executes a chunk of scripts sequentially while 
        maintaining monotonic state updates across the neighborhood.
        """
        
        def __init__(self, device, listfromlist, neighbours):
            Thread.__init__(self, name="Instance")
            self.device = device
            self.listfromlist = listfromlist
            self.neighbours = neighbours

        def set_data_for_all_devices(self, location, result):
            """
            Monotonic State Propagation.
            Logic: Updates all nodes in the neighborhood with the maximum value 
            found (result vs current), ensuring forward-only state flow.
            """
            for device in self.neighbours:
                # Concurrent update protection.
                self.device.lock.acquire()
                device.set_data(location, max(result, device.get_data(location)))
                self.device.lock.release()
            
            self.device.lock.acquire()
            self.device.set_data(location, max(result, self.device.get_data(location)))
            self.device.lock.release()

        def run(self):
            """
            Execution loop for the task chunk.
            Algorithm: neighborhood aggregation followed by monotonic result propagation.
            """
            script_data = []
            for (script, location) in self.listfromlist:
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                if script_data != []:
                    # Run logic and propagate results.
                    result = script.run(script_data)
                    self.set_data_for_all_devices(location, result)


    def run(self):
        """
        Main orchestration loop for the simulation timepoint.
        Algorithm: Balances work into 8 chunks and manages the worker lifecycle.
        """
        while True:
            # Refresh topology.
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break
            
            # Wait for supervisor signal.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            
            # Block Logic: Workload partitioning.
            list_of_scripts = self.listoflists(self.device.scripts, 8)
            instances = []
            
            # Spawns workers for each task chunk.
            for i in range(8):
                if len(list_of_scripts):
                    instances.append(self.Instance(self.device, list_of_scripts[i], neighbours))
            
            # Parallel execution phase.
            for index in range(len(instances)):
                instances[index].start()
            
            # Barrier Point: wait for all local chunk workers.
            for index in range(len(instances)):
                instances[index].join()
            
            # Global consensus rendezvous.
            self.device.barrier.wait()
