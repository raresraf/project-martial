"""
@c5e4e4e0-d063-49d2-9ff1-38b568482121/device.py
@brief Distributed sensor network simulation with spatial task partitioning.
This module implements a highly efficient parallel processing architecture where 
computational tasks are partitioned based on their target sensor location. By grouping 
scripts into per-location queues and assigning dedicated worker threads, the system 
naturally minimizes lock contention and ensures total order for updates at each 
spatial node. Global consistency is maintained via a two-phase semaphore barrier.

Domain: Spatial Partitioning, Parallel Processing, Multi-phase Barriers.
"""

import Queue
from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    Two-phase reusable barrier implementation using semaphores.
    Functional Utility: Employs a double-gate mechanism to ensure all threads have 
    exited the previous barrier phase before any thread can enter the next cycle.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: Number of participants in the rendezvous.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """Orchestrates the two-phase synchronization cycle."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First gate: coordinates thread arrival."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # All threads arrived: release the gate.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second gate: ensures total exit before reset."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # All threads cleared phase 1: release second gate.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class MyThread(Thread):
    """
    Diagnostic thread for validating barrier behavior.
    Functional Utility: Performs iterative steps to verify temporal synchronization.
    """

    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier

    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",


class Device(object):
    """
    Core node entity representing a sensor aggregation point.
    Functional Utility: Manages local data and coordinates the spatial partitioning 
    of tasks across the sensor field.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.barrier = None
        self.timepoint_done = Event()
        self.lock_data = Lock()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.all_devices = None
        self.locations = []
        self.lock_locations = []

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global topology discovery and synchronization setup.
        Logic: Aggregates all unique sensor locations in the network and 
        pre-allocates a mutex for each to support spatial partitioning.
        """
        # Discover local locations.
        for location in range(len(self.sensor_data)):
            if self.sensor_data.get(location) is not None:
                if location not in self.locations:
                    self.locations.append(location)

        self.all_devices = devices
        # Cross-discovery: Aggregates locations from all peer devices.
        for device in self.all_devices:
            for location in device.locations:
                if location not in self.locations:
                    self.locations.append(location)

        self.locations.sort()

        # Leader Election Logic: Only Device 0 performs global resource initialization.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))

            # Atomic Resource Allocation: Creating a dedicated lock for every spatial location.
            for _ in self.locations:
                lock = Lock()
                self.lock_locations.append(lock)

            # Propagation: Shares the barrier and global lock map with the network.
            for device in self.all_devices:
                device.set_barrier(self.barrier)
                device.set_lock_locations(self.lock_locations)


    def assign_script(self, script, location):
        """Registers a task for processing."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for the specified location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_barrier(self, barrier):
        """Injects the shared network-wide barrier."""
        self.barrier = barrier

    def set_lock_locations(self, lock_locations):
        """Injects the global spatial lock pool."""
        self.lock_locations = lock_locations

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main device orchestration thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Node-level simulation manager.
    Functional Utility: Implements spatial partitioning by creating dedicated 
    queues for each sensor location receiving scripts.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for the node manager.
        Algorithm: Iterative timepoint processing with location-based work splitting.
        """
        while True:
            # Topology Discovery Phase.
            neighbours = self.device.supervisor.get_neighbours()
            self.device.lockForLocations = []
            if neighbours is None:
                break
            
            # Synchronize with entire network before starting work.
            self.device.barrier.wait()

            # Wait for supervisor to finalize script assignments.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Block Logic: Spatial Partitioning.
            # Groups scripts into lists based on their target sensor location.
            queue_list = []
            index_list = []

            for item in self.device.scripts:
                (_, location) = item
                if location not in index_list:
                    index_list.append(location)
                    temp_queue = Queue.Queue()
                    temp_queue.put(item)
                    queue_list.append(temp_queue)
                else:
                    index = index_list.index(location)
                    queue_list[index].put(item)

            # Spawns a worker thread for each location that requires processing.
            th_list = []
            for queue in queue_list:
                worker = Thread(target=split_work, args=(self.device, neighbours, queue, ))
                worker.setDaemon(True)
                th_list.append(worker)
                worker.start()

            # Wait for all spatial workers to finish before progressing to next timepoint.
            for thr in th_list:
                thr.join()


def split_work(device, neighbours, queue_param):
    """
    Spatial worker function.
    Logic: Sequentially processes all scripts assigned to a specific location 
    while holding the corresponding global mutex to ensure network-wide consistency.
    """
    while True:
        try:
            # Non-blocking pull from the location-specific queue.
            (script, location) = queue_param.get(False)
        except Queue.Empty:
            break
        else:
            if location in device.locations:
                # Pre-condition: Acquire exclusive access to the spatial location.
                device.lock_locations[location].acquire()
                script_data = []
                
                # Neighborhood aggregation.
                for device_temp in neighbours:
                    data = device_temp.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Include local state.
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Process and propagate results to the neighborhood graph.
                    result = script.run(script_data)
                    for device_temp in neighbours:
                        device_temp.set_data(location, result)
                    device.set_data(location, result)
                
                # Finalize task and release spatial mutex.
                queue_param.task_done()
                device.lock_locations[location].release()
