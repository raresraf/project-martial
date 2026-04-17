"""
@c5e4e4e0-d063-49d2-9ff1-38b568482121/device.py
@brief Distributed sensor simulation with location-partitioned task queues and semaphore-based barrier synchronization.
* Algorithm: Data-parallel task distribution where scripts are grouped by sensor location into individual work queues.
* Functional Utility: Orchestrates simulation phases across a device cluster using phased semaphore barriers and per-location mutual exclusion.
"""

import Queue
from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    @brief Two-phase synchronization barrier implemented using counting semaphores.
    * Algorithm: Dual-stage arrival/release pattern to ensure consistent thread alignment across repeated cycles.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier state and its internal phase semaphores.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        @brief Synchronizes the calling thread through both stages of the barrier.
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
                # Logic: Final thread arrival triggers the release of the entire group.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Stage 2: Secondary synchronization to prevent thread overruns in tight loops.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class MyThread(Thread):
    """
    @brief Basic worker thread for validating barrier synchronization.
    """

    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier

    def run(self):
        """
        @brief Iterative execution demonstrating synchronized progression across barrier steps.
        """
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",


class Device(object):
    """
    @brief Encapsulates a simulation node with local data and lock management.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps its primary coordination thread.
        """
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
        self.locations = [] # Intent: Tracks all sensor locations relevant to this device and its peers.
        self.lock_locations = [] # Intent: Maps global locations to synchronization locks.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Cluster-wide resource initialization and lock distribution.
        Logic: Discovers all unique sensor locations across the network and establishes global locks.
        """
        # Logic: Aggregate local locations.
        for location in range(len(self.sensor_data)):
            if self.sensor_data.get(location) is not None:
                if location not in self.locations:
                    self.locations.append(location)

        self.all_devices = devices

        # Logic: Discover and aggregate locations from all neighboring peers.
        for device in self.all_devices:
            for location in device.locations:
                if location not in self.locations:
                    self.locations.append(location)

        self.locations.sort()

        # Invariant: Root node (ID 0) performs the heavy-lifting of primitive creation.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            
            # Logic: Pre-allocates a lock for every discovered global location.
            for _ in self.locations:
                lock = Lock()
                self.lock_locations.append(lock)

            # Logic: Distributes shared synchronization state to all devices in the cluster.
            for device in self.all_devices:
                device.set_barrier(self.barrier)
                device.set_lock_locations(self.lock_locations)

    def assign_script(self, script, location):
        """
        @brief Appends a processing task for the current simulation phase.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: All scripts for the timepoint have been delivered.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Standard data retrieval for sensor locations.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_barrier(self, barrier):
        """
        @brief Assigns the shared cluster-wide barrier.
        """
        self.barrier = barrier

    def set_lock_locations(self, lock_locations):
        """
        @brief Assigns the shared global lock set.
        """
        self.lock_locations = lock_locations

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
    @brief Coordination thread managing phase transitions and worker thread dispatching.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main control loop for the device node.
        Algorithm: Grouping tasks by location to minimize synchronization contention.
        """
        while True:
            # Logic: Neighbor discovery and barrier alignment.
            neighbours = self.device.supervisor.get_neighbours()
            self.device.lockForLocations = []
            if neighbours is None:
                break
            self.device.barrier.wait()

            # Block Logic: Waits for script delivery completion.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Task Organization: Partition scripts into location-specific queues.
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

            # Execution Phase: Spawns one worker thread per active sensor location.
            th_list = []
            for queue in queue_list:
                worker = Thread(target=split_work, args=(self.device, neighbours, queue, ))
                worker.setDaemon(True)
                th_list.append(worker)
                worker.start()

            # Logic: Wait for all location-specific workers to finish local processing.
            for thr in th_list:
                thr.join()

def split_work(device, neighbours, queue_param):
    """
    @brief Worker function that processes all tasks for a specific sensor location.
    Algorithm: Consumes location-partitioned queue with mandatory lock acquisition.
    """
    while True:
        try:
            # Logic: Non-blocking fetch to determine if all tasks for this location are done.
            (script, location) = queue_param.get(False)
        except Queue.Empty:
            break
        else:
            if location in device.locations:
                # Pre-condition: Must acquire global location lock for cluster-wide consistency.
                device.lock_locations[location].acquire()
                script_data = []
                
                # Distributed Aggregation Phase: Collect readings from Peers and Self.
                for device_temp in neighbours:
                    data = device_temp.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                # Logic: Execute the analysis logic and broadcast results.
                if script_data != []:
                    result = script.run(script_data)
                    for device_temp in neighbours:
                        device_temp.set_data(location, result)
                    device.set_data(location, result)
                
                queue_param.task_done()
                # Post-condition: Release global lock to allow other devices to process the same location.
                device.lock_locations[location].release()
