"""
@file device.py
@brief Distributed device simulation environment using queue-based task distribution.

Functional Intent: Implements a multi-threaded framework where devices coordinate 
parallel execution of data-processing scripts using a worker thread pool. 
Utilizes a centralized work queue for task dissemination within each device, 
enforces data consistency through location-specific shared locks, and synchronizes 
epoch-based execution steps via reusable condition barriers.

Domain: Production Systems, Distributed Computing, Parallel Processing.
"""

import Queue
from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond


class Device(object):
    """
    @brief Represents a computational entity in the simulated distributed network.
    
    Functional Utility: Manages local sensor state, task assignments, and a local 
    pool of worker threads. Interfaces with a global supervisor to identify 
    neighbors and establish shared synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device and prepares its internal orchestration state.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # Logic: Sequence of scripts to be executed in the current epoch.
        
        # Synchronization: Events for coordinating setup and epoch-based task assignment.
        self.timepoint_done = Event()
        self.event_setup = Event()

        self.devices = [] # Logic: Global registry of all peer devices in the network.
        self.barrier_device = None # Logic: Shared barrier for cross-device epoch synchronization.
        self.locations_lock = [] # Logic: Global pool of mutexes for protecting specific data addresses.
        self.data_set_lock = Lock() # Logic: Local mutex for protecting internal sensor state.

        # Control: Dedicated thread for high-level device management.
        self.thread = DeviceThread(self)
        self.thread.start()

        self.device_shutdown_order = False # Control: Flag to signal termination to worker threads.
        self.work_queue = Queue.Queue() # Logic: Producer-consumer queue for internal task dispatching.

        # Optimization: Pre-allocated synchronization for the internal worker pool (8 threads).
        self.worker_barrier = ReusableBarrierCond(8)
        self.data_semaphore = Semaphore(value=0) # Sync: Tracks available tasks for workers.
        self.worker_semaphore = Semaphore(value=0) # Sync: Tracks completed tasks for the manager.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Orchestrates the cluster-wide initialization of shared resources.
        
        Algorithm: Single-node master setup (Device 0).
        Logic: Only the designated master device initializes the global barrier and 
        shared lock pool to ensure all peers share identical synchronization objects.
        """
        # Pre-condition: Prevents redundant initialization by restricting setup to the first device.
        if self.device_id == 0:
            self.barrier_device = ReusableBarrierCond(len(devices))

            # Block Logic: Data-plane protection.
            # Invariant: Initializes 25 mutexes to regulate concurrent access to shared data locations.
            for _ in range(25):
                self.locations_lock.append(Lock())

            # Side Effect: Propagates shared object references to all participating devices.
            for dev in devices:
                dev.devices = devices
                dev.barrier_device = self.barrier_device
                dev.locations_lock = self.locations_lock
                dev.event_setup.set()

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing script for the current simulation step.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signaling None indicates that all tasks for this epoch have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves local sensor data (requires external locking for consistency).
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Thread-safe update of local sensor data.
        """
        with self.data_set_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Triggers a graceful shutdown of the device orchestration thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Management thread responsible for task orchestration and worker supervision.
    
    Logic: Coordinates the transition between epochs, task distribution to local 
    CPUs, and global consensus points.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core operational cycle for the device manager.
        
        Algorithm: Distributed epoch orchestration.
        Logic: 
        1. Waits for global setup confirmation.
        2. Spawns a pool of worker threads.
        3. Iteratively fetches neighbors and dispatches assigned scripts to the worker queue.
        4. Synchronizes at a global barrier to ensure network-wide step alignment.
        """
        # Pre-condition: Blocks execution until shared locks and barriers are available.
        self.device.event_setup.wait()

        # Block Logic: Internal worker pool lifecycle management.
        # Optimization: Scales to 8 parallel workers for high-throughput script processing.
        list_threads = []
        for i in range(8):
            thrd = WorkerThread(self.device, self.device.locations_lock, self.device.work_queue, i)
            list_threads.append(thrd)

        for thrd in list_threads:
            thrd.start()

        script_number = 0 # Invariant: Tracks total scripts dispatched to ensure full epoch completion.

        while True:
            # Block Logic: Neighbor discovery via supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            # Task Dispatch: Translates high-level scripts into granular worker tasks.
            for (script, location) in self.device.scripts:
                tup = (script, location, neighbours)

                self.device.work_queue.put(tup)
                self.device.data_semaphore.release() # Logic: Increment task availability for workers.

                script_number += 1

            self.device.timepoint_done.clear()

            # Synchronization: Ensures all devices have distributed their tasks before starting execution.
            self.device.barrier_device.wait()

        # Shutdown Sequence: Synchronization point to ensure workers have finished all epoch tasks.
        for _ in xrange(script_number):
            self.device.worker_semaphore.acquire()

        # Control: Broadcasts shutdown signal to the worker pool.
        self.device.device_shutdown_order = True
        for _ in xrange(8):
            self.device.data_semaphore.release()

        for thrd in list_threads:
            thrd.join()


class WorkerThread(Thread):
    """
    @brief Execution thread that processes scripts from the device's shared work queue.
    
    Algorithm: Neighborhood state reduction and update.
    """

    def __init__(self, device, locations_lock, work_queue, worker_id):
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.device = device
        self.locations_lock = locations_lock
        self.work_queue = work_queue
        self.worker_id = worker_id

    def run(self):
        """
        @brief Worker execution loop.
        
        Logic: 
        1. Waits for tasks signaled by the data semaphore.
        2. Retrieves script context (location, neighbors).
        3. Performs thread-safe data gathering across the neighborhood.
        4. Executes script and broadcasts the result to all peers.
        """
        while True:
            # Pre-condition: Wait for work or shutdown signal.
            self.device.data_semaphore.acquire()

            if self.device.device_shutdown_order is True:
                break

            tup = self.work_queue.get()
            script = tup[0]
            location = tup[1]
            neighbours = tup[2]

            # Synchronization: Mutual exclusion for specific data location to prevent race conditions.
            with self.locations_lock[location]:
                script_data = []
                
                # Block Logic: State aggregation from the neighborhood.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Functional Intent: Perform user-defined data transformation.
                    result = script.run(script_data)

                    # Block Logic: Update propagation to neighborhood peers.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
            
            # Synchronization: Signal task completion to the manager.
            self.device.worker_semaphore.release()

        # Finalization: Rendezvous with other workers before thread termination.
        self.device.worker_barrier.wait()
