"""
@da2ef9f9-56a9-4257-bd49-dcc942094b68/device.py
@brief Distributed sensor network simulation with self-recycling worker pool.
This module implements a dynamic parallel processing architecture where worker threads 
are managed via a central availability queue (thread_queue). After completing a 
computational task, workers atomically re-register themselves in the pool. The 
system utilizes a hierarchical locking strategy (Global Spatial Lock + Node-level 
Data Lock) and multi-stage barrier synchronization to ensure total network consistency.

Domain: Worker Recycling, Hierarchical Locking, Dynamic Parallel Dispatch.
"""

from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierCond
from Queue import Queue

# Configuration: defines the degree of parallelism per node.
THREAD_NR = 8

class Device(object):
    """
    Representation of a node in the distributed system.
    Functional Utility: Manages a recycled pool of worker threads and provides 
    dynamically initialized spatial locks for sensor locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.setup_finished = Event()
        # Mutex for protecting local sensor data state.
        self.dataLock = Lock()
        self.shared_lock = Lock()
        # Efficiency Logic: shared queue of idle worker threads.
        self.thread_queue = Queue(0)
        # Local barrier for internal thread group synchronization.
        self.wait_get_neighbours = ReusableBarrierCond(THREAD_NR)
        self.thread_pool = []
        self.neighbours = []

        # Spawns and registers the initial pool of workers.
        for i in range(0, THREAD_NR):
            thread = DeviceThread(self, i)
            self.thread_pool.append(thread)
            self.thread_queue.put(thread)
            thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global resource factory.
        Logic: Coordinator node (ID 0) initializes the shared network-wide barrier 
        and the global spatial lock pool.
        """
        if self.device_id == 0:
            # Network-wide synchronization sized for all threads in the group.
            self.reusable_barrier = ReusableBarrierCond(len(devices) * THREAD_NR)
            self.location_locks = {}
            # Propagation: Distribute shared resources to peer devices.
            for device in devices:
                if device.device_id != self.device_id:
                    device.set_location_locks(self.location_locks)
                    device.set_barrier(self.reusable_barrier)

            self.setup_finished.set()

    def set_barrier(self, reusable_barrier):
        """Injects the shared network barrier."""
        self.reusable_barrier = reusable_barrier
        self.setup_finished.set()

    def set_location_locks(self, location_locks):
        """Injects the global mapping of spatial locks."""
        self.location_locks = location_locks

    def assign_script(self, script, location):
        """
        Asynchronous Task Dispatch.
        Logic: Pulls an idle thread from the availability queue and assigns the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            # On-Demand Lock Initialization: ensure spatial mutex exists.
            if location not in self.location_locks:
                self.location_locks[location] = Lock()

            # Claim a worker.
            thread = self.thread_queue.get()
            thread.give_script(script, location)
        else:
            # Signal end of timepoint and flush remaining task assignments.
            for (s, l) in self.scripts:
                thread = self.thread_queue.get()
                thread.give_script(s, l)

            # Shutdown signals for the workers.
            for thread in self.thread_pool:
                thread.give_script(None, None)


    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all threads in the worker pool."""
        for i in range(THREAD_NR):
            self.thread_pool[i].join()


class DeviceThread(Thread):
    """
    Recyclable worker thread implementation.
    Functional Utility: Executes computational scripts and re-enters the 
    availability pool upon completion.
    """

    def __init__(self, device, ID):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id = ID
        # Thread-local task buffer.
        self.script_queue = Queue(0)

    def give_script(self, script, location):
        """Interface for the parent Device to assign work to this specific thread."""
        self.script_queue.put((script, location))

    def run(self):
        """
        Main worker execution loop.
        Algorithm: Multi-stage synchronization and task processing.
        """
        while True:
            # Block until global network setup is ready.
            self.device.setup_finished.wait()

            # Role-Based Logic: Designated thread handles topology discovery.
            if self.id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # Barrier Point: Wait for topology to be available to all local threads.
            self.device.wait_get_neighbours.wait()

            if self.device.neighbours is None:
                break

            # Block Logic: Task Processing phase.
            while True:
                # Wait for task assignment.
                (script, location) = self.script_queue.get()
                # Check for termination signal.
                if script is None:
                    break

                # Critical Section 1: Network-wide spatial lock for the location.
                self.device.location_locks[location].acquire()
                script_data = []

                # Critical Section 2: Node-level mutex for atomic data read from neighbors.
                for device in self.device.neighbours:
                    device.dataLock.acquire()
                    data = device.get_data(location)
                    device.dataLock.release()

                    if data is not None:
                        script_data.append(data)

                # Local state integration.
                self.device.dataLock.acquire()
                data = self.device.get_data(location)
                self.device.dataLock.release()
                
                if data is not None:
                   script_data.append(data)

                self.device.location_locks[location].release()

                if script_data != []:
                    # Compute result.
                    result = script.run(script_data)
                    
                    # Atomic result propagation across the neighborhood.
                    self.device.location_locks[location].acquire()
                    for device in self.device.neighbours:
                        device.dataLock.acquire()
                        device.set_data(location, result)
                        device.dataLock.release()

                    # Self-update.
                    self.device.dataLock.acquire()
                    self.device.set_data(location, result)
                    self.device.dataLock.release()
                    self.device.location_locks[location].release()

                # Lifecycle Recirculation: worker puts itself back into the pool.
                self.device.thread_queue.put(self)

            # Global Rendezvous point for entire network temporal consistency.
            self.device.reusable_barrier.wait()
