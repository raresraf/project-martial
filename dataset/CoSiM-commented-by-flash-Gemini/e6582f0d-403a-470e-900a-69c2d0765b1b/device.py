"""
@e6582f0d-403a-470e-900a-69c2d0765b1b/device.py
@brief Distributed sensor network simulation with managed parallel worker pool.
This module implements a coordinated processing framework where a set of persistent 
worker threads (WorkerThread) handle computational tasks via a shared node-level 
queue. Each simulation timepoint is coordinated through a node manager (DeviceThread) 
that discovery topology and dispatches workloads. Consistency is guaranteed through 
a shared pool of spatial locks and a monitor-based reusable synchronization barrier.

Domain: Parallel Worker Pools, Task Queuing, Distributed State Synchronization.
"""

from threading import Event, Thread, Lock
from Queue import Queue
from reusable_barrier_condition import ReusableBarrier


class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data, coordinates global synchronization 
    resource distribution, and provides a container for assigned parallel tasks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Spatial lock repository populated during setup.
        self.location_locks = {}
        self.barrier = None
        # Degree of local parallelism.
        self.num_threads = 8
        self.queue = Queue(self.num_threads)
        
        # Primary orchestration thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource factory.
        Logic: Coordinator node (first to run setup) initializes the shared 
        barrier and a network-wide pool of locks for all sensor locations.
        """
        if self.barrier is None:
            # Singleton setup: ensure all devices share the same rendezvous point.
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier
                # Discovery: pre-allocates a mutex for every sensor location in the group.
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
            # Propagation: share the lock map with all peers.
            for device in devices:
                device.location_locks = self.location_locks



    def assign_script(self, script, location):
        """Registers a task and signals completion of the assignment phase."""
        if script is not None:
            self.scripts.append((script, location))
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
        """Gracefully joins the orchestration thread."""
        self.thread.join()

class WorkerThread(Thread):
    """
    Persistent worker thread implementation.
    Functional Utility: Continuously consumes tasks from the node queue and 
    executes them while maintaining global spatial mutual exclusion.
    """

    def __init__(self, queue, device):
        Thread.__init__(self)
        self.queue = queue
        self.device = device

    def run(self):
        """
        Worker execution loop.
        Logic: Pulls (script, location, neighbors) from the queue and applies 
        computational logic under a network-wide spatial lock.
        """
        while True:
            # Block until a task is available.
            data_tuple = self.queue.get()

            # Exit Logic: Check for the poison pill signal.
            if data_tuple == (None, None, None):
                break

            # Critical Section: Spatial mutual exclusion for the target location.
            self.device.location_locks[data_tuple[1]].acquire()
            script_data = []
            
            # Aggregate neighborhood state.
            for device in data_tuple[2]:
                data = device.get_data(data_tuple[1])
                if data is not None:
                    script_data.append(data)
            
            # Include local state.
            data = self.device.get_data(data_tuple[1])
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Apply computational logic and propagate results to all peers.
                result = data_tuple[0].run(script_data)

                for device in data_tuple[2]:
                    device.set_data(data_tuple[1], result)
                
                self.device.set_data(data_tuple[1], result)
            
            # Release spatial mutex.
            self.device.location_locks[data_tuple[1]].release()



class DeviceThread(Thread):
    """
    Main orchestration thread for the node.
    Functional Utility: Coordinates between simulation timepoints and manages 
    the lifecycle of the internal parallel worker pool.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop.
        Algorithm: Iterative sequence of topology discovery, task dispatch, and consensus.
        """
        threads = []
        # Spawn the persistent worker pool.
        for i in range(self.device.num_threads):
            thread = WorkerThread(self.device.queue, self.device)
            threads.append(thread)
            threads[i].start()

        while True:
            # Refresh topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for supervisor to finalize script assignments.
            self.device.timepoint_done.wait()

            # Offload tasks to the parallel workers.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location, neighbours))

            # Global Rendezvous point for network-wide consensus.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()

        # Shutdown Logic: Dispatch poison pills to each worker in the pool.
        for i in range(self.device.num_threads):
            self.device.queue.put((None, None, None))

        for i in range(self.device.num_threads):
            threads[i].join()


from threading import Condition

class ReusableBarrier(object):
    """
    Monitor-based reusable barrier implementation.
    Functional Utility: Provides a temporal rendezvous point for a fixed group 
    of threads using a condition variable to signal threshold arrival.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: participants count.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()


    def wait(self):
        """Blocks the calling thread until the arrival threshold is met."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Threshold met: release all participants and reset for next cycle.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
