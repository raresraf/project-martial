"""
@de1ec49b-cd8c-49d6-b00b-cf861ca3f49e/device.py
@brief Distributed sensor network simulation with task recycling and steady-state processing.
This module implements a parallel processing model where a fixed pool of worker 
threads consumes tasks from a shared node-level queue. Assigned computational 
scripts are cached and recycled between simulation steps to simulate continuous, 
steady-state data processing. Consistency is enforced through a static, class-level 
pool of spatial locks and a robust two-phase synchronization barrier.

Domain: Steady-State Simulation, Task Recycling, Parallel Queue Processing.
"""

from threading import Event, Thread, Lock
from Queue import Queue
import reusable_barrier_semaphore

class Device(object):
    """
    Representation of a network node with persistent task buffers.
    Functional Utility: Manages local sensor data, coordinates global synchronization 
    resources, and maintains a recycled queue of computational tasks.
    """
    
    # Static Synchronization Resources.
    barrier = None
    # Global Repository: map of sensor locations to their respective mutexes.
    lockList = {}
    # Mutex for protecting the lockList itself during on-demand initialization.
    lockListLock = Lock()

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device node and its worker pool.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Local Barrier: coordinates the internal thread group.
        self.neighbours_done = reusable_barrier_semaphore.ReusableBarrier(8)
        self.neighbours = None
        
        # Core Task Queue.
        self.scripts = Queue()
        
        # Persistence Logic: maintains a copy of tasks to allow re-queuing.
        self.permanent_scripts = []
        self.threads = []
        self.startup_event = Event()

        # Spawns a pool of 8 persistent worker threads.
        for i in range(0, 8):
            self.threads.append(DeviceThread(self, i))
        for i in range(0, 8):
            self.threads[i].start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global resource initialization.
        Logic: Node 0 initializes the shared barrier. All nodes participate in 
        lazy-populating the static spatial lock pool based on their local data.
        """
        Device.lockListLock.acquire()
        if Device.barrier is None:
            # Atomic creation of the network-wide rendezvous point.
            Device.barrier = reusable_barrier_semaphore.ReusableBarrier(len(devices))
        Device.lockListLock.release()

        # Block Logic: Spatial Lock discovery.
        # Logic: Ensures a mutex exists for every sensor location managed by this node.
        the_keys = self.sensor_data.keys()
        for i in the_keys:
            Device.lockListLock.acquire()
            if i not in Device.lockList:
                Device.lockList[i] = Lock()
            Device.lockListLock.release()
        
        # Signal workers that initialization is complete.
        self.startup_event.set()

    def assign_script(self, script, location):
        """
        Registers a computational task.
        Logic: Appends to the permanent cache and enqueues for immediate processing.
        """
        if script is not None:
            self.scripts.put((script, location))
            self.permanent_scripts.append((script, location))
        else:
            # Block Logic: Simulation Phase Delimiter.
            # Logic: Dispatches 'poison pills' (None, None) to all 8 workers 
            # to signal the end of the current timepoint workload.
            for i in range(0, 8):
                self.scripts.put((script, location))

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully joins all persistent worker threads."""
        for i in range(0, 8):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    Simulated worker thread.
    Functional Utility: Executes scripts from the node's shared queue and 
    participates in role-based simulation step coordination.
    """

    def __init__(self, device, the_id):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.the_id = the_id

    def run(self):
        """
        Main worker execution loop.
        Algorithm: Iterative multi-phase synchronization:
        Topology Fetch -> Task Consumption -> Queue Recycling.
        """
        # Block until setup is finalized.
        self.device.startup_event.wait()
        while True:
            # Phase 1: Role-Based Topology Coordination.
            if self.the_id == 0:
                # Coordinator: fetches current network graph.
                Device.barrier.wait()
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # Local Synchronization: ensure all threads share the updated topology.
            self.device.neighbours_done.wait()

            if self.device.neighbours is None:
                break

            # Block Logic: Task Execution Phase.
            while True:
                (script, location) = self.device.scripts.get()
                script_data = []

                if script is None:
                    # Rendezvous Point: all threads reach end-of-step delimiter.
                    self.device.neighbours_done.wait()
                    if self.the_id == 0:
                        # Coordination Logic: Re-populates the queue for the next cycle.
                        for (script, location) in self.device.permanent_scripts:
                            self.device.scripts.put((script, location))
                    break

                # Critical Section: Spatial mutual exclusion across the entire network.
                if location is not None:
                    Device.lockList[location].acquire()
                    
                    # Aggregate neighborhood and local data.
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        # Apply computational logic and propagate results.
                        result = script.run(script_data)
                        for device in self.device.neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)
                    
                    # Release spatial mutex.
                    Device.lockList[location].release()

from threading import Semaphore, Lock
class ReusableBarrier():
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Implements a double-gate mechanism with semaphores to 
    ensure total temporal alignment across simulation cycles.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Executes the two-phase arrival and exit sequence."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement and semaphore signaling."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
