"""
@e0289284-1dd2-4a36-9748-9aa340f4df30/device.py
@brief Distributed sensor network simulation with persistent thread pool and lazy locking.
This module implements a coordinated parallel processing framework using a 
persistent 'ThreadPool' to handle computational tasks asynchronously. It features 
a lazy-initialization strategy for spatial synchronization, where mutexes for 
sensor locations are created and propagated across the entire network only when first 
referenced. Global temporal alignment is maintained via a two-phase semaphore barrier.

Domain: Parallel Worker Pools, Lazy Locking, Distributed State Synchronization.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

class ThreadPool(object):
    """
    Management layer for a persistent pool of computational workers.
    Functional Utility: Provides an asynchronous submission interface and 
    orchestrates the lifecycle of workers using a thread-safe task queue.
    """
    
    def __init__(self, thread_number, device):
        """
        Initializes the pool and spawns persistent workers.
        @param thread_number: size of the parallel pool.
        @param device: parent device context.
        """
        self.thread_number = thread_number
        self.queue = Queue(self.thread_number)
        self.threads = []
        self.device = device

        # Spawns persistent worker threads.
        for _ in xrange(thread_number):
            self.threads.append(Thread(target=self.execute))

        # Activates the pool.
        for thread in self.threads:
            thread.start()

    def execute(self):
        """
        Main execution loop for worker threads.
        Logic: Continuously pulls tasks from the queue and applies script logic 
        until a termination signal (poison pill) is received.
        """
        # Block on first task arrival.
        neighbours, script, location = self.queue.get()

        # Termination Logic: exits if all components of the task tuple are None.
        while neighbours is not None \
              and script is not None \
              and location is not None:

            # Execute the core computational task.
            self.run(neighbours, script, location)

            # Signal completion to support queue.join().
            self.queue.task_done()

            # Wait for next task.
            neighbours, script, location = self.queue.get()

        # Acknowledge the termination signal.
        self.queue.task_done()

    def run(self, neighbours, script, location):
        """
        Executes a single computational script.
        Logic: Atomically acquires the spatial lock for the location, gathers 
        neighborhood data, and propagates results.
        """
        script_data = []
        # Critical Section: Spatial mutual exclusion across the entire network.
        self.device.location_lock[location].acquire()
        
        # Aggregate neighborhood state.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Include local sensor state.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Apply computational logic.
            result = script.run(script_data)

            # Propagation: Updates all peers in the neighborhood.
            for device in neighbours:
                device.set_data(location, result)
            self.device.set_data(location, result)
            
        # Release the spatial mutex.
        self.device.location_lock[location].release()

    def submit(self, neighbours, script, location):
        """Enqueues a new task for parallel execution."""
        self.queue.put((neighbours, script, location))

    def wait(self):
        """Blocks until the internal task queue is completely drained."""
        self.queue.join()

    def end(self):
        """
        Gracefully terminates all pool threads.
        Logic: Flushes pending work and dispatches N poison pills.
        """
        self.wait()

        for _ in xrange(self.thread_number):
            self.submit(None, None, None)

        for thread in self.threads:
            thread.join()


class Barrier(object):
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
        """Executes the two-phase synchronization rendezvous."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement and semaphore release."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Arrival threshold reached: release the gate.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data and coordinates the lazy allocation 
    and propagation of shared synchronization resources.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        # Sparse repository of spatial locks (max 100 locations).
        self.location_lock = [None] * 100
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Main lifecycle management thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.all_devices = None
        self.recived_flag = False

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node initializes and propagates the shared barrier.
        """
        self.all_devices = devices

        if self.barrier is None:
            # Singleton setup: ensure all devices share the same barrier instance.
            self.barrier = Barrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Task and Lock management.
        Logic: Lazy initialization of spatial locks. If a location is new, 
        a mutex is created and distributed to all nodes in the network group.
        """
        if script is not None:
            # Lazy Mutex Initialization.
            if self.location_lock[location] is None:
                self.location_lock[location] = Lock()
                self.recived_flag = True

                # Propagation Logic: Broadcast the new lock to all peers.
                for device_number in xrange(len(self.all_devices)):
                    self.all_devices[device_number].location_lock[location] \
                        = self.location_lock[location]

            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Signal completion of current timepoint workload.
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


class DeviceThread(Thread):
    """
    Node-level orchestration thread.
    Functional Utility: Manages simulation timepoints and delegates tasks 
    to the persistent parallel thread pool.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Sized for 8 concurrent computational workers.
        self.thread_pool = ThreadPool(8, self.device)

    def run(self):
        """
        Main orchestration loop.
        Algorithm: Iterative sequence of topology refresh, task offloading, and global synchronization.
        """
        while True:
            # Fetch current topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Workload Execution.
            while True:
                # Wait for the supervisor to finalize script assignments.
                self.device.timepoint_done.wait()

                # Phase Check: ensures all tasks in the current batch are processed.
                if self.device.recived_flag:
                    # Offload tasks to the parallel workers.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
                    self.device.recived_flag = False
                else:
                    # Simulation step complete.
                    self.device.timepoint_done.clear()
                    self.device.recived_flag = True
                    break

            # Wait for the local pool to finish its assigned tasks.
            self.thread_pool.wait()

            # Global Rendezvous point for network-wide consensus.
            self.device.barrier.wait()

        # Shutdown pool resources.
        self.thread_pool.end()
