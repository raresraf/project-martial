"""
@f84a31ed-21e6-4bc4-8ccc-31494078b2a1/device.py
@brief Distributed sensor network simulation with hardware-optimized worker pools.
This module implements a dynamic processing model that optimizes parallelism by 
sizing the internal worker pool based on the system's CPU core count. Computational 
tasks are distributed to persistent threads (DeviceThread) using a round-robin 
strategy. Consistency is guaranteed through a network-wide pool of spatial locks, 
while temporal synchronization is enforced via a hierarchy of local and global 
monitor-based barriers.

Domain: Hardware-Aware Parallelism, Worker Pools, Spatial Mutual Exclusion.
"""

from threading import Event, Thread, Lock, Condition
import multiprocessing


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
        """Blocks the caller until the arrival threshold is reached."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Threshold met: release all participants and reset for next cycle.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Representation of a node in the sensor network simulation.
    Functional Utility: Manages local data state, coordinates global synchronization 
    resource distribution, and supervises a pool of hardware-optimized worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.timepoint_done = Event()
        self.got_neighbours = Event()

        self.neighbours = []
        self.thread_list = []

        self.device_barrier = None
        self.data_lock = None

        # Round-robin assignment pointer.
        self.counter = 0

        # Optimization Logic: sizes the worker pool based on hardware capability.
        self.nr_thread = multiprocessing.cpu_count()
        # Internal barrier for local thread group.
        barrier = ReusableBarrier(self.nr_thread)
        lock = Lock()
        for i in xrange(self.nr_thread):
            thread = DeviceThread(self, i, barrier, lock)
            self.thread_list.append(thread)
            thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource factory.
        Logic: Coordinator node (ID 0) initializes the shared barrier and 
        pre-allocates a mutex for every sensor location detected in the group.
        """
        for device in devices:
            if self.device_id > device.device_id:
                # Leader Election check: only the node with min ID proceeds.
                return None
        
        # Atomic Resource Allocation: network-wide rendezvous and spatial lock pool.
        self.device_barrier = ReusableBarrier(len(devices))
        self.data_lock = dict()
        for device in devices:
            for location in device.sensor_data:
                if location not in self.data_lock:
                    self.data_lock[location] = Lock()
        
        # Propagation: distribute resources to all peer nodes.
        for device in devices:
            device.set_barrier(self.device_barrier)
            device.set_data_lock(self.data_lock)

    def set_barrier(self, device_barrier):
        """Injects the shared network barrier."""
        self.device_barrier = device_barrier

    def set_data_lock(self, data_lock):
        """Injects the global mapping of spatial locks."""
        self.data_lock = data_lock

    def acquire_lock(self, location):
        """Claims exclusive access to a spatial location across the network."""
        self.data_lock[location].acquire()

    def release_lock(self, location):
        """Releases the spatial lock."""
        self.data_lock[location].release()

    def assign_script(self, script, location):
        """
        Task Distribution Logic.
        Algorithm: Round-robin assignment of scripts to the worker pool.
        """
        if script is not None:
            pos = self.counter
            self.thread_list[pos].script_list.append((script, location))
            # Signal the targeted worker that a new task is available.
            self.thread_list[self.counter].script_received.set()
            self.counter = (self.counter + 1) % self.nr_thread
        else:
            # Signal end of step workload to all local workers.
            self.timepoint_done.set()
            for thread in self.thread_list:
                thread.script_received.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully joins all persistent worker threads."""
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    """
    Worker implementation for the node.
    Functional Utility: Executes a subset of computational scripts while 
    participating in role-based simulation step coordination.
    """

    def __init__(self, device, thread_id, thread_barrier, thread_lock):
        Thread.__init__(self, name="D:%d T:%d" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id
        # Worker-local task buffer.
        self.script_list = []
        self.thread_barrier = thread_barrier
        self.thread_lock = thread_lock
        self.script_received = Event()

    def run_scripts(self, index, neighbours):
        """
        Computational Execution logic.
        Algorithm: Sequential processing of local buffer using spatial mutual exclusion.
        """
        size = len(self.script_list)
        while index < size:
            (script, location) = self.script_list[index]
            script_data = []
            
            # Critical Section: Spatial mutual exclusion across the entire network.
            self.device.acquire_lock(location)
            
            # Aggregate neighborhood state.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Include local state.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
                
            if script_data != []:
                # Apply computational logic and propagate results.
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            
            # Release spatial mutex.
            self.device.release_lock(location)
            index += 1
        return index

    def run(self):
        """
        Main worker execution loop.
        Algorithm: Iterative multi-phase synchronization: 
        Wait for Step -> Fetch Topology -> Parallel Process -> Local Barrier -> Global Barrier.
        """
        while True:
            # Phase 1: Role-Based Topology discovery.
            if self.thread_id == 0:
                # Coordinator: fetches current network graph.
                self.device.neighbours = \
                    self.device.supervisor.get_neighbours()
                self.device.got_neighbours.set()
            else:
                # Workers: block until coordinator finishes discovery.
                self.device.got_neighbours.wait()
            neighbours = self.device.neighbours

            if neighbours is None:
                break

            # Block Logic: Event-driven task processing.
            index = 0
            while not self.device.timepoint_done.is_set():
                # Process any currently queued scripts.
                index = self.run_scripts(index, neighbours)
                
                # Wait for more work assignment signals.
                self.script_received.wait()
                self.script_received.clear()
            
            # Final flush of tasks assigned just before timepoint_done.
            self.run_scripts(index, neighbours)

            # Local Rendezvous Point: ensure all threads finished processing before consensus.
            self.thread_barrier.wait()
            
            # Global Consensus Phase.
            if self.thread_id == 0:
                # Coordinator: resets events and waits for the entire network group.
                self.device.timepoint_done.clear()
                self.device.got_neighbours.clear()
                self.device.device_barrier.wait()
            
            # Local Rendezvous Point: ensure coordinator finished consensus.
            self.thread_barrier.wait()
