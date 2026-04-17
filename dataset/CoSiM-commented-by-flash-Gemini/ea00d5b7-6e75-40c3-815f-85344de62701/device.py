"""
@ea00d5b7-6e75-40c3-815f-85344de62701/device.py
@brief Distributed sensor network simulation with persistent worker pool and lazy lock propagation.
This module implements a coordinated parallel processing framework using a 
persistent worker pool to handle computational scripts asynchronously. It features 
a lazy-initialization strategy for spatial synchronization, where mutexes for 
sensor locations are created on-demand and immediately broadcast to all peer nodes 
to ensure network-wide consistency. Global temporal alignment is enforced via a 
re-entrant synchronization barrier.

Domain: Parallel Worker Pools, Lazy Lock Synchronization, Distributed State Management.
"""

from threading import Event, Thread, Lock
from Queue import Queue
from reentrantbarrier import Barrier

class Device(object):
    """
    Core network node representation.
    Functional Utility: Manages local data state, coordinates the lazy allocation 
    and propagation of shared spatial locks, and maintains a local parallel worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        # Dynamic Spatial Lock Registry.
        self.location_locks = {} 
        self.barrier = None 
        # State flag for simulation phase management.
        self.ready_to_get_script = False 
        self.all_devices = None 

        # Primary lifecycle management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """Injects the shared network barrier."""
        self.barrier = barrier

    def broadcast_barrier(self, devices):
        """Helper to distribute the shared barrier instance across the network."""
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(self.barrier)


    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) initializes the shared Barrier and 
        propagates it to all members of the group.
        """
        self.all_devices = devices
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            self.broadcast_barrier(devices)

    def assign_script(self, script, location):
        """
        Task and Lock management.
        Logic: Performs lazy initialization of spatial mutexes. If a location is new, 
        a mutex is created and distributed to all nodes in the network group.
        """
        if script is None:
            # Signal end of task assignment for current timepoint.
            self.timepoint_done.set()
            return
        else:
            # Block Logic: On-Demand lock creation.
            if self.location_locks.setdefault(location, None) is None:
                self.location_locks[location] = Lock()
                self.ready_to_get_script = True

            # Propagation Logic: Broadcast the new spatial lock to all peer nodes.
            self.broadcast_lock_for_location(location)

            self.scripts.append((script, location))
            self.script_received.set()

    def broadcast_lock_for_location(self, location):
        """Helper to ensure a shared lock instance is used for the spatial location across nodes."""
        for device_no in xrange(len(self.all_devices)):
            self.all_devices[device_no].location_locks[location] = self.location_locks[location]



    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data



    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    Node-level manager.
    Functional Utility: Manages simulation phases and delegates tasks to a persistent 
    pool of 8 execution worker threads via a shared queue.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_count = 8
        self.pool = Queue(self.thread_count)
        self.threads = []
        self.create_workers()
        self.start_workers()

    def create_workers(self):
        """Spawns persistent worker threads."""
        for _ in xrange(self.thread_count):
            self.threads.append(Thread(target=self.execute_script))

    def start_workers(self):
        """Activates the parallel execution pool."""
        for thread in self.threads:
            thread.start()


    def collect_data_from_neighbours(self, neighbours, location):
        """Gathers sensor state from all nodes in the neighborhood graph."""
        result = []
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    result.append(data)
        return result

    def execute_script(self):
        """
        Main worker execution loop.
        Logic: Continuously pulls tasks from the queue and applies script logic 
        under network-wide spatial locks until a poison pill is received.
        """
        # Block on first task arrival.
        neighbours, script, location = self.pool.get()

        while True:
            # Termination Logic: exit if poison pill (None, None, None) is detected.
            if neighbours is None and script is None and location is None:
                self.pool.task_done()
                break

            script_data = []
            # Critical Section: Spatial mutual exclusion across the entire network.
            self.device.location_locks[location].acquire()
            
            # Aggregate neighborhood and local state.
            collected_data = self.collect_data_from_neighbours(neighbours, location)
            if collected_data:
                script_data = script_data + collected_data
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Apply computational logic and propagate results to all nodes in the graph.
                result = script.run(script_data)

                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            
            # Release spatial mutex.
            self.device.location_locks[location].release()
            
            # Signal task completion and wait for next item.
            self.pool.task_done()
            neighbours, script, location = self.pool.get()

    def run(self):
        """
        Main orchestration loop for the node simulation.
        Algorithm: Iterative sequence of topology refresh, task dispatch, and consensus.
        """
        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Simulation Phase Coordination.
            while True:
                # Wait for task arrival signal.
                self.device.timepoint_done.wait()

                if not self.device.ready_to_get_script:
                    # Current step workload is dispatched.
                    self.device.timepoint_done.clear()
                    self.device.ready_to_get_script = True
                    break
                else:
                    # Offload assignments into the worker pool.
                    for (script, location) in self.device.scripts:
                        self.pool.put((neighbours, script, location))
                    self.device.ready_to_get_script = False


            # Wait for local pool to finish its tasks.
            self.pool.join()
            # Global Rendezvous point for network-wide consensus.
            self.device.barrier.wait()

        # Shutdown Logic: Dispatch poison pills to all workers.
        self.pool.join()
        for _ in xrange(self.thread_count):
            self.pool.put((None, None, None))
        for thread in self.threads:
            thread.join()
        # Resource Reclaim.
        self.device.location_locks.clear()
