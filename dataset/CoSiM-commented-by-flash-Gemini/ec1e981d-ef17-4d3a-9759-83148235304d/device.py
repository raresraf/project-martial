"""
@ec1e981d-ef17-4d3a-9759-83148235304d/device.py
@brief Distributed sensor network simulation with window-based task parallelization.
This module implements a dynamic parallel processing framework where computational 
scripts are executed by transient threads. Concurrency is managed via a sliding-window 
throttling mechanism that limits the number of active worker threads to 8. Global 
consistency is ensured through a network-wide pool of spatial locks and a 
reusable synchronization barrier for temporal alignment across simulation cycles.

Domain: Parallel Sliding Windows, Thread Throttling, Distributed State Synchronization.
"""

from _threading_local import local
from threading import Event, Thread, Lock, RLock, Condition, Semaphore
from barrier import *

class Device(object):
    """
    Representation of a node in the sensor network simulation.
    Functional Utility: Manages local data, coordinates the discovery and 
    distribution of global synchronization resources, and provides an interface 
    for script assignment.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        # Mutex for protecting local data state.
        self.lock = Lock()
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        # Primary node management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

        self.sync_barrier = None
        self.devices = []
        self.location_locks = {}
        # Cached neighborhood topology.
        self.nbs = []


    def __str__(self):
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """Injects the shared network barrier."""
        self.sync_barrier = barrier

    def sync_with_others(self):
        """Blocks until all devices in the network have reached the current phase."""
        self.sync_barrier.wait()

    def set_locks(self, locks):
        """Injects the global mapping of spatial locks."""
        self.location_locks = locks

    def get_lock(self, location):
        """Helper to retrieve the mutex for a specific spatial location."""
        return self.location_locks[location]

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) initializes the shared network barrier 
        and pre-allocates a mutex for every sensor location detected in the group.
        """
        self.devices = devices
        if self.device_id == 0:
            # Atomic setup of shared rendezvous point.
            barrier = ReusableBarrier(len(devices))
            locks = {}
            # Discovery: aggregate all spatial locations for lock allocation.
            for dev in devices:
                for loc in dev.sensor_data:
                    locks[loc] = Lock()
                dev.set_barrier(barrier)
            
            # Propagation: distribute resources to peer nodes.
            for dev in devices:
                dev.set_locks(locks)

    def assign_script(self, script, location):
        """Registers a task and signals the orchestration thread."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Finalize assignment phase for current step.
            self.timepoint_done.set()

    def get_scripts(self):
        """Prepares a batch of task contexts for the execution layer."""
        return [(self, s, l, self.nbs) for (s, l) in self.scripts]

    def get_data(self, l):
        """Atomic retrieval of local sensor data."""
        self.lock.acquire()
        ret = self.sensor_data[l] if l in self.sensor_data else None
        self.lock.release()
        return ret

    def set_data(self, location, data):
        """Atomic update of local sensor state."""
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    Main orchestration thread for the node.
    Functional Utility: Manages simulation timepoints and implements a 
    window-based distributor for transient computational threads.
    """
    
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Concurrency boundary: defines the size of the sliding parallel window.
        self.max_running_threads_cnt = 8

    @staticmethod
    def exec_script(invoker_device, script, location, neighbourhood):
        """
        Worker function for transient threads.
        Logic: Implements atomic read-modify-write within the global spatial lock.
        """
        script_data = []
        # Critical Section: Spatial mutual exclusion across the entire network.
        invoker_device.location_locks[location].acquire()
        
        # Aggregate neighborhood data.
        for device in neighbourhood:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Include local sensor data.
        data = invoker_device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Apply computational logic.
            result = script.run(script_data)
            
            # Propagation: distribute results back to all peers in the graph.
            for device in neighbourhood:
                device.set_data(location, result)
            invoker_device.set_data(location, result)

        # Release spatial mutex.
        invoker_device.location_locks[location].release()

    def run(self):
        """
        Main execution loop for node orchestration.
        Algorithm: Iterative sequence: 
        Wait -> Topology Refresh -> Windowed Parallel execution -> Global Barrier.
        """
        while True:
            # Topology Discovery Phase.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.nbs = neighbours
            
            # Block until assignment phase for the step is complete.
            self.device.timepoint_done.wait()

            # Phase Rendezvous: ensure total network alignment before execution.
            self.device.sync_with_others()
            
            scrpts = []
            threads = []

            scrpts.extend(self.device.get_scripts())
            running_threads_cnt = 0

            # Block Logic: Windowed Task Parallelization.
            # Algorithm: Spawns transient threads but throttles via join() 
            # to maintain a fixed execution density.
            for (d, s, l, n) in scrpts:
                thread = Thread(name="T",
                                target=DeviceThread.exec_script,
                                args=(d, s, l, n))
                threads.append(thread)
                thread.start()
                running_threads_cnt += 1
                
                # Throttling Logic: implement a FIFO window of joined threads.
                if running_threads_cnt >= self.max_running_threads_cnt:
                    wthread = threads.pop(0)
                    running_threads_cnt -= 1
                    wthread.join()

            # Wait for any remaining threads in the final window.
            for thread in threads:
                thread.join()

            # Phase reset for the next cycle.
            self.device.timepoint_done.clear()
