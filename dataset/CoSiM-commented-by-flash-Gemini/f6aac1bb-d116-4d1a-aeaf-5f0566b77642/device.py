"""
@f6aac1bb-d116-4d1a-aeaf-5f0566b77642/device.py
@brief Distributed sensor network simulation with global spatial locking and fork-join parallel compute.
This module implements a coordinated parallel processing framework where node managers 
(DeviceThread) utilize a 'Fork-Join' model to dispatch computational tasks to 
transient worker threads (ScriptSolver). Consistency is guaranteed through a 
centrally-allocated pool of spatial locks that ensure network-wide mutual exclusion 
for specific sensor locations. Global temporal alignment is maintained via a 
two-phase semaphore-based synchronization barrier.

Domain: Parallel Fork-Join, Global Spatial Mutex, Distributed Coordination.
"""

from threading import Event, Thread, Lock
from reusable_barrier_semaphore import ReusableBarrier


class Device(object):
    """
    Representation of a node in the sensor network simulation.
    Functional Utility: Manages local data state, coordinates the distribution 
    of global synchronization resources, and provides the interface for task assignment.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        # Spatial lock repository populated during setup.
        self.location_locks = []
        # Temporal rendezvous point.
        self.next_timepoint_barrier = ReusableBarrier(0)
        # Primary orchestration thread.
        self.thread = DeviceThread(self)

    def __str__(self):
        return "Device %d" % self.device_id

    @staticmethod
    def count_locations(devices, devices_nr):
        """
        Discovery Helper.
        Logic: Scans all nodes to determine the maximum sensor location index 
        across the entire network.
        @return: Total count of required spatial locks.
        """
        locations_number = 0
        for i in range(devices_nr):
            for location in devices[i].sensor_data.keys():
                if location > locations_number:
                    locations_number = location
        return locations_number + 1

    def setup_devices(self, devices):
        """
        Global synchronization resource factory.
        Logic: Coordinator node (ID 0) initializes the shared barrier and 
        pre-allocates a mutex for every sensor location detected in the group.
        """
        devices_nr = len(devices)
        next_timepoint_barrier = ReusableBarrier(devices_nr)

        if self.device_id == 0:
            # Block Logic: Atomic Global setup.
            locations_number = self.count_locations(devices, devices_nr)

            # Allocation: Create a dedicated lock for every spatial location.
            for i in range(locations_number):
                lock = Lock()
                self.location_locks.append(lock)

            # Propagation: distribute resources to all peer nodes and activate managers.
            for i in range(devices_nr):
                for j in range(locations_number):
                    devices[i].location_locks.append(self.location_locks[j])

                devices[i].next_timepoint_barrier = next_timepoint_barrier
                devices[i].thread.start()

    def assign_script(self, script, location):
        """Registers a task and signals the orchestration thread."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Finalize assignment phase for current step.
            self.script_received.set()

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
    Node-level simulation manager.
    Functional Utility: Orchestrates simulation phases and implements a fork-join 
    parallel processing model for computational scripts.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for node orchestration.
        Algorithm: Iterative sequence: Wait -> Fork -> Join -> Consensus.
        """
        scriptsolvers = []
        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until assignment phase is complete.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Phase 1: Fork.
            # Spawns a transient thread for every script in the current batch.
            for (script, location) in self.device.scripts:
                scriptsolvers.append(
                    ScriptSolver(self.device, script, neighbours, location))

            workers_nr = len(scriptsolvers)
            for index in range(workers_nr):
                scriptsolvers[index].start()

            # Phase 2: Join.
            # wait for all local parallel tasks to finalize.
            for index in range(workers_nr):
                scriptsolvers[index].join()

            # Phase Reset.
            scriptsolvers = []

            # Global Consensus rendezvous.
            self.device.next_timepoint_barrier.wait()


class ScriptSolver(Thread):
    """
    Transient computational worker.
    Functional Utility: Executes a script while maintaining network-wide 
    spatial mutual exclusion for the target location.
    """
    
    def __init__(self, device, script, neighbours, location):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location

    def collect_data(self, neighbours, location):
        """Gathers sensor state from all nodes in the neighborhood graph."""
        data_script = []
        # local state.
        own_data = self.device.get_data(location)
        if own_data is not None:
            data_script.append(own_data)

        # neighborhood state.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                data_script.append(data)

        return data_script

    def update_data(self, neighbours, location, run_result):
        """Propagates result to self and all neighbors."""
        self.device.set_data(location, run_result)
        for device in neighbours:
            device.set_data(location, run_result)

    def solve(self, script, neighbours, location):
        """
        Main execution sequence.
        Logic: Implements atomic read-modify-write within the global spatial lock.
        """
        # Critical Section: Spatial mutual exclusion across the entire network.
        self.device.location_locks[location].acquire()

        data_script = self.collect_data(neighbours, location)
        if data_script != []:
            # Apply computational logic.
            run_result = script.run(data_script)
            self.update_data(neighbours, location, run_result)

        # Release spatial mutex.
        self.device.location_locks[location].release()

    def run(self):
        self.solve(self.script, self.neighbours, self.location)

from threading import *

class ReusableBarrier():
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Uses a double-gate mechanism with semaphores to ensure 
    temporal alignment across simulation cycles.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """Orchestrates the two-phase arrival and exit protocol."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement and semaphore release."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
