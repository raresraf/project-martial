"""
@file device.py
@brief A distributed device simulation using a dynamic master-worker thread model.

This script models a network of devices operating in synchronized time steps. Each
device has a main "master" thread that, for each time step, dynamically spawns a
pool of "worker" threads to execute scripts. Synchronization uses a global
barrier and a two-tiered locking mechanism for data access.
"""

from threading import Event, Thread, Lock
# Assumes ReusableBarrierCond is defined in a local 'reusable_barrier' module.
from reusable_barrier import ReusableBarrierCond

class Device(object):
    """
    Represents a node in the distributed network. It manages its own state and a
    master control thread (`DeviceThread`).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # A per-device lock to protect its own sensor_data dictionary.
        self.lock_data = Lock()
        # A dictionary of global locks for each location, shared among all devices.
        self.locks_locations = {}
        # The shared barrier for synchronizing all devices.
        self.barrier = None
        self.worker_threads_no = 8
        self.worker_threads = []

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (locks and barrier) for all devices.
        This method is orchestrated by device 0.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))

            # Identify all unique locations across the entire system.
            all_locations = []
            for device in devices:
                for location in device.sensor_data:
                    if location not in all_locations:
                        all_locations.append(location)
                        # Create one global lock for each unique location.
                        self.set_lock_on_location(location)

            # Distribute the shared locks and barrier to all other devices.
            for device in devices:
                device.set_locks_locations(self.locks_locations)
                device.set_barrier(self.barrier)

    def set_barrier(self, pbarrier):
        """Sets the shared barrier instance for this device."""
        self.barrier = pbarrier

    def set_lock_on_location(self, plocation):
        """Initializes a lock for a given location (called by device 0)."""
        self.locks_locations[plocation] = Lock()

    def set_locks_locations(self, plocks):
        """Sets the shared dictionary of location locks for this device."""
        self.locks_locations = plocks

    def assign_script(self, script, location):
        """Assigns a script to the device for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its master thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The "master" thread for a device. It coordinates the execution of scripts for
    each timepoint by dynamically creating and managing a pool of worker threads.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def spread_scripts_to_threads(self):
        """Distributes the assigned scripts among the available worker threads."""
        script_no = 0
        for (script, location) in self.device.scripts:
            thread_idx = script_no % self.device.worker_threads_no
            self.device.worker_threads[thread_idx].add_script(script, location)
            script_no += 1

    def run(self):
        """Main execution loop, processing timepoints sequentially."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal

            # Wait for the supervisor to signal that all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()
            
            # Inefficiently create a new pool of worker threads for every timepoint.
            for i in range(self.device.worker_threads_no):
                worker_thread = DeviceWorkerThread(self, neighbours)
                self.device.worker_threads.append(worker_thread)

            # Assign this timepoint's scripts to the newly created workers.
            self.spread_scripts_to_threads()

            # Start all worker threads.
            for i in range(len(self.device.worker_threads)):
                self.device.worker_threads[i].start()

            # Wait for all worker threads to complete their assigned scripts.
            for i in range(len(self.device.worker_threads)):
                self.device.worker_threads[i].join()

            # Clean up the worker threads for this timepoint.
            del self.device.worker_threads[:]
            self.device.timepoint_done.clear()

            # Wait at the global barrier to synchronize with all other devices.
            self.device.barrier.wait()


class DeviceWorkerThread(Thread):
    """A worker thread that executes a subset of a device's scripts for a timepoint."""
    def __init__(self, device_thread, neighbours):
        super(DeviceWorkerThread, self).__init__()
        self.master_thread = device_thread
        self.device_neighbours = neighbours
        self.assigned_scripts = []

    def add_script(self, script, location):
        """Adds a script to this worker's personal queue."""
        if script is not None:
            self.assigned_scripts.append((script, location))

    def run(self):
        """Executes all assigned scripts, handling a complex locking scheme."""
        for (script, location) in self.assigned_scripts:
            # Tier 1 Lock: Acquire the global lock for this specific location.
            # This ensures no other device can work on this location concurrently.
            self.master_thread.device.locks_locations[location].acquire()

            script_data = []
            
            # Aggregate data from neighbors, using a per-device lock.
            for device in self.device_neighbours:
                # Tier 2 Lock: Acquire the specific lock for the neighbor's data structure.
                # This lock is likely redundant given the global location lock is already held.
                device.lock_data.acquire()
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
                device.lock_data.release()

            # Aggregate data from self, using the per-device lock.
            self.master_thread.device.lock_data.acquire()
            data = self.master_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)
            self.master_thread.device.lock_data.release()

            if script_data != []:
                result = script.run(script_data)
                
                # Disseminate results, again using per-device locks.
                for device in self.device_neighbours:
                    device.lock_data.acquire()
                    device.set_data(location, result)
                    device.lock_data.release()

                self.master_thread.device.lock_data.acquire()
                self.master_thread.device.set_data(location, result)
                self.master_thread.device.lock_data.release()

            # Release the global location lock.
            self.master_thread.device.locks_locations[location].release()
