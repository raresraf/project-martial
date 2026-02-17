"""
Models a distributed network of devices using a persistent worker pool.

Each device in this simulation is multi-threaded, running a fixed pool of
worker threads from initialization. One worker thread per device is designated
as a 'leader' to handle coordination tasks like fetching network data. The
system uses a multi-level barrier system for fine-grained synchronization
both within a device's threads and across all devices in the network.
"""

from threading import Event, Thread, Lock
from Queue import Queue, Empty
from barrier import Barrier


class Device(object):
    """
    Represents a single device in the network, managed by a pool of worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device and its persistent pool of worker threads.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): The initial sensor data.
            supervisor (object): The supervisor for network topology information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.num_threads = 8  # The number of worker threads per device.
        self.scripts = []
        self.jobs_queue = Queue() # A queue for scripts to be executed.
        self.neighbours = []

        # Synchronization primitives for coordinating worker threads within this device.
        self.scripts_received = Event()
        self.scripts_received_barrier = Barrier(self.num_threads)
        self.scripts_processed_barrier = Barrier(self.num_threads)
        self.neighbours_received_barrier = Barrier(self.num_threads)

        # Shared resources managed by the leader device.
        self.location_locks = {}
        self.timepoint_barrier = None

        # Create and start the persistent pool of worker threads.
        self.threads = [DeviceThread(self, i) for i in xrange(self.num_threads)]
        for i in xrange(self.num_threads):
            self.threads[i].start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for the entire network, managed by a leader device.

        The device with the lowest ID is elected leader and is responsible for creating
        the global synchronization barrier and location-based locks.

        Args:
            devices (list): A list of all devices in the network.
        """
        leader = min([device.device_id for device in devices])

        if self.device_id == leader:
            # The leader device discovers all unique locations and creates a lock for each.
            locations_set = set()
            for device in devices:
                locations_set.update(device.sensor_data.keys())
            locations = list(locations_set)
            self.location_locks = {location: Lock() for location in locations}

            # The leader creates the global barrier for end-of-timepoint synchronization.
            self.timepoint_barrier = Barrier(len(devices))

            # Distribute references to the shared resources to all other devices.
            for device in devices:
                device.location_locks = self.location_locks
                device.timepoint_barrier = self.timepoint_barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed. A 'None' script signals the end of assignments.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.jobs_queue.put((script, location))
        else:
            # All scripts for the timepoint have been queued; signal the workers.
            self.scripts_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its worker threads."""
        for i in xrange(self.num_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A persistent worker thread for a device. One thread per device acts as a leader.
    """

    def __init__(self, device, id_thread):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id_thread = id_thread

    def run(self):
        """
        The main execution loop for the worker thread, organized by synchronization barriers.
        """
        leader = 0
        while True:
            # --- Phase 1: Get Neighbors ---
            # The leader worker for this device fetches the neighbor list.
            if self.id_thread == leader:
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # All workers on this device wait until the neighbor list is available.
            self.device.neighbours_received_barrier.wait()

            if self.device.neighbours is None:
                # Shutdown signal received, exit the loop.
                break

            # --- Phase 2: Wait for and Process Scripts ---
            # All workers wait for the supervisor to finish assigning scripts.
            self.device.scripts_received.wait()
            self.device.scripts_received_barrier.wait()
            if self.id_thread == leader:
                self.device.scripts_received.clear() # Leader resets the event.

            # All workers process jobs from the queue until it's empty.
            while True:
                try:
                    (script, location) = self.device.jobs_queue.get_nowait()
                except Empty:
                    break
                self.run_script(script, location)
            
            # All workers wait here until all scripts for this timepoint are processed.
            self.device.scripts_processed_barrier.wait()

            # --- Phase 3: Global Synchronization ---
            # The leader worker is responsible for synchronizing with other devices.
            if self.id_thread == leader:
                # This logic appears to re-queue scripts, which would lead to an infinite
                # loop in the next iteration. It might be a bug or incomplete.
                for script in self.device.scripts:
                    self.device.jobs_queue.put(script)
                
                # Wait at the global barrier for all devices to finish the timepoint.
                self.device.timepoint_barrier.wait()

    def run_script(self, script, location):
        """
        Executes a single script, ensuring thread-safety via location-specific locks.
        """
        # Invariant: Acquire a lock for the specific location to prevent race
        # conditions when multiple scripts target the same data.
        with self.device.location_locks[location]:
            script_data = []
            
            # Aggregate data from neighbors and the local device.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: Execute script only if there is data.
            if script_data:
                result = script.run(script_data)
                
                # Distribute the result to all relevant devices.
                for device in self.device.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)