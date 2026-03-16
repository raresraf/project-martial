"""
This module implements a device simulation framework using a multi-level
leader-election pattern for coordination.

A global leader device initializes shared resources. Within each device, a local
leader worker thread manages communication with the supervisor and global
synchronization. The implementation uses a mix of threading primitives for complex
coordination, but contains some unusual design choices.
"""

from threading import Event, Thread, Lock
from Queue import Queue, Empty
from barrier import Barrier


class Device(object):
    """Represents a device that manages a pool of worker threads.

    A 'leader' device (lowest ID) is responsible for creating and distributing
    shared resources like locks and a global time-step barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device and its pool of worker threads."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.num_threads = 8
        self.scripts = []
        self.jobs_queue = Queue()
        self.neighbours = []

        # Events and Barriers for intra-device synchronization.
        self.scripts_received = Event()
        self.scripts_received_barrier = Barrier(self.num_threads)
        self.scripts_processed_barrier = Barrier(self.num_threads)
        self.neighbours_received_barrier = Barrier(self.num_threads)

        # Shared resources to be provided by the leader device.
        self.location_locks = {}
        self.timepoint_barrier = None

        # Note: DeviceThread here is used as a worker thread.
        self.threads = [DeviceThread(self, i) for i in xrange(self.num_threads)]
        for i in xrange(self.num_threads):
            self.threads[i].start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources via a leader election.

        The device with the lowest ID becomes the leader, creating and
        distributing locks for all data locations and a global barrier.
        """
        leader_id = min([device.device_id for device in devices])

        if self.device_id == leader_id:
            # Aggregate all unique data locations from all devices.
            locations_set = set()
            for device in devices:
                locations_set.update(device.sensor_data.keys())
            locations = list(locations_set)
            self.location_locks = {location : Lock() for location in locations}

            # Create the global barrier for synchronizing time steps.
            self.timepoint_barrier = Barrier(len(devices))

            # Distribute the shared resources to all other devices.
            for device in devices:
                device.location_locks = self.location_locks
                device.timepoint_barrier = self.timepoint_barrier

    def assign_script(self, script, location):
        """Assigns a script and adds it to the work queue."""
        if script is not None:
            self.scripts.append((script, location))
            self.jobs_queue.put((script, location))
        else:
            # A 'None' script signals that all scripts for this time step have been assigned.
            self.scripts_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all worker threads for a clean shutdown."""
        for i in xrange(self.num_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    Represents a worker thread within a Device.

    These threads perform the actual script execution. A thread with id_thread=0
    acts as a "leader" within the device's thread pool, handling supervisor
    communication and global synchronization.
    """

    def __init__(self, device, id_thread):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id_thread = id_thread

    def run(self):
        """The main loop for the worker thread."""
        leader = 0
        while True:
            # Block Logic: The leader worker fetches neighbors for the entire device.
            if self.id_thread == leader:
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # All workers wait here until the neighbor list is fetched.
            self.device.neighbours_received_barrier.wait()

            # Supervisor signals simulation end by returning None.
            if self.device.neighbours is None:
                break

            # Block Logic: All workers wait for the signal that scripts are assigned.
            self.device.scripts_received.wait()
            self.device.scripts_received_barrier.wait() # Synchronize after receiving event.
            self.device.scripts_received.clear()

            # Block Logic: Work-stealing loop. Each worker processes jobs from the queue.
            while True:
                try:
                    (script, location) = self.device.jobs_queue.get_nowait()
                except Empty:
                    break # The queue is empty for now.
                self.run_script(script, location)
            
            # All workers synchronize after processing is done for this step.
            self.device.scripts_processed_barrier.wait()

            # Block Logic: The leader worker handles end-of-step synchronization.
            if self.id_thread == leader:
                # Oddity: Re-queues all scripts. This may be a bug, as they have
                # just been processed.
                for script in self.device.scripts:
                    self.device.jobs_queue.put(script)
                
                # The leader is the only thread from the device to wait on the global barrier.
                self.device.timepoint_barrier.wait()

    def run_script(self, script, location):
        """Executes a single script."""
        # Use a `with` statement to ensure the lock is always released.
        with self.device.location_locks[location]:
            script_data = []
            
            # Gather data from neighbors.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)

                # Disseminate the result to all neighbors and self.
                for device in self.device.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
