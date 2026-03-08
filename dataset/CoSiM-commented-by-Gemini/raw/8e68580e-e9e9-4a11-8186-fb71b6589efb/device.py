# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices using a persistent thread pool
and a complex, custom work-distribution mechanism.
"""

from threading import Event, Thread, Lock
# This file likely contains the ReusableBarrier class defined in previous examples.
import reusable_barrier_semaphore


class Device(object):
    """
    Represents a single device, managing a pool of worker threads and a complex
    set of synchronization primitives for work distribution and time step management.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): The device's local sensor data.
        supervisor (object): A reference to the central supervisor.
        scripts (list): A list of scripts assigned for the current time step.
        number_threads_per_device (int): The size of the persistent worker thread pool.
        barrier_timepoint (ReusableBarrier): A global barrier to sync all threads at the end of a time step.
        barrier_get_neighbours (ReusableBarrier): A local barrier for device threads.
        barrier_reset_counters (ReusableBarrier): A local barrier for resetting state.
        script_access (Lock): A lock protecting the shared script index.
        script_access_index (int): A shared index used by threads to get work from the scripts list.
        finished_scripts (int): A flag indicating the current time step's work is done.
        locks_location_update_data (list): A globally shared list of locks for data locations.
        event_access_data (Event): An event to signal that new scripts are available.
        thread_list (list): The pool of `DeviceThread` worker instances.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device and its pool of worker threads."""

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        self.number_threads_per_device = 8
        self.number_locations = 100  # Assumes a hardcoded number of possible locations.

        # --- Synchronization Primitives ---
        self.barrier_timepoint = None
        self.barrier_get_neighbours = reusable_barrier_semaphore.ReusableBarrier(self.number_threads_per_device)
        self.barrier_reset_counters = reusable_barrier_semaphore.ReusableBarrier(self.number_threads_per_device)
        self.script_access = Lock()
        self.script_access_index = -1
        self.finished_scripts = 0
        self.exit_simulation = 0
        self.neighbours = []
        self.all_devices = []
        self.locks_location_update_data = [Lock() for _ in xrange(self.number_locations)]
        self.event_access_data = Event()

        # Create and start the persistent pool of worker threads.
        self.thread_list = [DeviceThread(self, i) for i in xrange(self.number_threads_per_device)]
        for thread in self.thread_list:
            thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources (global barrier, location locks) for all devices."""
        self.all_devices = devices
        if self.device_id == 0:  # Device 0 is the leader.
            self.barrier_timepoint = reusable_barrier_semaphore.ReusableBarrier(
                len(self.all_devices) * self.number_threads_per_device
            )
            # Distribute shared objects to all devices.
            for device in self.all_devices:
                device.barrier_timepoint = self.barrier_timepoint
                device.locks_location_update_data = self.locks_location_update_data

    def assign_script(self, script, location):
        """Adds a script to the list and signals workers that new data is available."""
        self.scripts.append((script, location))
        self.event_access_data.set()

    def get_data(self, location):
        """Retrieves sensor data. Not internally thread-safe."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data. Not internally thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its worker threads."""
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread that processes scripts using a complex, shared-index mechanism.

    NOTE: The work distribution logic here is extremely complex and brittle. It
    manually implements a form of work-stealing that could be more robustly and
    simply achieved using a standard `Queue` as seen in producer-consumer patterns.
    """

    def __init__(self, device, thread_id):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id
        self.script_to_access = 0

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            # --- Neighbor Discovery ---
            # The first thread on each device acts as a leader to get neighbors.
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                if self.device.neighbours is None:
                    self.device.exit_simulation = 1 # Signal all threads on this device to exit.

            # All local threads sync after neighbor discovery.
            self.device.barrier_get_neighbours.wait()

            if self.device.exit_simulation == 1:
                break

            # --- Custom Work-Stealing Loop ---
            while self.device.finished_scripts == 0:
                # Atomically get a unique index into the scripts list.
                with self.device.script_access:
                    self.device.script_access_index += 1
                    self.script_to_access = self.device.script_access_index
                
                # If this thread's index is out of bounds, wait for more scripts.
                if self.script_to_access >= len(self.device.scripts):
                    self.device.event_access_data.wait()

                # After waking up, re-check the finished flag.
                if self.device.finished_scripts == 1:
                    break
                
                with self.device.script_access:
                    # Clear the event; subsequent threads will block if they run out of work.
                    # This is a potential race condition.
                    self.device.event_access_data.clear()

                # Re-check the finished flag again.
                if self.device.finished_scripts == 1:
                    break
                
                # --- Script Execution ---
                (script, location) = self.device.scripts[self.script_to_access]
                
                # A None script is the sentinel for the end of a time step.
                if script is None:
                    self.device.finished_scripts = 1
                    self.device.event_access_data.set() # Wake up all waiting threads.
                    break

                # Acquire the global lock for this location before processing.
                with self.device.locks_location_update_data[location]:
                    script_data = []
                    # Aggregate data from neighbors and self.
                    if self.device.neighbours:
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    # Execute and disseminate results.
                    if script_data:
                        result = script.run(script_data)
                        for device in self.device.neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)

            # --- Reset and Synchronization Phase ---
            # All local threads wait here before resetting state.
            self.device.barrier_reset_counters.wait()
            
            # The leader thread resets the device's state for the next time step.
            if self.thread_id == 0:
                self.device.scripts.pop() # Remove the None sentinel script.
                self.device.event_access_data.clear()
                self.device.script_access_index = -1
                self.device.finished_scripts = 0

            # All threads from all devices wait here before starting the next time step.
            self.device.barrier_timepoint.wait()
