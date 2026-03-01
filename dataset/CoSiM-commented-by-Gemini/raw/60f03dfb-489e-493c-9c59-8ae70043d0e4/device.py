"""
@file device.py
@brief A framework for a distributed, time-stepped simulation of networked devices.

This script models a system of devices that operate in discrete, synchronized time steps.
Each device runs a fixed pool of threads to process assigned scripts in parallel. The
simulation manages distributed state, inter-device communication (via shared data
access), and a multi-level synchronization strategy to ensure coherent execution.
"""

from threading import Event, Thread, Condition, Lock


class Barrier(object):
    """
    A reusable barrier implementation that blocks a set of threads until all of
    them have called the wait() method.
    """

    def __init__(self, num_threads=0):
        """
        Initializes the barrier for a given number of threads.

        @param num_threads The number of threads that must wait on the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        
        # A condition variable to manage waiting and notification.
        self.cond = Condition()

    def wait(self):
        """
        Causes the calling thread to block until all `num_threads` threads have
        called this method. Then all threads are released.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # If this is the last thread to arrive, wake up all waiting threads.
        if self.count_threads == 0:
            self.cond.notify_all()
            # Reset the barrier for the next use.
            self.count_threads = self.num_threads
        else:
            # Wait for the other threads to arrive.
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a single node in the distributed network simulation. Each device
    manages its own sensor data, executes assigned scripts, and coordinates with
    neighboring devices using a fixed pool of worker threads.
    """
    
    # A global barrier to synchronize all Device instances at the end of a timepoint.
    bariera_devices = Barrier()
    # A collection of global locks, one for each location, to ensure mutual
    # exclusion when processing location-specific data.
    locks = []

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary holding the device's local data.
        @param supervisor A reference to the central supervisor object that manages the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Scripts assigned to this device for the current timepoint.
        self.scripts = []
        self.locations = []
        self.nr_scripturi = 0
        self.script_crt = 0

        # Event to signal that the device is ready to process the current timepoint.
        self.timepoint_done = Event()

        self.neighbours = []
        # Event to signal that the list of neighbors has been received.
        self.event_neighbours = Event()
        # Lock to protect the counter for scripts being processed.
        self.lock_script = Lock()
        # A local barrier for the device's own pool of threads.
        self.bar_thr = Barrier(8)

        # Each device has its own thread pool (1 main thread, 7 workers).
        self.thread = DeviceThread(self, 1) # The 'first' or 'main' thread.
        self.thread.start()
        self.threads = []
        for _ in range(7):
            tthread = DeviceThread(self, 0)
            self.threads.append(tthread)
            tthread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes static resources shared by all devices in the simulation.

        @param devices A list of all device objects in the simulation.
        """
        # Set up the global barrier for the total number of devices.
        Device.bariera_devices = Barrier(len(devices))
        
        # Initialize location-specific locks if not already done.
        if Device.locks == []:
            for _ in range(self.supervisor.supervisor.testcase.num_locations):
                Device.locks.append(Lock())

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device.

        @param script The script to execute.
        @param location The location associated with the script's data.
        """
        if script is not None:
            self.scripts.append(script)
            self.locations.append(location)
            self.nr_scripturi += 1
        else:
            # A None script signals the end of script assignment for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its worker threads."""
        self.thread.join()
        for tthread in self.threads:
            tthread.join()


class DeviceThread(Thread):
    """
    A worker thread for a Device. It executes scripts and handles synchronization.
    One thread per device is designated as 'first' to perform coordination tasks.
    """

    def __init__(self, device, first):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.first = first # Flag to identify the main thread for this device.

    def run(self):
        """The main execution loop for the thread, processing timepoints sequentially."""
        # This outer loop represents the progression through discrete timepoints.
        while True:
            # Block Logic (Main Thread Only): At the start of a timepoint, the main
            # thread fetches neighbors and resets state.
            if self.first == 1:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.script_crt = 0
                # Signal to other threads that neighbors are available.
                self.device.event_neighbours.set()

            # All threads wait here until the neighbors are set.
            self.device.event_neighbours.wait()
            
            # If neighbors is None, it's a signal to shut down.
            if self.device.neighbours is None:
                break

            # All threads wait here until all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()

            # This inner loop processes all assigned scripts for the current timepoint.
            # Threads within a device work together to complete the list of scripts.
            while True:
                # Block Logic: Atomically get the index of the next script to process.
                self.device.lock_script.acquire()
                index = self.device.script_crt
                self.device.script_crt += 1
                self.device.lock_script.release()

                # If the index is out of bounds, all scripts have been processed.
                if index >= self.device.nr_scripturi:
                    break

                location = self.device.locations[index]
                script = self.device.scripts[index]

                # Block Logic (Critical Section): Acquire a global lock for the specific location.
                # This ensures that data for this location is not modified by another
                # device at the same time, preventing race conditions.
                Device.locks[location].acquire()

                script_data = []
                # Aggregate data from all neighbors for the given location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Include the device's own data.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Execute the script on the aggregated data.
                    result = script.run(script_data)

                    # Disseminate the result back to all neighbors and self.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the lock for the location.
                Device.locks[location].release()

            # Block Logic: All 8 threads of this device synchronize here to ensure
            # all scripts for this device are complete for the timepoint.
            self.device.bar_thr.wait()
            
            # The main thread resets events for the next timepoint.
            if self.first == 1:
                self.device.event_neighbours.clear()
                self.device.timepoint_done.clear()
            
            # A second local barrier to prevent race conditions during state reset.
            self.device.bar_thr.wait()
            
            # Block Logic (Main Thread Only): The main thread now waits on the global
            # barrier. This synchronizes ALL devices, ensuring that none can
            # start the next timepoint until all have finished the current one.
            if self.first == 1:
                Device.bariera_devices.wait()
