"""
@file device.py
@brief A simulation of distributed devices that execute scripts by spawning new threads for each task.
@details This module provides another implementation of a distributed device network.
In this version, each device's main thread dynamically creates, starts, and joins
a new thread for each assigned script. Synchronization is handled by a shared barrier
and a shared list of locks for data locations.
"""

from threading import Event, Thread, Condition, RLock


class ReusableBarrier(object):
    """
    @brief A barrier for thread synchronization using a Condition variable.
    @details This implementation uses a counter and a `threading.Condition` to block threads
    until a specified number of them have called the `wait()` method.

    @note The name `ReusableBarrier` is potentially misleading. This implementation may not be safely
    reusable in all concurrent scenarios. If a thread loops and calls `wait()` again before all
    other threads have woken up from the previous `wait()`, a race condition can occur. A more robust
    reusable barrier typically uses a two-phase or "turnstile" approach.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier for a given number of threads.
        @param num_threads The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        @brief Causes a thread to wait at the barrier.
        @details The last thread to arrive notifies all waiting threads and resets the counter.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # Pre-condition: The last thread arrives at the barrier.
        # Invariant: All threads are waiting on the condition variable.
        if self.count_threads == 0:
            # Wake up all waiting threads.
            self.cond.notify_all()
            # Reset the counter for the next use.
            self.count_threads = self.num_threads
        else:
            # Wait to be notified by the last thread.
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    @brief Represents a single device in the simulation.
    @details This device uses a single main thread (`DeviceThread`) which in turn spawns
    temporary `ScriptThread`s to perform work.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that all scripts for a timepoint have been received.
        self.script_received = Event()
        self.can_be_write = RLock()
        self.scripts = []
        self.devices = []

        # A list of (location, lock) tuples, shared and populated by all devices.
        self.locationl = []
        # A list of unique locations, also shared.
        self.locations = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # The shared barrier object.
        self.barrier = None


    def __str__(self):
        """@brief Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (barrier and lock lists) for all devices.
        """
        self.devices = devices
        
        # Block Logic: The device with ID 0 creates the shared barrier.
        # Other devices get a reference to this barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        else:
            self.barrier = self.devices[0].barrier

        # Block Logic: The lock infrastructure is shared from device 0.
        # This is a centralized approach to lock management.
        if self.device_id != 0:
            self.locationl = self.devices[0].locationl
            self.locations = self.devices[0].locations

        # Populate the shared list of locks based on this device's sensor data.
        for loc in self.sensor_data:
            self.locationl.append((loc, RLock()))
            self.locations.append(loc)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device or signals the end of a timepoint.
        """
        # A new location is dynamically added to the shared lock list if not present.
        if location not in self.locations:
            for dev in self.devices:
                dev.locationl.append((location, RLock()))
                dev.locations.append(location)
        
        # A None script indicates all scripts for the timepoint have been assigned.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """@brief Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in\
                                             self.sensor_data else None

    def set_data(self, location, data):
        """@brief Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """@brief Shuts down the device by joining its main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief The main control thread for a Device instance.
    @details This thread spawns new worker threads for each script in batches.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """@brief The main execution loop for the device's control thread."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait until all scripts for the timepoint are received.
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            threads = []
            i = 1
            
            # Block Logic: Spawns a new thread for each script.
            # This is done in batches of 8, which is an inefficient approach due to the high overhead
            # of thread creation and destruction. A thread pool would be more performant.
            for (script, location) in self.device.scripts:
                threads.append(ScriptThread(self.device, neighbours,
                                            location, script))
                # Process threads in batches of 8.
                if i % 8 == 0:
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    threads = []
                i = i+1
            
            # Start and join any remaining threads in the last, possibly smaller, batch.
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            # Synchronize with all other devices.
            self.device.barrier.wait()
            # Wait for the timepoint done signal before proceeding to the next cycle.
            self.device.timepoint_done.wait()

class ScriptThread(Thread):
    """
    @brief A short-lived worker thread created to execute a single script.
    """

    def __init__(self, device, neighbours, location, script):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.location = location
        self.script = script

    def run(self):
        """@brief Executes a single script on data from the device and its neighbors."""
        
        # Block Logic: Find and acquire the lock for the specified data location.
        # This linear search for the lock can be inefficient if the number of locations is large.
        for (loc, lock) in self.device.locationl:
            if loc == self.location:
                lock.acquire()
        
        # Gather data from neighbors and the local device.
        script_data = []
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)
        
        # Execute the script and broadcast the result.
        if script_data != []:
            result = self.script.run(script_data)
            
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        
        # Block Logic: Find and release the lock for the location.
        for (loc, lock) in self.device.locationl:
            if loc == self.location:
                lock.release()
