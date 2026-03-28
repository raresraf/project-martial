"""
Defines a device simulation model using a thread-per-script concurrency pattern.

This module contains a `Device` class whose main `DeviceThread` spawns a new,
short-lived `ScriptThread` for each script assigned to it within a timepoint.
Synchronization is managed by joining these threads and then waiting on a global
barrier.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    A reusable barrier implementation for a fixed number of threads.

    This barrier uses a two-phase protocol with semaphores to ensure that all
    threads wait at the barrier until the last one arrives, after which all are
    released and the barrier can be used again.
    """
    def __init__(self, num_threads):
        """Initializes the barrier for `num_threads`."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have called wait()."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0: # Last thread to arrive
                for _ in range(self.num_threads):
                    threads_sem.release() # Release all waiting threads
                count_threads[0] = self.num_threads # Reset for next use
        threads_sem.acquire() # All other threads wait here until released.


class Device(object):
    """
    Represents a device node that processes scripts by spawning threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()
        self.location_locks = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for all devices in the simulation.

        If this is the master device (ID 0), it creates a global barrier and a
        shared list of locks for all unique data locations, then distributes
        these resources to all other devices.
        """
        if 0 == self.device_id:
            # Create a barrier for all devices.
            self.barrier = ReusableBarrier(len(devices))

            # Aggregate all unique locations from all devices.
            locations = []
            for device in devices:
                for location in device.sensor_data:
                    if location not in locations:
                        locations.append(location)

            # Create a shared list of locks corresponding to each unique location.
            self.location_locks = []
            for _ in range(len(locations)):
                self.location_locks.append(Lock())

            # Distribute the shared resources to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be processed or signals the end of a timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates the sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class ScriptThread(Thread):
    """
    A short-lived thread created to execute a single data processing script.
    """
    def __init__(self, device, script, location, neighbours):
        """Initializes the thread with its specific script and context."""
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script. The entire operation is synchronized on the
        location-specific lock to ensure data consistency.
        """
        # Use a `with` statement to automatically acquire and release the lock for the location.
        with self.device.location_locks[self.location]:
            script_data = []
            # Gather data from neighbors and the parent device.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # If data was found, run the script and propagate the result.
            if script_data:
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    The main control thread for a device. It spawns and manages `ScriptThread`s.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation loop for the device.
        
        For each timepoint, it waits for scripts, spawns a thread for each one,
        waits for them to complete, and then synchronizes with other devices.
        """
        while True:
            # Get neighbors for the current timepoint. A None value is the shutdown signal.
            vecini = self.device.supervisor.get_neighbours()
            if vecini is None:
                break
            
            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            threads = []
            
            # For each assigned script, create and start a new worker thread.
            if len(vecini) != 0:
                for (script, locatie) in self.device.scripts:
                    thread = ScriptThread(self.device, script, locatie, vecini)
                    threads.append(thread)
                    thread.start()
                # Wait for all spawned threads for this timepoint to finish.
                for thread in threads:
                    thread.join()
            
            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Wait at the global barrier for all other devices to finish their timepoint.
            self.device.barrier.wait()
