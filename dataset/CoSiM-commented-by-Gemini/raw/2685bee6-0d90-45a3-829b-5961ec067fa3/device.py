"""
Models a distributed system of devices executing computational scripts concurrently.

This module defines the core components for a device simulation, including the
Device itself, threading and pooling helpers, and synchronization primitives.
It appears to simulate a scenario where devices, each with their own sensor data,
execute scripts that can read/write data from/to their neighbors. The system
operates in synchronized timepoints, coordinated by a barrier.
"""
from threading import Event, Thread, Lock, Condition
from Queue import Queue


class ReusableBarrierCond(object):
    """
    A reusable synchronization barrier implemented using a Condition variable.

    This barrier blocks a specified number of threads until all of them have
    reached the barrier. Once all threads are waiting, they are all released,
    and the barrier resets for the next synchronization point.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Causes a thread to wait at the barrier.

        The thread will block until `num_threads` have called this method.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # When the last thread arrives, notify all waiting threads to proceed.
        if self.count_threads == 0:
            self.cond.notify_all()
            # Reset the barrier for the next use.
            self.count_threads = self.num_threads
        else:
            # Wait until notified by the last thread.
            self.cond.wait()
        
        self.cond.release()



class Device(object):
    """
    Represents a single device in the distributed system simulation.

    Each device runs in its own thread, manages local sensor data, and executes
    scripts that may involve communication with neighboring devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data,
                                keyed by location.
            supervisor (object): A supervisor object that manages the overall simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Event to signal the arrival of a new script.
        self.script_received = Event()
        self.scripts = []
        # Event to signal the end of a timepoint's script assignments.
        self.timepoint_done = Event()
        self.barrier = None
        # Per-location locks to ensure thread-safe access to sensor data.
        self.data_locks = {location: Lock() for location in sensor_data}
        # The main execution thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the barrier for synchronization among all devices.
        
        This method is intended to be called on a single device (e.g., device 0)
        to create and distribute a shared barrier instance to all other devices.
        
        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Only device 0 should create the shared barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.
        
        If `script` is None, it signals the end of script assignments for the
        current timepoint.
        
        Args:
            script (object): The script object to execute.
            location (str): The data location the script is associated with.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that no more scripts are coming in this time step.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safely retrieves sensor data from a given location.

        Args:
            location (str): The key for the desired sensor data.

        Returns:
            The data associated with the location, or None if the location is not found.
        """
        if location in self.sensor_data:
            self.data_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Thread-safely updates sensor data at a given location and releases the lock.

        Args:
            location (str): The key for the sensor data to update.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_locks[location].release()

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()

class ThreadPool(object):
    """
    A simple thread pool for executing script tasks concurrently.
    """
    def __init__(self, helper, no_threads):
        """
        Initializes the thread pool.

        Args:
            helper (Helper): The Helper object that owns this pool.
            no_threads (int): The number of worker threads in the pool.
        """
        self.helper = helper
        self.queue = Queue(no_threads)
        # Pre-populate the queue with initial scripts.
        for i in range(len(helper.scripts)):
            self.queue.put((helper.scripts[i][0], helper.scripts[i][1], helper.neighbours))
        
        self.threads = [Thread(target=self.run) for _ in range(no_threads)]
        for thread in self.threads:
            thread.start()

    def run(self):
        """The main execution loop for worker threads."""
        while True:
            script, location, neighbours = self.queue.get()

            # A None script is used as a sentinel to terminate the thread.
            if not neighbours or not script:
                self.queue.task_done()
                break

            # --- Data Gathering Phase ---
            # Collect data from all neighboring devices for the script's execution context.
            script_data = []
            for device in neighbours:
                if device.device_id != self.helper.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Include the local device's data.
            data = self.helper.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # --- Script Execution and Data Update Phase ---
            if script_data:
                result = script.run(script_data)
                # Propagate the result back to all neighbors.
                for device in neighbours:
                    if device.device_id != self.helper.device.device_id:
                        device.set_data(location, result)
                self.helper.device.set_data(location, result)

            self.queue.task_done()

    def join(self):
        """Blocks until all tasks in the queue have been processed."""
        self.queue.join()

    def close(self):
        """Shuts down the thread pool gracefully."""
        self.join()

        # Send sentinel values to terminate each worker thread.
        for _ in self.threads:
            self.queue.put((None, None, None))

        for thread in self.threads:
            thread.join()


class Helper(object):
    """
    A helper class that manages script execution and the thread pool for a Device.
    """
 
    def __init__(self, device):
        """
        Initializes the helper for a given device.
        
        Args:
            device (Device): The device this helper serves.
        """
        self.device = device
        self.neighbours = None
        self.scripts = None
        self.pool = None

    def set_neighbours_and_scripts(self, neighbours, scripts):
        """
        Configures the helper with the current set of neighbors and scripts.
        Initializes or adds work to the thread pool.
        """
        self.scripts = scripts
        self.neighbours = neighbours
        if not self.pool:
            self.pool = ThreadPool(self, 8)
        else:
            # If the pool already exists, add new scripts to its queue.
            for i in range(len(scripts)):
                self.pool.queue.put((scripts[i][0], scripts[i][1], neighbours))

    def close_pool(self):
        """Closes the associated thread pool if it exists."""
        if self.pool:
            self.pool.close()


class DeviceThread(Thread):
    """
    The main control thread for a single Device.
    
    This thread orchestrates the device's lifecycle, including fetching neighbor
    information, processing scripts, and synchronizing at timepoints.
    """

    def __init__(self, device):
        """
        Initializes the device thread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.helper = None

    def run(self):
        """The main loop for the device thread."""
        while True:
            # Determine the device's neighbors for the current time step.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbors signals a system shutdown.
                break
            
            self.helper = Helper(self.device)
            
            # Inner loop for processing scripts within a single timepoint.
            while True:
                # Wait for either new scripts or a signal that the timepoint is done.
                if (self.device.script_received.is_set() or
                self.device.timepoint_done.is_set()):
                    
                    if self.device.script_received.is_set():
                        # A new script has been assigned; process it.
                        self.device.script_received.clear()
                        self.helper.set_neighbours_and_scripts(neighbours,
							self.device.scripts)
                    else:
                        # All scripts for this timepoint have been assigned.
                        self.device.timepoint_done.clear()
                        # This seems to immediately re-set script_received, which
                        # may be part of the protocol to prepare for the next timepoint.
                        self.device.script_received.set()
                        break
            
            # All work for the timepoint is done; clean up the thread pool.
            self.helper.close_pool()
            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()
