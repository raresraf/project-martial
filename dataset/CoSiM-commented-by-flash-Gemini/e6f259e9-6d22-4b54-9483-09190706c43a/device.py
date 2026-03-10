"""
This module implements a simulation framework for distributed devices,
focusing on concurrent execution of scripts and synchronized data processing.
It utilizes a `Device` class to represent each simulated entity,
a `DeviceThread` for main control flow, and a `ReusableBarrier` for global
synchronization across devices. Location-specific locks are managed through a
shared dictionary (`semafor`).
"""


from threading import Event, Thread, Condition, Semaphore, Lock, RLock # RLock and Semaphore are imported but not used directly in this file's classes.

class ReusableBarrier():
    """
    A reusable barrier synchronization primitive implemented using a `Condition` variable.

    This barrier allows a fixed number of threads (`num_threads`) to wait for
    each other to reach a common point before any can proceed. It is designed
    to be reusable across multiple synchronization points within a larger simulation loop.
    """
    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must arrive
                               at the barrier before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Counter for threads currently waiting at the barrier.
        self.cond = Condition() # A condition variable used for synchronization (waiting and notifying).
                                                 

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        `num_threads` threads have also called `wait()`.

        The last thread to arrive releases all waiting threads and resets the barrier.
        """
        self.cond.acquire() # Acquires the lock associated with the condition variable.
        self.count_threads -= 1 # Decrements the counter of threads yet to arrive.
        if self.count_threads == 0: # Conditional Logic: If this is the last thread to arrive.
            self.cond.notify_all() # Notifies all threads waiting on this condition.
            self.count_threads = self.num_threads # Resets the counter for barrier reusability.
        else:
            self.cond.wait(); # Blocks (waits) until notified by the last arriving thread.
        self.cond.release();                     


class Device(object):
    """
    Represents a simulated device within a distributed system.

    Each device manages its sensor data, interacts with a central supervisor,
    and dispatches scripts for execution. It launches a dedicated `DeviceThread`
    to handle its control flow and participates in global synchronization.
    Location-specific locks are managed through a shared `semafor` dictionary.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor readings
                                (e.g., {location_id: data_value}).
            supervisor (object): An object representing the central supervisor,
                                 used for coordination (e.g., getting neighbors).
        """
        self.device_id = device_id # Unique identifier for this device.
        self.sensor_data = sensor_data # Dictionary storing sensor data for this device.
        self.supervisor = supervisor # Reference to the global supervisor.
        self.script_received = Event() # Event to signal that scripts have been assigned.
        self.scripts = [] # List to store scripts assigned to this device.
        self.devices = None # Reference to the list of all devices in the simulation.
        self.timepoint_done = None # Reference to the global ReusableBarrier for timepoint synchronization.
        self.semafor = {} # Shared dictionary of Locks, one per data location, for distributed access control.
        # The main thread for the device, responsible for supervisor interaction and script execution.
        self.thread = DeviceThread(self)
        # Note: The thread is started in `setup_devices` to ensure global synchronization primitives are ready.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of global synchronization primitives (a global barrier and shared locks).

        This method identifies device 0 (root device) which initializes and distributes
        the `ReusableBarrier` and the `semafor` (location-specific locks) dictionary
        to all other devices. After setup, the device's `DeviceThread` is started.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices # Stores reference to all devices.
        # Block Logic: Only device with ID 0 acts as the coordinator for global setup.
        if self.device_id == 0:
            # Functional Utility: Creates a global ReusableBarrier for all devices.
            self.timepoint_done = ReusableBarrier(len(self.devices))
            
            # Block Logic: Initializes `self.semafor` (shared location locks) by finding all unique locations
            # across all devices and creating a Lock for each.
            for device in self.devices:
                for location, data in device.sensor_data.iteritems(): # Note: `iteritems()` is Python 2.x specific.
                    if location not in self.semafor:
                        self.semafor.update({location:Lock()}) # Creates a Lock for each unique location.
            # Also includes its own sensor data locations in the `semafor` initialization.
            for location, data in self.sensor_data.iteritems():
                if location not in self.semafor:
                    self.semafor.update({location:Lock()})
        else:
            # Block Logic: Non-root devices retrieve the globally initialized `timepoint_done` barrier
            # and `semafor` locks from device 0.
            for device in self.devices:
                if device.device_id == 0:
                    self.timepoint_done = device.timepoint_done
                    self.semafor = device.semafor
        
        self.thread.start() # Starts the DeviceThread after synchronization primitives are set up.

    def assign_script(self, script, location):
        """
        Assigns a script for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location)) # Adds the script and its location to a list.
        else:
            # If script is None, it signifies the end of script assignments for the current timepoint.
            # Functional Utility: Signals that all scripts for the current timepoint have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Note: This method does NOT acquire any locks itself.
        Locking for data access is expected to be handled by the caller (`DeviceThread`).

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if not found.
        """
        value = None
        if location in self.sensor_data:
            value = self.sensor_data[location]
        return value

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Note: This method does NOT acquire any locks itself.
        Locking for data modification is expected to be handled by the caller (`DeviceThread`).

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Performs a graceful shutdown of the main device thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for coordinating with the supervisor
    and executing assigned scripts.

    In this implementation, the `DeviceThread` itself executes the scripts sequentially
    for its device, rather than dispatching them to a separate pool of worker threads.
    It handles fetching neighbor information, waiting for script assignments,
    executing scripts with appropriate locking, and participating in global synchronization.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The `Device` instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for the DeviceThread.

        Architectural Intent: Continuously fetches neighbor information from the supervisor,
        waits for scripts to be assigned, executes those scripts one by one (acquiring
        and releasing location-specific locks), updates data, and participates in a
        global barrier for timepoint synchronization.
        """
        while True:
            # Block Logic: Fetches the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Functional Utility: Participates in the global barrier. This likely marks the end
            # of the current timepoint's script execution phase for all devices.
            self.device.timepoint_done.wait()
            # Termination Condition: If no neighbors are returned (None), it signifies
            # that the simulation is ending for this device, and the loop breaks.
            if neighbours is None:
                break

            # Block Logic: Waits for the `script_received` event to be set, indicating that
            # all scripts for the current timepoint have been assigned to the device.
            self.device.script_received.wait()
            
            # Block Logic: Iterates through each assigned script and executes it.
            # In this design, the `DeviceThread` itself executes the scripts sequentially.
            for (script, location) in self.device.scripts:
                
                # Functional Utility: Acquires the location-specific lock from the shared `semafor` dictionary.
                # This ensures exclusive access to the data at `location` across all devices during script processing.
                self.device.semafor[location].acquire()
                script_data = [] # List to store all data relevant to the script.
                
                # Block Logic: Collects sensor data from neighboring devices.
                for device in neighbours:
                    data = device.get_data(location) # `get_data` does not acquire locks internally.
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collects sensor data from its own device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If any data was collected, executes the script and updates data.
                if script_data != []:
                    # Functional Utility: Executes the assigned script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Updates data on neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result) # `set_data` does not acquire locks internally.
                    
                    # Block Logic: Updates data on its own device.
                    self.device.set_data(location, result)
                
                self.device.semafor[location].release() # Releases the location-specific lock.
            
            # Functional Utility: Participates in the global barrier. This likely marks the end
            # of the current timepoint's script execution phase for all devices.
            self.device.timepoint_done.wait()
            
            self.device.script_received.clear() # Clears the event to prepare for the next timepoint.
