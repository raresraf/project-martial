

"""
This module provides core components for a distributed device simulation framework.
It includes a reusable barrier for thread synchronization, a Device class for managing
simulated devices and their sensor data, and thread classes for executing scripts
and handling inter-device communication within the simulated environment.
"""

from threading import Event, Thread, Lock, Semaphore, RLock


class ReusableBarrier(object):
    """
    A reusable barrier synchronization primitive that allows a fixed number of threads
    to wait for each other to reach a common execution point. It uses two phases
    to ensure reusability, preventing a "lost wakeup" problem if a thread
    arrives too early in a subsequent cycle.
    """

    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier with a specified number of participating threads.

        Args:
            num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Two counters for alternating phases of the barrier.
        # This prevents threads from a new cycle from proceeding before all threads
        # from the previous cycle have passed the barrier.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()  # Mutex to protect access to the thread counters
        # Semaphores for blocking and releasing threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks the calling thread until all other threads have also called wait().
        Once all threads have arrived, they are all released to proceed.
        This method alternates between two internal phases to ensure proper synchronization.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current count of threads
                                  remaining in this phase.
            threads_sem (Semaphore): The semaphore associated with this phase
                                     to control thread release.
        """
        with self.count_lock:
            # Decrement the count of threads yet to reach the barrier in this phase.
            count_threads[0] -= 1
            # Check if this is the last thread to arrive at the barrier.
            if count_threads[0] == 0:
                # If all threads have arrived, release them.
                i = 0
                while i < self.num_threads:
                    # Release threads one by one.
                    threads_sem.release()
                    i += 1
                # Reset the counter for the next cycle.
                count_threads[0] = self.num_threads
        # Acquire the semaphore to block the thread until all threads in this phase
        # have been released by the last arriving thread.
        threads_sem.acquire()


class Device(object):
    """
    Represents a simulated device in a distributed system. Each device manages its
    own sensor data, interacts with a central supervisor, and can execute assigned scripts.
    It also handles synchronization with other devices for coordinated actions.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for
                                various locations, e.g., {'location_id': data_value}.
            supervisor (Supervisor): A reference to the central supervisor managing
                                     the distributed system.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []  # List to store (script, location) tuples to be executed
        
        self.devices = []  # To hold references to other devices in the system
        self.script_received = Event()  # Event to signal when a script has been assigned
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's tasks
        # Each device runs in its own thread to simulate concurrent operation.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up communication and synchronization mechanisms among a group of devices.

        Args:
            devices (list): A list of Device objects that are part of the same group.
        """
        # Populate the internal list of devices.
        for device in devices:
            self.devices.append(device)
        # For synchronization across all devices in a group, a ReusableBarrier is used.
        # It is initialized once on the first device and shared across the group.
        self.devices[0].barrier = ReusableBarrier(len(self.devices))
        # A dictionary to hold RLock objects for each location, ensuring atomic
        # updates to sensor data for a specific location across multiple threads/devices.
        self.devices[0].locations_lock = {}

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location on this device.
        If script is None, it signals that no more scripts are coming for the current timepoint.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (str): The identifier for the location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device.

        Args:
            location (str): The identifier of the location for which to retrieve data.

        Returns:
            Any: The sensor data associated with the location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None # Explicitly return None if location not in sensor_data

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location on this device.

        Args:
            location (str): The identifier of the location to update.
            data (Any): The new sensor data value for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device's operational thread.
        This method waits for the device's thread to complete its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    A dedicated thread for a Device instance. This thread manages the device's
    operational lifecycle, including fetching neighbor devices, executing assigned
    scripts concurrently, and synchronizing with other devices at specific timepoints.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Number of concurrent threads to execute scripts for a given timepoint.
        self.num_threads = 8 

    def run(self):
        """
        The main execution loop for the device thread.
        It continuously fetches scripts, executes them, and synchronizes with other devices.
        """
        while True:
            # List to hold worker threads for script execution.
            threads = []
            
            # Pre-condition: Fetches information about neighboring devices from the supervisor.
            # Invariant: `neighbours` will be None if the simulation is ending.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Break the loop if the supervisor signals the end of the simulation.
                break
            
            # Block until scripts for the current timepoint are received and assigned.
            self.device.script_received.wait()

            # Block Logic: Prepare worker threads for each assigned script.
            for (script, location) in self.device.scripts:
                thread = MyThread(self, script, location, neighbours)
                threads.append(thread)

            # Calculate the number of full rounds of concurrent script execution
            # and any remaining scripts (leftovers).
            rounds = len(self.device.scripts) / self.num_threads
            leftovers = len(self.device.scripts) % self.num_threads
            
            # Block Logic: Execute scripts in batches using multiple threads.
            # Invariant: Each loop iteration processes `self.num_threads` scripts concurrently.
            while rounds > 0:
                # Start a batch of worker threads.
                for j in xrange(self.num_threads):
                    threads[j].start()
                # Wait for all threads in the current batch to complete.
                for j in xrange(self.num_threads):
                    threads[j].join()
                # Remove processed threads from the list.
                for j in xrange(self.num_threads):
                    threads.pop(0)
                rounds -= 1
            
            # Block Logic: Process any remaining scripts that didn't form a full batch.
            for j in xrange(leftovers):
                threads[j].start()
            for j in xrange(leftovers):
                threads[j].join()
            for j in xrange(leftovers):
                threads.pop(0)

            # Post-condition: All scripts for the current timepoint have been executed.
            # Clear the threads list to prepare for the next timepoint.
            del threads[:]
            
            # Synchronize with other devices using the reusable barrier.
            # All devices must reach this point before any can proceed to the next timepoint.
            self.device.devices[0].barrier.wait()
            
            # Reset the event to wait for new scripts in the next simulation timepoint.
            self.device.script_received.clear()


class MyThread(Thread):
    """
    A worker thread responsible for executing a specific script associated with a
    particular location on a device. It gathers relevant sensor data from
    neighboring devices and its own device, runs the script with this data,
    and then propagates the results back to the involved devices.
    """

    def __init__(self, device_thread, script, location, neighbours):
        """
        Initializes the MyThread worker.

        Args:
            device_thread (DeviceThread): The parent DeviceThread managing this worker.
            script (Script): The script to be executed.
            location (str): The specific location the script pertains to.
            neighbours (list): A list of neighboring Device objects to gather data from.
        """
        Thread.__init__(self)
        self.device_thread = device_thread
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the assigned script.
        This involves acquiring a lock for the specific location, gathering data
        from the device and its neighbors, running the script, and then
        updating the data on all involved devices.
        """
        # Pre-condition: Ensure a Reentrant Lock (RLock) exists for the current location.
        # This lock guarantees exclusive access to the sensor data for this location
        # across all devices during the script execution and data update phase,
        # preventing race conditions.
        if self.location not in\
                self.device_thread.device.devices[0].locations_lock:
            self.device_thread.device.devices[0].locations_lock[self.location]\
                = RLock()
        # Acquire the RLock for the specific location.
        with self.device_thread.device.devices[0].locations_lock[self.location]:
            script_data = []  # Accumulator for all data relevant to the script.
            
            # Block Logic: Gather sensor data from all neighboring devices for the current location.
            # Invariant: Each 'device' in 'self.neighbours' is a valid Device object.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Gather sensor data from the current device itself for the current location.
            data = self.device_thread.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            # Pre-condition: Check if any data was collected to run the script.
            if script_data != []:
                # Execute the script with the collected data.
                result = self.script.run(script_data)
                
                # Block Logic: Propagate the result of the script execution back to all neighboring devices.
                # Invariant: Each 'device' in 'self.neighbours' will have its data updated.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                # Block Logic: Update the current device's own sensor data with the result.
                self.device_thread.device.set_data(self.location, result)
