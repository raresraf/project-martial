
"""
This module implements a Device class representing a simulated entity
that processes scripts and interacts with other devices. It also includes
a ReusableBarrier for synchronizing multiple threads within this simulation.
The design supports distributed processing of tasks (scripts) among devices
and ensures proper synchronization at various timepoints.
"""

from threading import *


MAX_THREADS = 8

class ReusableBarrier(object):
    """
    A reusable barrier synchronization primitive for a fixed number of threads.
    It ensures that a group of threads pause at a certain point until all
    threads have reached that point, and then allows them to proceed.
    This implementation uses two phases to allow for reusability.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier with the specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # count_threads1 and count_threads2 are lists used to hold a single integer
        # representing the count of threads yet to reach the barrier in phase 1 and 2, respectively.
        # Using a list allows the integer to be modified within a Lock's context.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # count_lock protects access to count_threads and ensures atomic decrement operations.
        self.count_lock = Lock()
        # threads_sem1 and threads_sem2 are semaphores used to block and release threads
        # for phase 1 and 2 of the barrier, respectively. Initialized to 0 so threads block immediately.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` threads have called this method.
        This method executes two phases of the barrier to ensure reusability.
        """
        # First phase of the barrier.
        self.phase(self.count_threads1, self.threads_sem1)
        # Second phase of the barrier.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Internal method implementing a single phase of the barrier.

        Args:
            count_threads (list): A list containing the current count of threads
                                  yet to reach this phase of the barrier.
            threads_sem (Semaphore): The semaphore associated with this phase,
                                     used to block and release threads.
        """
        with self.count_lock:
            # Decrement the count of threads yet to reach the barrier.
            count_threads[0] -= 1
            # If this thread is the last one to reach the barrier (count becomes 0).
            if count_threads[0] == 0:
                # Release all waiting threads by releasing the semaphore `num_threads` times.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the count for the next use of this phase.
                count_threads[0] = self.num_threads

        # Acquire the semaphore, blocking until released by the last thread.
        threads_sem.acquire()

dictionary = dict() # Global dictionary to store ReusableBarrier instances, keyed by the number of devices.

class Device(object):
    """
    Represents a simulated device capable of holding sensor data and executing scripts.
    Each device runs in its own thread and interacts with a supervisor and other devices.
    It uses a barrier for synchronization with other devices in a group.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding sensor data, where keys are locations
                                and values are data points.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.lock = Lock()  # Lock to protect concurrent access to device's data.
        self.barrier = None # Will hold a ReusableBarrier instance for group synchronization.
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been received by the device.
        self.script_received = Event()
        # List to store assigned scripts, each as a tuple (script_object, location).
        self.scripts = []

        # Event to signal that the device has completed processing scripts for a timepoint.
        self.timepoint_done = Event()
        # Dictionary to hold locks, potentially for fine-grained access control to specific data locations.
        self.locks = dict()
        # Create and start a dedicated thread for this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the barrier for group synchronization based on the number of devices.
        It uses a global dictionary to reuse barrier instances for groups of the same size.

        Args:
            devices (list): A list of Device objects that will share the same barrier.
        """
        # If a barrier for this group size doesn't exist, create one and store it globally.
        if not dictionary.has_key(len(devices)):
            dictionary[len(devices)] = ReusableBarrier(len(devices))
        self.barrier = dictionary[len(devices)] # Assign the barrier to this device.

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution.

        Args:
            script (Script): The script object to be executed.
            location (str): The data location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script to the device's queue.
            self.script_received.set() # Signal that a script has been received.
        else:
            # If script is None, it signals the end of scripts for the current timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (str): The location for which to set data.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data: # Only update if the location already exists.
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device's dedicated thread.
        """
        self.thread.join() # Wait for the device's thread to complete its execution.


class DeviceThread(Thread):
    """
    A dedicated thread for a Device instance, responsible for managing script execution
    and interactions with other devices and the supervisor.
    """
    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def build_lists(self, lst, number):
        """
        Distributes a list of scripts (`lst`) into a specified number of sub-lists (`number`).
        This is used to prepare scripts for parallel processing by `Instance` threads.

        Args:
            lst (list): The master list of scripts (tuples of script_object, location).
            number (int): The number of sub-lists (threads) to distribute the scripts among.

        Returns:
            list: A list of lists, where each inner list contains a subset of the original scripts.
        """
        result_list = []
        # Calculate base size for each sub-list.
        size = int(len(lst) / number)
        # Initialize sub-lists with the base size.
        for i in range(number):
            row = size * i
            column = size * (i + 1)
            result_list.append(lst[row: column])
        # Distribute remaining scripts (if any) among the sub-lists.
        for i in range(len(lst) - size * number):
            row = i % number
            column = i + (size * number)
            result_list[row].append(lst[column])
        return result_list

    class Instance(Thread):
        """
        A nested thread class within DeviceThread, responsible for executing a subset of scripts
        assigned to a device and interacting with neighboring devices.
        """
        def __init__(self, device, big_list, neighbours):
            """
            Initializes an Instance thread.

            Args:
                device (Device): The parent Device instance.
                big_list (list): A subset of scripts to be processed by this instance.
                neighbours (list): A list of neighboring Device objects.
            """
            Thread.__init__(self, name="Instance")
            self.neighbours = neighbours
            self.big_list = big_list
            self.device = device

        def run(self):
            """
            The main execution logic for the Instance thread.
            It processes its assigned scripts, retrieves data from neighbors and itself,
            executes the script, and updates data across all relevant devices.
            """
            script_data = [] # List to accumulate data required by the script.
            for (script, location) in self.big_list:
                # Gather data from neighboring devices at the specified location.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Gather data from the current device at the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # If any data was collected, execute the script.
                if script_data != []:
                    result = script.run(script_data) # Execute the script with collected data.
                    self.all_devices_set_data(location, result) # Update data based on script result.

        def my_max(self, a, b):
            """
            A utility function to return the maximum of two values.

            Args:
                a (any): First value.
                b (any): Second value.

            Returns:
                any: The maximum of 'a' and 'b'.
            """
            if a >= b:
                return a
            return b

        def all_devices_set_data(self, location, result):
            """
            Updates the data at a specific location across all neighboring devices and
            the current device, ensuring that the updated value is the maximum
            between the current value and the script's result.

            Args:
                location (str): The data location to update.
                result (any): The result from the script execution.
            """
            for device in self.neighbours:
                with self.device.lock: # Acquire lock to protect concurrent access to device data.
                    # Update neighbor's data with the maximum value.
                    device.set_data(location, self.my_max(result, device.get_data(location)))
            with self.device.lock: # Acquire lock to protect concurrent access to current device data.
                result_data = self.my_max(result, self.device.get_data(location))
                self.device.set_data(location, result_data)


    def run(self):
        """
        The main execution loop for the DeviceThread.
        It continuously waits for timepoint completion signals, retrieves neighbors,
        distributes scripts among worker Instance threads, waits for their completion,
        and then synchronizes with other DeviceThreads via a barrier.
        """
        while True:
            threads_scripts = list()
            # Get a list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors are returned, it might be a signal to terminate.
            if neighbours is None:
                break
            # Wait for the current timepoint's scripts to be fully assigned and processed by the main thread.
            self.device.timepoint_done.wait()
            # Clear the event for the next timepoint.
            self.device.timepoint_done.clear()
            # Distribute the device's scripts into sub-lists for parallel processing.
            list_of_scripts = self.build_lists(self.device.scripts, MAX_THREADS)
            # Create and start Instance threads for each sub-list of scripts.
            for i in range(MAX_THREADS):
                if list_of_scripts != []: # Ensure the list is not empty before creating an Instance.
                    threads_scripts.append(self.Instance(self.device, list_of_scripts[i], neighbours))
            for script_thread in threads_scripts:
                script_thread.start()
            # Wait for all Instance threads to complete their script processing.
            for script_thread in threads_scripts:
                script_thread.join()
            # Synchronize with other DeviceThreads at the barrier before proceeding to the next timepoint.
            self.device.barrier.wait()