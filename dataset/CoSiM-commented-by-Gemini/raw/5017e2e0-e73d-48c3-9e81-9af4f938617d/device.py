from threading import *

# Defines the number of worker threads each device will spawn.
MAX_THREADS = 8

class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This barrier implementation uses a two-phase system with two semaphores
    to allow threads to wait for each other multiple times. All threads
    block on `wait()` until the last thread arrives, at which point all
    are released.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that will synchronize on this barrier.
        """
        self.num_threads = num_threads
        # Counters for each phase, stored in a list to be mutable.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Semaphores to block/release threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier. Consists of two phases
        to ensure reusability.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier synchronization.

        Args:
            count_threads (list): The mutable counter for the current phase.
            threads_sem (Semaphore): The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread has arrived, release all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # All threads wait here until the semaphore is released.
        threads_sem.acquire()

# Global dictionary to store ReusableBarrier instances, keyed by the number of devices.
# This allows groups of devices of the same size to share a barrier.
dictionary = dict()

class Device(object):
    """
    Represents a single computational device in a distributed system.

    Each device runs its own thread, processes scripts, interacts with sensor
    data, and synchronizes with other devices via a supervisor and a barrier.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data.
            supervisor (object): The supervisor object that manages this device.
        """
        self.lock = Lock()
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script is ready to be processed.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that a timepoint (computation step) is complete.
        self.timepoint_done = Event()
        self.locks = dict()
        # The main thread that drives the device's lifecycle.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for a group of devices.

        Args:
            devices (list): A list of all devices in the synchronization group.
        """
        # Note: 'has_key' is deprecated (Python 2). Use 'in' for Python 3.
        if not dictionary.has_key(len(devices)):
            # If no barrier exists for this group size, create one.
            dictionary[len(devices)] = ReusableBarrier(len(devices))
        self.barrier = dictionary[len(devices)]

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script (object): The script object to execute.
            location (any): The data location the script is associated with.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of a timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device instance.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def build_lists(self, lst, number):
        """
        Distributes a list of items into a specified number of sublists.

        This is used to partition scripts among worker threads.

        Args:
            lst (list): The list to partition.
            number (int): The number of sublists to create.

        Returns:
            list: A list of sublists.
        """
        result_list = []
        if not lst:
            return result_list
        size = int(len(lst) / number)
        for i in range(number):
            result_list.append(lst[size * i: size * (i + 1)])
        # Distribute the remainder elements.
        for i in range(len(lst) % number):
            result_list[i].append(lst[i + (size * number)])
        return result_list

    class Instance(Thread):
        """
        A worker thread that executes a subset of a device's scripts.
        """
        def __init__(self, device, big_list, neighbours):
            """
            Initializes the worker instance.

            Args:
                device (Device): The parent device.
                big_list (list): The subset of scripts for this worker to run.
                neighbours (list): A list of neighboring Device objects.
            """
            Thread.__init__(self, name="Instance")
            self.neighbours = neighbours
            self.big_list = big_list
            self.device = device

        def run(self):
            """
            Main execution logic for the worker thread.
            
            It iterates through its assigned scripts, gathers data from its
            device and its neighbors, runs the script, and updates the data.
            """
            for (script, location) in self.big_list:
                script_data = []
                # Gather data from neighbors.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                if script_data:
                    # Run the script and update data on all relevant devices.
                    result = script.run(script_data)
                    self.all_devices_set_data(location, result)

        def my_max(self, a, b):
            """Custom max function."""
            return a if a >= b else b

        def all_devices_set_data(self, location, result):
            """
            Updates the data at a given location for this device and all its
            neighbors with a new result, ensuring the new value is the maximum.
            
            Note: The lock is on the parent device, which does not guarantee
            thread safety for operations on neighboring devices.
            """
            # Update data on neighboring devices.
            for device in self.neighbours:
                with self.device.lock:
                    device.set_data(location, self.my_max(result, device.get_data(location)))
            # Update data on the local device.
            with self.device.lock:
                result_data = self.my_max(result, self.device.get_data(location))
                self.device.set_data(location, result_data)

    def run(self):
        """
        The main loop of the device thread. It coordinates script execution
        and synchronization for each timepoint.
        """
        while True:
            # Get the list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown by returning None.
                break

            # Wait for the supervisor to signal the end of a timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Partition scripts among worker threads.
            list_of_scripts = self.build_lists(self.device.scripts, MAX_THREADS)
            threads_scripts = []
            if list_of_scripts:
                for i in range(MAX_THREADS):
                    if i < len(list_of_scripts) and list_of_scripts[i]:
                        threads_scripts.append(self.Instance(self.device, list_of_scripts[i], neighbours))

            # Start and join all worker threads for this timepoint.
            for script_thread in threads_scripts:
                script_thread.start()
            for script_thread in threads_scripts:
                script_thread.join()

            # Synchronize with all other devices at the barrier.
            self.device.barrier.wait()
