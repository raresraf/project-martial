
"""
@5017e2e0-e73d-48c3-9e81-9af4f938617d/device.py
@brief Implements a simulated device for a distributed sensor network,
       including a reusable barrier for synchronization and thread management
       for script execution and data aggregation.
"""
from threading import *


MAX_THREADS = 8

class ReusableBarrier(object):
    """
    @brief A reusable barrier synchronization primitive for coordinating
           multiple threads.

    This barrier allows a fixed number of threads to wait until all have
    reached a certain point, then releases them all simultaneously. It can
    be reused multiple times after all threads have passed.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes a new instance of the ReusableBarrier.

        @param num_threads: The total number of threads that must reach
                            the barrier before it releases.
        """
        self.num_threads = num_threads
        # Two counters are used to manage the two phases of the barrier,
        # preventing lost wake-ups in a reusable scenario.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # Mutex to protect access to the thread counters.
        self.count_lock = Lock()
        # Semaphores to block and release threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` threads
               have reached this barrier across two phases.

        This method ensures that all participating threads synchronize twice,
        which is crucial for reusing the barrier safely.
        """
        # Phase 1: All threads synchronize at the first semaphore.
        self.phase(self.count_threads1, self.threads_sem1)
        # Phase 2: All threads synchronize at the second semaphore.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Manages a single synchronization phase of the barrier.

        Decrements a counter and, if it reaches zero, releases all waiting
        threads via a semaphore. Otherwise, the thread waits.

        @param count_threads: A list containing the current count of threads
                              waiting in this phase. (List is used for mutability)
        @param threads_sem: The semaphore used to block and release threads
                            in this specific phase.
        """
        with self.count_lock:
            # Atomically decrement the count of threads waiting.
            count_threads[0] -= 1
            # Check if this is the last thread to arrive at the barrier.
            if count_threads[0] == 0: 
                # Release all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads  
        # Acquire the semaphore; if not the last thread, it blocks here.
        threads_sem.acquire() 

# Global dictionary to store ReusableBarrier instances, keyed by the number
# of devices to ensure a single barrier per group size.
dictionary = dict()

class Device(object):
    """
    @brief Represents a simulated device in a sensor network.

    Each device has a unique ID, manages its sensor data, interacts with
    a supervisor, and executes scripts in a dedicated thread. It utilizes
    a barrier for synchronization with other devices.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's sensor readings.
        @param supervisor: A reference to the supervisor object for inter-device communication.
        """
        # Lock for protecting access to device's internal state, especially sensor data.
        self.lock = Lock()
        # Barrier for synchronizing multiple devices. Initialized in setup_devices.
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        # List to store scripts along with their target locations.
        self.scripts = []
        # Event to signal when a timepoint (script execution cycle) is complete.
        self.timepoint_done = Event()
        # Dictionary to hold various locks, though currently not explicitly used in provided code.
        self.locks = dict()
        # The dedicated thread for running device operations.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the synchronization barrier for the group of devices.

        It ensures that a single ReusableBarrier instance is created and shared
        among devices of the same group size.

        @param devices: A list of all devices in the network, used to determine
                        the size of the barrier.
        """
        # Block Logic: Ensures that a barrier is created only once for a given number of devices.
        # Invariant: The 'dictionary' stores unique barrier instances based on the number of threads.
        if not dictionary.has_key(len(devices)):
            dictionary[len(devices)] = ReusableBarrier(len(devices))
        self.barrier = dictionary[len(devices)]

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.

        If a script is provided, it's added to the device's script queue, and an
        event is set to notify the device's thread. If no script, it signals
        that the current timepoint is done.

        @param script: The script object to be executed, or None to signal timepoint completion.
        @param location: The data location relevant to the script.
        """
        # Block Logic: Manages the assignment of new scripts or signals timepoint completion.
        if script is not None:
            self.scripts.append((script, location))
            # Signal the DeviceThread that a new script has arrived.
            self.script_received.set()
        else:
            # Signal that no more scripts are coming for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The location for which to retrieve data.
        @return: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The location at which to set the data.
        @param data: The new data value to be set.
        """
        # Block Logic: Updates the sensor data if the location is valid.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its processing thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device to handle script execution
           and inter-device communication.

    This thread continuously fetches scripts, executes them, and synchronizes
    with other device threads using a barrier.
    """
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread with a reference to its parent device.

        @param device: The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def build_lists(self, lst, number):
        """
        @brief Divides a list of scripts into sub-lists, distributing them
               among a specified number of threads.

        This method implements a load-balancing strategy to assign scripts
        to worker threads.

        @param lst: The original list of scripts to be distributed.
        @param number: The number of sub-lists (threads) to distribute the scripts among.
        @return: A list of lists, where each inner list contains scripts for a thread.
        """
        result_list = []        
        size = int(len(lst) / number)
        # Block Logic: Distributes scripts evenly among 'number' of threads.
        # Invariant: Each thread initially receives 'size' number of scripts.
        for i in range(number):
            row = size * i
            column = size * (i + 1)
            result_list.append(lst[row: column])
        # Block Logic: Distributes remaining scripts (if any) to threads in a round-robin fashion.
        for i in range(len(lst) - size * number):
            row = i % number
            column = i + (size * number)
            result_list[row].append(lst[column])
        return result_list

    class Instance(Thread):
        """
        @brief A worker thread for executing a subset of scripts on a device.

        Each instance processes a portion of the assigned scripts, collects
        data from neighboring devices, runs the script logic, and updates
        sensor data.
        """
        def __init__(self, device, big_list, neighbours):
            """
            @brief Initializes a script execution instance.

            @param device: The parent Device object.
            @param big_list: A sub-list of scripts to be executed by this instance.
            @param neighbours: A list of neighboring Device objects.
            """
            Thread.__init__(self, name="Instance")
            self.neighbours = neighbours            
            self.big_list = big_list
            self.device = device

        def run(self):
            """
            @brief The main execution logic for the script instance thread.

            It iterates through its assigned scripts, gathers sensor data from
            the local device and its neighbors, executes the script's `run` method,
            and then updates relevant sensor data across devices.
            """
            script_data = []
            # Block Logic: Processes each script assigned to this instance.
            # Invariant: 'script_data' accumulates relevant sensor information before script execution.
            for (script, location) in self.big_list:
                # Collect data from neighboring devices at the specified location.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Collect data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                # If sensor data was collected, execute the script.
                if script_data != []:
                    result = script.run(script_data)
                    # Update sensor data across all relevant devices.
                    self.all_devices_set_data(location, result)

        def my_max(self, a, b):
            """
            @brief Helper function to return the maximum of two values.

            @param a: The first value.
            @param b: The second value.
            @return: The greater of the two values.
            """
            # Inline: Compares two values and returns the larger one.
            if a >= b:
                return a
            return b


        def all_devices_set_data(self, location, result):
            """
            @brief Updates sensor data across the local device and its neighbors
                   based on the script's result.

            It applies a maximum-value update strategy to the sensor data.

            @param location: The location of the sensor data to update.
            @param result: The result from the script execution.
            """
            # Block Logic: Iterates through neighbor devices to update their sensor data.
            # Invariant: Each device's data at 'location' is updated with the maximum
            #            of its current value and the 'result'.
            for device in self.neighbours:
                # Use device's lock to ensure atomic update of shared sensor data.
                with self.device.lock:
                    device.set_data(location, self.my_max(result, device.get_data(location)))
            # Update the local device's sensor data.
            with self.device.lock:
                result_data = self.my_max(result, self.device.get_data(location))
                self.device.set_data(location, result_data)


    def run(self):
        """
        @brief The main loop for the DeviceThread.

        It continuously waits for scripts, distributes them to worker instances,
        executes them, and synchronizes with other device threads using a barrier
        after each timepoint's processing is complete.
        """
        # Block Logic: Main loop for device thread operation.
        # Invariant: The thread continues processing timepoints until the supervisor
        #            signals termination.
        while True:
            threads_scripts = list()
            # Retrieve the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: If there are no neighbors, the device thread terminates.
            if neighbours is None:
                break
            # Block Logic: Waits for the current timepoint's scripts to be assigned.
            # Invariant: 'timepoint_done' is cleared to prepare for the next timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            # Distribute the device's scripts among MAX_THREADS worker instances.
            list_of_scripts = self.build_lists(self.device.scripts, MAX_THREADS)
            # Block Logic: Creates and starts worker threads for script execution.
            for i in range(MAX_THREADS):
                if list_of_scripts != []: # Ensure there are scripts to process
                    threads_scripts.append(self.Instance(self.device, list_of_scripts[i], neighbours))
            # Start all worker threads.
            for script_thread in threads_scripts:
                script_thread.start()
            # Wait for all worker threads to complete their script execution.
            for script_thread in threads_scripts:
                script_thread.join()
            # Synchronize with other devices at the barrier before proceeding to the next timepoint.
            self.device.barrier.wait()