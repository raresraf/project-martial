


"""
This module implements a simulated distributed device system.

It defines classes for:
- ReusableBarrier: A reusable barrier for synchronizing multiple threads in phases.
  This version uses lists for thread counters to enable reusability across barrier waits.
- Device: Represents a single device in the distributed system, managing its sensor data,
  communication with a supervisor, and multi-threaded script execution. It utilizes
  `RLock` for reentrant locking to protect shared data and script assignments.
- DeviceThread: The primary thread for a Device, acting as an orchestrator. It fetches
  neighbor information, distributes scripts to dynamically created worker threads
  for concurrent processing, and manages overall timepoint synchronization.
"""

from threading import Event, Lock, Thread, RLock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier synchronization mechanism for multiple threads using semaphores.
    This barrier allows a fixed number of threads to wait at a synchronization point,
    and once all threads arrive, they are all released simultaneously. The counters
    are stored in lists (`count_threads1`, `count_threads2`) to allow for reusability
    of the barrier instance across multiple `wait` calls without complex state resetting.
    """

    def __init__(self, num_threads):
        """
        Initializes the reusable barrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate
                               in the barrier synchronization.
        """
        self.num_threads = num_threads
        # Counters for the two phases of the barrier. Stored in lists to be mutable
        # when passed to the 'phase' method.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # Lock to protect the shared counters during decrements and resets.
        self.count_lock = Lock()
        # Semaphores for the two phases of threads to wait on.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all threads have reached this barrier.
        This method orchestrates a two-phase synchronization to ensure reusability
        without deadlocks and to handle threads arriving out of order for subsequent waits.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current count of threads
                                  remaining for this phase.
            threads_sem (Semaphore): The semaphore threads wait on for this phase.
        """
        with self.count_lock: # Protect shared counter access.
            count_threads[0] -= 1 # Decrement the count of threads remaining.
            if count_threads[0] == 0: # If this is the last thread to arrive:
                for i in range(self.num_threads):
                    threads_sem.release() # Release all waiting threads.
                count_threads[0] = self.num_threads # Reset counter for next use.
        threads_sem.acquire() # Wait for all threads to be released.


class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, interacts with a supervisor,
    and orchestrates multi-threaded script execution. It uses `RLock` instances
    to protect access to its sensor data and script assignments.
    """

    # Global barrier for synchronizing all devices, initialized by the master device.
    barrier = None

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for the device.
            supervisor (object): A reference to a supervisor object for inter-device communication.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        self.scripts = [] # List to store assigned scripts and their locations.
        # Event to signal that the current timepoint's processing is complete.
        self.timepoint_done = Event()
        # The main thread for this device, responsible for orchestration.
        self.thread = DeviceThread(self)
        self.thread.start() # Start the DeviceThread upon Device initialization.
        self.lock = RLock() # Reentrant lock to protect access to sensor_data.
        self.script_lock = RLock() # Reentrant lock to protect access to scripts list.
        self.run_lock = RLock() # Another reentrant lock, purpose needs context.

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global barrier for all devices in the simulation.
        This method is designed to be called by a single, designated master device (device_id 0).

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        if self.device_id is 0:
            # Inline: The master device initializes the barrier for all devices.
            # The number of threads for the barrier is the total number of DeviceThreads across all Devices.
            # Assuming DeviceThread.num_threads (8) is hardcoded or configured elsewhere.
            self.barrier = ReusableBarrier(8 * len(devices))
            # Inline: Distribute the initialized barrier to all other devices.
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (str): The location identifier in the sensor data to which the script applies.
        """
        with self.script_lock: # Protects access to the scripts list.
            if script is not None:
                self.scripts.append((script, location))
                self.script_received.set() # Signal that a script has been received.
            else:
                self.timepoint_done.set() # Signal that the timepoint is done if no script is assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location, protected by a reentrant lock.

        Args:
            location (str): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        with self.lock: # Ensures exclusive access to sensor_data.
            result = self.sensor_data[location] if location in self.sensor_data else None
        return result

    def set_data(self, location, data):
        """
        Sets sensor data for a given location, protected by a reentrant lock.

        Args:
            location (str): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        with self.lock: # Ensures exclusive access to sensor_data.
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by waiting for its main DeviceThread to complete.
        """
        self.thread.join() # Wait for the main DeviceThread to finish.


class DeviceThread(Thread):
    """
    The main orchestrating thread for a `Device`.
    This thread is responsible for fetching neighbor information from the supervisor,
    managing timepoint progression, and dynamically spawning worker `Thread` instances
    to execute scripts concurrently.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent Device object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.
        It continuously fetches neighbor data, orchestrates concurrent script execution
        via worker threads, and manages synchronization for timepoint progression.
        """
        while True:
            # Block Logic: Fetch neighbor information from the supervisor.
            # If the supervisor returns None, it signals termination for this device.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit the loop, signaling device shutdown.

            neighbours.append(self.device) # Add the current device itself to the list of neighbors for local data access.

            # Block Logic: Wait for the current timepoint to be marked as done, signifying that
            # all scripts for the previous timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Dynamically create and manage worker threads for concurrent script execution.
            # `num_threads` determines the degree of parallelism for script processing within this device.
            num_threads = 8 # Hardcoded number of concurrent worker threads.
            threads = [Thread(target=self.concurrent_work,
                              args=(neighbours, i, num_threads)) for i in range(num_threads)]

            for thread in threads:
                thread.start() # Start each worker thread.

            for thread in threads:
                thread.join() # Wait for all worker threads to complete their assigned work.

            # Block Logic: Synchronize all devices at the global barrier.
            # This ensures all devices have completed their concurrent work for the timepoint
            # before proceeding to the next timepoint.
            self.device.barrier.wait()
            # Inline: Clear the timepoint_done event, preparing it for the next timepoint's signal.
            self.device.timepoint_done.clear()

    def concurrent_work(self, neighbours, thread_id, num_threads):
        """
        Executes a subset of the assigned scripts concurrently within a worker thread.
        This method is designed to be run by dynamically spawned `Thread` instances.

        Args:
            neighbours (list): A list of Device objects, including the current device itself and its neighbors.
            thread_id (int): A unique identifier for the current worker thread.
            num_threads (int): The total number of worker threads spawned.
        """
        # Block Logic: Iterates through the scripts assigned to this specific worker thread.
        for (script, location) in self.keep_assigned(self.device.scripts, thread_id, num_threads):
            script_data = []
            # Block Logic: Collects data from all neighboring devices (and the current device)
            # at the specified location for the script.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: If data was collected, execute the script and update device data.
            if script_data != []:
                # Inline: Execute the script's `run` method with the collected data.
                result = script.run(script_data)

                # Block Logic: Update the sensor data of all involved devices with the result.
                # In this specific implementation, it takes the maximum of the new result and existing data.
                for device in neighbours:
                    res = max(result, device.get_data(location))
                    device.set_data(location, res)

    def keep_assigned(self, scripts, thread_id, num_threads):
        """
        Filters a list of scripts, returning only those assigned to a specific worker thread.
        This implements a round-robin assignment strategy.

        Args:
            scripts (list): The full list of scripts assigned to the Device.
            thread_id (int): The ID of the current worker thread.
            num_threads (int): The total number of worker threads available.

        Returns:
            list: A subset of scripts assigned to the `thread_id`.
        """
        assigned_scripts = []
        # Inline: Assigns scripts to worker threads in a round-robin fashion.
        # A script 'i' is assigned to 'thread_id' if 'i % num_threads' equals 'thread_id'.
        for i, script in enumerate(scripts):
            if i % num_threads is thread_id:
                assigned_scripts.append(script)

        return assigned_scripts
