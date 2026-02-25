"""
@3722b56a-c47e-4bf2-b452-4ea7f5549ce0/device.py
@brief Implements a distributed device simulation framework, including a reusable barrier for synchronization and device-specific functionalities for sensor data processing and script execution.
* Algorithm: Thread-based simulation with barrier synchronization.
* Concurrency: Uses `threading.Semaphore` and `threading.Lock` for inter-thread synchronization and mutual exclusion.
"""

from threading import Event, Thread
from threading import Semaphore, Lock

"""
    A reusable barrier synchronization primitive for coordinating multiple threads.
    This barrier allows a fixed number of threads to wait until all threads
    have reached a specific point, and then resets itself for reuse.
    """
class ReusableBarrier(object):
    """
    @brief A re-usable barrier mechanism for synchronizing multiple threads.
    This barrier allows a set number of threads to wait for each other at a synchronization
    point, and once all threads have arrived, they are all released simultaneously.
    It can be used multiple times after all threads have passed through.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that must reach the
                                barrier before any can proceed.
        """
        self.num_threads = num_threads
        # Tracks the number of threads waiting in the first phase of the barrier.
        self.count_threads1 = [self.num_threads]
        # Tracks the number of threads waiting in the second phase of the barrier.
        self.count_threads2 = [self.num_threads]

        # A lock to protect access to the thread count.
        self.count_lock = Lock()

        # Semaphore for releasing threads in the first phase.
        self.threads_sem1 = Semaphore(0)

        # Semaphore for releasing threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all other threads have also
        called this method. This method involves two phases to ensure the
        barrier can be reused.
        """

        self.phase(self.count_threads1, self.threads_sem1)
        # Phase 2: Reset the barrier and wait for all threads to clear
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Manages a single phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current count of threads
                                  remaining in this phase. (Uses a list to allow
                                  modification within the `with` statement).
            threads_sem (Semaphore): The semaphore used to release threads
                                     once the count reaches zero.
        """
        with self.count_lock:
            # Decrement the count of threads waiting in this phase.
            count_threads[0] -= 1
            # If this is the last thread to arrive, release all waiting threads.
            if count_threads[0] == 0:
                # Release all threads by calling release() num_threads times.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the thread count for the next use of the barrier.
                count_threads[0] = self.num_threads
        # Acquire the semaphore, effectively waiting until all threads are released.
        threads_sem.acquire()



class Device(object):
    """
    Represents a simulated device within a multi-device system.

    Each device has unique sensor data, can execute scripts, and interacts
    with a supervisor and other devices through synchronization mechanisms
    like a reusable barrier and locks for managing shared resources (locations).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor readings,
                                 where keys are locations and values are data.
            supervisor (Supervisor): A reference to the supervisor object
                                     that manages device interactions.
        """
        # Barrier for synchronizing all devices. Initialized by device_id 0.
        self.barrier = None 
        # Event to signal that initial setup for the device is complete.
        self.InitializationEvent = Event() 
        # Dictionary to store locks for specific locations (shared resources).
        self.LockLocation = None 
        # Lock to protect access to the LockLocation dictionary.
        self.LockDict = Lock() 

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been assigned.
        self.script_received = Event()
        # List of scripts assigned to this device, each with its associated location.
        self.scripts = []
        # Event to signal that processing for the current timepoint is done.
        self.timepoint_done = Event()
        # The thread dedicated to running this device's logic.
        self.thread = DeviceThread(self)

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources and starts the device thread.
        Only device with device_id 0 initializes the shared barrier and LockLocation.

        Args:
            devices (list): A list of all Device instances in the system.
        """
        # The device with device_id 0 is responsible for initializing shared resources.
        if self.device_id == 0:
            # Get the total number of devices to initialize the barrier.
            n = len(devices)
            # Initialize the reusable barrier for all devices.
            self.barrier = ReusableBarrier(n)   
            # Initialize a dictionary to hold locks for various locations.
            self.LockLocation = {}  

            # Iterate through all devices to set up their shared resources.
            for idx in range(len(devices)):
                d = devices[idx]

                # Assign the shared LockLocation dictionary and barrier to each device.
                d.LockLocation = self.LockLocation
                d.barrier = self.barrier
                # If it's not device 0, signal that initialization is complete for it.
                if d.device_id == 0:
                    pass
                else:
                    d.InitializationEvent.set()
        else:
            # Non-device 0 waits for the shared resources to be initialized by device 0.
            self.InitializationEvent.wait()

        # Start the thread for this device.
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location.

        If `script` is None, it signals that no more scripts are to be executed
        for the current timepoint, and the `timepoint_done` event is set.

        Args:
            script (Script or None): The script object to assign, or None to signal
                                     the end of scripts for a timepoint.
            location (str): The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Invariant: If script is None, it means the current timepoint's script assignments are done.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            Any: The sensor data if the location exists, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (str): The location where the data should be set.
            data (Any): The new data to set for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its associated thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Represents a thread that manages the execution of scripts on a Device.

    This thread continually synchronizes with other device threads using a
    reusable barrier, processes assigned scripts, and interacts with the
    supervisor to get information about neighboring devices.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously performs the following steps:
        1. Waits at a reusable barrier to synchronize with other device threads.
        2. Retrieves information about neighboring devices from the supervisor.
        3. If there are no neighbors (indicating shutdown), breaks the loop.
        4. Waits until all scripts for the current timepoint are assigned.
        5. Iterates through assigned scripts, acquiring locks for locations,
           executing scripts with collected data, and updating data.
        6. Clears the `timepoint_done` event for the next cycle.
        """
        while True:
            # Wait at the barrier to synchronize with other devices.
            self.device.barrier.wait()

            # Get information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If `neighbours` is None, it signals the end of the simulation.
            if neighbours is None:
                # If there are no neighbors, it implies a shutdown signal, so break the loop.
                break

            # Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            dev_scripts = self.device.scripts

            # Process each assigned script.
            for (script, location) in self.device.scripts:
                # Acquire a lock to safely access and modify the LockLocation dictionary.
                self.device.LockDict.acquire()

                # If a lock for this location doesn't exist, create one.
                if location not in self.device.LockLocation.keys():
                    self.device.LockLocation[location] = Lock()


                # Acquire the specific lock for the current location to prevent race conditions.
                self.device.LockLocation[location].acquire()

                # Release the lock for LockDict as it's no longer needed for this operation.
                self.device.LockDict.release()

                script_data = []
                # Collect data from neighboring devices for the current location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Collect data from the current device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: Execute the script only if relevant data was collected.
                if script_data != []:
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Update the data in neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    # Update the data in the current device.
                    self.device.set_data(location, result)

                # Release the lock for the current location.
                self.device.LockLocation[location].release()

            # Clear the event to prepare for the next timepoint's scripts.
            self.device.timepoint_done.clear()
from threading import Semaphore, Lock

class ReusableBarrier(object):
    

    def __init__(self, num_threads):
        

        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]

        
        self.count_lock = Lock()

        
        self.threads_sem1 = Semaphore(0)

        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        

        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        

        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                
                for i in range(self.num_threads):
                    
                    
                    threads_sem.release()
                
                count_threads[0] = self.num_threads
        
        threads_sem.acquire()
        
