"""
This module implements a device simulation framework that utilizes multiple threads
and a reusable barrier for synchronization. It defines:
- ReusableBarrierSem: A re-usable barrier mechanism for threads using semaphores.
- Device: Represents a simulated device managing sensor data and script assignments,
          and orchestrating its own pool of worker threads.
- MyThread: Worker threads responsible for executing individual scripts.
- DeviceThread: The main thread for a Device, coordinating script execution and data sharing.

The system uses Events, Locks, and Semaphores for inter-thread communication and data consistency.
"""


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    Implements a reusable barrier synchronization mechanism using semaphores and a lock.
    This barrier allows a fixed number of threads to wait until all have reached a certain point
    before any are allowed to proceed, and can then be reset for subsequent synchronizations.
    """
    
    
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               # Protects access to the count_threads variables.
        self.threads_sem1 = Semaphore(0)         # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0)         # Semaphore for the second phase.
    
    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this barrier.
        Uses a two-phase approach to allow reusability.
        """
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """
        First phase of the barrier. Threads decrement a counter and the last thread
        releases all waiting threads for this phase.
        """
        with self.counter_lock: # Block Logic: Ensure atomic access to the counter.
            self.count_threads1 -= 1
            # Block Logic: If this is the last thread in phase 1, release all waiting threads.
            if self.count_threads1 == 0:
                for i in range(self.num_threads): # Block Logic: Release each waiting thread.
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads       # Reset counter for next use.
        self.threads_sem1.acquire() # Block Logic: Wait for all threads to reach this point.
    
    def phase2(self):
        """
        Second phase of the barrier. Threads decrement a counter and the last thread
        releases all waiting threads for this phase.
        """
        with self.counter_lock: # Block Logic: Ensure atomic access to the counter.
            self.count_threads2 -= 1
            # Block Logic: If this is the last thread in phase 2, release all waiting threads.
            if self.count_threads2 == 0:
                for i in range(self.num_threads): # Block Logic: Release each waiting thread.
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads       # Reset counter for next use.
        self.threads_sem2.acquire() # Block Logic: Wait for all threads to reach this point.


class Device(object):
    """
    Represents a simulated device in a distributed system. Each device manages
    its sensor data, assigns and executes scripts via its `DeviceThread`,
    and interacts with a supervisor. Synchronization is handled through shared
    locks and a global barrier.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are assigned.
        self.scripts = [] # List to store assigned scripts (script, location) tuples.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.thread = DeviceThread(self) # The main thread for this device.

        
        self.neighbours = [] # List to store neighboring devices.
        self.alldevices = [] # List of all devices in the simulation.
        self.barrier = None # Global barrier for device synchronization.
        self.threads = [] # List to hold MyThread instances (worker threads).
        self.threads_number = 8 # Number of worker threads for this device.
        self.locks = [None] * 100 # List to store location-specific locks. Each lock protects access to a location's data.

        self.thread.start() # Start the main device thread.

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs initial setup for all devices in the simulation.
        This includes setting up the global barrier and populating the `alldevices` list.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Block Logic: If the global barrier is not yet set, initialize it for all devices.
        if self.barrier is None:
            # Inline: Initialize the barrier with the total number of devices.
            barrier = ReusableBarrierSem(len(devices))
            self.barrier = barrier # Set the barrier for the current device.
            # Block Logic: Propagate the initialized barrier to all other devices that don't have one.
            for d in devices:
                if d.barrier is None:
                    d.barrier = barrier
        
        # Block Logic: Populate the list of all devices, filtering out any None entries.
        for device in devices:
            if device is not None:
                self.alldevices.append(device)


    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific location.
        If a script is provided, it's added to the device's script list.
        A location-specific lock is either created or retrieved from another device.
        If no script is provided (None), it signals that a timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None to signal timepoint completion.
            location (str): The location associated with the script.
        """
        
        
        no_lock_for_location = 0; # Flag to indicate if a lock for the location has been found/created.
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
            # Block Logic: Search through all devices to find an existing lock for the location.
            for device in self.alldevices:

                # Block Logic: If an existing lock is found, use it.
                if device.locks[location] is not None:
                    self.locks[location] = device.locks[location]
                    no_lock_for_location = 1; # Set flag to indicate lock found.
                    break;
            # Block Logic: If no existing lock was found, create a new one for this location.
            if no_lock_for_location == 0:
                self.locks[location] = Lock()
            self.script_received.set() # Signal that a new script is available.
        else:
            self.timepoint_done.set() # Signal that processing for the current timepoint is complete.

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            Any or None: The sensor data if available for the location, otherwise None.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (str): The location for which to set data.
            data (Any): The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its main device thread.
        """
        
        self.thread.join()


class MyThread(Thread):

    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        self.device.locks[self.location].acquire()
        script_data = []
        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)
        self.device.locks[self.location].release()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            self.device.neighbours = neighbours

            count = 0
            
            for (script, location) in self.device.scripts:
                
                if count >= self.device.threads_number:
                    break
                count = count + 1
                thread = MyThread(self.device, location, script, neighbours)
                self.device.threads.append(thread)

            
            for thread in self.device.threads:
                thread.start()
            for thread in self.device.threads:
                thread.join()
            self.device.threads = []

            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
