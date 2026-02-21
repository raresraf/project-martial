


from threading import *


class Device(object):
    """
    Represents a device in a simulated distributed system.
    Each device manages sensor data, processes scripts, and coordinates
    with a supervisor and other devices through a shared reusable barrier.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding the device's sensor data,
                                keyed by location.
            supervisor (Supervisor): The supervisor object responsible for
                                     managing devices and providing neighborhood information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        # List to store assigned scripts.
        self.scripts = []
        # Event to signal that the device has completed processing for the current timepoint.
        self.timepoint_done = Event()
        # The dedicated thread for this device's execution logic.
        self.thread = DeviceThread(self)
        # Global barrier for synchronizing all devices. Set by setup_devices.
        self.bar = None
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures a shared MyReusableBarrier among a group of devices.
        Only the device at index 0 in the 'devices' list initializes the barrier,
        which is then accessed by all other devices through this central device.

        Args:
            devices (list): A list of all Device objects participating in the simulation.
        """
        self.devices=devices # Store the list of all devices for later access.
        # Only the device at index 0 in the provided list initializes the barrier.
        if self==devices[0]:
            self.bar = MyReusableBarrier(len(devices))
        
        # This 'pass' statement indicates no further action is taken by individual devices
        # beyond setting the 'devices' list and (for devices[0]) initializing the barrier.
        pass

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.
        If `script` is None, it signals the completion of script assignments for the timepoint.

        Args:
            script (Script): The script object to be executed.
            location: The data location pertinent to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a script has been received.
        else:
            # Signal that all scripts for this timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location: The key for the sensor data.

        Returns:
            The sensor data for the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location: The key for the sensor data.
            data: The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its execution thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The dedicated thread of execution for a Device object.
    It manages the device's operational logic within the simulation loop,
    including supervisor interaction, sequential script execution,
    and coordinating with other devices using a global reusable barrier.
    """
    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device object associated with this thread.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device thread.
        It continuously processes timepoints, synchronizes with other devices
        using a global barrier, executes scripts sequentially, and updates data.
        """
        while True:
            # Retrieve current neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors are returned, it signifies the end of the simulation.
            if neighbours is None:
                break

            # Wait until scripts for the current timepoint have been assigned.
            # This Event is set by the assign_script method when script is None.
            self.device.timepoint_done.wait()
            # Clear the event for the next cycle.
            self.device.timepoint_done.clear()

            # Iterate through and execute all assigned scripts for the current timepoint.
            # WARNING: This implementation lacks explicit locking mechanisms (e.g., mutexes)
            # for protecting shared data (sensor_data on local and neighboring devices)
            # during read/write operations within this loop. This can lead to race
            # conditions and data inconsistency in a concurrent environment if multiple
            # DeviceThreads attempt to modify the same data locations simultaneously.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Collect data from neighboring devices for the current location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Collect data from the local device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute the script if there is data to process.
                if script_data != []:
                    # Execute the script with the aggregated data.
                    result = script.run(script_data)

                    # Propagate the script's result back to all neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Update the local device's data with the script's result.
                    self.device.set_data(location, result)

            # Clear the list of scripts after they have been processed for the current timepoint.
            self.device.scripts = []
            
            # Synchronize with all other devices at the global barrier before starting the next timepoint.
            # The barrier is accessed through the first device in the shared 'devices' list.
            self.device.devices[0].bar.wait()


class MyReusableBarrier():
    """
    Implements a reusable barrier synchronization mechanism using a two-phase
    approach with semaphores and a lock. Threads arriving at the barrier
    wait until all expected threads have arrived across two distinct phases,
    after which all are released simultaneously, and the barrier is reset for reuse.
    """
    def __init__(self, num_threads):
        """
        Initializes the MyReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Counter for threads in the first phase.
        self.count_threads1 = self.num_threads
        # Counter for threads in the second phase.
        self.count_threads2 = self.num_threads
        
        # Lock to protect access to the thread counters.
        self.counter_lock = Lock()       
        # Semaphore for the first phase of waiting.
        self.threads_sem1 = Semaphore(0) 
        # Semaphore for the second phase of waiting.
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """
        Causes the calling thread to wait until all `num_threads` have reached this barrier.
        The barrier is reusable after all threads have passed through both phases.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        First phase of the reusable barrier synchronization.
        Threads decrement a counter and the last thread releases all others.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            # Reset the second phase counter for reusability.
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire() # Wait until released by the last thread in phase1.
         
    def phase2(self):
        """
        Second phase of the reusable barrier synchronization.
        Threads decrement a counter and the last thread releases all others.
        This phase allows the barrier to be reused.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            # Reset the first phase counter for reusability.
            self.count_threads1 = self.num_threads
         
        self.threads_sem2.acquire() # Wait until released by the last thread in phase2.

