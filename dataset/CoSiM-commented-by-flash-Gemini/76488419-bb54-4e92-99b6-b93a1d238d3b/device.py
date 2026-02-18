"""
@76488419-bb54-4e92-99b6-b93a1d238d3b/device.py
@brief Implements device behavior within a distributed sensor network simulation.
This module defines the core logic for individual devices, including data processing,
inter-device communication, and thread management for concurrent operations.
It simulates a scenario where devices collect sensor data, execute scripts based on
this data and data from neighbors, and then propagate results.
"""

from threading import Event, Thread, Lock
from reusableBarrier import ReusableBarrier

class MyThread(Thread):
    """
    @brief Manages script execution and data exchange for a specific location within a device.
    This thread is responsible for acquiring locks on neighbor devices, retrieving data,
    executing a script with collected data, and then updating neighbor devices with the result.
    """
    def __init__(self, d, location, script, neighbours):
        """
        @brief Initializes a MyThread instance.
        @param d: The DeviceThread instance containing the parent Device.
        @param location: The specific sensor data location this thread will process.
        @param script: The script to be executed.
        @param neighbours: A list of neighboring Device instances.
        """
        Thread.__init__(self)
        self.d = d
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief Executes the script, collects data from neighbors and self, and propagates results.
        Block Logic:
        1. Iterates through neighbor devices to acquire locks and collect sensor data for the given location.
           Invariant: Locks are acquired on neighbor devices to ensure exclusive access to shared data.
        2. Collects its own sensor data for the given location.
        3. If data is collected, executes the assigned script with the combined data.
        4. Propagates the script result back to neighbor devices and its own device, releasing locks as it goes.
           Invariant: Locks are released after data propagation to allow other threads access.
        """
        script_data = []

        # Block Logic: Iterates through neighbor devices to acquire locks and collect sensor data.
        # Precondition: `self.neighbours` contains valid Device instances.
        # Invariant: For each neighbor, if a lock exists for `self.location`, it is acquired.
        for device in self.neighbours:
            keys = device.dictionar.keys()
            if self.location in keys:
                lock = device.dictionar[self.location]
                if lock is not None:
                    lock.acquire()


            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Retrieves sensor data from the current device for the specified location.
        data = self.d.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script if any data was collected and propagates the result.
        if script_data != []:
            
            result = self.script.run(script_data)
            
            # Block Logic: Updates neighbor devices with the script result and releases associated locks.
            # Invariant: Locks are released for each neighbor device after updating its data.
            for device in self.neighbours:
                device.set_data(self.location, result)
                keys = device.dictionar.keys()
                if self.location in keys:
                    lock = device.dictionar[self.location]
                    if lock is not None:
                        lock.release()
            
            # Block Logic: Updates the current device's sensor data with the script result.
            self.d.device.lock.acquire()
            self.d.device.set_data(self.location, result)
            self.d.device.lock.release()

class Device(object):
    """
    @brief Represents a single device in the distributed sensor network.
    Manages sensor data, script assignment, and communication with a supervisor and neighbor devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing initial sensor data for various locations.
        @param supervisor: The supervisor object managing the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.lock = Lock()
        self.dictionar = {}
        # Block Logic: Initializes a dictionary of locks for each sensor data location.
        # Invariant: Each location in `sensor_data` (if not None) will have an associated Lock object.
        for location in self.sensor_data:
            if location != None:
                self.dictionar[location] = Lock()
            else:
                self.dictionar[location] = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up a reusable barrier for device synchronization.
        Only the device with ID 0 initializes the barrier, which is then shared among all devices.
        @param devices: A list of all Device instances in the network.
        Precondition: This method is called once by a designated device (e.g., device 0) at startup.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            # Block Logic: Assigns the created barrier to all devices in the network for synchronization.
            # Invariant: All devices will share the same ReusableBarrier instance.
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location on the device.
        Signals that a script has been received, or that a timepoint is done if no script.
        @param script: The script object to assign.
        @param location: The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The sensor data location to retrieve.
        @return The data associated with the location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
        @param location: The sensor data location to update.
        @param data: The new data to set.
        Precondition: `location` must be a key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device thread, ensuring it completes its operations.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief Manages the continuous operation of a Device, including fetching neighbors,
    executing assigned scripts, and synchronizing with other device threads.
    Time Complexity: O(T * S * N) where T is the number of timepoints, S is the number of scripts per device,
    and N is the number of neighbors.
    """

    def __init__(self, device):
        """
        @brief Initializes a DeviceThread instance.
        @param device: The parent Device instance this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device thread.
        Block Logic:
        1. Continuously fetches neighbor devices from the supervisor.
        2. Waits for a `timepoint_done` signal, indicating new scripts or a completed timepoint.
           Invariant: The thread pauses until a new timepoint begins or is signaled as done.
        3. Clears the `timepoint_done` signal.
        4. Creates and starts `MyThread` instances for each assigned script and waits for their completion.
           Invariant: All scripts for the current timepoint are executed concurrently via `MyThread` instances.
        5. Synchronizes with other device threads using a barrier to ensure all devices complete their timepoint.
           Invariant: All device threads reach this barrier before proceeding to the next timepoint.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break

            self.device.timepoint_done.wait()


            self.device.timepoint_done.clear()


            my_thread_list = []
            
            for (script, location) in self.device.scripts:
                my_thread = MyThread(self, location, script, neighbours)
                my_thread_list.append(my_thread)
                my_thread.start()
            for thread in my_thread_list:
                thread.join()

            
            self.device.barrier.wait()

