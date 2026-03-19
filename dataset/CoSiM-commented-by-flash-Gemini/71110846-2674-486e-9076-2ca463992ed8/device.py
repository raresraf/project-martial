


"""
This module defines the Device and DeviceThread classes, which represent
individual processing units in a distributed system, handling sensor data,
script execution, and inter-device communication. It also includes the
ScriptRunner class for executing assigned scripts.
"""

from threading import Event, Thread, Lock
import barrier
import runner


class Device(object):
    """
    Represents a single device or processing unit within a distributed system.
    Each device manages its own sensor data, executes scripts, and interacts
    with other devices through a supervisor and a shared barrier mechanism.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): Initial sensor data for the device,
                                mapping locations to data values.
            supervisor (Supervisor): The supervisor managing this device
                                     and its interactions with others.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that the current timepoint's tasks are completed.
        self.timepoint_done = Event()


        self.thread = DeviceThread(self)
        self.barr = None
        self.devices = []
        self.runners = []
        # List of locks, where each lock protects a specific data location.
        self.locks = [None] * 50
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
        Configures the device with a list of other devices in the system
        and establishes a shared barrier for synchronization.

        Args:
            devices (list): A list of Device objects representing
                            other devices in the system.
        """
        # Precondition: The device's barrier is not yet initialized.
        # Invariant: After this block, all devices in the provided list
        # will share the same barrier object.
        if self.barr is None:
            # Initialize a reusable barrier with the total number of participating devices.
            barr = barrier.ReusableBarrierSem(len(devices))
            self.barr = barr
            # Assign the newly created barrier to all devices that don't have one yet.
            for dev in devices:
                if dev.barr is None:
                    dev.barr = barr
        
        # Invariant: After this loop, the internal 'devices' list will contain
        # references to all valid devices from the input list.
        for dev in devices:
            if dev is not None:
                self.devices.append(dev)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.
        Manages the acquisition of locks for the given location to prevent
        race conditions during script execution.

        Args:
            script (Script): The script object to be executed.
                             If None, it signifies the end of a timepoint.
            location (int): The identifier for the data location
                            the script operates on.
        """
        ok = 0
        # Precondition: A script is provided for execution.
        # Invariant: If a script is provided, it's added to the scripts list,
        # and a lock for the location is ensured to exist and be shared.
        if script is not None:
            self.scripts.append((script, location))
            
            # Block Logic: Ensures that a lock exists for the specified location.
            # If a lock for this location already exists on another device,
            # it is reused; otherwise, a new lock is created.
            if self.locks[location] is None:
                for device in self.devices:
                    if device.locks[location] is not None:
                        self.locks[location] = device.locks[location]
                        ok = 1
                        break
                if ok == 0:
                    self.locks[location] = Lock()
            # Signal that a script has been received, allowing the device thread to proceed.
            self.script_received.set()
        else:
            # If no script is provided, signal that the timepoint's script assignment is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location.

        Args:
            location (int): The identifier of the data location.

        Returns:
            Any: The sensor data at the given location, or None if the
                 location does not exist in the device's sensor_data.
        """
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location.

        Args:
            location (int): The identifier of the data location to update.
            data (Any): The new data value to set for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device,
        waiting for its background thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    A dedicated thread for a Device object, responsible for continuously
    checking for new scripts, executing them, and managing synchronization
    between timepoints.
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
        The main execution loop for the device thread.

        Continuously retrieves neighbor information, waits for timepoint
        completion signals, processes assigned scripts, and synchronizes
        with other devices via a barrier.
        """
        # Block Logic: Main loop for the device thread, continuously processing
        # timepoints until a shutdown signal (None neighbors) is received.
        while True:
            # Precondition: The supervisor provides the current neighbors for the device.
            # Invariant: If neighbors are None, it signals the thread to terminate.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits for a timepoint to be marked as done, indicating
            # that all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Iterates through all scripts assigned for the current timepoint,
            # creating and managing ScriptRunner instances for parallel execution.
            for (script, location) in self.device.scripts:
                run = runner.ScriptRunner(self.device, script, location,
                                          neighbours)
                self.device.runners.append(run)

                n = len(self.device.runners)
                x = n / 8
                r = n % 7
                
                # Block Logic: Acquires a lock for the current location to ensure
                # exclusive access to shared data during script execution.
                self.device.locks[location].acquire()
                # Block Logic: Starts ScriptRunner threads in batches of 8 for concurrent execution.
                for i in xrange(0, x):
                    for j in xrange(0, 8):
                        self.device.runners[i * 8 + j].start()
                
                # Block Logic: Handles the remaining ScriptRunner threads if the total
                # number is not a multiple of 8, ensuring all runners are started.
                if n >= 8:
                    for i in xrange(len(self.device.runners) - r,
                                    len(self.device.runners)):
                        self.device.runners[i].start()
                
                # Block Logic: If there are fewer than 8 runners, start all of them directly.
                else:
                    for i in xrange(0, n):
                        self.device.runners[i].start()
                # Block Logic: Waits for all ScriptRunner threads to complete their execution.
                for i in xrange(0, n):
                    self.device.runners[i].join()
                

                # Block Logic: Releases the lock for the current location,
                # allowing other devices to access the shared data.
                self.device.locks[location].release()
                
                self.device.runners = []

            # Block Logic: Resets the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Block Logic: Waits at a barrier to synchronize with all other devices
            # before proceeding to the next timepoint.
            self.device.barr.wait()


from threading import Thread


class ScriptRunner(Thread):
    """
    A dedicated thread for executing a single assigned script on a device.
    It retrieves necessary data from the device and its neighbors, runs the
    script, and then propagates the results back to the devices.
    """

    def __init__(self, device, script, location, neighbours):
        """
        Initializes a new ScriptRunner instance.

        Args:
            device (Device): The Device object on which the script will run.
            script (Script): The script object to execute.
            location (int): The data location relevant to this script execution.
            neighbours (list): A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the assigned script.

        This method retrieves data from neighboring devices and the local
        device, runs the script with this data, and then updates the
        corresponding data locations on both the neighbors and the local device.
        """
        script_data = []
        # Block Logic: Gathers sensor data from all neighboring devices
        # for the specified location.
        for device in self.neighbours:
            # Precondition: The neighbor device has sensor data at the specified location.
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Gathers sensor data from the local device for the specified location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Precondition: There is collected data for the script to process.
        # Invariant: If data is present, the script is run and its results are
        # propagated to the devices.
        if script_data != []:
            # Block Logic: Executes the script with the collected data.
            result = self.script.run(script_data)
            # Block Logic: Updates the sensor data on all neighboring devices
            # with the result of the script execution.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Block Logic: Updates the sensor data on the local device
            # with the result of the script execution.
            self.device.set_data(self.location, result)
