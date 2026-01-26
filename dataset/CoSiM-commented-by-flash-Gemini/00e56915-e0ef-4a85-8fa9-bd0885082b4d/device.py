

"""
This module implements a simulation of a distributed system of devices.
Each `Device` processes sensor data, executes assigned scripts, and coordinates
with a `supervisor` and other `neighbours`. Thread synchronization is managed
using `threading.Event`, `threading.Lock`, `threading.Condition`, and a custom
`CondBarrier` class.
"""

from threading import Event, Thread, Condition, Lock


class Device(object):
    """
    Represents a single device in a simulated environment.
    Manages its unique ID, local sensor data, and a list of scripts to execute.
    Coordinates with a supervisor for neighbor information and uses a shared
    conditional barrier (`CondBarrier`) and lock for synchronization with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        :param device_id: A unique integer identifier for the device.
        :param sensor_data: A dictionary containing the device's sensor readings.
        :param supervisor: A reference to the central supervisor managing the devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal when a script is received (currently unused).
        self.scripts = []  # List of (script, location) tuples to execute.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's scripts.
        self.thread = DeviceThread(self)  # The dedicated thread for this device's operations.
        self.barr = None  # Reference to a shared CondBarrier for device synchronization.
        self.lock = None  # Reference to a shared Lock for protecting critical sections.
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the device.

        :return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared conditional barrier (`CondBarrier`)
        and a shared global lock (`threading.Lock`) among all devices.
        The barrier is created only once by the first device (device with ID 0)
        and then distributed to all others.

        :param devices: A list of all Device instances in the system.
        """
        # Block Logic: Initializes and distributes the CondBarrier.
        # The barrier is created only by the first device (device 0) and then shared.
        if devices[0].barr is None and self.device_id == devices[0].device_id:
            barr = CondBarrier(len(devices))  # Create a new conditional barrier.
            for i in devices:
                i.barr = barr  # Distribute the barrier to all devices.
        
        # Block Logic: Initializes and distributes the shared global lock.
        # A single lock is created and shared among all devices to protect shared resources.
        lock = Lock()
        for d in devices:
            d.lock = lock

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific sensor data location.
        If `script` is `None`, it signals that the current timepoint is done.

        :param script: The script (callable) to execute, or None to signal timepoint completion.
        :param location: The sensor data location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))  # Add the script to the list.
        else:
            self.timepoint_done.set()  # Signal that this timepoint is complete.
            self.script_received.set() # Also sets script_received, though this event is unused here.

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from the device's local sensor data.

        :param location: The key identifying the sensor data location.
        :return: The sensor data value, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a specified location within the device's local sensor data.

        :param location: The key identifying the sensor data location.
        :param data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates a graceful shutdown of the device's operational thread.
        """
        self.thread.join()  # Wait for the device's thread to complete its execution.

class CondBarrier():
    """
    Implements a reusable conditional barrier for synchronizing a fixed number of threads.
    Threads arriving at the barrier will block until all `num_threads` have arrived,
    after which all waiting threads are released simultaneously.
    """
    def __init__(self, num_threads):
        """
        Initializes the conditional barrier.

        :param num_threads: The total number of threads that must reach the barrier
                            before all waiting threads are released.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads  # Current count of threads waiting.
        self.cond = Condition()  # The condition variable used for blocking and notifying.

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this barrier.
        Once the last thread arrives, all threads are released, and the barrier resets.
        """
        self.cond.acquire()  # Acquire the condition variable's lock.
        self.count_threads -= 1  # Decrement the count of threads yet to arrive.
        if self.count_threads == 0:
            self.cond.notify_all()  # If this is the last thread, wake up all waiting threads.
            self.count_threads = self.num_threads  # Reset the barrier count for next use.
        else:
            self.cond.wait()  # If not the last thread, wait until notified.
        self.cond.release()  # Release the condition variable's lock.


class DeviceThread(Thread):
    """
    Manages the execution lifecycle of a Device. This includes fetching
    neighbor information from a supervisor, waiting for timepoint signals,
    executing assigned scripts, and synchronizing with other devices using
    a conditional barrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        :param device: The Device instance this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device thread.
        It continuously fetches neighbor data, processes scripts in timepoints,
        executes them while acquiring a shared lock, and synchronizes with
        other devices using a conditional barrier.
        """
        # Main loop for continuous device operation across multiple timepoints.
        while True:
            # Block Logic: Fetch neighbor information from the supervisor.
            # If the supervisor returns None, it signals the termination of the simulation.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # Exit the main loop if no more neighbors or simulation ends.

            # Wait for the `timepoint_done` event to be set, indicating that
            # scripts for the current timepoint are ready for execution.
            self.device.timepoint_done.wait()

            # Block Logic: Process and execute assigned scripts for the current timepoint.
            # A shared lock protects the data access and script execution across devices.
            for (script, location) in self.device.scripts:
                self.device.lock.acquire()  # Acquire the global shared lock.
                script_data = []  # Buffer to collect data for the current script.

                # Gather data from neighboring devices.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from the local device itself.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # If there's data to process, run the script.
                if script_data != []:
                    # Execute the script with the collected data.
                    # Invariant: `script` is a callable object with a `run` method.
                    result = script.run(script_data)

                    # Update sensor data on neighboring devices with the script's result.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Update local device's sensor data with the script's result.
                    self.device.set_data(location, result)
                self.device.lock.release()  # Release the global shared lock.
            
            # Reset the `timepoint_done` event for the next timepoint cycle.
            self.device.timepoint_done.clear()
            # Synchronize with all other devices using the shared conditional barrier.
            # Invariant: All devices must reach this point before any can proceed to the next timepoint.
            self.device.barr.wait()
