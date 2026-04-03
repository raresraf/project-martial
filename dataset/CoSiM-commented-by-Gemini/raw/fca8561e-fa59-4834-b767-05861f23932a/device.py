
"""
This module defines a simulated Device for a distributed system, likely for a
sensor network or a parallel computing environment. It includes the Device class
itself, a DeviceThread for managing the device's lifecycle, and a WorkerThread
for executing tasks.
"""

from threading import Event, Thread, Lock
from Queue import Queue
from barrier import ReusableBarrier

class Device(object):
    """
    Represents a single device in the simulated distributed system.

    Each device has a unique ID, its own sensor data, and a connection to a
    supervisor. It manages a thread (DeviceThread) to handle its operations and
    can be assigned scripts to execute on its data. It uses synchronization
    primitives like Events and Locks to coordinate with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: A dictionary representing the device's sensor data.
            supervisor: A supervisor object that manages the device network.
        """
        self.thread_number = 8
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        # Events and locks for synchronization.
        self.timepoint_done = Event()
        self.location_locks = {}
        self.devices_barrier = None
        self.setup_devices_done = Event()
        self.neighbours = []
        
        # The main thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's knowledge of other devices in the system,
        including a shared barrier and locks for synchronization.
        
        This method is typically called by the master device (device_id 0).
        """
        if self.device_id is 0:
            # The master device creates the shared barrier and master lock.
            self.devices_barrier = ReusableBarrier(len(devices))
            self.location_locks["master_lock"] = Lock()
            for dev in devices:
                if dev.device_id != self.device_id:
                    # Share the barrier and locks with other devices.
                    dev.devices_barrier = self.devices_barrier
                    dev.location_locks = self.location_locks
                # Signal that the device setup is complete for this device.
                dev.setup_devices_done.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific location.
        If the script is None, it signals that the timepoint is done.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.

        Returns:
            The data at the given location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Sets the sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control loop for a Device.

    This thread manages a pool of WorkerThreads. It gets a list of neighbours
    from the supervisor, waits for scripts to be assigned, puts them in a queue,
    and then waits for all devices in the simulation to complete the timepoint
    using a ReusableBarrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The Device instance that this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = []
        self.script_queue = Queue()
        self.worker_pool = []
        # Create a pool of worker threads to execute scripts concurrently.
        for _ in range(self.device.thread_number):
            self.worker_pool.append(WorkerThread(self))

    def run(self):
        """The main execution loop for the device thread."""
        # Start all worker threads.
        for worker in self.worker_pool:
            worker.start()

        while True:
            # Get the list of neighboring devices from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                # If the supervisor signals no more neighbors, break the loop to shutdown.
                break

            # Wait until the device setup is complete.
            self.device.setup_devices_done.wait()
            # Wait for the signal that a timepoint (a unit of simulation time) is done.
            self.device.timepoint_done.wait()

            # Add all assigned scripts for the current timepoint to the queue.
            for (script, location) in self.device.scripts:
                self.script_queue.put((script, location))

            # Wait for all scripts in the queue to be processed by the workers.
            self.script_queue.join()
            # Wait at the barrier for all other devices to finish their timepoint.
            self.device.devices_barrier.wait()
            # Clear the timepoint_done event to prepare for the next timepoint.
            self.device.timepoint_done.clear()
        
        # Send a signal to each worker thread to terminate.
        for _ in range(len(self.worker_pool)):
            self.script_queue.put(None)

        # Wait for all worker threads to finish.
        for worker in self.worker_pool:
            worker.join()

class WorkerThread(Thread):
    """
    Executes scripts in a separate thread to allow for parallel execution.
    
    Workers pull a script and a location from the queue, acquire a lock for that
    location to ensure data consistency, gather data from the device and its
    neighbors at that location, run the script, and then update the data on the
    device and all its neighbors with the result.
    """

    def __init__(self, device_thread):
        Thread.__init__(self)
        self.device_thread = device_thread

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            # Get a script and location from the queue.
            script_pair = self.device_thread.script_queue.get()

            # If the script is None, it's a signal to terminate.
            if script_pair is None:
                break

            script, location = script_pair

            # Acquire the master lock to safely check for and create a location-specific lock.
            with self.device_thread.device.location_locks["master_lock"]:
                if location not in self.device_thread.device.location_locks:
                    self.device_thread.device.location_locks[location] = Lock()

            # Acquire the lock for the specific location to prevent race conditions.
            self.device_thread.device.location_locks[location].acquire()

            script_data = []
            
            # Gather data from all neighboring devices at the specified location.
            for device in self.device_thread.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from the current device.
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)
                
            # If there is data to process, run the script.
            if script_data:
                result = script.run(script_data)

                # Update the data on all neighboring devices with the result.
                for device in self.device_thread.neighbours:
                    device.set_data(location, result)
                
                # Update the data on the current device.
                self.device_thread.device.set_data(location, result)

            # Signal that the task is done.
            self.device_thread.script_queue.task_done()
            # Release the lock for the location.
            self.device_thread.device.location_locks[location].release()
