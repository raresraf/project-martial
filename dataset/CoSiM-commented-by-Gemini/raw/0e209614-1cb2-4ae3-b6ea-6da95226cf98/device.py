"""
Models a device in a simulated environment for distributed script execution.

This module defines a `Device` class that encapsulates the state and behavior of a single processing
node. Each device runs a main control thread (`DeviceThread`) and a pool of worker threads
(`DeviceWorker`) to execute scripts on sensor data. The system uses synchronization primitives
to coordinate execution across multiple devices, ensuring that they operate in discrete timepoints
and safely access shared location data.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device in the distributed simulation.

    Manages the device's ID, its local sensor data, assigned scripts, and its set of neighbors.
    It orchestrates the execution of scripts through a work queue and a pool of worker threads,
    and synchronizes with other devices using barriers and locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor readings.
            supervisor (Supervisor): An external object responsible for providing neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # A list to store scripts assigned to this device, each with a target location.
        self.scripts = []
        # A queue to hold tasks (script, location) for the worker threads.
        self.work_queue = Queue()
        # A list of neighboring Device objects.
        self.neighbours = []

        # An event to signal that the device has completed its work for the current timepoint.
        self.timepoint_done = Event()
        # An event to signal that the global setup (barriers, locks) is complete.
        self.setup_ready = Event()
        # An event to signal that the neighbors for the current timepoint have been set.
        self.neighbours_set = Event()
        # A semaphore to protect access to the `scripts` list.
        self.scripts_mutex = Semaphore(1)
        # A mutex for creating location-specific locks, shared across all devices.
        self.location_locks_mutex = None
        # A dictionary mapping locations to Lock objects for resource protection.
        self.location_locks = {}
        # A reusable barrier for synchronizing all devices at the end of each timepoint.
        self.timepoint_barrier = None

        # The main control thread for this device.
        self.thread = DeviceThread(self)
        
        # A pool of worker threads to execute scripts.
        self.workers = [DeviceWorker(self) for _ in range(8)]

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs setup and synchronization for a group of devices.

        Initializes the shared timepoint barrier and location locks. Device 0 acts
        as the coordinator for this one-time setup. All threads are started after setup.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Pre-condition: Ensure device 0 acts as the coordinator for one-time setup.
        if self.device_id == 0:
            self.timepoint_barrier = ReusableBarrierCond(len(devices))
            self.location_locks_mutex = Lock()
            self.setup_ready.set() # Signal that the shared objects are ready.
        else:
            # All other devices wait for device 0 to complete setup.
            device = next(device for device in devices if device.device_id == 0)
            device.setup_ready.wait()
            # Copy references to the shared synchronization objects.
            self.timepoint_barrier = device.timepoint_barrier
            self.location_locks = device.location_locks
            self.location_locks_mutex = device.location_locks_mutex

        # Invariant: After this block, all locations this device can access have a corresponding lock.
        with self.location_locks_mutex:
            for location in self.sensor_data.keys():
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()

        # Start the main device thread and all worker threads.
        self.thread.start()
        for worker in self.workers:
            worker.start()


    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        This method is called by the supervisor. It waits until the device has its
        neighbor list for the current timepoint, then adds the work to the queue.
        A `None` script is a signal to terminate the timepoint.

        Args:
            script (Script): The script object to execute.
            location (str): The location context for the script execution.
        """
        # Wait until the device's neighbors for the current step are known.
        self.neighbours_set.wait()

        if script is not None:
            with self.scripts_mutex:
                self.scripts.append((script, location))
            self.work_queue.put((script, location))
        else:
            # A None script signals the end of script assignment for this timepoint.
            self.neighbours_set.clear()
            self.timepoint_done.set() # Signal that this device has no more new work.

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Updates the sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device and its associated threads gracefully."""
        # Wait for the main device thread to finish.
        self.thread.join()
        # Send a termination signal (None, None) to each worker thread.
        for worker in self.workers:
            self.work_queue.put((None, None))

        # Wait for all worker threads to complete.
        for worker in self.workers:
            worker.join()



class DeviceThread(Thread):
    """
    The main control thread for a Device.

    This thread orchestrates the device's lifecycle, managing the progression
    through discrete timepoints. It fetches neighbor information, dispatches
    work from the previous timepoint, and synchronizes with other devices.
    """

    def __init__(self, device):
        """
        Initializes the device's main thread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device

    def run(self):
        """
        The main execution loop for the device's control logic.
        """
        # Invariant: The loop continues as long as the supervisor provides neighbors.
        while True:
            # Fetches the list of neighbors for the upcoming computation step.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # A None response from the supervisor signals the end of the simulation.
            if self.device.neighbours is None:
                break


            # Re-queue all persistent scripts for the new timepoint.
            for (script, location) in self.device.scripts:
                self.device.work_queue.put((script, location))

            
            # Signal to the supervisor that the device is ready to receive new scripts.
            self.device.neighbours_set.set()

            # Wait until the supervisor signals that all scripts for this timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block until all items in the work queue have been processed.
            self.device.work_queue.join()

            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()

            # Synchronize with all other devices, waiting for them to complete the current timepoint.
            self.device.timepoint_barrier.wait()



class DeviceWorker(Thread):
    """
    A worker thread that executes scripts on sensor data.

    Each device has a pool of these threads to process tasks from a shared work queue
    in parallel.
    """

    def __init__(self, device):
        """
        Initializes a worker thread.

        Args:
            device (Device): The parent device this worker belongs to.
        """
        Thread.__init__(self, name="Device %d Worker" % device.device_id)
        self.device = device


    def run(self):
        """
        The main execution loop for the worker.
        """
        # Invariant: The loop continues until a termination signal is received.
        while True:
            # Block and wait for a task from the work queue.
            (script, location) = self.device.work_queue.get(block=True)

            # Pre-condition: A (None, None) tuple is the signal to terminate.
            if script is None and location is None:
                # Mark the task as done before breaking the loop.
                self.device.work_queue.task_done()
                break

            
            
            

            # Acquire a lock for the specific location to ensure exclusive access
            # to the sensor data of this device and its neighbors at that location.
            with self.device.location_locks[location]:
                script_data = []

                # Gather data from all neighbors at the specified location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from the local device itself.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute the script only if there is data to process.
                if script_data != []:
                    result = script.run(script_data)

                    # Broadcast the result back to all neighbors.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    # Update the local device's data with the result.
                    self.device.set_data(location, result)
            
            # Signal that the current task from the queue is complete.
            self.device.work_queue.task_done()
