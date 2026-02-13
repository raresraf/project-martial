"""
This module simulates a distributed system of devices that execute scripts concurrently.

It defines two main classes:
- Child: A thread pool manager that executes tasks for a parent 'Device'. It handles
  a queue of scripts and a set of worker threads to run them.
- Device: Represents a node in the system, which receives scripts, communicates with
  neighboring devices, and uses a 'Child' instance to perform the actual computation.

The system is designed around a timepoint-based synchronization model, where devices
operate in discrete steps, coordinated by a barrier.
"""
from Queue import Queue
from threading import Thread, Event, Lock
from myBarrier import MyBarrier


class Child(object):
    """
    Manages a pool of worker threads for a parent device.

    This class encapsulates a work queue and a fixed number of threads that
    continuously pull tasks from the queue and execute them. It provides a simple
    interface for submitting work and waiting for its completion.
    """

    def __init__(self):
        """
        Initializes the Child thread pool.
        """
        # A queue to hold tasks, with a fixed capacity.
        self.que = Queue(8)
        # A reference to the parent device.
        self.device = None
        # A list to hold the worker Thread objects.
        self.threads = []

        # Create and start the worker threads.
        for _ in xrange(8):
            new_thread = Thread(target=self.run)
            self.threads.append(new_thread)

        for thread in self.threads:
            thread.start()

    def set_device(self, device):
        """Sets the parent device for this Child instance."""
        self.device = device

    def submit(self, neighbours, script, location):
        """
        Submits a new task to the work queue.

        Args:
            neighbours (list): A list of neighboring devices.
            script (Script): The script to be executed.
            location (str): The location context for the script.
        """
        self.que.put((neighbours, script, location))

    def wait(self):
        """Blocks until all items in the queue have been processed."""
        self.que.join()

    def end_threads(self):
        """Shuts down all worker threads gracefully."""
        # First, wait for any remaining tasks to complete.
        self.wait()
        
        # Submit a termination signal (None) for each worker thread.
        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)

        # Wait for all threads to finish execution.
        for thread in self.threads:
            thread.join()

    def run(self):
        """
        The target function for each worker thread.

        It continuously fetches tasks from the queue, executes the script on data
        gathered from the device and its neighbors, and writes the result back.
        """
        # Invariant: The loop continues until a termination signal is received.
        while True:

            (neighbours, script, location) = self.que.get()

            # Pre-condition: A (None, None, None) tuple is the termination signal.
            if neighbours is None and script is None:
                self.que.task_done()
                return

            script_data = []
            
            # Aggregate data from all neighbors for the given location.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    
                    if data is not None:
                        script_data.append(data)

            # Include the local device's data.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            
            # Execute the script only if there is data to process.
            if script_data != []:

                rezult = script.run(script_data)

                # Distribute the result back to the neighbors and the local device.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, rezult)

                self.device.set_data(location, rezult)

            self.que.task_done()


class Device(object):
    """
    Represents a single node in the distributed simulation.

    Each device has a unique ID, local sensor data, and a list of scripts to
    execute. It synchronizes with other devices using a barrier and offloads
    script execution to a Child thread pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (Supervisor): The central controller of the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # An event to signal when a new script has been received.
        self.script_received = Event()

        self.scripts = []
        # An event to signal that the device is done with its work for a timepoint.
        self.timepoint_done = Event()
        
        # The synchronization barrier shared by all devices.
        self.barrier = None
        
        # A dictionary of locks to protect access to data at each location.
        self.data_lock = {location : Lock() for location in sensor_data}
        
        # The main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared barrier for all devices.

        Device 0 is responsible for creating the barrier and distributing it
        to all other devices.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        # Pre-condition: Device 0 acts as the coordinator for barrier setup.
        if self.device_id == 0:
            self.barrier = MyBarrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        Args:
            script (Script): The script to be executed.
            location (str): The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A 'None' script signals the end of the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data for a given location, acquiring a lock.

        @note Asymmetric Locking: This function acquires a lock that is expected
        to be released by `set_data`. This pattern is fragile and can lead to
        deadlocks if `set_data` is not called for every `get_data`.
        """
        if location in self.sensor_data:
            self.data_lock[location].acquire()
            return self.sensor_data[location]
        else:
            return None
        

    def set_data(self, location, data):
        """
        Updates data for a given location, releasing a lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_lock[location].release()

    def shutdown(self):
        """Shuts down the main device thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    Orchestrates the device's operation across timepoints, managing script
    execution and synchronization with other devices.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Creates a Child thread pool for executing scripts.
        self.child_threads = Child()

    def run(self):
        """The main execution loop of the device thread."""
        self.child_threads.set_device(self.device)
        # Invariant: The loop continues as long as the supervisor provides neighbors.
        while True:
            # Get the set of neighbors for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # 'None' neighbors signal the end of the simulation.
                break

            # Invariant: This inner loop processes scripts for the current timepoint.
            while True:
                # Pre-condition: If no script has been received, wait for a signal.
                if not self.device.script_received.isSet():
                    # This waits for either a new script or the end-of-timepoint signal.
                    self.device.timepoint_done.wait()
                    
                    # If still no script after waking, it means end of timepoint.
                    if not self.device.script_received.isSet():
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break
                else:
                    # A script has been received.
                    self.device.script_received.clear()
                    
                    # Submit all currently assigned scripts to the worker pool.
                    for (script, location) in self.device.scripts:
                        self.child_threads.submit(neighbours, script, location)

            
            # Wait for all submitted scripts for this timepoint to complete.
            self.child_threads.wait()

            
            # Synchronize with all other devices at the end of the timepoint.
            self.device.barrier.wait()

        # Cleanly shut down the worker threads.
        self.child_threads.end_threads()
