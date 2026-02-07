
"""
This module defines the core components for a distributed device simulation.

It includes classes for managing device state, inter-device communication,
and concurrent script execution within a simulated environment. The system
is designed around a threaded model where each device operates concurrently
and interacts with its neighbors.
"""

import Queue
from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

class ScriptThread(Thread):
    """
    A worker thread for executing scripts on a device.

    Each ScriptThread is associated with a single device and processes scripts
    from its parent device's script queue. It handles data aggregation from
    neighboring devices, script execution, and result propagation.
    """

    def __init__(self, device):
        """
        Initializes the script thread.

        Args:
            device: The parent device instance that this thread belongs to.
        """
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop for the script-processing thread.

        It continuously fetches scripts from the queue, gathers necessary data,
        executes the script, and distributes the results. The loop terminates
        when a special sentinel value (None, None) is received.
        """
        while True:

            # Dequeue a script and its target location. This is a blocking call.
            (script, location) = self.device.scripts_queue.get()

            # Sentinel check: If a (None, None) tuple is received, it signals
            # the thread to terminate. The sentinel is put back in the queue
            # to signal other ScriptThreads to terminate as well.
            if (script, location) == (None, None):
                self.device.scripts_queue.put((None, None))
                break


            script_data = []

            # Acquire a lock for the specific data location to ensure atomicity
            # of data gathering and updates.
            with self.device.lcks[location]:
                # Data Aggregation: Collect data from all neighboring devices
                # for the specified location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Collect data from the local device itself.
                data = self.device.get_data(location)

                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Execute the script with the aggregated data.
                    result = script.run(script_data)
                    
                    # Result Propagation: Distribute the outcome of the script
                    # execution back to all neighbors and the local device.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

            # Mark the task as done in the queue.
            self.device.scripts_queue.task_done()


class Device(object):
    """
    Represents a single device in the distributed simulation.

    Each device has a unique ID, local sensor data, and a list of neighbors.
    It manages a pool of ScriptThreads to execute tasks concurrently and
    communicates with a central supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data.
            supervisor: The central supervisor managing the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.start_scripts = Event()
        
        # A reusable barrier for synchronizing devices at the end of a timepoint.
        self.timepoint_done = ReusableBarrierCond(0)
        
        self.used_barrier = False
        
        self.neighbours = []

        # Queue for pending scripts to be executed by the thread pool.
        self.scripts_queue = Queue.Queue()
        
        # A dictionary of locks, one for each data location, to prevent race conditions.
        self.lcks = {}

        # The main thread for this device, handling communication and coordination.
        self.thread = DeviceThread(self)
        
        self.thread.start()

        # A pool of worker threads for executing scripts.
        self.thread_pool = []
        
        self.init_thread_pool(self.thread_pool)


    def init_thread_pool(self, pool):
        """
        Initializes and starts a pool of ScriptThreads for the device.

        Args:
            pool (list): The list to be populated with ScriptThread instances.
        """

        # Create a fixed-size pool of 8 script-executing threads.
        for i in xrange(8):
            thread = ScriptThread(self)
            pool.append(thread)

        # Start all the threads in the pool.
        for i in xrange(len(pool)):
            pool[i].start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources (locks and barriers) among a group of devices.

        This method ensures that all devices in the simulation share the same
        lock objects for corresponding data locations and a common barrier for
        synchronization.

        Args:
            devices (list): A list of all device instances in the simulation.
        """

        # For each piece of sensor data, create a shared lock and distribute it
        # to all other devices if it doesn't already exist.
        for i in self.sensor_data.keys():
            
            if self.lcks.has_key(i) is False:
                
                self.lcks[i] = Lock()
                
                for j in xrange(len(devices)):
                    if devices[j].device_id != self.device_id:
                        devices[j].lcks[i] = self.lcks[i]

        # Initialize and share a ReusableBarrierCond for synchronization.
        # This is done only once per group of devices.
        if self.used_barrier is False:
            
            self.timepoint_done.count_threads = len(devices)
            self.timepoint_done.num_threads = len(devices)
            
            for i in xrange(len(devices)):
                devices[i].used_barrier = True
                if devices[i].device_id != self.device_id:
                    devices[i].timepoint_done = self.timepoint_done


    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        If the script is not None, it is added to a temporary list. If the script
        is None, it signals that all scripts for the current timepoint have been
        received.

        Args:
            script: The script object to be executed.
            location: The data location the script will operate on.
        """
        if script is not None:
            # Append the script to a list for the current timepoint.
            self.scripts.append((script, location))
        else:
            # Signal that all scripts for this cycle have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves data from a specific location on the device.

        Args:
            location: The key for the desired data.

        Returns:
            The data if the location exists, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates data at a specific location on the device.

        Args:
            location: The key for the data to be updated.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Gracefully shuts down the device and its associated threads.
        """

        # Wait for all script-processing threads to complete.
        for i in xrange(len(self.thread_pool)):
            self.thread_pool[i].join()

        # Wait for the main device thread to complete.
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device.

    This thread is responsible for managing the device's lifecycle, including
    fetching neighbor information, waiting for scripts to be assigned, queueing
    them for execution, and synchronizing with other devices.
    """

    def __init__(self, device):
        """
        Initializes the main device thread.

        Args:
            device: The parent device instance.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device

    def run(self):
        """
        The main loop for the device thread.

        Orchestrates the device's operation in each simulation timepoint.
        It terminates when the supervisor indicates no more neighbors (end of simulation).
        """
        while True:

            # Get the current list of neighbors from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # If get_neighbours returns None, it's a signal to shut down.
            # A sentinel is placed in the scripts queue to terminate worker threads.
            if self.device.neighbours is None:
                self.device.scripts_queue.put((None, None))
                break

            # Wait until all scripts for the current timepoint are assigned.
            self.device.script_received.wait()


            self.device.script_received.clear()

            # Enqueue all assigned scripts for execution by the ScriptThread pool.
            for (script, location) in self.device.scripts:
                self.device.scripts_queue.put((script, location))

            # Block until all scripts in the queue are processed.
            self.device.scripts_queue.join()
            
            # If a barrier is being used, wait here for all other devices
            # to finish their work for the current timepoint.
            if self.device.used_barrier is True:
                self.device.timepoint_done.wait()
