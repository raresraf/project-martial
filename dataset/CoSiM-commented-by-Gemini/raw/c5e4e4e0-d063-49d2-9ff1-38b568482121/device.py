
"""
@file raw/c5e4e4e0-d063-49d2-9ff1-38b568482121/device.py
@brief Implements a simulation of a distributed network of devices that process
       sensor data in synchronized time steps.

This module defines a system for simulating a network of devices that operate on
shared sensor data. The key components are:
- A reusable barrier (`ReusableBarrierSem`) to synchronize all devices at the
  beginning of each time step.
- A `Device` class representing a node in the network, which manages its own
  sensor data and executes scripts.
- A `DeviceThread` that contains the main execution loop for each device,
  orchestrating synchronization and data processing.
- A `split_work` function that performs the actual data aggregation and script
  execution for a specific location, using locks to ensure data consistency
  across the network.
"""
import Queue
from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    A reusable barrier implemented using semaphores, allowing a set of threads
    to synchronize at a point in execution multiple times.

    This is a classic two-phase barrier. Threads wait in `phase1`. Once all
    threads have arrived, they are all released. `phase2` ensures that no
    thread starts the next `phase1` cycle until all threads have completed
    the first phase, preventing race conditions.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must reach the
                               barrier before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0) # Controls entry to the first phase
        self.threads_sem2 = Semaphore(0) # Controls entry to the second phase

    def wait(self):
        """
        Causes a thread to wait at the barrier. All participating threads must
        call this method to proceed.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread arrived, release all waiting threads for this phase
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use

        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase to ensure all threads have passed phase1 before reset."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread arrived, release all for the full barrier completion
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use

        self.threads_sem2.acquire()

class MyThread(Thread):
    """
    A simple example thread class to demonstrate the usage of the
    ReusableBarrierSem. This does not appear to be part of the main
    device simulation logic.
    """

    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier

    def run(self):
        """The thread's execution logic."""
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "
",


class Device(object):
    """
    Represents a single device (node) in the distributed simulation network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping location IDs to sensor
                                values for this device.
            supervisor: An object responsible for providing network topology
                        (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.barrier = None
        self.timepoint_done = Event() # Signals that scripts for a time step are assigned
        self.lock_data = Lock()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.all_devices = None
        self.locations = [] # List of all unique sensor locations in the system
        self.lock_locations = [] # List of locks, one for each unique location


    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global synchronization context for the entire device network.
        This method is intended to be called on a single "master" device (e.g., id 0).
        It creates and distributes the shared barrier and location-specific locks
        to all devices in the network.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Collect all unique locations from this device's sensor data
        for location in range(len(self.sensor_data)):
            if self.sensor_data.get(location) is not None:
                if location not in self.locations:
                    self.locations.append(location)


        self.all_devices = devices

        # Aggregate all unique locations from all devices in the network
        for device in self.all_devices:
            for location in device.locations:
                if location not in self.locations:
                    self.locations.append(location)



        self.locations.sort()

        # The device with ID 0 acts as the master for setting up synchronization
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))

            # Create one lock for each unique sensor location
            for _ in self.locations:
                lock = Lock()
                self.lock_locations.append(lock)

            # Distribute the shared barrier and location locks to all devices
            for device in self.all_devices:
                device.set_barrier(self.barrier)
                device.set_lock_locations(self.lock_locations)


    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location.

        Args:
            script: The script object with a `run` method.
            location (int): The location ID for the data processing.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is a sentinel indicating the end of script assignment for a timepoint
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location for this device."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_barrier(self, barrier):
        """Sets the shared barrier object for this device."""
        self.barrier = barrier

    def set_lock_locations(self, lock_locations):
        """Sets the shared list of location-specific locks."""
        self.lock_locations = lock_locations

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location on this device.
        This provides the mechanism for scripts to propagate their results.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            

    def shutdown(self):
        """Joins the device's thread to cleanly shut down."""
        self.thread.join()


class DeviceThread(Thread):
    """The main worker thread for a Device."""

    def __init__(self, device):
        """Initializes the thread for a given device."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device simulation.
        This loop represents the progression of discrete time steps.
        """
        while True:
            # Supervisor provides network topology for the current time step
            neighbours = self.device.supervisor.get_neighbours()
            self.device.lockForLocations = []
            if neighbours is None:
                break # Simulation ends when supervisor provides no neighbours

            # --- Synchronization Point 1: Start of Time Step ---
            # All devices wait here until every device is ready for the new time step.
            self.device.barrier.wait()

            # --- Synchronization Point 2: Script Assignment ---
            # Waits until the supervisor has finished assigning all scripts for this step.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Group assigned scripts by their target location to be processed in parallel.
            queue_list = []
            index_list = []

            for item in self.device.scripts:

                (_, location) = item

                if location not in index_list:
                    index_list.append(location)
                    temp_queue = Queue.Queue()
                    temp_queue.put(item)
                    queue_list.append(temp_queue)
                else:
                    index = index_list.index(location)
                    queue_list[index].put(item)


            # Create and start a worker thread for each location that has scripts.
            th_list = []

            for queue in queue_list:
                worker = Thread(target=split_work, args=(self.device, neighbours, queue, ))
                worker.setDaemon(True)
                th_list.append(worker)
                worker.start()

            # Wait for all per-location worker threads to complete.
            for thr in th_list:
                thr.join()


def split_work(device, neighbours, queue_param):
    """
    Worker function that processes all scripts for a single location.
    This function is executed in its own thread.

    Args:
        device (Device): The parent device object.
        neighbours (list): A list of neighboring Device objects.
        queue_param (Queue.Queue): A queue containing all scripts for one location.
    """
    while True:
        try:
            # Dequeue a script to execute
            (script, location) = queue_param.get(False)
        except Queue.Empty:
            # No more scripts for this location in this time step
            break
        else:
            if location in device.locations:
                # --- Critical Section: Per-Location Data Processing ---
                # Acquire a lock specific to this location to ensure that only one
                # thread in the entire distributed system can modify data for this
                # location at a time.
                device.lock_locations[location].acquire()
                
                script_data = []
                
                # Gather data for the current location from all neighbors.
                for device_temp in neighbours:
                    data = device_temp.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data for the current location from the device itself.
                data = device.get_data(location)

                if data is not None:
                    script_data.append(data)

                # Only run the script if there is data to process.
                if script_data != []:
                    
                    # Execute the script on the aggregated data.
                    result = script.run(script_data)
                    
                    # --- Data Broadcast ---
                    # Update the data on all neighbors with the new result.
                    for device_temp in neighbours:
                        device_temp.set_data(location, result)
                    
                    # Update the data on the local device.
                    device.set_data(location, result)
                
                queue_param.task_done()
                device.lock_locations[location].release()
