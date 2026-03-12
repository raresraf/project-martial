


"""
This module implements a simulated distributed device system.

It defines classes for:
- ReusableBarrierSem: A reusable barrier for synchronizing multiple threads in phases.
- Device: Represents a single device in the distributed system, managing its sensor data,
  communication with a supervisor, and multi-threaded script execution. It uses static
  class members for shared resources like location-specific locks and a global barrier.
- DeviceThread: Worker threads for a Device, also performing master-like duties for
  the thread with index 0, including coordinating with the supervisor and distributing
  scripts.
"""

from threading import Lock, Event, Thread, Semaphore, Condition


class ReusableBarrierSem():
    """
    A reusable barrier synchronization mechanism for multiple threads using semaphores.
    This barrier allows a fixed number of threads to wait at a synchronization point,
    and once all threads arrive, they are all released simultaneously. It can then
    be reused for subsequent synchronization points.
    """


    
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate
                               in the barrier synchronization.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all threads have reached this barrier.
        This method orchestrates a two-phase synchronization to ensure reusability
        without deadlocks.
        """
        self.phase1()
        self.phase2()
    def phase1(self):
        """
        The first phase of the barrier synchronization.
        Threads decrement a shared counter and the last thread to reach zero
        releases all waiting threads for this phase.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    def phase2(self):
        """
        The second phase of the barrier synchronization, necessary for reusability.
        Similar to phase1, threads decrement a counter, and the last thread
        releases all waiting threads for this phase.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, interacts with a supervisor,
    and executes scripts in a multi-threaded fashion. This version uses
    static class members to manage global shared resources like
    location-specific locks and a global barrier for all DeviceThreads.
    """
    # List of tuples: (location_id, Lock_object) to protect access to sensor data at each location.
    # This is a static member, meaning all Device instances share the same set of locks.
    location_locks = []
    # A global barrier for synchronizing all DeviceThread instances across all Devices.
    barrier = None
    # The number of worker threads (DeviceThread instances) per Device.
    nr_t = 8
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for the device.
            supervisor (object): A reference to a supervisor object for inter-device communication.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.neighbours_event = Event()
        self.threads = []
        for i in xrange(Device.nr_t):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(Device.nr_t):
            self.threads[i].start()
    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global barrier for all DeviceThreads across all devices.
        This method is typically called once by the supervisor or a designated master device
        during the initialization phase of the distributed system.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        Device.barrier = ReusableBarrierSem(Device.nr_t * len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific data location.
        If the location does not yet have a lock associated with it, a new lock is created.

        Args:
            script (object): The script object to be executed.
            location (str): The location identifier in the sensor data to which the script applies.
        """
        
        if location not in [elem[0] for elem in Device.location_locks]:
            Device.location_locks.append((location, Lock()))

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (str): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location.

        Args:
            location (str): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device and its associated threads.
        Waits for all DeviceThread instances to complete their execution.
        """
        
        for i in xrange(Device.nr_t):
            self.threads[i].join()

class DeviceThread(Thread):
    """
    A worker thread associated with a Device.
    The thread with index 0 also performs master-like duties for its Device,
    such as fetching neighbor information and managing event flags.
    Other threads (index > 0) primarily focus on executing assigned scripts.
    """

    

    def __init__(self, device, index):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device object this thread belongs to.
            index (int): The unique index of this thread among the Device's worker threads.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.index = index
        self.neighbours = None

    def run(self):
        """
        The main execution loop for the DeviceThread.
        This method handles synchronization, fetching neighbor data, script execution,
        and data updates based on the thread's index.
        """
        
        while True:
            # Block Logic: Synchronization and neighbor information retrieval.
            # The thread with index 0 acts as a coordinator, fetching neighbors
            # and signaling other threads. Other threads wait for this signal.
            if self.index == 0:
                self.neighbours = self.device.supervisor.get_neighbours()
                # Inline: Signal to other DeviceThreads in this Device that neighbors are updated.
                self.device.neighbours_event.set()
            else:
                # Inline: Wait for the coordinating thread (index 0) to fetch neighbors.
                self.device.neighbours_event.wait()
                self.neighbours = self.device.threads[0].neighbours
            
            # Pre-condition: 'neighbours' is populated.
            # Invariant: If neighbours is None, it's a termination signal.
            if self.neighbours is None:
                break

            # Block Logic: Wait for the current timepoint to be marked as done for script processing.
            self.device.timepoint_done.wait()

            # Block Logic: Process assigned scripts for this thread.
            # Scripts are distributed among worker threads using a round-robin approach.
            for j in range(self.index, len(self.device.scripts), Device.nr_t):
                location = self.device.scripts[j][1]
                script = self.device.scripts[j][0]

                # Block Logic: Acquire lock for the specific data location to ensure exclusive access.
                for i in range(len(Device.location_locks)):
                    if location == Device.location_locks[i][0]:
                        Device.location_locks[i][1].acquire()

                script_data = []
                # Block Logic: Collect data from neighboring devices.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Block Logic: Collect data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Execute script if data is available and distribute results.
                if script_data != []:
                    # Inline: Execute the script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Distribute results to neighboring devices.
                    for device in self.neighbours:
                        device.set_data(location, result)
                    # Block Logic: Set the result back to the local device's sensor data.
                    self.device.set_data(location, result)

                # Block Logic: Release lock for the specific data location.
                for i in range(len(Device.location_locks)):
                    if location == Device.location_locks[i][0]:
                        Device.location_locks[i][1].release()

            # Block Logic: Synchronize all DeviceThreads across all Devices.
            # All threads wait here until every DeviceThread has completed its script processing for the timepoint.
            Device.barrier.wait()
            # Block Logic: Clear timepoint_done event for the next cycle.
            # Only the thread with index 0 is responsible for clearing this event.
            if self.index == 0:
                self.device.timepoint_done.clear()

            # Block Logic: Clear neighbours_event for the next cycle.
            # Only the thread with index 0 is responsible for clearing this event.
            if self.index == 0:
                self.device.neighbours_event.clear()
            # Block Logic: Another synchronization point to ensure all threads have cleared events
            # or are ready for the next timepoint.
            Device.barrier.wait()
