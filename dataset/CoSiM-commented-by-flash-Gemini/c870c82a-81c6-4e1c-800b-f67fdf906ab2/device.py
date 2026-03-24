"""
This module defines a distributed simulation framework with a master-worker thread model
for each simulated device, utilizing a reusable semaphore-based barrier for synchronization.

It includes:
- ReusableBarrierSem: A semaphore-based reusable barrier for thread synchronization.
- Device: Represents a simulated device with sensor data and script execution capabilities.
- DeviceThread: The master thread for a Device, responsible for coordination and task distribution.
- Worker: Worker threads for a Device, executing scripts concurrently.
"""


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    A reusable barrier synchronization primitive implemented using semaphores.
    It coordinates multiple threads, ensuring all threads arrive at the barrier
    before any are allowed to proceed. Supports two phases for reusability.
    """
    

    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrierSem with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that must reach the barrier.
        """

        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all threads have reached this point.
        This method orchestrates the two phases of the barrier.
        """

        
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        First phase of the barrier. Threads decrement a counter, and the last thread
        to reach the barrier releases all waiting threads for phase 1 by releasing semaphores.
        """

        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """
        Second phase of the barrier. Similar to phase 1, threads decrement a counter,
        and the last thread to reach releases all waiting threads for phase 2.
        This completes one full cycle of the reusable barrier.
        """

        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a simulated computational device in a distributed system.
    Each device has a unique ID, sensor data, and interacts with a supervisor.
    It manages a master thread (DeviceThread) and multiple worker threads (Worker)
    to execute scripts and process data, utilizing various synchronization primitives.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping data locations to initial sensor readings.
            supervisor (Supervisor): The supervisor object responsible for coordinating devices.
        """

        
        self.device_id = device_id


        self.sensor_data = sensor_data
        self.lock = {}

        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_done = Event()
        self.terminate = Event()
        self.neighbours = []

        self.barrier = None
        # Block Logic: Initializes a reusable barrier for synchronizing the master thread and 8 worker threads.
        self.threads_barrier = ReusableBarrierSem(9) # 1 master + 8 workers
        # Block Logic: Initializes and starts the master thread (DeviceThread) for this device.
        # It passes essential synchronization objects and events.
        self.master = DeviceThread(self, self.terminate, self.barrier, self.threads_barrier, \
                                    self.setup_done)
        self.master.start()

        self.threads = []

        # Block Logic: Initializes and starts 8 worker threads for this device.
        # Each worker is linked to the master thread and shares termination and synchronization objects.
        for _ in range(8):
            thread = Worker(self.master, self.terminate, self.threads_barrier)

            self.threads.append(thread)
            thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device {device_id}".
        """

        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the shared barrier and locks for all devices.
        This method is designed to be called by a coordinating entity (e.g., supervisor).
        Only the device with device_id 0 initializes the global barrier and lock system.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """

        

        # Block Logic: Only device with ID 0 acts as the coordinator to set up shared resources.
        if self.device_id == 0:
            # Initializes a global barrier for all devices in the simulation.
            self.barrier = ReusableBarrierSem(len(devices))
            # Initializes a dictionary of locks, one for each device, to manage access.
            for dev in devices:
                self.lock[dev] = Lock()
            # Distributes the global barrier and shared locks to all other devices.
            # Signals that setup is complete for each device.
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier
                    dev.lock = self.lock
                    dev.setup_done.set()

            # Signals that setup is complete for the coordinating device itself.
            self.setup_done.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific data location on this device.
        Signals either that a new script is available or that the timepoint is done.

        Args:
            script (object): The script object to be executed, or None to signal timepoint completion.
            location (int): The identifier for the data location where the script should run.
        """

        

        # Block Logic: If a script is provided, it's added to the device's script list
        # and a signal is set to indicate new scripts are available.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        # If no script is provided (script is None), it signals that the current timepoint is done.
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (int): The identifier for the data location.

        Returns:
            Any: The sensor data at the specified location, or None if the location is not found.
        """

        

        # Block Logic: Returns the sensor data for the given location if it exists, otherwise returns None.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location. Data is updated only if the location exists.

        Args:
            location (int): The identifier for the data location.
            data (Any): The new sensor data to set.
        """

        
        # Block Logic: Updates sensor data for the given location if it exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device gracefully by signaling termination to all threads
        and waiting for them to complete their execution.
        """

        

        # Block Logic: Signals all threads to terminate their execution.
        self.terminate.set()
        # Block Logic: For each worker thread, signals that a script has been received (to wake it up if waiting)
        # and then waits for the thread to complete its execution.
        for i in range(8):
            self.threads[i].script_received.set() # Wake up worker if it's waiting
            self.threads[i].join()
        # Block Logic: Waits for the master thread to complete its execution.
        self.master.join()


class DeviceThread(Thread):
    """
    The master thread for a Device. It is responsible for orchestrating
    the simulation workflow for its device, including coordinating with
    the supervisor, synchronizing timepoints, and distributing scripts to worker threads.
    """
    

    def __init__(self, device, terminate, barrier, threads_barrier, setup_done):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device instance this thread is associated with.
            terminate (Event): An Event to signal thread termination.
            barrier (ReusableBarrierSem): The global barrier for device-level synchronization.
            threads_barrier (ReusableBarrierSem): The internal barrier for master-worker synchronization.
            setup_done (Event): An Event to signal completion of device setup.
        """

        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = []
        self.terminate = terminate
        # The global barrier for device-level synchronization.
        self.barrier = barrier
        # The internal barrier for master-worker synchronization within this device.
        self.threads_barrier = threads_barrier
    def run(self):
        """
        The main execution loop for the DeviceThread (master thread).
        It orchestrates the simulation workflow, including waiting for setup,
        synchronizing with other devices, fetching neighbor information,
        and distributing scripts to worker threads.
        """



        # Block Logic: Waits for the device's setup to be completed.
        self.setup_done.wait()
        # Block Logic: Synchronizes with other devices at the global barrier after setup.
        self.device.barrier.wait()

        while True:
            # Block Logic: Synchronizes with other devices at the global barrier for the start of a timepoint.
            self.device.barrier.wait()

            # Block Logic: Fetches the latest neighbor information from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: If no neighbors (e.g., simulation termination signal), break the loop.
            if self.neighbours is None:
                break

            # Block Logic: Waits for a signal that a new timepoint has begun and scripts are assigned.
            self.device.timepoint_done.wait()
            # Clears the event for the next timepoint.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes with other devices at the global barrier before script distribution.
            self.device.barrier.wait()

            # Block Logic: Distributes the assigned scripts among the 8 worker threads.
            scripts = []
            for i in range(8):
                scripts.append([])

            for i in range(len(self.device.scripts)):
                scripts[i%8].append(self.device.scripts[i])

            for i in range(8):
                self.device.threads[i].scripts = scripts[i]
                # Signals each worker thread that new scripts are available.
                self.device.threads[i].script_received.set()

            # Block Logic: If not terminating, waits for all worker threads to complete their assigned scripts
            # before proceeding to the next timepoint.
            if not self.terminate.is_set():
                self.threads_barrier.wait()


class Worker(Thread):
    """
    A worker thread for a Device, managed by the DeviceThread (master).
    Workers are responsible for executing assigned scripts at specific locations,
    gathering data from the device and its neighbors, and propagating results.
    """
    

    def __init__(self, master, terminate, barrier):
        """
        Initializes a new Worker thread.

        Args:
            master (DeviceThread): The master DeviceThread this worker is associated with.
            terminate (Event): An Event to signal thread termination.
            barrier (ReusableBarrierSem): The internal barrier for master-worker synchronization.
        """


        Thread.__init__(self)
        self.master = master
        self.script_received = Event()
        self.terminate = terminate
        self.scripts = []
        self.barrier = barrier

    @staticmethod
    def append_data(device, location, script_data):
        """
        Appends sensor data from a specified device and location to a list,
        with proper locking to ensure data consistency.

        Args:
            device (Device): The device from which to get data.
            location (int): The data location to retrieve.
            script_data (list): The list to append the retrieved data to.
        """

        
        # Block Logic: Acquires a lock specific to the device to ensure exclusive access
        # while retrieving data from its sensor_data.
        device.lock[device].acquire()
        data = device.get_data(location)
        device.lock[device].release()
        # If data is successfully retrieved, append it to the script_data list.
        if data is not None:
            script_data.append(data)

    @staticmethod
    def set_data(device, location, result):
        """
        Sets sensor data for a specific location on a device, with proper locking
        to ensure data consistency.

        Args:
            device (Device): The device on which to set data.
            location (int): The data location to update.
            result (Any): The new data value to set.
        """

        
        # Block Logic: Acquires a lock specific to the device to ensure exclusive access
        # while setting data in its sensor_data.
        device.lock[device].acquire()
        device.set_data(location, result)
        device.lock[device].release()

    def run(self):
        """
        The main execution loop for the Worker thread.
        It waits for script assignments, executes them, and then synchronizes
        with other workers via a barrier.
        """

        while True:
            # Block Logic: Waits for a signal that new scripts have been assigned.
            self.script_received.wait()
            self.script_received.clear() # Clears the event for the next cycle.

            # Block Logic: If the termination signal is set, the worker thread breaks its loop and exits.
            if self.terminate.is_set():
                break
            # Block Logic: Processes assigned scripts if available.
            if self.scripts is not None:
                for (script, location) in self.scripts:
                    # Block Logic: Collects data relevant to the script from neighbors and the local device.
                    script_data = []
                    if self.master.neighbours is not None:
                        # Collects data from each neighboring device.
                        for device in self.master.neighbours:
                            self.append_data(device, location, script_data)

                    # Collects data from the master's associated device.
                    self.append_data(self.master.device, location, script_data)

                    # Block Logic: If data was collected, executes the script and propagates the result.
                    if script_data != []:
                        # Executes the script with the gathered data.
                        result = script.run(script_data)

                        if self.master.neighbours is not None:
                            # Propagates the result to neighboring devices.
                            for device in self.master.neighbours:
                                self.set_data(device, location, result)
                        # Propagates the result to the master's associated device.
                        self.set_data(self.master.device, location, result)

            # Block Logic: Synchronizes with other worker threads via an internal barrier
            # after processing scripts for the current timepoint.
            self.barrier.wait()
