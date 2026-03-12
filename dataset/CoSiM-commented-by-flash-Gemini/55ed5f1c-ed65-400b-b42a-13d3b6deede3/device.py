


"""
This module implements a simulated distributed device system.

It defines classes for:
- ReusableBarrierSem: A reusable barrier for synchronizing multiple threads in phases.
- Device: Represents a single device in the distributed system, managing its sensor data,
  communication with a supervisor, and multi-threaded script execution.
- DeviceThread: The master thread for a Device, coordinating with the supervisor and
  distributing scripts to worker threads.
- Worker: A thread responsible for executing assigned scripts, collecting data from
  local and neighboring devices, and updating results.
"""


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
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
                for _ in range(self.num_threads):
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
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, interacts with a supervisor,
    and executes scripts in a multi-threaded fashion, coordinating with
    other devices using barriers and locks.
    """

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
        self.lock = {}

        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_done = Event()
        self.terminate = Event()
        self.neighbours = []

        self.barrier = None
        self.threads_barrier = ReusableBarrierSem(9)
        self.master = DeviceThread(self, self.terminate, self.barrier, self.threads_barrier, \
                                    self.setup_done)
        self.master.start()

        self.threads = []

        for _ in range(8):
            thread = Worker(self.master, self.terminate, self.threads_barrier)

            self.threads.append(thread)
            thread.start()

    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up inter-device communication and synchronization mechanisms.
        This method is typically called by the supervisor or a designated master device (device_id 0).

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        

        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                self.lock[dev] = Lock()
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier
                    dev.lock = self.lock
                    dev.setup_done.set()

            self.setup_done.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (str): The location identifier in the sensor data to which the script applies.
        """
        

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
        

        return self.sensor_data[location] if location in self.sensor_data else None

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
        Signals termination and waits for all worker and master threads to complete.
        """
        

        self.terminate.set()
        for i in range(8):
            self.threads[i].script_received.set()
            self.threads[i].join()
        self.master.join()


class DeviceThread(Thread):
    """
    The master thread for a Device. This thread is responsible for overall
    device coordination, including synchronizing with other devices, fetching
    neighbor information from the supervisor, distributing assigned scripts
    to worker threads, and managing timepoint progression.
    """

    def __init__(self, device, terminate, barrier, threads_barrier, setup_done):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent Device object this thread manages.
            terminate (Event): An Event to signal thread termination.
            barrier (ReusableBarrierSem): The global device-level barrier.
            threads_barrier (ReusableBarrierSem): The barrier for synchronizing master and worker threads.
            setup_done (Event): An Event to signal when device setup is complete.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = []
        self.terminate = terminate
        self.barrier = barrier
        self.threads_barrier = threads_barrier
        self.setup_done = setup_done

    def run(self):
        """
        The main execution loop for the master device thread.
        It manages the device's lifecycle from setup, through timepoints,
        to script distribution and eventual termination.
        """
        # Wait until the device's initial setup is complete
        self.setup_done.wait()
        # Synchronize with all other devices at the global barrier after setup
        self.device.barrier.wait()

        # Main loop for processing timepoints and scripts
        while True:
            # Synchronize with other devices before starting a new timepoint
            self.device.barrier.wait()

            # Get updated neighbor information from the supervisor
            self.neighbours = self.device.supervisor.get_neighbours()

            # If no neighbors (or a termination signal from supervisor), break the loop
            if self.neighbours is None:
                break

            # Wait for the current timepoint's data processing to be conceptually done
            self.device.timepoint_done.wait()
            # Reset the timepoint_done event for the next timepoint
            self.device.timepoint_done.clear()
            # Synchronize with other devices after timepoint processing
            self.device.barrier.wait()

            # Prepare list of scripts to distribute among worker threads
            scripts = []
            for i in range(8): # There are 8 worker threads
                scripts.append([])

            # Distribute scripts in a round-robin fashion to worker threads
            for i in range(len(self.device.scripts)):
                scripts[i%8].append(self.device.scripts[i])

            # Assign scripts to each worker thread and signal them to start processing
            for i in range(8):
                self.device.threads[i].scripts = scripts[i]
                self.device.threads[i].script_received.set()

            # If not terminating, wait for all worker threads to complete their tasks
            if not self.terminate.is_set():
                self.threads_barrier.wait()


class Worker(Thread):
    """
    A worker thread associated with a Device. Workers are responsible for
    executing individual scripts, collecting necessary data from the local
    device and its neighbors, performing computations, and updating the
    sensor data based on script results.
    """

    def __init__(self, master, terminate, barrier):
        """
        Initializes a Worker thread.

        Args:
            master (DeviceThread): Reference to the master DeviceThread.
            terminate (Event): An Event to signal thread termination.
            barrier (ReusableBarrierSem): The barrier for synchronizing with the master thread.
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
        A static method to safely append data from a device's sensor data
        to a script's input data list, using the device's specific lock.

        Args:
            device (Device): The device from which to retrieve data.
            location (str): The data location to retrieve.
            script_data (list): The list to which the retrieved data will be appended.
        """
        
        device.lock[device].acquire()
        data = device.get_data(location)
        device.lock[device].release()
        if data is not None:
            script_data.append(data)

    @staticmethod
    def set_data(device, location, result):
        """
        A static method to safely set data on a device's sensor data
        at a specific location, using the device's specific lock.

        Args:
            device (Device): The device on which to set data.
            location (str): The data location to update.
            result (any): The new data value to set.
        """
        
        device.lock[device].acquire()
        device.set_data(location, result)
        device.lock[device].release()

    def run(self):
        """
        The main execution loop for the worker thread.
        It continuously waits for scripts, executes them, and updates data,
        until a termination signal is received.
        """
        while True:
            # Wait for scripts to be assigned by the master thread
            self.script_received.wait()
            # Clear the event for the next script assignment
            self.script_received.clear()

            # Check for termination signal
            if self.terminate.is_set():
                break # Exit the loop if termination is requested

            # If scripts are assigned, process them
            if self.scripts is not None:
                for (script, location) in self.scripts:
                    # Initialize list to collect data for the current script
                    script_data = []
                    # If the master device has neighbors, collect data from them
                    if self.master.neighbours is not None:
                        for device in self.master.neighbours:
                            self.append_data(device, location, script_data)

                    # Also collect data from the master device itself
                    self.append_data(self.master.device, location, script_data)

                    # If data was collected, run the script and distribute results
                    if script_data != []:
                        # Execute the script with the collected data
                        result = script.run(script_data)

                        # If the master device has neighbors, distribute results to them
                        if self.master.neighbours is not None:
                            for device in self.master.neighbours:
                                self.set_data(device, location, result)

                        # Distribute results to the master device itself
                        self.set_data(self.master.device, location, result)

            # Synchronize with the master thread after processing all assigned scripts
            self.barrier.wait()
