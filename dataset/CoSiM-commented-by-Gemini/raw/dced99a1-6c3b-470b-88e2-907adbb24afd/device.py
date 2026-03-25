from Queue import Queue
from threading import Semaphore, Lock, Event, Thread


class Device(object):
    """
    Represents a device in a simulated distributed system. Each device has a
    main thread that, in turn, manages a pool of worker threads for executing
    scripts in each timepoint.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The initial sensor data for the device.
            supervisor (object): The supervisor managing the simulation.
        """
        self.device_id = device_id
        self.read_data = sensor_data
        self.supervisor = supervisor
        # A queue to hold scripts to be executed by worker threads.
        self.active_queue = Queue()
        # A temporary list to store scripts for the current timepoint.
        self.scripts = []
        # The main thread for this device.
        self.thread = DeviceThread(self)
        self.time = 0

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for all devices and starts the main thread.
        This method should be called once before the simulation begins.

        Args:
            devices (list): A list of all device objects in the simulation.
        """
        if self.device_id == 0:
            # Device 0 is responsible for creating a shared barrier for all devices.
            self.new_round = ReusableBarrierSem(len(devices))
            self.devices = devices
            for device in self.devices:
                device.new_round = self.new_round
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. When a None script is received, it
        populates the active queue for the worker threads.

        Args:
            script (object): The script to be executed. If None, it signals
                             the end of script assignment for the timepoint.
            location (int): The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # When all scripts for a timepoint are assigned (script is None),
            # move them to the active queue for the workers.
            for (script, location) in self.scripts:
                self.active_queue.put((script, location))
            # Add sentinel values to the queue to signal worker threads to terminate.
            for x in range(8):
                self.active_queue.put((-1, -1))

    def get_data(self, location):
        """
        Retrieves data for a given location.

        Args:
            location (int): The location to query.

        Returns:
            The data at the specified location, or None if not available.
        """
        return self.read_data.get(location)

    def set_data(self, location, data):
        """
        Updates data at a given location.

        Args:
            location (int): The location to update.
            data (any): The new data value.
        """
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        """Shuts down the device by waiting for its main thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It manages the lifecycle of worker
    threads for each simulation timepoint.
    """

    def __init__(self, device):
        """
        Initializes the main device thread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers_number = 8

    def run(self):
        """
        The main loop. For each timepoint, it spawns a new set of worker threads,
        waits for them to complete, and then synchronizes with other devices.
        """
        neighbours = self.device.supervisor.get_neighbours()
        while True:
            self.workers = []
            self.device.neighbours = neighbours
            # A None neighbours list is the signal to terminate.
            if neighbours is None:
                break

            # Block Logic: Create and start a new pool of worker threads for the timepoint.
            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            # Wait for all worker threads to finish their tasks for the timepoint.
            for worker in self.workers:
                worker.join()
            
            # Invariant: All devices synchronize here before starting the next timepoint.
            self.device.new_round.wait()
            # Pre-condition: Get neighbors for the next timepoint.
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):
    """
    A worker thread that executes scripts from the device's active queue.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Continuously fetches and executes scripts from the queue until a
        sentinel value is encountered.
        """
        while True:
            # Get a script from the queue.
            script, location = self.device.active_queue.get()
            # The sentinel value (-1) signals the end of work for this timepoint.
            if script == -1:
                break
            
            script_data = []
            matches = []
            # Gather data from neighboring devices at the specified location.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    matches.append(device)
                    script_data.append(data)
            
            # Include the current device's data.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
                matches.append(self.device)

            # Block Logic: Execute the script and conditionally update data.
            if len(script_data) > 1:
                result = script.run(script_data)
                for device in matches:
                    old_value = device.get_data(location)
                    # The update only occurs if the new result is greater than the old value.
                    if old_value < result:
                        device.set_data(location, result)


class ReusableBarrierSem:
    """
    A reusable barrier implemented using semaphores, for synchronizing a fixed
    number of threads. It uses a two-phase mechanism.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all threads have called this method."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()