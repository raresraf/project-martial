from threading import Event, Thread, Lock, Semaphore


class ReusableBarrierSem():
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This barrier uses a two-phase semaphore system to ensure that threads can
    wait at the barrier multiple times. Threads are blocked until the last
    thread arrives, at which point all are released.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that will synchronize.
        """
        self.num_threads = num_threads
        # Counters for each phase.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        # Semaphores for blocking/releasing in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread releases all threads waiting in this phase.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset counter for the next use.
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """The second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread releases all threads waiting in this phase.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset counter for the next use.
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a computational device in a distributed network.

    Each device maintains its own sensor data, executes scripts, and
    communicates with neighboring devices. It operates on a fixed pool of
    worker threads and uses a complex synchronization scheme involving
    barriers, locks, and events to coordinate with other devices.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and starts its worker threads.

        Args:
            device_id (int): Unique ID for the device.
            sensor_data (dict): The local data store for the device.
            supervisor (object): The central supervisor managing the devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        self.barrier = None
        # A list of locks for fine-grained, location-based synchronization.
        self.lock_location = None
        # A lock to protect access to the shared script list.
        self.lock_script = Lock()
        # A lock to manage updates to the list of neighbors.
        self.lock_neighbours = Lock()
        # A list of flags indicating which scripts are available to be run.
        self.available = []
        self.neighbours = None
        # An event to signal that the centralized initialization is complete.
        self.init_done = Event()
        # A flag to trigger neighbor list updates.
        self.update_neighbours = True
        
        # Create and start a fixed pool of 8 worker threads.
        self.threads = []
        for i in range(8):
            self.threads.append(DeviceThread(self))
            self.threads[i].start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization objects for a group of devices.

        This method implements a centralized setup where device_id 0 is
        responsible for creating the shared barrier and location locks. Other
        devices wait for device 0 to finish and then copy its references.

        Args:
            devices (list): The list of all devices in the system.
        """
        if self.device_id == 0:
            # Device 0 acts as the primary initializer.
            self.barrier = ReusableBarrierSem(len(devices))
            # Create a lock for each of the 200 possible data locations.
            self.lock_location = [Lock() for _ in range(200)]
            self.init_done.set()
        else:
            # Other devices find device 0 and wait for it to complete init.
            for device in devices:
                if device.device_id == 0:
                    device.init_done.wait()
                    # Copy references to shared objects.
                    self.barrier = device.barrier
                    self.lock_location = device.lock_location
                    return

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals the end of a timepoint.

        Args:
            script (object): The script to run, or None to signal end of timepoint.
            location (any): The data location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.available.append(True)
        else:
            # A None script is a signal from the supervisor.
            # 1. All devices synchronize at the barrier.
            self.barrier.wait()
            # 2. Reset internal state for the next timepoint.
            self.reset()
            # 3. Signal to waiting worker threads that the timepoint is done.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Ensures all worker threads have completed."""
        for thread in self.threads:
            if thread.isAlive():
                thread.join()

    def reset(self):
        """Resets the device's state for the next timepoint."""
        with self.lock_neighbours:
            self.update_neighbours = True
        # Mark all scripts as available for the worker threads to claim.
        for i in range(len(self.available)):
            self.available[i] = True


class DeviceThread(Thread):
    """
    A worker thread for a Device.

    Multiple instances of this thread run concurrently, processing scripts
    from the parent device's shared script list.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop for the worker thread.
        
        It dynamically claims available scripts, executes them, and synchronizes
        at the end of each computational step (timepoint).
        """
        while True:
            # Lazily update the list of neighbors if needed.
            with self.device.lock_neighbours:
                if self.device.update_neighbours:
                    self.device.neighbours = self.device.supervisor.get_neighbours()
                    self.device.update_neighbours = False
            
            neighbours = self.device.neighbours
            if neighbours is None:
                # Supervisor signals shutdown by providing None for neighbors.
                break

            # Iterate through all scripts assigned to the parent device.
            for (script, location) in self.device.scripts:
                # Atomically check and claim an available script.
                with self.device.lock_script:
                    index = self.device.scripts.index((script, location))
                    if not self.device.available[index]:
                        # If script is already taken by another thread, skip it.
                        continue
                    self.device.available[index] = False

                # Acquire the lock for this specific data location to ensure
                # that only one thread across the entire system can modify it.
                with self.device.lock_location[location]:
                    script_data = []
                    # Gather data from all neighbors for the script.
                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    # Gather data from the local device.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data:
                        # Run the script with the gathered data.
                        result = script.run(script_data)
                        # Broadcast/overwrite the result to all neighbors and local device.
                        for device in neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)

            # Wait until the supervisor signals the end of the current timepoint.
            self.device.timepoint_done.wait()
