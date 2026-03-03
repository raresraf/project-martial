"""
This module implements a distributed device simulation framework where each
script execution is handled by a newly spawned thread.

The architecture is characterized by:
- A main thread per device (`DeviceThread`) that orchestrates the simulation step.
- A "thread-per-script" model where the main thread spawns a new thread for
  each script to be executed in the current time step.
- A shared, custom-implemented two-phase reusable barrier (`ReusableBarrierSem`)
  for synchronizing all devices at the end of a step.
- Lazy, on-demand creation of locks for data locations, which contains a
  potential race condition.
"""

from threading import Lock, Thread, Event, Semaphore


class Device(object):
    """
    Represents a single device in the simulation network.

    It holds sensor data, receives scripts from a supervisor, and manages a
    main execution thread (`DeviceThread`) that handles the simulation logic.

    Attributes:
        device_id (int): Unique identifier for the device.
        sensor_data (dict): The device's sensor readings, keyed by location.
        supervisor (object): The central simulation supervisor.
        scripts (list): A list of (script, location) tuples for the current step.
        timepoint_done (Event): An event signaling that all scripts for the
                                current step have been assigned.
        thread (DeviceThread): The main execution thread for this device.
        barrier (ReusableBarrierSem): Shared barrier for inter-device sync.
        location_locks (dict): Shared dictionary of locks for data locations.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.location_locks = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and shares synchronization objects across all devices.

        Device 0 is responsible for creating the shared barrier and the dictionary
        for location locks. Other devices will receive a reference to these objects.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        Device.devices_no = len(devices)
        if self.device_id == 0:
            # Device 0 creates the shared synchronization primitives.
            self.barrier = ReusableBarrierSem(len(devices))
            self.location_locks = {}
        else:
            # Other devices get references from Device 0.
            self.barrier = devices[0].barrier
            self.location_locks = devices[0].location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be run or signals the end of assignments.

        Args:
            script (object): The script to execute, or None.
            location (str): The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that all scripts for this step are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location. Access is unsynchronized here."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data for a given location. Access is unsynchronized here."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, adopting a "thread-per-script" model.

    In each simulation step, this thread waits for scripts, then spawns a new
    worker thread for each script. It waits for these workers to complete before
    synchronizing with other devices at a global barrier.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_scripts(self, script, location, neighbours):
        """
        The target function for worker threads. Executes a single script.

        It handles locking for the specific data location, creating the lock if it
        doesn't exist. It gathers data, runs the script, and updates the data on
        all relevant devices.

        Args:
            script (object): The script to run.
            location (str): The location to operate on.
            neighbours (list): A list of neighboring devices.
        """
        # Lazy initialization of locks.
        # WARNING: This check-then-create pattern is not atomic and can lead to a
        # race condition if two threads for the same new location execute this
        # block concurrently. A lock should protect the creation itself.
        lock_location = self.device.location_locks.get(location)
        if lock_location is None and location is not None:
            self.device.location_locks[location] = Lock()
            lock_location = self.device.location_locks[location]
        
        with lock_location:
            script_data = []
            # Gather data from neighbors and the local device.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Run the script and update data on all involved devices.
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)

    def run(self):
        """Main simulation loop for the device thread."""
        while True:
            # Get neighbors for the upcoming step from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # None from supervisor is the shutdown signal.
                break
            
            # Wait until the supervisor signals that all scripts are assigned.
            self.device.timepoint_done.wait()
            
            tlist = []
            # Spawn a new thread for each assigned script.
            for (script, location) in self.device.scripts:
                thread = Thread(target=self.run_scripts, args=(script, location, neighbours))
                tlist.append(thread)
                thread.start()
            
            # Wait for all script-threads for this step to complete.
            for thread in tlist:
                thread.join()
            
            # Reset for the next time step.
            self.device.timepoint_done.clear()
            
            # Synchronize with all other devices before starting the next step.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    A custom implementation of a reusable barrier using Semaphores.
    This is a two-phase barrier to ensure that threads from a previous `wait()`
    call have all exited the barrier before it can be used again.
    """

    def __init__(self, num_threads):
        """Initializes the barrier."""
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait until all threads reach the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive releases all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase to ensure safe reuse of the barrier."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread releases all threads for the next cycle.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
