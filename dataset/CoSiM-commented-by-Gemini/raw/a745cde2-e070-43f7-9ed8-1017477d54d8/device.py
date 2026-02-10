from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    Implements a two-phase reusable barrier for thread synchronization.
    
    This barrier ensures that a specified number of threads all reach a
    synchronization point before any of them are allowed to proceed. It is
    "reusable" because it resets itself after all threads have passed,
    allowing it to be used multiple times, for instance, in a loop for
    time-stepped simulations.

    Algorithm: The barrier uses two phases, each controlled by a semaphore and a
    counter. This prevents threads that have completed the barrier from racing
    ahead and re-entering it before all threads from the previous phase have
    been released.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a fixed number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        # Counter for threads entering the first phase.
        self.count_threads1 = [self.num_threads]
        # Counter for threads entering the second phase.
        self.count_threads2 = [self.num_threads]
        # Lock to ensure atomic updates to the counters.
        self.count_lock = Lock()
        # Semaphore for the first phase. Blocks threads until all have arrived.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second phase. Ensures all threads from phase 1 have exited
        # before the barrier can be reused.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier until all participating threads
        have called this method.
        """
        # First synchronization phase to ensure all threads arrive.
        self.phase(self.count_threads1, self.threads_sem1)
        # Second synchronization phase to ensure all threads are released before reset.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the barrier synchronization.

        Args:
            count_threads (list[int]): A list containing the counter for the current phase.
            threads_sem (Semaphore): The semaphore to block and release threads for this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Invariant: The last thread to arrive (when count becomes 0) is responsible
            # for releasing all waiting threads.
            if count_threads[0] == 0:
                # Release all waiting threads for this phase.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # Block until the last thread releases the semaphore.
        threads_sem.acquire()

class MyThread(Thread):
    """
    A worker thread for executing a single script on a device.

    Functional Utility: This thread encapsulates the logic for a single computational
    task within a time-step. It gathers data from the local device and its neighbors,
    executes a script on the collected data, and then distributes the result back
    to the same set of devices.
    """
    def __init__(self, neighbours, script, location, device):
        """
        Initializes the script-executing thread.

        Args:
            neighbours (list[Device]): A list of neighboring Device objects.
            script (object): The script to be executed. Must have a `run` method.
            location (any): The data location identifier to operate on.
            device (Device): The parent device on which this thread is running.
        """
        Thread.__init__(self)
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.device = device
        self.script_data = []

    def run(self):
        """
        The main execution logic for the thread.
        """
        # Block Logic: Data gathering phase.
        # Pre-condition: `self.neighbours` and `self.device` are initialized.
        # Invariant: Collects data from all neighbors and the local device associated
        # with the specified `location`.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                self.script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            self.script_data.append(data)

        # Block Logic: Script execution and data propagation phase.
        # Pre-condition: `self.script_data` is populated.
        # Invariant: If data was gathered, the script is run and its result is
        # written back to the local device and all its neighbors.
        if self.script_data != []:
            # Execute the script with the collected data.
            result = self.script.run(self.script_data)
            
            # Distribute the result to all neighbors.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Set the result on the local device.
            self.device.set_data(self.location, result)
        
        # Clean up data for the next run.
        self.script_data = []

class Device(object):
    """
    Represents a single device in the simulated network.

    Architectural Role: A Device is an active entity that holds sensor data,
    manages assigned scripts, and communicates with its neighbors under the
    coordination of a supervisor. It runs its own thread (`DeviceThread`) to
    manage its lifecycle within the time-stepped simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): A dictionary representing the device's local data.
            supervisor (object): The central supervisor object that manages the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been assigned.
        self.script_received = Event()
        self.scripts = []
        # Event to signal the start of a new time-step.
        self.timepoint_done = Event()
        # The main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the synchronization barrier for all devices.
        
        Note: This method contains a race condition. It should only be called
        by a single, designated thread (in this case, device 0) before any
        other device threads start interacting with the barrier.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for i in xrange(len(devices)):
                devices[i].barrier = barrier

    def assign_script(self, script, location):
        """Assigns a script to be executed in the next time-step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script is treated as a signal to start the time-step.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from the device's local sensor data store."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates data in the device's local sensor data store."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a Device instance.

    Functional Utility: This thread orchestrates the device's participation
    in the time-stepped simulation. It waits for a signal to begin a time-step,
    spawns worker threads (`MyThread`) for each assigned script, and ensures
    synchronization with all other devices at the end of the step using a barrier.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            # Retrieve the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown by returning None.
                break
            
            # Wait for the supervisor to signal the start of a time-step.
            self.device.timepoint_done.wait()

            # Block Logic: Script execution for the current time-step.
            # Pre-condition: `self.device.scripts` has been populated.
            # Invariant: For each assigned script, a `MyThread` is created and
            # executed in parallel. The main device thread waits for all of them
            # to complete.
            freds = []
            for (script, location) in self.device.scripts:
                fred = MyThread(neighbours, script, location, self.device)
                freds.append(fred)
            
            # Start all script-executing threads.
            for i in freds:
                i.start()
            
            # Wait for all script-executing threads to complete.
            for i in freds:
                i.join()
            
            # Reset the time-step event and wait at the global barrier for all
            # other devices to finish their time-step.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()