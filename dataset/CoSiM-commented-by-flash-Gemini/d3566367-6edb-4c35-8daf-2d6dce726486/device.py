
"""
This module provides a framework for simulating a distributed system of devices
that operate in synchronized time steps. It includes custom synchronization
primitives and a threaded model for parallel script execution.
"""

from threading import Event, Thread, Lock, Semaphore, RLock




class ReusableBarrier(object):
    """
    A reusable, two-phase barrier for synchronizing a fixed number of threads.

    This barrier ensures that all participating threads wait for each other at a
    synchronization point before any of them are allowed to continue. It can be
    reused after all threads have passed through it. The implementation uses
    two phases to prevent threads from one iteration from passing the barrier
    before all threads from the previous iteration have been released.
    """

    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have arrived.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Manages one phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current thread count for the phase.
            threads_sem (Semaphore): The semaphore to block/release threads for the phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread to arrive releases all waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads # Reset for next use.
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device in a distributed network simulation.

    Each device runs a pool of threads to execute scripts that can read and write
    data locally and from neighboring devices. It manages synchronization with
    other devices to ensure they operate in lock-step time points.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): A dictionary mapping locations to sensor values.
            supervisor: An object responsible for providing neighborhood information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.scripts = []
        self.timepoint_done = Event()

        self.threads = []
        self.no_threads = 8 # Number of worker threads per device.
        
        # A barrier to synchronize all devices in the simulation at the end of a timepoint.
        self.timepoint_barrier = None
        
        # Locks to protect access to individual sensor data locations.
        self.locks = []
        
        # Lock to protect access to the shared scripts list.
        self.scripts_lock = Lock()
        
        # A barrier to synchronize all worker threads within this device.
        self.internal_barrier = ReusableBarrier(self.no_threads)

        # Lock to manage the end of a timepoint signaling.
        self.end_timepoint = Lock()
        
        # A buffer for scripts that are processed in a timepoint.
        self.last_scripts = []

        # Device 0 is special and handles the initialization of shared resources.
        if device_id == 0:
            self.init_event = Event()

        for thread_id in range(self.no_threads):
            thread = DeviceThread(self, thread_id)
            self.threads.append(thread)


    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and shares synchronization primitives among all devices.

        Device 0 is responsible for creating the shared barrier and locks. Other
        devices wait for this initialization to complete before copying references
        to these shared objects.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            # Device 0 sets up the barrier for all devices.
            self.timepoint_barrier = ReusableBarrier(len(devices))

            no_location = 0
            for device in devices:
                no_location += len(device.sensor_data)

            # Create a reentrant lock for each sensor location across all devices.
            self.locks = [RLock() for _ in range(no_location)]
            self.init_event.set() # Signal that initialization is complete.
        else:
            # Other devices wait for Device 0 to finish initialization.
            for device in devices:
                if device.device_id == 0:
                    device.init_event.wait()
                    # Copy references to the shared objects.
                    self.timepoint_barrier = device.timepoint_barrier
                    self.locks = device.locks

        # Start all worker threads after setup.
        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device's threads.

        A `None` script is a special signal to indicate the end of script
        submissions for the current timepoint.

        Args:
            script: The script object to be executed.
            location (int): The data location the script will operate on.
        """
        if script is not None:
            with self.scripts_lock:
                self.scripts.append((script, location))
        else:
            # A None script signals the end of the current timepoint's script assignment.
            self.end_timepoint.acquire()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a given location in a thread-safe manner.

        Args:
            location (int): The location index of the data.

        Returns:
            The data at the specified location, or None if the location is not
            local to this device.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()

        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data at a given location in a thread-safe manner.

        Args:
            location (int): The location index of the data.
            data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """Waits for all worker threads to complete their execution."""
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread within a Device.

    This thread executes scripts, fetching data from its own device and neighboring
    devices, and synchronizes with other threads and devices using barriers.
    """

    def __init__(self, device, thread_id):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
            thread_id (int): The unique ID for this thread within the device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """Main execution loop for the worker thread."""
        # Thread 0 of each device has special setup duties.
        if self.thread_id == 0:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device in self.device.neighbours:
                self.device.neighbours.remove(self.device)

        while True:
            # Thread 0 also handles script list management.
            if self.thread_id == 0:
                with self.device.scripts_lock:
                    self.device.scripts += self.device.last_scripts
                    self.device.last_scripts = []

            # Synchronize all threads within the device before starting a new phase.
            self.device.internal_barrier.wait()
            
            neighbours = self.device.neighbours
            if neighbours is None:
                break # Shutdown signal.

            # --- First Script Processing Phase ---
            while len(self.device.scripts) != 0:
                script = None

                with self.device.scripts_lock:
                    if len(self.device.scripts) != 0:
                        script, location = self.device.scripts.pop(0)
                        self.device.last_scripts.append((script, location))

                if script:
                    script_data = []
                    # Gather data from neighbors.
                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                    # Gather local data.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    # Run script and broadcast result.
                    if script_data:
                        result = script.run(script_data)
                        for device in neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)

		    # Wait for the signal that no more new scripts will be added for this timepoint.
            self.device.timepoint_done.wait()

            # --- Second Script Processing Phase (for late-arriving scripts) ---
            while len(self.device.scripts) != 0:
                script = None

                with self.device.scripts_lock:
                    if len(self.device.scripts) != 0:
                        script, location = self.device.scripts.pop(0)
                        self.device.last_scripts.append((script, location))

                if script:
                    script_data = []
                    # Logic is identical to the first phase.
                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data:
                        result = script.run(script_data)
                        for device in neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)

            # Synchronize all threads within the device before ending the timepoint.
            self.device.internal_barrier.wait()

            # Thread 0 handles the end-of-timepoint synchronization with other devices.
            if self.thread_id == 0:
                self.device.timepoint_barrier.wait() # Wait for all devices to finish.
                
                # Get updated neighbors for the next timepoint.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                if self.device.neighbours and self.device in self.device.neighbours:
                    self.device.neighbours.remove(self.device)
                
                # Reset for the next timepoint.
                self.device.timepoint_done.clear()
                self.device.end_timepoint.release()
