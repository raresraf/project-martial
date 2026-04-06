"""
Models a device in a distributed network simulation for sensor data processing.

This module defines the core components of a concurrent, multi-threaded simulation
where multiple `Device` objects interact. Each device manages its own sensor data,
executes assigned scripts, and synchronizes with other devices. The architecture
uses a main `DeviceThread` per device, which in turn manages a pool of `ComputationThread`
workers to parallelize script execution. Threading primitives like Locks, Events,
and a custom reusable barrier are used extensively for synchronization and
inter-thread communication.
"""

from threading import Event, Thread, Lock, Semaphore, Condition


class ReusableBarrierCond(object):
    """
    A custom, reusable barrier implementation using a `threading.Condition`.

    This barrier allows a fixed number of threads to synchronize and wait for each
    other to reach a certain point before proceeding. Once all threads have
    arrived, they are all released, and the barrier resets for the next use.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.
        
        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        
        self.cond = Condition()

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have arrived.
        """
        
        self.cond.acquire()
        self.count_threads -= 1
        # Pre-condition: Check if the current thread is the last one to arrive.
        if self.count_threads == 0:
            """
            Block Logic: If this is the last thread, it notifies all waiting
            threads to proceed and resets the barrier's thread count for reuse.
            """
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            """
            Block Logic: If other threads are still pending, this thread waits on
            the condition variable, releasing the lock until notified.
            """
            self.cond.wait()
        
        self.cond.release()


class SignalType(object):
    """
    Defines constants representing the types of signals for inter-thread communication.
    
    This acts as a simple enumeration to provide clarity and avoid magic numbers.
    """
    SCRIPT_RECEIVED = 1
    TIMEPOINT_DONE = 2
    TERMINATION = 3


class Device(object):
    """
    Represents a single device in the distributed network simulation.

    Manages its own state, sensor data, and a list of scripts to execute. It
    communicates with a central supervisor and other devices, and orchestrates its
    own internal worker thread (`DeviceThread`).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device and starts its main worker thread.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor values.
            supervisor: A reference to the central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        # --- Synchronization Primitives ---
        self.devices_barrier = None
        self.signal_received = Event()
        self.signal_type = None
        self.timepoint_work_done = Event()
        self.signal_sent = Event()
        self.data_locks = {}
        self.scripts_lock = Lock()

        # --- Worker Thread ---
        self.thread = DeviceThread(self)
        self.thread.start()

        # Initialize a lock for each piece of sensor data to ensure thread-safe access.
        for location in sensor_data:
            self.data_locks[location] = Lock()

        self.devices_lock = {}

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        One-time setup method, called by the master device (ID 0).

        It initializes and distributes a shared reusable barrier and a set of shared
        data locks to all devices in the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """

        # Pre-condition: This setup should only be run by one designated master device.
        if self.device_id == 0:
            devices_barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                device.devices_barrier = devices_barrier
                for location in device.sensor_data:
                    self.devices_lock[location] = Lock()

                device.devices_lock = self.devices_lock

    def assign_script(self, script, location):
        """
        Assigns a script for execution or signals the end of a timepoint.

        This method is called by an external entity (e.g., the supervisor) to
        drive the device's behavior.

        Args:
            script: The script object to be executed.
            location: The location associated with the script. Can be None to signal
                      the end of a timepoint.
        """
        # Pre-condition: Check if a new script is being assigned or if it's a timepoint signal.
        if script is not None:
            # Add the script to the list in a thread-safe manner.
            with self.scripts_lock:
                self.scripts.append((script, location))

            # Signal the DeviceThread that a new script has been received.
            self.signal_type = SignalType.SCRIPT_RECEIVED
            self.signal_received.set()
            
            # Wait for the DeviceThread to acknowledge receipt of the signal.
            self.signal_sent.wait()
            self.signal_sent.clear()

        else:
            # Signal the DeviceThread that all scripts for the current timepoint are assigned.
            self.signal_type = SignalType.TIMEPOINT_DONE
            self.signal_received.set()
            
            # Wait for acknowledgement.
            self.signal_sent.wait()
            self.signal_sent.clear()
            
            # Wait for the DeviceThread to confirm all work for the timepoint is complete.
            self.timepoint_work_done.wait()
            self.timepoint_work_done.clear()

    def get_data(self, location):
        """
        Thread-safely retrieves sensor data for a given location.

        Args:
            location: The location key for the desired sensor data.

        Returns:
            The sensor data, or None if the location is not managed by this device.
        """
        # Pre-condition: Check if the location exists for this device.
        if location in self.sensor_data:
            with self.data_locks[location]:
                return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Thread-safely updates sensor data for a given location.

        Args:
            location: The location key for the sensor data to update.
            data: The new value for the sensor data.
        """
        # Pre-condition: Check if the location exists for this device.
        if location in self.sensor_data:
            with self.data_locks[location]:
                self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully shuts down the device by stopping its worker thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main worker thread for a Device.

    It manages the device's lifecycle, including fetching neighbor information from
    the supervisor, handling signals from the main `Device` object, and delegating
    script execution to a pool of `ComputationThread` workers.
    """

    def __init__(self, device):
        """Initializes the device's main thread and its computation thread pool."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        self.neighbours = None

        # --- Synchronization Primitives ---
        self.signal_received = Event()
        self.signal_type = None
        self.scripts_index = 0

        self.new_timepoint = Semaphore(0)
        self.signal_lock = Lock()

        # --- Computation Thread Pool ---
        self.num_threads = 8
        self.timepoint_computation_done = [Event() for _ in range(self.num_threads)]
        self.threads = [ComputationThread(self, count) for count in range(self.num_threads)]
        for count in range(self.num_threads):
            self.threads[count].start()

        self.neighbour_locks = [Lock() for _ in range(self.num_threads)]

    def acquire_neighbours(self):
        """Acquires all locks associated with neighbor interactions."""
        for lock in self.neighbour_locks:
            lock.acquire()

    def release_neighbours(self):
        """Releases all locks associated with neighbor interactions."""
        for lock in self.neighbour_locks:
            lock.release()

    def run(self):
        """The main execution loop for the device thread."""
        
        # Invariant: The loop continues as long as the supervisor indicates the
        # simulation is active.
        while True:
            # Safely get the latest list of neighbors from the supervisor.
            self.acquire_neighbours()
            
            self.neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: Check if the supervisor signaled termination.
            if self.neighbours is None:
                """
                Block Logic: On termination signal, propagate the signal to worker
                threads and wait for them to finish before breaking the loop.
                """
                self.signal_type = SignalType.TERMINATION
                self.device.signal_sent.set()
                self.release_neighbours()

                # Wait for all computation threads to acknowledge termination.
                for computation_thread_done in self.timepoint_computation_done:
                    computation_thread_done.wait()
                    computation_thread_done.clear()
                break

            self.release_neighbours()

            # Invariant: This inner loop processes signals for a single timepoint.
            while True:
                # Wait for a signal (e.g., SCRIPT_RECEIVED, TIMEPOINT_DONE) from the Device object.
                self.device.signal_received.wait()
                self.device.signal_received.clear()

                # Propagate the signal to worker threads and acknowledge receipt.
                self.signal_type = self.device.signal_type
                
                self.signal_received.set()
                self.device.signal_sent.set()

                # Pre-condition: Check if the timepoint's work is complete.
                if self.signal_type == SignalType.TIMEPOINT_DONE:
                    """
                    Block Logic: If all scripts for the timepoint are assigned, wait for
                    all computation threads to finish their work, then clean up for the
                    next timepoint.
                    """
                    # Wait for all computation threads to finish processing scripts.
                    for computation_thread_done in self.timepoint_computation_done:
                        computation_thread_done.wait()
                        computation_thread_done.clear()

                    # Reset the script index for the next timepoint.
                    self.scripts_index = 0

                    # Signal back to the Device object that the timepoint is fully processed.
                    self.device.timepoint_work_done.set()
                    break

            # Synchronize with all other devices at the end of the timepoint.
            self.device.devices_barrier.wait()

        # Clean up: Join all computation threads upon simulation termination.
        for computation_thread in self.threads:
            computation_thread.join()


class ComputationThread(Thread):
    """
    A worker thread that executes scripts for a device.

    Each `DeviceThread` manages a pool of these threads to process scripts in
    parallel.
    """

    def __init__(self, device_thread, thread_id):
        """Initializes a computation worker thread."""
        Thread.__init__(self, name="Computing Thread %d" % thread_id)

        self.device_thread = device_thread
        self.thread_id = thread_id

    def run(self):
        """
        The main execution loop for the computation thread.
        
        Pulls scripts from a shared list and executes them.
        """
        
        # Invariant: The thread continues to run as long as the simulation is active.
        while True:
            # Wait for a signal from the parent DeviceThread to start processing.
            self.device_thread.signal_received.wait()
            
            # Acquire a lock to ensure exclusive access during critical sections.
            self.device_thread.neighbour_locks[self.thread_id].acquire()

            # Pre-condition: Check for a termination signal.
            if self.device_thread.signal_type == SignalType.TERMINATION:
                self.device_thread.neighbour_locks[self.thread_id].release()
                # Signal that this thread has completed its shutdown.
                self.device_thread.timepoint_computation_done[self.thread_id].set()
                break

            # Invariant: This loop processes all available scripts for the current timepoint.
            while True:
                # Safely access the shared script list.
                self.device_thread.device.scripts_lock.acquire()

                # Pre-condition: Check if all scripts have been claimed by worker threads.
                if len(self.device_thread.device.scripts) == self.device_thread.scripts_index:
                    """
                    Block Logic: No more scripts to process for this timepoint. Release
                    the lock, signal completion, and break the script-processing loop.
                    """
                    self.device_thread.device.scripts_lock.release()
                    
                    self.device_thread.timepoint_computation_done[self.thread_id].set()

                    break

                # Atomically claim the next available script.
                index = self.device_thread.scripts_index
                (script_todo, location) = self.device_thread.device.scripts[index]
                self.device_thread.scripts_index += 1

                self.device_thread.device.scripts_lock.release()

                script_data = []
                
                # Gather necessary data from neighbor devices.
                for device in self.device_thread.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the local device.
                data = self.device_thread.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: Only run the script if there is data to process.
                if script_data:
                    # Execute the script's logic.
                    result = script_todo.run(script_data)

                    # Propagate the result to neighbor devices.
                    for device in self.device_thread.neighbours:
                        device.set_data(location, result)

                    # Update the local device's data.
                    self.device_thread.device.set_data(location, result)

            
            # Pre-condition: If this was a single script assignment, clear the signal
            # to wait for the next one.
            if self.device_thread.signal_type == SignalType.SCRIPT_RECEIVED:
                self.device_thread.signal_received.clear()

            self.device_thread.neighbour_locks[self.thread_id].release()
