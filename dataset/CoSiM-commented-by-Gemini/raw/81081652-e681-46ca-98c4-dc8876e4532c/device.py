"""
This module provides a multi-threaded framework for simulating a network of devices.

The simulation operates in discrete time steps, synchronized by a reusable barrier.
For each step, a `DeviceThread` spawns multiple short-lived `DeviceSubThread`s,
one for each assigned script.

The locking strategy is two-tiered:
1. A shared list of location-specific locks ensures that operations on any given
   location are serialized across the entire system.
2. Each device also has its own individual lock, which is used to protect writes
   to its own `sensor_data`.
"""

from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads, implemented
    using a two-phase protocol with semaphores.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # A list is used to create a mutable integer reference for the counter.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes a single phase of the barrier protocol."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a device in the simulation, holding its state and managing its
    main control thread.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        # A lock specific to this device instance, used to protect its data during writes.
        self.lock = Lock()
        # A list of locks shared among all devices, one for each location.
        self.locationlock = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (barrier, location locks)
        to all devices. Designed to be called by a single master device (device_id 0).
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            locationlock = [Lock() for _ in xrange(100)]
            for device in devices:
                device.locationlock = locationlock
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """Assigns the shared barrier to this device."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """Assigns a script to be executed in the next timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A 'None' script is a sentinel signaling that all scripts for the
            # current timepoint have been assigned.
            self.script_received.set() # This event seems unused in this implementation.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data. Note: Not thread-safe. Relies on the caller
        to manage synchronization.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Sets sensor data. Note: Not thread-safe. Relies on the caller
        to manage synchronization (e.g., by holding `self.lock`).
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to complete."""
        self.thread.join()

class DeviceThread(Thread):
    """The main control thread for a device, which spawns worker threads."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Termination signal.
            
            # Wait for the supervisor to signal the start of the timepoint.
            self.device.timepoint_done.wait()
            
            # --- Worker Thread Management ---
            subthreads = []
            for (script, location) in self.device.scripts:
                # Create and start a new worker thread for each script.
                new_thread = DeviceSubThread(self, neighbours, script, location)
                subthreads.append(new_thread)
                new_thread.start()
            
            # Wait for all worker threads for this timepoint to finish.
            for subthread in subthreads:
                subthread.join()
            
            self.device.scripts = [] # Clear scripts for the next round.
            self.device.timepoint_done.clear()
            
            # Synchronize with all other devices before the next timepoint begins.
            self.device.barrier.wait()


class DeviceSubThread(Thread):
    """A short-lived worker thread that executes a single script."""
    def __init__(self, devicethread, neighbours, script, location):
        Thread.__init__(self, name="Device SubThread %d" % devicethread.device.device_id)
        self.neighbours = neighbours
        self.devicethread = devicethread
        self.script = script
        self.location = location

    def run(self):
        """Executes the script logic with a two-tiered locking strategy."""
        # Acquire the shared lock for this specific location to prevent any other
        # thread from working on this location concurrently.
        self.devicethread.device.locationlock[self.location].acquire()
        
        script_data = []
        
        # --- Data Aggregation Phase ---
        # Reading is safe because the location lock is held.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.devicethread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # --- Computation and Write-back Phase ---
        if script_data:
            result = self.script.run(script_data)
            
            # For writing, acquire each device's individual lock to ensure
            # atomic updates to its sensor_data.
            for device in self.neighbours:
                with device.lock:
                    device.set_data(self.location, result)
            
            with self.devicethread.device.lock:
                self.devicethread.device.set_data(self.location, result)
        
        # Release the location lock, allowing another thread to process this location.
        self.devicethread.device.locationlock[self.location].release()
