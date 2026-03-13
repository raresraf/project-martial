"""
A simulation framework for a network of communicating devices using a
thread-per-task concurrency model and global location-based locking.

This script defines a system where each `Device`'s main thread (`DeviceThread`)
spawns a new worker thread (`DeviceSubThread`) for each script it is assigned
in a time-step. This approach ensures task isolation but can be less scalable
than a thread pool model.

The main components are:
- ReusableBarrier: A semaphore-based reusable barrier for synchronization.
- Device: Represents a node. The primary device (id 0) is responsible for
  creating and distributing a shared barrier and a global dictionary of
  per-location locks to all other devices.
- DeviceThread: The main control loop for a device. It waits for script
  assignments, creates a new worker thread for each script, and then waits
  for all workers to complete before synchronizing with other devices.
- DeviceSubThread: A short-lived worker thread that handles the execution
  of a single script. It uses the global location-based locking mechanism
  to ensure data consistency.

Key Architectural Points:
- **Thread-per-Task Model**: For each time-step, a new thread is created for
  every assigned script, which can be resource-intensive.
- **Global Location-Based Locking**: A central dictionary maps data locations
  to locks. A worker thread acquires the specific lock for its script's
  location at the beginning of its execution and holds it until it's finished.
  This serializes all operations on a given location, preventing race
  conditions and deadlocks.
- **Inter-Device Synchronization**: A global `ReusableBarrier` ensures that
  all `DeviceThread`s complete their work for a time-step before any device
  proceeds to the next.
"""
from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    Implements a reusable barrier using two semaphores for two-phase signaling.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier protocol."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a device node that spawns worker threads for each task.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.all_devices = []
        self.scripts = []
        self.data_lock = Lock() # A per-device lock, seems inconsistent with location_locks.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = ReusableBarrier(0)
        self.location_locks = {} # Shared dictionary of global locks.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes the shared barrier and location locks.
        """
        if self.device_id == 0:
            # Create a global lock for each possible location.
            for location in xrange(100):
                self.location_locks[location] = Lock()
            self.barrier = ReusableBarrier(len(devices))

        # Distribute shared resources to all devices.
        for dev in devices:
            if self.device_id == 0:
                dev.barrier = self.barrier
                dev.location_locks = self.location_locks
            self.all_devices.append(dev)

    def assign_script(self, script, location):
        """Assigns a script to the device for the current time-step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that all scripts for this step have been assigned.
            self.scripts_received.set()

    def get_data(self, location):
        """Retrieves sensor data."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Safely updates sensor data using a local device lock.
        
        NOTE: This uses a different locking mechanism (`data_lock`) than the
        worker threads (`location_locks`), which could lead to race conditions
        if used concurrently.
        """
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a Device, spawning a new thread for each script.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.assigned_scripts = {}

    def run(self):
        """Main orchestration loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for supervisor to signal script assignment is complete.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            # Create a new worker thread for each assigned script.
            device_threads = [
                DeviceSubThread(self.device, script, location, neighbours)
                for (script, location) in self.device.scripts
            ]

            # Start and wait for all worker threads for this step to complete.
            for thread in device_threads:
                thread.start()
            for thread in device_threads:
                thread.join()

            # --- Global Sync Point ---
            # Wait for all other devices to finish their work for the time-step.
            self.device.barrier.wait()

class DeviceSubThread(Thread):
    """A worker thread that executes one script for a specific location."""
    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script for the given location, ensuring exclusive access
        to that location across the entire system.
        """
        # --- Global Location Lock ---
        # Acquire the global lock for this location to serialize all work on it.
        self.device.location_locks[self.location].acquire()
        
        try:
            script_data = []
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from the local device.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Execute script and disseminate results.
            if script_data:
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)
        finally:
            # Release the lock to allow other threads to work on this location.
            self.device.location_locks[self.location].release()
