"""
This module simulates a network of devices executing scripts in parallel.

The architecture uses a complex, two-level barrier synchronization system to keep
all threads and devices operating in lock-step. Work is distributed to a pool
of threads on each device from a shared queue.
"""

from threading import Event, Thread, Semaphore, Lock

class Device(object):
    """
    Represents a single device in the distributed simulation.

    Each device manages a pool of worker threads and is responsible for
    executing scripts that may involve communicating with neighboring devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): A dictionary of local sensor data.
            supervisor: The supervisor object providing network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_scripts = []
        self.neighbours = []
        self.timepoint_done = Event()
        
        self.initialization = Event()
        
        self.threads = []
        for k in xrange(8):
            self.threads.append(DeviceThread(self, k))
        
        # This lock protects both the `timepoint_scripts` list and the
        # `locked_locations` dictionary. The name is slightly misleading.
        self.locations_lock = Lock()
        self.locked_locations = None
        # A global barrier for ALL threads across ALL devices.
        self.devices_barrier = None
        # A local barrier for threads within THIS device.
        self.device_barrier = ReusableBarrier(len(self.threads))

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and shares global synchronization objects.

        Device 0 creates the global barrier and the shared dictionary for
        location-specific locks.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        if self.device_id == 0:
            # The global barrier must sync every thread from every device.
            self.devices_barrier = ReusableBarrier(len(devices)*len(self.threads))
            self.locked_locations = {}

            # Share the global objects with all other devices.
            for device in devices:
                device.locked_locations = self.locked_locations
                device.devices_barrier = self.devices_barrier
                device.initialization.set() # Signal that setup is complete.
        else:
            # Wait for Device 0 to finish its setup.
            self.initialization.wait()

        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals the end of a timepoint.

        Args:
            script: The script to execute, or None to signal end of assignments.
            location (int): The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Release the worker threads to start processing the current timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data from the device's local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets data in the device's local sensor data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to complete."""
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """A worker thread belonging to a Device."""

    def __init__(self, device, thread_id):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device.
            thread_id (int): The ID of this thread within the device's pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """Main execution loop for the worker thread."""
        while True:
            # 1. Global Sync: All threads from all devices wait here.
            self.device.devices_barrier.wait()

            # 2. Intra-Device Setup: Thread 0 gets neighbors for this timepoint.
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # 3. Intra-Device Sync: All threads on this device wait for setup to complete.
            self.device.device_barrier.wait()
            neighbours = self.device.neighbours
            if neighbours is None: # Supervisor signals shutdown.
                break

            # 4. Wait for Supervisor: Threads wait for the signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            
            # 5. Work Queue Setup: Thread 0 copies scripts to a temporary work queue.
            if self.thread_id == 0:
                self.device.timepoint_scripts = self.device.scripts[:]
            
            # 6. Intra-Device Sync: Wait for Thread 0 to finish creating the work queue.
            self.device.device_barrier.wait()
            
            # 7. Work-Stealing Loop
            while True:
                with self.device.locations_lock:
                    if len(self.device.timepoint_scripts) == 0:
                        # No more scripts for this timepoint.
                        break
                    (script, location) = self.device.timepoint_scripts.pop()

                    # Lazily create a lock for this location if it doesn't exist.
                    if location not in self.device.locked_locations:
                        self.device.locked_locations[location] = Lock()
                    
                    # Acquire the specific lock for this location.
                    self.device.locked_locations[location].acquire()
                
                # --- Critical Section for Location ---
                script_data = []
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

                self.device.locked_locations[location].release()
            
            # 8. Intra-Device Sync: Wait for all threads on this device to finish their work.
            self.device.device_barrier.wait()
            
            # 9. Reset for next timepoint.
            if self.thread_id == 0:
                self.device.timepoint_done.clear()

class ReusableBarrier(object):
    """
    A reusable, two-phase barrier for synchronizing a fixed number of threads.
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
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Manages one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads        
        
        threads_sem.acquire()                    
