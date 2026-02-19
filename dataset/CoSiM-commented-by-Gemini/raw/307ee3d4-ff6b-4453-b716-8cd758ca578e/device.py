"""
@file device.py
@brief Defines an advanced, multi-threaded device simulation framework.

This module implements a simulation where each device is a container for a pool
of worker threads. It uses a two-level barrier system for fine-grained
synchronization, both within a single device and across the entire network of
devices.
"""


from threading import Event, Thread, Semaphore, Lock

class Device(object):
    """
    Represents a device, acting as a manager for a pool of worker threads.

    This class holds the device's state and orchestrates the setup and
    synchronization primitives for its internal worker threads and for
    communication with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its internal thread pool."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_scripts = []
        self.neighbours = []
        self.timepoint_done = Event()
        
        self.initialization = Event()
        
        # Each device has its own pool of worker threads.
        self.threads = []
        for k in xrange(8):
            self.threads.append(DeviceThread(self, k))
        
        # Lock for safely accessing the device's script list.
        self.locations_lock = Lock()
        # Shared dictionary for location-based data locks.
        self.locked_locations = None
        # Global barrier to sync all threads from all devices.
        self.devices_barrier = None
        # Local barrier to sync all threads within this device.
        self.device_barrier = ReusableBarrier(len(self.threads))

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of global synchronization objects.

        Functional Utility: Device 0 acts as the master, creating a global barrier
        for all threads in the simulation and a shared dictionary for data locks.
        It signals other devices once this setup is complete.
        """
        if self.device_id == 0:
            # The dictionary of locks is shared across all devices.
            self.locked_locations = {}
            # This barrier synchronizes every thread from every device.
            self.devices_barrier = ReusableBarrier(len(devices)*len(self.threads))

            # Distribute the shared objects to all other devices.
            for device in devices:
                device.locked_locations = self.locked_locations
                device.devices_barrier = self.devices_barrier
                # Signal to other devices that initialization is done.
                device.initialization.set()
        else:
            # Non-master devices wait for the master to complete setup.
            self.initialization.wait()

        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be processed by the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of script assignment for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data for a given sensor location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data for a given sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads of this device to terminate."""
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread within a device's thread pool.

    Executes a portion of the device's scripts for each timepoint using a
    work-pulling model and complex, multi-level synchronization.
    """

    def __init__(self, device, thread_id):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            # 1. Global Sync: All threads from all devices wait here for the timepoint to begin.
            self.device.devices_barrier.wait()

            # 2. Per-Device Setup: Thread 0 of each device performs setup tasks.
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # 3. Local Sync: All threads within this device wait for thread 0 to finish setup.
            self.device.device_barrier.wait()
            
            neighbours = self.device.neighbours
            if neighbours is None:
                # Supervisor signals the end of the simulation.
                break

            # 4. Wait for Scripts: All threads wait for the supervisor to assign scripts.
            self.device.timepoint_done.wait()

            # 5. Script Distribution: Thread 0 copies the scripts for this timepoint to a shared list.
            if self.thread_id == 0:
                self.device.timepoint_scripts = self.device.scripts[:]

            # 6. Local Sync: Wait for thread 0 to finish copying scripts.
            self.device.device_barrier.wait()

            # 7. Work Pulling: All threads in the pool pull scripts from the shared list until it's empty.
            while True:
                # Safely get a script from the shared list.
                self.device.locations_lock.acquire()
                if len(self.device.timepoint_scripts) == 0:
                    self.device.locations_lock.release()
                    break
                (script, location) = self.device.timepoint_scripts.pop()

                # Ensure a lock exists for the location, then acquire it.
                if location not in self.device.locked_locations:
                    self.device.locked_locations[location] = Lock()
                self.device.locked_locations[location].acquire()
                self.device.locations_lock.release()

                # --- Critical Section for Data Processing ---
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
                # --- End Critical Section ---
                
                self.device.locked_locations[location].release()

            # 8. Local Sync: Wait for all threads in this device to finish their scripts.
            self.device.device_barrier.wait()
            
            # 9. Cleanup: Thread 0 resets the event for the next timepoint.
            if self.thread_id == 0:
                self.device.timepoint_done.clear()

class ReusableBarrier(object):
    """A two-phase reusable barrier for thread synchronization."""
    def __init__(self, num_threads):
        """Initializes the barrier for a fixed number of threads."""
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
        """Executes a single synchronization phase."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all others.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for reuse.
                count_threads[0] = self.num_threads
        
        # Block until released by the last thread.
        threads_sem.acquire()                    
