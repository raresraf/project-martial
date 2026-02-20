"""
This module defines a device simulation model where each device operates
on a single thread. Scripts are processed sequentially within this thread.

Synchronization between devices is achieved with a reusable barrier, and
access to shared data locations is serialized using a dictionary of locks that
is itself protected by a global lock during modification.
"""

from threading import Event, Thread
from threading import Semaphore, Lock

class ReusableBarrier(object):
    """
    A reusable two-phase barrier implemented with semaphores, for synchronizing
    multiple threads at a specific point in their execution, multiple times.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Use a list to hold the count, allowing it to be modified by reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the caller until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes a single phase of the two-phase barrier protocol."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
        

class Device(object):
    """
    Represents a single device that processes scripts sequentially in its own
    dedicated control thread.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.barrier = None 
        self.InitializationEvent = Event() # Used for master/slave setup.
        self.LockLocation = None # Shared dictionary of location-specific locks.
        self.LockDict = Lock() # A lock to protect the LockLocation dictionary itself.

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources using a master/slave pattern.
        Device 0 acts as the master. Other devices wait on an event until the
        master has finished setting up the shared barrier and lock dictionary.
        """
        if self.device_id == 0:
            # Master device creates the shared resources.
            n = len(devices)
            self.barrier = ReusableBarrier(n)   
            self.LockLocation = {}  

            for idx in range(len(devices)):
                d = devices[idx]
                d.LockLocation = self.LockLocation
                d.barrier = self.barrier
                if d.device_id == 0:
                    pass
                else:
                    # Signal slave devices that setup is complete.
                    d.InitializationEvent.set()
        else:
            # Slave devices wait for the master's signal.
            self.InitializationEvent.wait()

        # All threads are started only after setup is complete.
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to the device or signals the end of a timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data.
        @note This method is not thread-safe by itself.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data.
        @note This method is not thread-safe by itself.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The single control thread for a device, which processes all its scripts
    sequentially for each timepoint.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop. It synchronizes with other devices, processes
        all its scripts sequentially, and repeats.
        """
        while True:
            # Phase 1: All devices synchronize at the barrier before starting.
            self.device.barrier.wait()

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for supervisor to assign all scripts for this timepoint.
            self.device.timepoint_done.wait()

            # Block Logic: Sequentially process all assigned scripts.
            for (script, location) in self.device.scripts:
                # Use a global lock to protect the dictionary of location locks
                # during the check-and-add operation.
                self.device.LockDict.acquire()
                if location not in self.device.LockLocation.keys():
                    self.device.LockLocation[location] = Lock()
                
                # Acquire the specific lock for this location.
                self.device.LockLocation[location].acquire()
                self.device.LockDict.release()

                # --- Critical Section for this location starts ---
                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
                # --- Critical Section for this location ends ---
                self.device.LockLocation[location].release()

            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()
