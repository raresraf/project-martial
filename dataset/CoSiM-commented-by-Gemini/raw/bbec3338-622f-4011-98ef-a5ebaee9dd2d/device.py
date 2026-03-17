"""
This module defines a framework for a discrete-time, multi-threaded simulation of interconnected devices. 
It models a system where devices can run computational scripts on local and neighboring data, 
synchronize their operations at specific time points, and update their state based on script results.
The simulation relies heavily on Python's threading primitives to manage concurrency and synchronization.
"""

from threading import Event, Thread, RLock, Lock, Semaphore, Condition


class Device(object):
    """
    Represents a single device in the distributed simulation. Each device runs in its own thread,
    manages its own sensor data, and executes assigned scripts.
    """
    
    # A class-level barrier to synchronize all device threads at the end of a timepoint.
    barrier = None

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device's state.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The initial local sensor data for this device.
            supervisor (object): A supervisor object that manages device interactions.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been assigned.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that the current timepoint processing is complete.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # Lock to protect access to the scripts list during assignment.
        self.run_script = RLock()
        # Semaphore to limit the number of concurrently running scripts.
        self.scripts_sem = Semaphore(8)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared ReusableBarrier for all devices.
        This method is intended to be called once by a single device (e.g., device 0)
        to set up the synchronization primitive for the entire system.
        """
        if Device.barrier == None and self.device_id == 0:
            Device.barrier = ReusableBarrier(len(devices))

    def assign_script(self, script, location):
        """
        Assigns a new script to be executed by the device in the current timepoint.
        This method is called by the supervisor.
        """
        self.run_script.acquire()
        # Signal the device thread that new work has arrived.
        self.script_received.set()
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is a signal from the supervisor that the current timepoint is over.
            self.timepoint_done.set()
        self.run_script.release()

    def get_data(self, location):
        """Retrieves sensor data from a specific location on this device."""
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        """Updates sensor data at a specific location on this device."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to complete its execution."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main worker thread for a Device. It orchestrates the device's lifecycle,
    including script execution and synchronization with other devices.
    """

    def __init__(self, device):
        """
        Initializes the device thread.

        Args:
            device (Device): The parent device object this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device. The loop represents the progression
        of the simulation through discrete time steps.
        """
        while True:
            # Pre-condition: At the start of the loop, the device is waiting for instructions for the next timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # The supervisor signals termination by returning None for neighbours.
                break

            # Block Logic: Waits until the supervisor assigns at least one script.
            self.device.script_received.wait()

            # Block Logic: Executes all assigned scripts for the current timepoint in parallel.
            self.device.run_script.acquire()
            dictionar = {}
            i = 0
            
            # Invariant: Iterates through all scripts assigned for the current timepoint.
            for (script, location) in self.device.scripts:
                # Limits the number of concurrent script executions.
                self.device.scripts_sem.acquire()
                thread = MyThread(self.device, neighbours, location, script)
                dictionar[i] = thread
                dictionar[i].start()
                i = i + 1
            self.device.run_script.release()
            
            # Block Logic: Waits for all spawned script threads to complete.
            for idx in range(0, len(dictionar)):
                dictionar[idx].join()

            # Synchronization Point: Waits for all other devices to finish their script computations for this timepoint.
            Device.barrier.wait()
            # Synchronization Point: Waits for the supervisor to signal that the timepoint is officially over.
            self.device.timepoint_done.wait()

class MyThread(Thread):
    """
    A thread designed to execute a single script on data from a specific location,
    gathered from the parent device and its neighbors.
    """
    # Class-level dictionary of locks to ensure that operations on a given data 'location'
    # are serialized across all threads and devices, preventing race conditions.
    lockForLocations = {}

    def __init__(self, device, neighbours, location, script):
        """
        Initializes a script execution thread.

        Args:
            device (Device): The parent device.
            neighbours (list): A list of neighboring Device objects.
            location (any): The identifier for the data location to be processed.
            script (object): The script object with a `run` method to be executed.
        """
        Thread.__init__(self)
        self.location = location
        self.script = script
        self.device = device
        self.neighbours = neighbours

        # Invariant: Ensures a lock exists for every location that will be accessed.
        if location not in MyThread.lockForLocations:
            MyThread.lockForLocations[location] = Lock()

    def run(self):
        """

        Executes the script. This involves acquiring a location-specific lock,
        gathering data, running the script, and distributing the results.
        """
        MyThread.lockForLocations[self.location].acquire()
        script_data = []
        
        # Block Logic: Gathers data from all neighboring devices for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Also include the data from the current device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # The script is executed only if there is data to process.
            result = self.script.run(script_data)

            # Block Logic: Propagates the script's result back to all neighbors.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Also update the local device's data.
            self.device.set_data(self.location, result)
        
        MyThread.lockForLocations[self.location].release()
        # Releases the semaphore permit, allowing another script to run.
        self.device.scripts_sem.release()

class ReusableBarrier(object):
    """
    A simple, reusable barrier implementation that blocks a set of threads until all of them
    have called the `wait` method. It resets automatically after all threads have passed.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a specific number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        # Counter for threads that have arrived at the barrier.
        self.count_threads = self.num_threads
        # A Condition variable to manage waiting and notification.
        self.cond = Condition()

    def wait(self):
        """
        Causes a thread to wait at the barrier. The last thread to arrive will
        notify all waiting threads and reset the barrier for future use.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Pre-condition: All threads have reached the barrier.
            # The last thread notifies all waiting threads to proceed.
            self.cond.notify_all()
            # Invariant: Resets the barrier counter for the next synchronization point.
            self.count_threads = self.num_threads
        else:
            # Threads wait until the last thread arrives and notifies them.
            self.cond.wait()
        self.cond.release()