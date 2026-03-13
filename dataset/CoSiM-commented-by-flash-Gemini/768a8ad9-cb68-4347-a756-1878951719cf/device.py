"""
This module implements a simulation framework for distributed devices,
focusing on concurrent execution of scripts and synchronized data processing.
It utilizes a `Device` class to represent each simulated entity,
a `DeviceThread` for main control flow, and `MyThread` workers for
executing individual scripts. A `ReusableBarrier` facilitates global
synchronization across devices, and a shared `lock_hash` manages
location-specific data access.
"""


from threading import Semaphore, Event, Lock, Thread, Condition # Condition imported for ReusableBarrier

class ReusableBarrier(object):
    """
    A reusable double-barrier synchronization primitive implemented using semaphores.

    This barrier allows a fixed number of threads (`num_threads`) to wait for
    each other to reach a common point before any can proceed. It is designed
    to be reusable across multiple synchronization points within a larger simulation loop.
    The implementation uses a two-phase (double turnstile) approach to prevent
    threads from one cycle from proceeding before all threads have exited the
    previous cycle, ensuring safe reuse.
    """
    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must arrive
                               at the barrier before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase of the barrier (entry).


        self.count_threads2 = self.num_threads # Counter for the second phase of the barrier (exit).
        self.counter_lock = Lock() # Lock to protect the counters during concurrent updates.
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase, blocking threads until all arrive.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase, blocking threads until all have passed phase 1.

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        `num_threads` threads have also called `wait()`. This is the public
        entry point for the double-barrier mechanism.
        """
        
        self.phase1() # Executes the first synchronization phase.


        self.phase2() # Executes the second synchronization phase, enabling reusability.

    def phase1(self):
        """
        Manages the first phase (entry turnstile) of the double-barrier synchronization.

        Block Logic:
        A thread entering this phase decrements a shared counter. If it is the
        last thread to arrive (counter reaches zero), it unblocks all other
        waiting threads by releasing the first semaphore `num_threads` times.
        It then resets the counter to prepare for the next barrier cycle.
        All threads, including the last one, then attempt to acquire the semaphore,
        blocking until it is fully released.
        """
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in xrange(self.num_threads): # Note: `xrange` is Python 2.x specific.
                    self.threads_sem1.release() # Releases all threads blocked on `threads_sem1`.
                self.count_threads1 = self.num_threads # Resets the counter for barrier reusability.

        self.threads_sem1.acquire() # Blocks the current thread until it's released by the semaphore.

    def phase2(self):
        """
        Manages the second phase (exit turnstile) of the double-barrier synchronization.

        Block Logic:
        This phase ensures that no thread can re-enter `phase1` of a new cycle
        before all threads have completed `phase1` of the current cycle.
        The logic mirrors `phase1` but uses a second counter and semaphore.
        When the last thread passes, it resets the barrier to a clean state,
        making it fully reusable.
        """
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in xrange(self.num_threads): # Note: `xrange` is Python 2.x specific.
                    self.threads_sem2.release() # Releases all threads blocked on `threads_sem2`.
                self.count_threads2 = self.num_threads # Resets the counter for barrier reusability.

        self.threads_sem2.acquire() # Blocks the current thread until it's released by the semaphore.

class Device(object):
    """
    Represents a single node in the distributed simulation.

    Each device maintains its own sensor data, a list of scripts to execute,
    and a dedicated control thread (`DeviceThread`). It communicates with other
    devices through shared synchronization primitives (`ReusableBarrier`, `lock_hash`).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's internal state
                                or sensor readings, keyed by location.
            supervisor (object): A central supervisor object responsible for
                                 providing network topology information (neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.none_script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.timepoint_end = 0
        self.barrier = None
        self.lock_hash = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """Assigns a shared `ReusableBarrier` to the device."""
        self.barrier = barrier

    def set_locks(self, lock_hash):
        """Assigns a shared dictionary of location-based locks."""
        self.lock_hash = lock_hash

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects for the network.

        This method is intended to be called on a single "leader" device. It
        creates a global `ReusableBarrier` for all devices and a `lock_hash`
        containing a `Lock` for each unique data location across the entire
        network. It then distributes these shared objects to all other devices.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        
        ids_list = []
        for dev in devices:
            ids_list.append(dev.device_id)


        if self.device_id == min(ids_list):
            
            self.barrier = ReusableBarrier(len(devices))
            self.lock_hash = {}

            for dev in devices:
                for location in dev.sensor_data:
                    if location not in self.lock_hash:
                        self.lock_hash[location] = Lock()

            
            
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.set_barrier(self.barrier)
                    dev.set_locks(self.lock_hash)


    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        This is the primary method for dispatching work. If the script is `None`,
        it signals that no more scripts are expected for the current time step.

        Args:
            script (object): The script object to be executed, which must have a `run` method.
            location (str): The location context for the script execution.
        """

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.none_script_received.set()

    def get_data(self, location):
        """
        Retrieves data for a specific location from the device's sensor data.
        
        Args:
            location (str): The key for the desired data.
            
        Returns:
            The data associated with the location, or `None` if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates the data for a specific location in the device's sensor data.
        
        Args:
            location (str): The key for the data to be updated.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Signals the device's control thread to terminate and waits for it to join."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control loop for a single `Device`.

    This thread orchestrates the device's participation in the simulation.
    It waits for scripts to be assigned, dispatches them to `MyThread` worker
    threads for concurrent execution, and then synchronizes with all other
    devices at a global barrier before starting the next time step.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent `Device` object this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.semaphore = Semaphore(value=8)

    def run(self):
        """
        The main execution loop of the device.

        Workflow per time step:
        1. Fetches the current list of neighbors from the supervisor.
        2. Waits until the supervisor signals that all scripts for the step have been assigned.
        3. Creates and starts a `MyThread` worker for each assigned script.
        4. Waits for all its local worker threads to complete.
        5. Enters the global barrier, waiting for all other devices in the simulation to reach this point.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.none_script_received.wait()
            self.device.none_script_received.clear()

            thread_list = []

            
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, neighbours, script, location,
                    self.semaphore)


                thread.start()
                thread_list.append(thread)

            for i in xrange(len(thread_list)):
                thread_list[i].join()

            
            
            
            self.device.barrier.wait()

class MyThread(Thread):
    """
    A worker thread responsible for executing one script at a specific location.

    It manages the fine-grained data aggregation, script execution, and result
    propagation, ensuring thread safety through a combination of a device-level
    semaphore and a location-specific global lock.
    """

    def __init__(self, device, neighbours, script, location, semaphore):
        """
        Initializes a worker thread.

        Args:
            device (Device): The parent device that owns this worker.
            neighbours (list): A list of neighboring `Device` objects.
            script (object): The script to execute.
            location (str): The location context for data access.
            semaphore (Semaphore): A semaphore to limit concurrent workers per device.
        """

        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.semaphore = semaphore

    def run(self):
        """
        The core execution logic for the worker thread.

        Workflow:
        1. Acquire a semaphore slot to limit concurrency on the parent device.
        2. Acquire the global lock for the target `location` to ensure exclusive data access.
        3. Aggregate data from its own device and all neighbors at the specified `location`.
        4. Execute the script with the aggregated data.
        5. Propagate the result back to its own device and all neighbors at that `location`.
        6. Release the location lock.
        7. Release the semaphore slot.
        """
        self.semaphore.acquire()

        self.device.lock_hash[self.location].acquire()

        script_data = []

        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)

            
            self.device.set_data(self.location, result)

        
        self.device.lock_hash[self.location].release()

        
        self.semaphore.release()