from threading import Event, Thread, Lock
from Queue import Queue
# Note: The 'reusable_barrier_semaphore' is imported but the ReusableBarrier
# class is also defined at the end of this file. This documentation assumes
# the local definition is used.
import reusable_barrier_semaphore

class Device(object):
    """
    Represents a device in a distributed simulation, managing a fixed-size
    pool of worker threads to execute scripts concurrently.
    """
    
    # Class-level barrier for synchronizing all Device instances.
    barrier = None
    
    # A dictionary of locks, keyed by location, to ensure exclusive access
    # to sensor data at specific locations across all devices.
    lockList = {}
    
    # A lock to protect access to the shared lockList.
    lockListLock = Lock()

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device, creating its own pool of 8 worker threads.
        
        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's local sensor data.
            supervisor (object): The supervisor managing device interactions.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # A local barrier to synchronize the 8 worker threads of this device.
        self.neighbours_done = ReusableBarrier(8)
        self.neighbours = None
        
        # A queue for scripts to be processed in the current timepoint.
        self.scripts = Queue()
        
        # A list to store all assigned scripts, allowing them to be re-run
        # in subsequent timepoints.
        self.permanent_scripts = []
        self.threads = []
        # An event to signal worker threads that setup is complete.
        self.startup_event = Event()
        for i in range(8):
            self.threads.append(DeviceThread(self, i))
        for i in range(8):
            self.threads[i].start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """

        Initializes shared, class-level resources for the simulation. This
        includes the global device barrier and the dictionary of location locks.
        
        Args:
            devices (list): The list of all devices in the simulation.
        """
        # Atomically initialize the global barrier once.
        Device.lockListLock.acquire()
        if Device.barrier is None:
            Device.barrier = ReusableBarrier(len(devices))
        Device.lockListLock.release()

        # Create a lock for each unique sensor location.
        the_keys = self.sensor_data.keys()
        for i in the_keys:
            Device.lockListLock.acquire()
            if i not in Device.lockList:
                Device.lockList[i] = Lock()
            Device.lockListLock.release()
        
        # Signal to this device's worker threads that they can begin execution.
        self.startup_event.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. Scripts are added to a persistent list
        and a queue for immediate processing.
        
        Args:
            script (object): The script to execute. If None, it acts as a
                             sentinel to end the current timepoint's work.
            location (any): The location context for the script.
        """
        if script is not None:
            self.scripts.put((script, location))
            self.permanent_scripts.append((script, location))
        else:
            # When script assignment is done, add a sentinel for each worker thread.
            for i in range(8):
                self.scripts.put((None, None))

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all of its worker threads."""
        for i in range(8):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread within a Device's thread pool. Implements a leader-follower
    pattern where thread 0 handles global synchronization.
    """

    def __init__(self, device, the_id):
        """
        Initializes the worker thread.
        
        Args:
            device (Device): The parent device.
            the_id (int): The ID of this thread (0-7) within the device's pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.the_id = the_id

    def run(self):
        """
        The main loop of the worker thread. It synchronizes at the start of
        each timepoint, processes scripts from a queue, and then re-synchronizes.
        """
        # Wait until the device's setup_devices method is complete.
        self.device.startup_event.wait()
        while True:
            # Block Logic: Leader thread (id 0) handles global synchronization.
            if self.the_id == 0:
                # Synchronize with all other devices (via their leader threads).
                Device.barrier.wait()
                # Fetch neighbors for the new timepoint.
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # Invariant: All threads in the local pool wait here to ensure the
            # 'neighbours' list is updated before they proceed.
            self.device.neighbours_done.wait()

            # A None neighbours list is the signal to terminate.
            if self.device.neighbours is None:
                break

            # Block Logic: Process all scripts for the current timepoint.
            while True:
                (script, location) = self.device.scripts.get()
                
                # A None script is a sentinel indicating the end of the timepoint.
                if script is None:
                    # All local threads must reach this point before the leader can
                    # re-queue scripts for the next timepoint.
                    self.device.neighbours_done.wait()
                    if self.the_id == 0:
                        # Leader thread re-populates the queue for the next round.
                        for (s, l) in self.device.permanent_scripts:
                            self.device.scripts.put((s, l))
                    break # Exit script processing loop, start next timepoint.

                # If it's a valid script, execute it.
                if location is not None:
                    # Acquire the global lock for this location.
                    Device.lockList[location].acquire()
                    script_data = []
                    # Gather data from neighbors.
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    # Gather data from self.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    # Run script and update data on all relevant devices.
                    if script_data:
                        result = script.run(script_data)
                        for device in self.device.neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)
                    
                    Device.lockList[location].release()

class ReusableBarrier():
    """
    A reusable barrier for a fixed number of threads, implemented with two
    semaphores to create two phases, preventing race conditions on reuse.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the thread until all participating threads have called wait."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all others.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire()