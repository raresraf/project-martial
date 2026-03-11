"""
This module simulates a network of devices that execute scripts on sensor data
in a parallelized, multi-threaded environment.

NOTE: This single file appears to be a combination of multiple modules.
It defines classes that seem to belong in separate `barrier` and `threadpool`
files. The classes `Solve` and `ThreadPoll` and the function `comparator` are
defined multiple times, suggesting a copy-paste error. The documentation below
attempts to describe the code as-is, noting its unusual structure and potential bugs.
"""

from threading import Event, Thread, Lock, Semaphore

# The 'barrier' and 'threadpool' modules are referenced but not provided.
# The code below seems to be the content of those modules.
from barrier import ReusableBarrierSem
from threadpool import ThreadPoll, comparator

class Device(object):
    """
    Represents a single device in the simulated network.
    It manages its own state, sensor data, and a control thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (object): An object to get neighbor information from.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.script_received.set()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.lock = Event() # Used for sequential setup.
        self.lock.clear()
        self.datalock = Lock() # This lock appears to be unused.
        self.personal_lock = [] # Fine-grained locks for each sensor location.
        self.bariera = None     # Shared barrier for simulation step synchronization.
        self.all = None
        self.no_devices = None

        # Initialize a lock for each potential sensor location up to the max key.
        if self.sensor_data:
            crt = max(self.sensor_data.keys())
            for _ in xrange(crt + 1):
                self.personal_lock.append(Lock())

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's synchronization objects in a sequential chain.
        This ensures shared resources are correctly distributed before the simulation starts.
        """
        self.all = sorted(devices, cmp=comparator)
        self.no_devices = len(self.all)

        if self.device_id == 0:
            # Device 0 is the leader and creates the shared barrier.
            self.bariera = ReusableBarrierSem(len(self.all)) # Assumed from missing import.
            self.lock.set() # Signal that its setup is complete.
        else:
            # Follower devices wait for the previous device in the chain to finish.
            prev_device = self.all[self.device_id - 1]
            prev_device.lock.wait()
            self.bariera = prev_device.bariera # Inherit the shared barrier.
            self.lock.set() # Signal completion.

        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be executed for a specific location."""
        if script is not None:
            self.script_received.wait()
            self.scripts.append((script, location))
        else:
            # A None script signals the end of script assignments for the current step.
            self.script_received.clear()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's control thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a single Device, managing its lifecycle in the simulation.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        thread_pool = ThreadPoll(8)
        while True:
            # All devices synchronize at the barrier before starting a new step.
            self.device.bariera.wait()

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None neighbor list signals the end of the simulation.
                thread_pool.close()
                break

            # Wait until all scripts for the current step have been assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            work_list = []
            # Prepare the list of tasks for the thread pool.
            for (script, location) in self.device.scripts:
                work_list.append((location, script, self.device, neighbours))

            # Dispatch the work to the thread pool and wait for completion.
            thread_pool.put_work(work_list)

            # Signal that the device is ready to receive scripts for the next step.
            self.device.script_received.set()

# The following definitions appear to be duplicated from other modules.

def comparator(device_a, device_b):
    """
    Comparator function to sort devices based on their device_id.
    """
    if device_a.device_id > device_b.device_id:
        return 1
    else:
        return -1

class Solve(Thread):
    """
    Represents a worker thread that executes a single script task.
    """
    def __init__(self, sem, free_threads, working_threads):
        Thread.__init__(self)
        self.free_threads = free_threads
        self.sem = sem
        self.working_threads = working_threads
        self.work = Event()
        self.free = Event()
        self.done = 0
        self.work.clear()
        self.free.set()
        self.location = None
        self.script = None
        self.device = None
        self.neighbours = None

    def set_work(self, location, script, device, neighbours):
        """Assigns a new task to the thread."""
        self.location = location
        self.script = script
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """Main execution loop of the worker thread."""
        while 1:
            self.work.wait() # Wait for a task.

            if self.done == 1:
                break # Exit if shutdown is signaled.

            script_data = []

            # Create a sorted list of devices to ensure consistent lock acquisition order.
            list_neighbours = self.neighbours
            list_neighbours.append(self.device)
            list_neighbours = set(list_neighbours)
            list_neighbours = sorted(list_neighbours, cmp=comparator)

            # Acquire locks for the location on all relevant devices.
            for device in list_neighbours:
                if self.location in device.sensor_data:
                    device.personal_lock[self.location].acquire()

            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # This appears to be a BUG: it re-appends the data from the last neighbor.
            # It likely intended to append data from `self.device`.
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Execute the script and update data on all relevant devices.
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)

            # Release locks in reverse order of acquisition to prevent deadlocks.
            for device in reversed(list_neighbours):
                if self.location in device.sensor_data:
                    device.personal_lock[self.location].release()

            # Signal that the thread is free and the task is complete.
            self.free.set()
            self.work.clear()
            self.free_threads.append(self)
            self.sem.release()

class ThreadPoll(object):
    """
    A custom thread pool for managing and dispatching tasks to worker threads.
    """
    def __init__(self, no_threads):
        """
        Initializes the thread pool.

        Args:
            no_threads (int): The number of worker threads in the pool.
        """
        self.no_threads = no_threads
        self.free_threads = []
        self.working_threads = []
        self.all_threads = []
        self.workdone = Event()
        self.sem = Semaphore(self.no_threads)

        # Create and start the worker threads.
        for _ in xrange(0, no_threads):
            tmp = Solve(self.sem, self.free_threads, self.working_threads)
            self.free_threads.append(tmp)
            self.all_threads.append(tmp)

        for current_thread in self.free_threads:
            current_thread.start()

    def put_work(self, work_list):
        """
        Assigns a list of tasks to the thread pool and waits for completion.
        """
        for (location, script, device, neighbours) in work_list:
            self.sem.acquire() # Wait for a free thread.
            current_thread = self.free_threads.pop(0)
            current_thread.set_work(location, script, device, neighbours)
            current_thread.free.clear()
            current_thread.work.set() # Signal the thread to start working.

        # Wait for all threads to finish their assigned tasks.
        for current_thread in self.all_threads:
            current_thread.free.wait()

    def close(self):
        """Terminates all worker threads in the pool."""
        for current_thread in self.all_threads:
            current_thread.done = 1
            current_thread.work.set()

        for current_thread in self.all_threads:
            current_thread.join()

