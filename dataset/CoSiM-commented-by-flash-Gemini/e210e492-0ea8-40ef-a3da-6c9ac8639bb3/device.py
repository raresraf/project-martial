"""
This module provides a sophisticated simulation framework for distributed devices,
employing a thread pool for concurrent script execution and a centralized object
for managing shared state.

Architecture:
- `Device`: Represents a network node. It submits incoming scripts to a queue for processing.
- `DeviceThread`: The main control loop for a `Device`. It manages a `ThreadPool`
  and orchestrates the script execution and synchronization for each time step.
- `RunScript`: A task object (not a thread) that contains the logic for executing
  a single script. These tasks are submitted to the `ThreadPool`.
- `ThreadPool`: A standard thread pool that manages a fixed number of `Worker`
  threads to execute `RunScript` tasks concurrently.
- `SharedDeviceData`: A central object that holds all state shared across devices,
  including a global synchronization barrier and location-specific locks.
- `CyclicBarrier`: A classic barrier implementation used for global time-step synchronization.
"""

from threading import Thread, Lock, Condition, Semaphore, Event
from Queue import Queue
# The 'utils' module was likely merged into this file.
# from utils import SharedDeviceData
# from utils import ThreadPool

class Device(object):
    """
    Represents a single device node in the simulation.
    
    This class acts as a high-level manager, delegating script execution to a
    `DeviceThread` and handling script assignment via a thread-safe queue.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor (object): A supervisor object to get network information (neighbors).
        """
        self.device_id = device_id
        self.num_cores = 8  
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.new_scripts = Queue()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes the central `SharedDeviceData` object.

        This method should be called on a single "leader" device (e.g., device_id 0).
        It creates the one instance of `SharedDeviceData` that will be used by all
        devices for synchronization and shared lock management.
        
        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        if self.device_id == 0:
            shared_data = SharedDeviceData(len(devices))
            
            # Pre-populate locks for initial data locations.
            for data in self.sensor_data:
                if data not in shared_data.location_locks:
                    shared_data.location_locks[data] = Lock()

            for dev in devices:
                dev.shared_data = shared_data

    def assign_script(self, script, location):
        """
        Assigns a new script to the device by placing it in a queue.
        A `None` script acts as a sentinel value to signal the end of a time step.

        Args:
            script (object): The script to execute.
            location (str): The location context for the script.
        """
        self.new_scripts.put((script, location))

    def get_data(self, location):
        """Safely retrieves data from the device's sensor readings."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Safely updates data in the device's sensor readings."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main control thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control and orchestration thread for a single `Device`.

    It manages a `ThreadPool` to execute scripts, processes incoming scripts from
    a queue, and handles synchronization at the end of each simulation step.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop of the device simulation.

        Workflow per time step:
        1. Initialize a thread pool.
        2. In a loop, wait for new scripts to be assigned via a queue.
        3. For each new script, ensure a lock for its location exists in the
           central `SharedDeviceData`, creating one if necessary.
        4. Submit the script as a `RunScript` task to the thread pool.
        5. A `None` script signals the end of assignments for the current time step.
        6. Once the signal is received, shut down the pool and wait for all tasks to complete.
        7. Wait at the global `timepoint_barrier` for all other devices to finish.
        """
        thread_pool = ThreadPool(self.device.num_cores)
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Process scripts from the previous time step.
            for (script, location) in self.device.scripts:
                thread_pool.submit(RunScript(script, location, neighbours,
                                             self.device))
            
            # Wait for and process new scripts for the current time step.
            while True:
                (script, location) = self.device.new_scripts.get()
                if script is None: # Sentinel value marks end of time step.
                    break

                # Ensure a lock exists for the script's location, creating one if new.
                self.device.shared_data.ll_lock.acquire()
                if location not in self.device.shared_data.location_locks:
                    self.device.shared_data.location_locks[location] = Lock()
                self.device.shared_data.ll_lock.release()

                thread_pool.submit(RunScript(script, location, neighbours,
                                             self.device))
                self.device.scripts.append((script, location))

            thread_pool.shutdown() 
            thread_pool.wait_termination(False) 

            # Synchronize with all other devices before the next time step.
            self.device.shared_data.timepoint_barrier.wait()

        thread_pool.wait_termination() 

class RunScript(object):
    """
    A task object representing a single script execution.

    This object is created by `DeviceThread` and executed by a `Worker`
    in the `ThreadPool`.
    """

    def __init__(self, script, location, neighbours, device):
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        Executes the script logic.
        
        It acquires the appropriate location-specific lock from the shared data,
        gathers data, runs the script, propagates the result, and releases the lock.
        """
        # Get the location-specific lock from the central shared data.
        self.device.shared_data.ll_lock.acquire()
        lock = self.device.shared_data.location_locks[self.location]
        self.device.shared_data.ll_lock.release()

        script_data = []

        lock.acquire()  

        # Aggregate data from neighbors and self.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Execute script and propagate results.
            result = self.script.run(script_data)

            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

        lock.release() 

# --- Utility Classes (likely from a 'utils.py' file) ---

class CyclicBarrier(object):
    """A classic cyclic barrier implementation using a Condition variable."""

    def __init__(self, parties):
        self.parties = parties
        self.count = 0
        self.condition = Condition()

    def wait(self):
        """
        Waits until all `parties` have called this method.
        When the last party arrives, all are notified and released, and the
        barrier resets for the next cycle.
        """
        self.condition.acquire()
        self.count += 1
        if self.count == self.parties:
            self.condition.notifyAll() 
            self.count = 0  
        else:
            self.condition.wait()
        self.condition.release()

class ThreadPool(object):
    """A standard thread pool implementation."""

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.task_queue = Queue() 
        self.num_tasks = Semaphore(0) # Semaphore to count pending tasks.
        self.stop_signal = Event() 
        self.shutdown_signal = Event()

        self.threads = []
        for i in xrange(0, num_threads):
            self.threads.append(Worker(self.task_queue,
                                       self.num_tasks,
                                       self.stop_signal))
        
        for i in xrange(0, num_threads):
            self.threads[i].start()

    def submit(self, task):
        """Submits a task to the pool's queue for execution."""
        if self.shutdown_signal.is_set():
            return 
        self.task_queue.put(task)
        self.num_tasks.release()

    def shutdown(self):
        """Signals that no new tasks will be submitted."""
        self.shutdown_signal.set()

    def wait_termination(self, end=True):
        """
        Waits for all submitted tasks to complete and optionally shuts down all workers.
        
        Args:
            end (bool): If True, workers will be permanently stopped.
                        If False, pool can be reused.
        """
        self.task_queue.join()
        if end is True:
            self.stop_signal.set() 
            for i in xrange(0, self.num_threads):
                self.task_queue.put(None) 
                self.num_tasks.release()

            for i in xrange(0, self.num_threads):
                self.threads[i].join()
        else:
            self.shutdown_signal.clear()


class Worker(Thread):
    """A worker thread that consumes and executes tasks from a queue."""

    def __init__(self, task_queue, num_tasks, stop_signal):
        Thread.__init__(self)
        self.task_queue = task_queue
        self.num_tasks = num_tasks
        self.stop_signal = stop_signal

    def run(self):
        """Continuously fetches tasks from the queue and runs them."""
        while True:
            self.num_tasks.acquire()
            if self.stop_signal.is_set():
                break
            
            task = self.task_queue.get()
            
            task.run()
            self.task_queue.task_done()

class SharedDeviceData(object):
    """A centralized container for all data shared across the simulation."""

    def __init__(self, num_devices):
        self.num_devices = num_devices
        # The global barrier for synchronizing all devices at each time step.
        self.timepoint_barrier = CyclicBarrier(num_devices)
        
        # A dictionary to hold locks for each data location.
        self.location_locks = {}

        # A meta-lock to ensure thread-safe access to the location_locks dictionary itself.
        self.ll_lock = Lock()