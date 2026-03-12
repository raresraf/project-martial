"""
Models a device in a distributed sensor network simulation using a thread pool.

This script defines a device's behavior with a more modular approach, using
helper classes for thread pooling and shared data management. Each device's main
thread creates a new thread pool for each time step to execute script tasks.
"""


from threading import Thread, Lock, Condition, Semaphore, Event
from Queue import Queue

class Device(object):
    """
    Represents a single device node in the simulated network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device."""
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
        Initializes and distributes the shared data object to all devices.
        Device 0 creates the single `SharedDeviceData` instance.
        """
        if self.device_id == 0:
            shared_data = SharedDeviceData(len(devices))
            
            for data in self.sensor_data:
                if data not in shared_data.location_locks:
                    shared_data.location_locks[data] = Lock()

            for dev in devices:
                dev.shared_data = shared_data

    def assign_script(self, script, location):
        """Adds a new script to a queue to be processed by the DeviceThread."""
        self.new_scripts.put((script, location))

    def get_data(self, location):
        """Retrieves sensor data. Not internally thread-safe."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. Not internally thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device, managing its time-stepped execution.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main time-stepped loop.

        In each timepoint, it creates a new thread pool, submits all known and new
        scripts as tasks, waits for their completion, and then synchronizes at a
        global barrier.
        """
        # NOTE: Creating a new thread pool in each iteration is inefficient.
        thread_pool = ThreadPool(self.device.num_cores)
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Submit persistent scripts from previous timepoints.
            for (script, location) in self.device.scripts:
                thread_pool.submit(RunScript(script, location, neighbours,
                                             self.device))

            # Process newly assigned scripts for the current timepoint.
            while True:
                (script, location) = self.device.new_scripts.get()
                if script is None: # Sentinel marks end of new scripts.
                    break

                # Thread-safely create a lock for the location if it's new.
                self.device.shared_data.ll_lock.acquire()
                if location not in self.device.shared_data.location_locks:
                    self.device.shared_data.location_locks[location] = Lock()
                self.device.shared_data.ll_lock.release()

                thread_pool.submit(RunScript(script, location, neighbours,
                                             self.device))
                self.device.scripts.append((script, location))

            thread_pool.shutdown() # Prevent new task submissions.
            thread_pool.wait_termination(False) # Wait for current tasks to finish.

            # Wait for all devices to complete the timepoint.
            self.device.shared_data.timepoint_barrier.wait()

        thread_pool.wait_termination() # Final cleanup of the last threadpool.

class RunScript(object):
    """
    A runnable task object that wraps a script execution for the thread pool.
    """

    def __init__(self, script, location, neighbours, device):
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        Executes the script for a specific location in a thread-safe manner.
        """
        # Safely get the lock for the target location from the shared dictionary.
        self.device.shared_data.ll_lock.acquire()
        lock = self.device.shared_data.location_locks[self.location]
        self.device.shared_data.ll_lock.release()
        
        script_data = []

        # Acquire the location-specific lock to ensure exclusive access to sensor data.
        lock.acquire()  

        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Execute the script with the collected data.
            result = self.script.run(script_data)

            # Broadcast the result to all neighbours and self.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

        lock.release() 



class CyclicBarrier(object):
    """
    A custom, non-reusable cyclic barrier implementation.

    NOTE: This implementation is not truly "cyclic" or "reusable" in a
    thread-safe manner, as a fast thread could loop around and enter a new
    wait cycle before slow threads have exited the previous one, leading to deadlocks.
    """

    def __init__(self, parties):
        """Initializes the barrier for a given number of parties (threads)."""
        self.parties = parties
        self.count = 0
        self.condition = Condition()

    def wait(self):
        """
        Blocks until all parties have called wait(). The last party to arrive
        resets the count and notifies all waiting parties.
        """
        self.condition.acquire()
        self.count += 1
        if self.count == self.parties:
            self.condition.notifyAll() 
            self.count = 0  # Reset for the next "cycle".
        else:
            self.condition.wait()
        self.condition.release()

class ThreadPool(object):
    """
    A thread pool for executing tasks concurrently.
    """

    def __init__(self, num_threads):
        """Initializes the pool and starts the worker threads."""
        self.num_threads = num_threads

        self.task_queue = Queue() 
        self.num_tasks = Semaphore(0) # Counts pending tasks.
        self.stop_signal = Event()    # Signals workers to terminate permanently.
        self.shutdown_signal = Event() # Signals the pool to stop accepting new tasks.

        self.threads = []
        for i in xrange(0, num_threads):
            self.threads.append(Worker(self.task_queue,
                                       self.num_tasks,
                                       self.stop_signal))
        
        for i in xrange(0, num_threads):
            self.threads[i].start()

    def submit(self, task):
        """Submits a new task to the pool if not shut down."""
        if self.shutdown_signal.is_set():
            return 
        self.task_queue.put(task)
        self.num_tasks.release()

    def shutdown(self):
        """Stops the pool from accepting new tasks."""
        self.shutdown_signal.set()

    def wait_termination(self, end=True):
        """
        Waits for all tasks to complete and optionally terminates the worker threads.
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
    """
    A worker thread that executes tasks from a queue.
    """

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
    """
    A container for data and synchronization objects shared across all devices.
    """

    def __init__(self, num_devices):
        self.num_devices = num_devices
        self.timepoint_barrier = CyclicBarrier(num_devices)
        
        self.location_locks = {}
        
        self.ll_lock = Lock()