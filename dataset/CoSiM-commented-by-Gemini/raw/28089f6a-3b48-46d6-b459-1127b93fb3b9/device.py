
"""
Models a distributed system of devices executing computational scripts concurrently.

This module presents another variation of a device simulation framework. It features
a Condition-variable-based barrier and a thread pool manager (`MyThread`) for
executing script tasks. However, this implementation appears to have significant
logical bugs related to its concurrency control (unbalanced/misused Condition
variables for locking) and script execution logic (potential for N^2 execution of
N scripts). The synchronization flow between the main device thread and the
supervisor is also complex and relies on a non-thread-safe boolean flag.
"""
from Queue import Queue
from threading import Thread, Condition, Event

class ReusableBarrier(object):
    """
    A reusable synchronization barrier implemented using a Condition variable.
    Functionally similar to `threading.Barrier` available in newer Python versions.
    """

    def __init__(self, num_threads):
        

        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread arrives, notify all waiting threads.
            self.cond.notify_all()
            self.count_threads = self.num_threads # Reset for reuse.
        else:
            self.cond.wait()
        self.cond.release()

class MyThread(object):
    """
    A thread pool manager (confusingly named MyThread).

    This class creates, starts, and manages a pool of worker threads that
    process tasks from a shared queue.
    
    NOTE: The `run_script` method contains a nested loop that causes it to
    re-execute all scripts for every task pulled, leading to incorrect N^2
    behavior instead of N.
    """

    def __init__(self, threads_count):
        

        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None
        self.create(threads_count)
        self.start()

    def create(self, threads_count):
        """Creates the worker threads for the pool."""
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

    def start(self):
        """Starts all worker threads in the pool."""
        for thread in self.threads:
            thread.start()

    def set_device(self, device):
        """Sets the parent device associated with this thread pool."""
        self.device = device

    def execute(self):
        """The main loop for a worker thread."""
        while True:

            # Get a task from the queue.
            neighbours, script, location = self.queue.get()

            # Sentinel value to terminate the thread.
            if neighbours is None and script is None:
                self.queue.task_done()
                return

            self.run_script(neighbours, script, location)
            self.queue.task_done()

    def run_script(self, neighbours, script, location):
        """
        Executes scripts.
        
        BUG: This method iterates through `self.device.scripts` which contains ALL
        scripts for the device, not just the single `script` passed from the queue.
        This causes every script to be executed multiple times.
        """
        for (script, location) in self.device.scripts:
            script_data = []
            
            # --- Data Gathering Phase ---
            # BUG: The locking protocol is broken. get_data acquires a lock
            # that is never released in this scope.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                # --- Data Update Phase ---
                # BUG: set_data releases a lock that was acquired by get_data.
                # This leads to unbalanced lock/unlock and potential deadlock or errors.
                for device in neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)

    def put(self, neighbours, script, location):
        """Adds a new task to the thread pool's queue."""
        self.queue.put((neighbours, script, location))

    def wait_threads(self):
        """Blocks until the queue is empty."""
        self.queue.join()

    def end_threads(self):
        """Shuts down the thread pool gracefully."""
        self.wait_threads()

        # Add sentinel tasks to stop each worker thread.
        for _ in xrange(len(self.threads)):
            self.put(None, None, None)

        for thread in self.threads:
            thread.join()

class Device(object):
    """Represents a single device in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # NOTE: This boolean flag is not thread-safe for script assignment.
        self.script_received = False
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        # NOTE: Using a Condition object as a Lock is unconventional. A Lock is more appropriate.
        self.location_cond = {location: Condition() for location in sensor_data}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Creates and distributes the shared barrier."""
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            self.send_barrier(devices, self.barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        """Static method to assign the barrier to all other devices."""
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """Sets the barrier for this device."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """Assigns a script or signals the end of the timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received = True # Not thread-safe
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Acquires a lock and returns data.
        BUG: The lock acquired here is expected to be released by a call to set_data,
        which is a dangerous and fragile design.
        """
        if location in self.sensor_data:
            self.location_cond[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Updates data and releases a lock.
        BUG: This method releases a lock it did not acquire, breaking encapsulation.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_cond[location].release()

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    """The main control thread for a single Device."""

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = MyThread(8) # The thread pool manager.

    def run(self):

        self.threads.set_device(self.device)

        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # This inner loop's logic for script handling is complex and racy.
            while True:

                
                if self.device.script_received or self.device.timepoint_done.wait():
                    if self.device.script_received: # Race condition here
                        self.device.script_received = False
                        
                        # Add all scripts to the queue for the thread pool to process.
                        for (script, location) in self.device.scripts:
                            self.threads.put(neighbours, script, location)
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_received = True
                        break

            
            # Wait for the thread pool to finish all tasks for this time step.
            self.threads.wait_threads()

            
            # Synchronize with all other devices before the next time step.
            self.device.barrier.wait()

        
        # Cleanly shut down the thread pool.
        self.threads.end_threads()
