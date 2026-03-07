"""
This Python 2 module implements a flawed distributed device simulation.

The file, misleadingly named MyQueue.py, contains a full framework including
Device, DeviceThread, a ReusableBarrier, and a custom thread pool (MyQueue).
The design attempts to use a thread pool to process tasks concurrently and
synchronize devices with a barrier.

However, the implementation contains several significant design flaws:
- A race condition in the initialization of the MyQueue workers.
- An unsafe and deadlock-prone locking mechanism where get_data() acquires a
  lock that set_data() is expected to release.
- An overly complex and likely buggy main loop in the DeviceThread.
"""
from Queue import Queue 
from threading import Thread

class MyQueue():
    """
    A custom thread pool implementation for processing device tasks.

    This class creates a set of worker threads that pull tasks from a shared queue.
    NOTE: This implementation has a critical race condition. The worker threads
    are started in __init__ and immediately try to access `self.device`, but
    `self.device` is only assigned later in the `DeviceThread.run` method.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the thread pool and starts the worker threads.

        Args:
            num_threads (int): The number of worker threads to create.
        """
        
        self.queue = Queue(num_threads)
        self.threads = []
        self.device = None

        
        for _ in xrange(num_threads):
            thread = Thread(target=self.run)
            self.threads.append(thread)
        
        
        for thread in self.threads:
            thread.start()
    
    def run(self):
        """The target function for each worker thread."""
        
        while True:
            
            # Get a task from the queue.
            neighbours, script, location = self.queue.get()

            
            # Sentinel value to terminate the thread.
            if neighbours is None and script is None:
                self.queue.task_done()
                return
        
            script_data = []
            
            # Gather data from neighboring devices.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            
            # Gather data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                # Execute the script.
                result = script.run(script_data)
                
                
                # Update data on neighbors and the local device.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                
                self.device.set_data(location, result)
            
            self.queue.task_done()
    
    def finish(self):
        """Shuts down the thread pool gracefully."""
        
        
        # Wait for all tasks in the queue to be processed.
        self.queue.join()

        
        # Send a sentinel value for each thread to signal termination.
        for _ in xrange(len(self.threads)):
           self.queue.put((None, None, None))

        
        # Wait for all worker threads to finish.
        for thread in self.threads:
            thread.join()

# The file re-imports modules and defines more classes below.
from threading import Thread, Event, Lock, Semaphore
from MyQueue import MyQueue # This import is circular and redundant.

class ReusableBarrier():
    """A reusable barrier using a two-phase semaphore protocol."""
    
    def __init__(self, num_threads):
        """Initializes the barrier."""
        
        self.num_threads = num_threads
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
        """Executes one phase of the barrier synchronization."""
        
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()

class Device(object):
    """
    Represents a device in the simulation.

    NOTE: The locking mechanism implemented in get_data/set_data is unsafe.
    It requires the caller of get_data to ensure set_data is called to release
    the lock, which is a fragile and deadlock-prone design.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device."""
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.location_locks = {location: Lock() for location in self.sensor_data}
        self.scripts_available = False
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up and distributes the shared barrier."""
        
        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        
        
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Acquires a lock and returns data from a location. Unsafe design.
        """
        
        
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]     
        else:
            return None

    def set_data(self, location, data):
        """
        Sets data at a location and releases a lock. Unsafe design.
        """
        
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
        else:
            return None

    def shutdown(self):
        """Shuts down the device's thread."""
        
        self.thread.join()

class DeviceThread(Thread):
    """The main execution thread for a device."""
    

    def __init__(self, device):
        """Initializes the thread and its custom thread pool (`MyQueue`)."""
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = MyQueue(8)

    def run(self):
        """
        The main simulation loop.

        NOTE: This loop has a convoluted structure for handling script arrival
        and is likely buggy. It also introduces a race condition by setting
        `self.queue.device` here, after the queue's worker threads have already started.
        """
        
        
        # This assignment is not thread-safe.
        self.queue.device = self.device
        while True:
            
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            # Complex and potentially buggy loop to wait for scripts.
            while True:
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    if self.device.scripts_available:
                        self.device.scripts_available = False

                        
                        # Submit tasks to the thread pool.
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
            
                    else:
            
                        
                        self.device.timepoint_done.clear()
                        self.device.scripts_available = True
                        break
            
            
            # Wait for all submitted tasks to complete.
            self.queue.queue.join()
            # Synchronize with other devices.
            self.device.barrier.wait()

        
        # Cleanly shut down the thread pool.
        self.queue.finish()
