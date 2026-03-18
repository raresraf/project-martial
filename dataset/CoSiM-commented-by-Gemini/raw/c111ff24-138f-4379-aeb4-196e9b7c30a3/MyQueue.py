
"""
This module implements a simulation of a distributed system of devices.

It defines the core components for a multi-threaded device simulation, including
a custom worker pool queue, a two-phase reusable barrier for synchronization,
and the Device and DeviceThread classes that model the behavior of each node
in the distributed network.
"""
from Queue import Queue 
from threading import Thread, Event, Lock, Semaphore

class MyQueue():
    """
    A self-contained, multi-threaded worker pool.

    This class encapsulates a job queue and a fixed number of worker threads
    that process jobs from that queue concurrently. It's designed to process
    script execution tasks for a single device.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the worker pool and starts the worker threads.

        Args:
            num_threads (int): The number of worker threads to create in the pool.
        """
        self.queue = Queue(num_threads)
        self.threads = []
        self.device = None # The parent device, to be assigned after initialization.

        # Block Logic: Creates and starts a pool of worker threads that will
        # immediately block on the queue, waiting for jobs.
        for _ in xrange(num_threads):
            thread = Thread(target=self.run)
            self.threads.append(thread)
        
        for thread in self.threads:
            thread.start()
    
    def run(self):
        """
        The main execution loop for each worker thread.

        A worker thread continuously fetches a job from the queue, executes it,
        and signals completion. A job consists of gathering data from neighboring
        devices, running a script, and broadcasting the result.
        """
        while True:
            # This call blocks until a job is available.
            neighbours, script, location = self.queue.get()

            # Pre-condition: A sentinel value of (None, None, None) signals
            # the worker thread to terminate.
            if neighbours is None and script is None:
                self.queue.task_done()
                return
        
            script_data = []
            
            # Block Logic: Aggregates input data for the script by querying the
            # local device and its neighbors.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Core Logic: Executes the script with the collected data.
                result = script.run(script_data)
                
                # Block Logic: Distributes the script's output back to the local
                # device and all its neighbors.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                self.device.set_data(location, result)
            
            self.queue.task_done()
    
    def finish(self):
        """
        Gracefully shuts down the worker pool.

        This method waits for all pending jobs to complete, then posts sentinel
        values to the queue to terminate each worker thread, and finally waits
        for all threads to exit.
        """
        # Synchronization Point: Ensures all enqueued jobs are processed before shutdown.
        self.queue.join()

        # Block Logic: Sends a termination signal (sentinel) to each worker thread.
        for _ in xrange(len(self.threads)):
           self.queue.put((None, None, None))

        # Synchronization Point: Waits for all worker threads to terminate cleanly.
        for thread in self.threads:
            thread.join()

class ReusableBarrier():
    """
    A reusable, two-phase barrier for synchronizing multiple threads.

    This barrier ensures that all participating threads wait at the barrier
    point until every thread has arrived. The two-phase implementation prevents
    fast threads from re-entering and passing through the barrier for a second
    time before slow threads have exited it from the first time.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier for a specified number of threads.

        Args:
            num_threads (int): The number of threads that must synchronize.
        """
        self.num_threads = num_threads
        # The counters are wrapped in lists to make them mutable objects,
        # allowing them to be modified by reference within the `phase` method.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 # Protects access to the counters.
        self.threads_sem1 = Semaphore(0)         # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0)         # Semaphore for the second phase.
 
    def wait(self):
        """
        Blocks the calling thread until all threads have reached the barrier.
        
        This is achieved by passing through two internal synchronization phases.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the barrier synchronization.

        Args:
            count_threads (list): A mutable list containing the countdown for the phase.
            threads_sem (Semaphore): The semaphore used for blocking and releasing threads.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Invariant: When the last thread arrives (counter becomes 0), it is
            # responsible for releasing all waiting threads by signaling the semaphore
            # `num_threads` times. It then resets the counter for the next use.
            if count_threads[0] == 0:            
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        # Blocks the thread until it is released by the last arriving thread.
        threads_sem.acquire()

class Device(object):
    """
    Represents a node in the simulated distributed system.
    
    Each device has an ID, sensor data, and runs a dedicated thread (`DeviceThread`)
    to manage its execution lifecycle.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        Args:
            device_id (int): Unique identifier for the device.
            sensor_data (dict): A dictionary of sensor readings keyed by location.
            supervisor (Supervisor): A central object to get network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # Event to signal a batch of scripts is ready.
        self.barrier = None
        self.location_locks = {location: Lock() for location in self.sensor_data}
        self.scripts_available = False
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes the synchronization barrier.
        
        The device with ID 0 is responsible for creating the `ReusableBarrier` and
        sharing it with all other devices.
        
        Args:
            devices (list): A list of all devices in the system.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed or signals the end of a script batch.

        Args:
            script (Script): The script object to execute.
            location (str): The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            # Signals that a full batch of scripts for a timepoint has been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safely retrieves sensor data for a given location.

        Args:
            location (str): The data location to read from.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]     
        else:
            return None

    def set_data(self, location, data):
        """
        Thread-safely updates sensor data for a given location.

        Args:
            location (str): The data location to write to.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
        else:
            return None

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a Device's lifecycle.

    This thread orchestrates the device's main loop: fetching network topology,
    dispatching script jobs to its worker pool (`MyQueue`), and synchronizing
    with other devices using a shared barrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = MyQueue(8) # The dedicated worker pool for this device.

    def run(self):
        """
        The main execution loop for the device.
        """
        self.queue.device = self.device
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: This inner loop manages the asynchronous arrival of scripts.
            # It waits until either new scripts are available or a 'timepoint_done'
            # event signals the end of a batch.
            while True:
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    if self.device.scripts_available:
                        self.device.scripts_available = False

                        # Dispatches all available scripts to the worker queue.
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
            
                    else:
                        # Resets the event for the next cycle and breaks the inner loop.
                        self.device.timepoint_done.clear()
                        self.device.scripts_available = True
                        break
            
            # Synchronization Point: Waits for the worker pool to process all jobs for this cycle.
            self.queue.queue.join()
            # Synchronization Point: Waits for all other devices to complete their cycle.
            self.device.barrier.wait()

        # Initiates a clean shutdown of the worker pool.
        self.queue.finish()
