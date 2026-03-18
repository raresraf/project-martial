
"""
Models a distributed system of interconnected devices that execute computational
scripts based on shared sensor data. The system employs a multi-threaded,
barrier-synchronized architecture to simulate parallel processing and data
exchange cycles.
"""
from threading import Event, Thread, Condition, Lock
from Queue import Queue

class Device(object):
    """
    Represents a single computational node in the distributed system.

    Each device manages its own sensor data, executes assigned scripts, and
    communicates with neighboring devices under the coordination of a central
    supervisor. It uses locks to ensure thread-safe access to its data during
    concurrent script executions.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding the device's sensor readings,
                                keyed by location.
            supervisor (Supervisor): An object responsible for providing network
                                     topology information (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []       # Stores tuples of (script, location) to be executed.
        self.locks = {}         # A dictionary of locks to protect sensor_data access, keyed by location.
                                    
        self.no_more_scripts = Event()  # An event to signal changes in the script list.
                                            
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the synchronization barrier for a group of devices.

        If this is the designated primary device (device_id == 0), it creates the
        shared barrier and distributes it to all other devices in the network.

        Args:
            devices (list): A list of all Device objects in the system.
        """
        # Invariant: The device with ID 0 acts as the master for setting up
        # the synchronization barrier for all devices in the simulation.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))

        # Block Logic: Distributes the barrier reference to all other devices.
        for device in devices:
            if device is not self:
                device.set_barrier(self.barrier)


    def assign_script(self, script, location):
        """
        Assigns a new script to the device or signals the end of script assignments.

        Args:
            script (Script): The script object to be executed.
            location (str): The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: Signals to the device's thread that the current
            # batch of scripts has been fully assigned.
            self.no_more_scripts.set()

    def get_data(self, location):
        """
        Thread-safely retrieves sensor data from a specified location.

        Acquires a lock for the location before reading.

        Args:
            location (str): The key for the desired sensor data.

        Returns:
            The sensor data, or None if the location is invalid.
        """
        if location in self.sensor_data:
            # Pre-condition: A lock must be acquired to ensure exclusive access
            # to the sensor data for this location.
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Thread-safely updates sensor data at a specified location.

        Releases the lock for the location after writing.

        Args:
            location (str): The key for the sensor data to update.
            data: The new data to be stored.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    
    def set_barrier(self, barrier):
        """
        Assigns a synchronization barrier to the device.

        Args:
            barrier (ReusableBarrier): The barrier instance to use for synchronization.
        """
        self.barrier = barrier

    def shutdown(self):
        """
        Gracefully shuts down the device and its associated threads.
        """
        # Block Logic: Ensures all child worker threads complete their execution
        # before the main device thread is joined.
        for thread in self.thread.child_threads:
            if thread.is_alive():
                thread.join()
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages a pool of worker threads to execute a device's scripts in parallel.

    This thread orchestrates the main processing loop for a device, fetching
    neighbor information, dispatching script execution jobs to a worker pool,
    and synchronizing with other devices at the end of each processing cycle.
    """

    def __init__(self, device):
        """
        Initializes the device's main execution thread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue()        # A queue for dispatching script jobs to worker threads.
        self.child_threads = []     # A pool of worker threads for script execution.
        self.max_threads = 8        # The number of worker threads in the pool.


    def run(self):
        """
        The main execution loop for the device.

        This loop continuously fetches neighbors, processes scripts, and synchronizes.
        It manages a worker thread pool to execute scripts concurrently.
        """
        # Block Logic: Initializes a lock for each sensor data location to ensure
        # thread-safe access during concurrent script execution.
        for location, data in self.device.sensor_data.iteritems():
            self.device.locks[location] = Lock()

        # Block Logic: Spawns a fixed-size pool of worker threads that will process
        # script execution jobs from the queue.
        for i in xrange(self.max_threads):
            thread = Thread(target=process_scripts, args=(self.queue,))
            self.child_threads.append(thread)
            thread.start()

        # Invariant: This loop represents one full cycle of the distributed algorithm:
        # data gathering, computation, and synchronization.
        while True:
            # Retrieves the current network topology from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            if neighbours is None:
                # Block Logic: Termination sequence. A sentinel value (None) is placed
                # on the queue for each worker thread to signal it to exit.
                for i in xrange(len(self.child_threads)):
                    self.queue.put(None)
                
                self.queue.join() # Waits for all worker threads to finish processing.
                break

            done_scripts = 0
            
            # Block Logic: Dispatches all currently assigned scripts as jobs to the
            # worker thread pool via the queue.
            for (script, location) in self.device.scripts:
                job = {}
                job['script'] = script
                job['location'] = location
                job['device'] = self.device
                job['neighbours'] = neighbours
                self.queue.put(job)     
                done_scripts += 1       

            # Functional Utility: Waits for a signal indicating that the current batch of
            # scripts has been assigned. This prevents the loop from proceeding
            # prematurely if scripts are assigned incrementally.
            self.device.no_more_scripts.wait()
            self.device.no_more_scripts.clear()
            
            # Block Logic: Handles scripts that may have been added after the initial
            # dispatch but before the `no_more_scripts` event was set.
            if done_scripts < len(self.device.scripts):
                for (script, location) in self.device.scripts[done_scripts:]:
                    job = {}
                    job['script'] = script
                    job['location'] = location
                    job['device'] = self.device
                    job['neighbours'] = neighbours
                    self.queue.put(job)     

            # Synchronization Point: Blocks until all jobs in the queue for the current
            # cycle have been processed by the worker threads.
            self.queue.join()

            # Synchronization Point: Blocks until all devices in the system have
            # reached this point, ensuring all devices proceed to the next cycle together.
            self.device.barrier.wait()

def process_scripts(queue):
    """
    The target function for worker threads, processing script execution jobs.

    This function runs in a loop, taking jobs from a shared queue. Each job
    involves gathering data, executing a script, and broadcasting the result.

    Args:
        queue (Queue): The shared job queue.
    """
    while True:
        job = queue.get()
        
        # Pre-condition: A None job is a sentinel indicating the thread should terminate.
        if job is None:
            queue.task_done()
            break
        
        script = job['script']
        location = job['location']
        mydevice = job['device']
        neighbours = job['neighbours']

        script_data = []
        
        # Block Logic: Gathers data from all neighboring devices for the script's
        # specified location, forming the input for the script.
        for device in neighbours:
            if device is not mydevice:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        
        # Block Logic: Gathers data from the local device itself.
        data = mydevice.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Core Logic: Executes the computational script with the aggregated data.
            result = script.run(script_data)

            # Block Logic: Broadcasts the computed result by updating the data on all
            # neighboring devices and the local device.
            for device in neighbours:
                if device is not mydevice:
                    device.set_data(location, result)
            
            mydevice.set_data(location, result)
        
        queue.task_done()



class ReusableBarrier(object):
    """
    A simple, reusable barrier for thread synchronization.

    This barrier allows a specified number of threads to wait until all of them
    have reached the barrier. It then releases all of them and resets itself
    for subsequent use.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait on the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition() # A condition variable to manage waiting and notification.
                                                 
    def wait(self):
        """
        Causes a thread to block until all threads have reached the barrier.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # Invariant: When the last thread arrives (count_threads == 0),
        # it notifies all waiting threads and resets the barrier for the next cycle.
        if self.count_threads == 0: 
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
