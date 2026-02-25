"""
This script implements a simulated network of devices that perform computations
in synchronized time steps. It uses a queue-based model for distributing work
to a pool of worker threads within each device.

The architecture is designed for parallel, distributed computation where each device
operates on its own data and data from its neighbors. A global barrier ensures
all devices complete a time step before the simulation proceeds, and location-based
locks ensure data integrity during updates.
"""

from threading import Event, Semaphore, Lock, Thread
from Queue import Queue

class Device(object):
    """
    Represents a single device (node) in the simulation network.

    It manages its own sensor data, a main control thread, and a collection of
    scripts to be executed at each timepoint.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and starts its main control thread.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): A dictionary of sensor data, keyed by location.
            supervisor (object): A supervisor object that provides network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that all scripts for a timepoint have been received.
        self.scripts_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        # Each device has a main control thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Number of worker threads to spawn for executing scripts.
        self.no_th = 8

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources for the entire simulation.
        This method should be called once on a single device (e.g., device 0).

        Args:
            devices (list): A list of all device objects in the simulation.
        """
        if self.device_id == 0:
            # Create a global barrier for synchronizing all devices between timepoints.
            barrier = ReusableBarrierSem(len(devices))

            # Create a global set of locks, one for each unique data location.
            lock_for_loct = {}
            for device in devices:
                for location in device.sensor_data:
                    if location not in lock_for_loct:
                        lock_for_loct[location] = Lock()

            # Distribute the shared barrier and locks to all devices.
            for device in devices:
                device.barrier = barrier
                device.lock_for_loct = lock_for_loct

    def assign_script(self, script, location):
        """
        Adds a script to the list of tasks for the current timepoint.

        Args:
            script (object): The script to be executed.
            location (any): The data location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A 'None' script signifies the end of script assignment for this timepoint.
            self.scripts_received.set()

    def get_data(self, location):
        """Retrieves data from the device's sensor data store."""
        return self.sensor_data.get(location)

    def set_data(self, location, data, source=None):
        """Updates data in the device's sensor data store."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main control thread to terminate the device cleanly."""
        self.thread.join()

class DeviceThread(Thread):
    """The main control thread that orchestrates a device's operations per timepoint."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = []
        self.neighbours = []

    def run(self):
        """Main execution loop for a device, synchronizing timepoints."""
        while True:
            # Get the current list of neighbors from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                # 'None' neighbors indicates simulation shutdown.
                break

            # Wait until all scripts for the current timepoint are assigned.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear()
            
            # Populate a queue with the assigned scripts for worker threads to process.
            self.queue = Queue()
            for script in self.device.scripts:
                self.queue.put_nowait(script)

            # Spawn a pool of worker threads to process the script queue.
            for _ in range(self.device.no_th):
                SolveScript(self.device, self.neighbours, self.queue).start()
            
            # Wait for all scripts in the queue to be processed by worker threads.
            self.queue.join()
            
            # Wait at the global barrier until all devices have completed this timepoint.
            self.device.barrier.wait()

class SolveScript(Thread):
    """A worker thread that processes computational scripts from a shared queue."""

    def __init__(self, device, neighbours, queue):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.queue = queue

    def run(self):
        """Pulls tasks from the queue and executes them until the queue is empty."""
        try:
            # This loop structure is unusual. The work is driven by queue.get(), not the for loop.
            # The for loop will run for every script, but each thread will only pull some from the queue.
            for _ in self.device.scripts:
                (script, location) = self.queue.get(False)
                
                # Acquire a lock for the specific data location to prevent race conditions.
                self.device.lock_for_loct[location].acquire()

                script_data = []
                # Gather data from neighboring devices and the current device.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Execute the script with the aggregated data.
                    result = script.run(script_data)
                    
                    # Broadcast the result back to all participating devices.
                    for device in self.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the lock for the data location.
                self.device.lock_for_loct[location].release()
                
                # Signal that this task from the queue is complete.
                self.queue.task_done()
        except:
            # A broad exception handler catches errors, including Queue.Empty when
            # the queue is exhausted, allowing the thread to terminate.
            pass

class ReusableBarrierSem():
    """A reusable, two-phase barrier for synchronizing a fixed number of threads."""

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """Blocks the calling thread until all threads reach the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """The second synchronization phase, ensuring all threads from phase 1 are clear."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
