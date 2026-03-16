"""
This module implements a simulation of a synchronized, distributed device network.

It models a set of devices that operate in discrete, synchronized time steps. At each
step, devices can execute scripts that process data from their local neighborhood
(the device itself and its immediate neighbors). This is a common pattern in simulations
of sensor networks, IoT systems, or other parallel computations.

The synchronization is achieved using a reusable barrier, ensuring that all devices
complete a time step before any can proceed to the next. The system employs a
fine-grained locking mechanism to ensure data consistency when scripts access and
modify sensor data at specific "locations".

Classes:
    Device: Represents a node in the distributed network.
    DeviceThread: The execution thread for a Device, containing the main simulation loop.
    ReusableBarrier: A classic two-phase reusable barrier implementation for synchronization.
    MyThread: A simple example thread class to demonstrate the barrier.
"""
from rr import ReusableBarrier
from threading import Event, Thread, Lock


# Global dictionary to hold locks for specific data locations. This allows for
# fine-grained locking, where operations on different locations can run in parallel,
# but operations on the same location are serialized.
L_LOCKS = {}
# A global lock to protect access to the `L_LOCKS` dictionary itself, preventing
# race conditions when multiple threads try to create a new location lock simultaneously.
LOCK = Lock()
# The global barrier instance used to synchronize all device threads at each time step.
BARRIER = None

class Device(object):
    """
    Represents a single device (or node) in the simulated network.

    Each device has its own sensor data, a set of scripts to execute, and runs
    within its own thread. It interacts with a supervisor to get information
    about its neighbors.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor readings,
                                keyed by location.
            supervisor (Supervisor): An object that provides network topology information,
                                     such as the neighbors of this device.
        """
        self.event = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the startup of all devices in the simulation.

        This method implements a custom startup synchronization protocol. Device 0 acts
        as the master: it creates the global barrier and then signals all other
        devices to start their main execution threads. Other devices wait on an
        event until the master is ready.
        """
        if self.device_id > 0:
            # Worker devices wait for the signal from the master device.
            self.event.wait()
            self.thread.start()
        else:
            # Device 0 is the master and initializes the global barrier.
            global BARRIER
            BARRIER = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id > 0:
                    # Signal all worker devices to start.
                    device.event.set()

            self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a specific location.

        This method is called by the simulation supervisor to dispatch tasks
        for the next time step.
        """
        if script is None:
            # A None script is a signal that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()
        else:
            self.scripts.append((script, location))


    def get_data(self, location):
        """Retrieves sensor data from a specific location on this device."""
        return self.sensor_data[location] if location in self.sensor_data \
	else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location on this device."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device.

    This thread contains the primary simulation loop, which runs in synchronized
    time steps (timepoints).
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The core simulation loop for the device.

        The loop is structured as a series of synchronized time steps. In each step,
        the device executes assigned scripts, which involves gathering data from
        itself and its neighbors, computing a result, and propagating that result
        back to the neighborhood.
        """
        while True:
            # -- Synchronization Point --
            # All devices wait here until every device in the simulation is ready
            # to begin the next time step.
            global BARRIER
            BARRIER.wait()
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbors is the signal to shut down the simulation.
                break

            # Wait for the supervisor to assign all scripts for this timepoint.
            self.device.timepoint_done.wait()
            cs = self.device.scripts

            
            # -- Script Execution Phase --
            # Process all scripts assigned for the current time step.
            for (script, location) in self.device.scripts:
                # -- Critical Section for Location Lock --
                # This section ensures that only one device thread can be creating a lock
                # for a new location at any given time.
                global LOCK
                LOCK.acquire()

                # Acquire a lock specific to the data location. This ensures that even
                # on different devices, scripts operating on the same location are serialized.
                global L_LOCKS
                if not location in L_LOCKS.keys():
                    L_LOCKS[location] = Lock()
                L_LOCKS[location].acquire()
                LOCK.release()

                script_data = []
                
                # -- Data Gathering Phase --
                # Collect data from all neighboring devices for the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Also collect data from the current device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    # -- Computation Phase --
                    # Execute the script with the gathered data.
                    result = script.run(script_data)

                    
		    
                    # -- Data Propagation Phase --
                    # Write the result back to all neighboring devices and the current device.
                    # This simulates a consensus or data update mechanism.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                
                # Release the lock for this location, allowing other scripts to process it.
                L_LOCKS[location].release()

            
            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()
from threading import Thread, Lock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a two-phase approach to allow the barrier to be used
    multiple times. Threads wait on one semaphore for the first phase and a second
    semaphore for the second phase, which prevents threads from a previous `wait()`
    call from interfering with threads in a subsequent one.
    """


    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        # Use a list to hold the counter so it can be passed by reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
	
        # Semaphore for the first phase of the barrier.
        self.threads_sem1 = Semaphore(0)
        
        # Semaphore for the second phase of the barrier.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have arrived.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier wait.

        Args:
            count_threads (list): A list containing the counter for the current phase.
            threads_sem (Semaphore): The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
	    
            if count_threads[0] == 0:
                # The last thread to arrive releases all other waiting threads.
                for i in range(self.num_threads):
	   	    
                    threads_sem.release()
		
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
	
        # All threads wait here until the last thread releases the semaphore.
        threads_sem.acquire()

class MyThread(Thread):
    """
    An example thread class to demonstrate the usage of the ReusableBarrier.
    """
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier

    def run(self):
        """
        The thread's execution logic, demonstrating repeated barrier synchronization.
        """
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + \
            " after barrier, in step " + str(i) + "\n",