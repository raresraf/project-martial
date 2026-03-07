"""
This module provides a Python 2 implementation for a distributed device simulation.

The framework is built for simulating a network of devices that perform computations
in synchronized steps. It features a more advanced synchronization model than a simple
barrier, incorporating fine-grained locking for specific data locations and a
semaphore to limit concurrency. The device with the lowest ID is designated to
coordinate the setup of shared synchronization primitives.
"""
from threading import Semaphore, Event, Lock, Thread

class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a two-phase semaphore protocol to ensure that threads
    from one synchronization round do not interfere with threads from the next.
    It is intended for use in Python 2, as suggested by `xrange`.
    """
    

    def __init__(self, num_threads):
        """
        Initializes the barrier for a specified number of threads.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the two-phase barrier protocol."""
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread to arrive opens the gate for the first phase.
                for _ in xrange(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use

        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase, ensuring all threads have passed phase 1 before reset."""
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread to arrive opens the gate for the second phase.
                for _ in xrange(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use

        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a device in the simulated network.

    Each device has its own thread of execution and manages its own sensor data.
    It coordinates with other devices using a shared barrier and location-specific locks.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): A dictionary holding the device's sensor data.
            supervisor (obj): An object responsible for providing network topology (neighbors).
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.none_script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.timepoint_end = 0
        self.barrier = None
        self.lock_hash = None

    def __str__(self):
        """Returns a string representation of the device."""
        
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """Assigns a shared barrier instance to the device."""
        
        self.barrier = barrier

    def set_locks(self, lock_hash):
        """Assigns a shared dictionary of location-based locks."""
        
        self.lock_hash = lock_hash

    def setup_devices(self, devices):
        """
        Initializes and distributes synchronization primitives (barrier and locks).

        The device with the minimum ID takes the lead in creating a shared ReusableBarrier
        and a dictionary of locks, one for each unique data location across all devices.
        These are then distributed to all other devices in the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        
        ids_list = []
        for dev in devices:
            ids_list.append(dev.device_id)


        # The device with the minimum ID is elected as the master for setup.
        if self.device_id == min(ids_list):
            
            self.barrier = ReusableBarrier(len(devices))
            self.lock_hash = {}

            # Create a lock for each unique sensor data location.
            for dev in devices:
                for location in dev.sensor_data:
                    if location not in self.lock_hash:
                        self.lock_hash[location] = Lock()

            
            
            # Distribute the barrier and locks to other devices.
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.set_barrier(self.barrier)
                    dev.set_locks(self.lock_hash)


    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution in the next time step.

        Args:
            script (obj): The script to be executed. If None, it signals that
                          all scripts for the current step have been assigned.
            location (any): The data location the script will operate on.
        """
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.none_script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the device's main thread to shut it down gracefully."""
        
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a Device."""
    

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Semaphore to limit the number of concurrently running script threads.
        self.semaphore = Semaphore(value=8)

    def run(self):
        """
        The main execution loop for the device.

        This loop continuously waits for scripts, executes them in parallel worker
        threads (`MyThread`), and then synchronizes with all other devices at a barrier.
        """
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal from supervisor.

            
            # Wait for the signal that all scripts for this step have been assigned.
            self.device.none_script_received.wait()
            self.device.none_script_received.clear()

            thread_list = []

            
            # Create and start a worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, neighbours, script, location,
                    self.semaphore)
                thread.start()
                thread_list.append(thread)

            # Wait for all worker threads to complete.
            for i in xrange(len(thread_list)):
                thread_list[i].join()

            
            
            
            # Synchronize with all other devices before proceeding to the next time step.
            self.device.barrier.wait()

class MyThread(Thread):
    """
    A worker thread for executing a single script on a specific data location.
    """
    

    def __init__(self, device, neighbours, script, location, semaphore):
        """
        Initializes the worker thread.

        Args:
            device (Device): The parent device.
            neighbours (list): A list of neighboring devices.
            script (obj): The script to execute.
            location (any): The data location to operate on.
            semaphore (Semaphore): A semaphore to limit overall concurrency.
        """
        

        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.semaphore = semaphore

    def run(self):
        """
        Executes the script logic.

        This involves acquiring the concurrency semaphore, locking the specific
        data location, gathering data, running the script, updating data, and
        finally releasing the lock and semaphore.
        """
        
        self.semaphore.acquire()

        # Acquire a lock for the specific data location to prevent race conditions.
        self.device.lock_hash[self.location].acquire()

        script_data = []

        
        # Gather data from neighboring devices.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        # Gather data from the local device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            # Execute the script with the gathered data.
            result = self.script.run(script_data)

            
            # Broadcast the result to all neighbors.
            for device in self.neighbours:
                device.set_data(self.location, result)

            
            # Update the local device's data.
            self.device.set_data(self.location, result)

        
        # Release the lock for the data location.
        self.device.lock_hash[self.location].release()

        
        # Release the concurrency-limiting semaphore.
        self.semaphore.release()
