"""
This module provides a Python 2 implementation of a simulated device for a
concurrent system.

It features a custom two-phase `ReusableBarrier` implemented with Semaphores
and a simple, coarse-grained locking strategy using a single global lock for
all script executions within a device.
"""



from threading import Lock, Semaphore, Event, Thread

class ReusableBarrier(object):
    """
    A reusable barrier implemented using semaphores and a lock.

    This barrier synchronizes a fixed number of threads at a certain point.
    It employs a two-phase signaling mechanism (`phase`) to ensure that no
    thread can start a new wait cycle before all threads have completed the
    previous one, preventing race conditions.
    """
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        # Counters are stored in a list to be mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock() 
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """Blocks until all threads have called this method."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Represents one phase of the two-phase barrier.

        Args:
            count_threads (list): The counter for the current phase.
            threads_sem (Semaphore): The semaphore for signaling in this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0: 
                # Last thread to arrive releases the semaphore for all waiting threads.
                i = 0
                while i < self.num_threads:
                    threads_sem.release() 
                    i += 1                
                # Reset the counter for the next use of the barrier.
                count_threads[0] = self.num_threads  
        threads_sem.acquire() 
                              
                              

class Device(object):
    """
    Represents a single device in the simulated network.

    Each device runs in its own thread and coordinates with others using a
    shared barrier and a global lock.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.barrier = None
        self.lock = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization objects for all devices.

        The first device in the list acts as a leader to create a shared
        barrier and a single global lock, which are then distributed to all
        other devices.
        """
        if devices[0].barrier is None:
            if self.device_id == devices[0].device_id:
                bariera = ReusableBarrier(len(devices))
                my_lock = Lock()
                for device in devices:
                    device.barrier = bariera
                    device.lock = my_lock



    def assign_script(self, script, location):
        """Assigns a script to the device for execution."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""

        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a device's lifecycle."""

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device.

        This loop processes assigned scripts in a serial manner, using a
        coarse-grained global lock for each script execution.
        """
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break



            # Wait for a signal to begin processing for the current timepoint.
            self.device.timepoint_done.wait()

            
            # Process each script sequentially.
            for (script, location) in self.device.scripts:
                # Acquire the global lock, preventing parallel execution of scripts
                # even if they operate on different locations.
                self.device.lock.acquire()
                script_data = []
                
                # Gather data from neighbors.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Include the device's own data.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = self.script.run(script_data)

                    
                    # Propagate the result to neighbors and the device itself.


                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                # Release the global lock.
                self.device.lock.release()
            
            # Signal completion of the timepoint and wait at the barrier for all devices.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
