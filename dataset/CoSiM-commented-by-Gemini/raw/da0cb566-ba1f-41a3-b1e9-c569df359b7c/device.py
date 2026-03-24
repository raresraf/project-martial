"""
This module contains a simulation of a distributed device network.

It defines a `Device` class and its main `DeviceThread`. The implementation is
notable for its reliance on global variables for synchronization primitives
(BARRIER, LOCK, L_LOCKS) and a master-worker setup where device 0 initializes
these resources.

A key feature is that script execution within a single device is sequential,
not concurrent, as the `DeviceThread` itself processes each script in a loop.

The file appears to be a concatenation of multiple files, including a
`ReusableBarrier` implementation and a `MyThread` test class.
"""

from rr import ReusableBarrier
from threading import Event, Thread, Lock


# Global variables for shared synchronization state. Using globals is
# generally discouraged in favor of passing shared objects.
L_LOCKS = {}
LOCK = Lock()
BARRIER = None

class Device(object):
    """Represents a single device in the network."""
    

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its main control thread."""
        
        self.event = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

    def __str__(self):
        """String representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and starts the device threads with a shared barrier.

        Device 0 (master) creates the global barrier and then signals all other
        devices to start their main threads.
        """
        
        if self.device_id > 0:
            # Worker devices wait for the master to create the barrier.
            self.event.wait()
            self.thread.start()
        else:
            # Master device (id 0) creates the global barrier.
            global BARRIER
            BARRIER = ReusableBarrier(len(devices))
            # Signal all worker devices that they can now start.
            for device in devices:
                if device.device_id > 0:
                    device.event.set()

            self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be executed, or signals the end of a timepoint."""
        
        if script is None:
            # A 'None' script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()
        else:
            self.scripts.append((script, location))


    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        
        return self.sensor_data[location] if location in self.sensor_data \
	else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device.
    
    This thread processes all assigned scripts sequentially within its run loop.
    """
    

    def __init__(self, device):
        """Initializes the device's main thread."""
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main device loop. It synchronizes at a barrier, waits for scripts for
        the current timepoint, and then executes them sequentially.
        """
        
	
        while True:
            global BARRIER
            BARRIER.wait()
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals termination.
                break

            # Wait for the supervisor to signal all scripts have been assigned.
            self.device.timepoint_done.wait()
            cs = self.device.scripts

            
            # NOTE: All scripts are executed sequentially within this single DeviceThread.
            # This is not a concurrent execution model for scripts on the same device.
            for (script, location) in self.device.scripts:
                global LOCK
                LOCK.acquire() # Acquire global lock to safely check/create location lock.

                global L_LOCKS
                # Lazily initialize a lock for the location if it doesn't exist.
                if not location in L_LOCKS.keys():
                    L_LOCKS[location] = Lock()


                L_LOCKS[location].acquire() # Acquire the specific lock for this location.
                LOCK.release() # Release the global lock.

                script_data = []
                
                # Gather data from neighbors and self.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    # Run the script and propagate the result.
                    result = script.run(script_data)

                    
		    
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                # Release the location-specific lock.
                L_LOCKS[location].release()

            
            # Reset for the next timepoint.
            self.device.timepoint_done.clear()
# The following appears to be content from a different file, concatenated here.
from threading import Thread, Lock, Semaphore

class ReusableBarrier():
    """
    A custom reusable barrier for thread synchronization.
    
    This implementation appears to be from a Python 2 context, using a list
    to hold a mutable integer count.
    """
    
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]


        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
	
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all threads have reached the barrier."""
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier protocol."""
        
        with self.count_lock:
            count_threads[0] -= 1
	    
            if count_threads[0] == 0:
                for i in range(self.num_threads):
	   	    
                    threads_sem.release()
		
                count_threads[0] = self.num_threads
	
        
        threads_sem.acquire()

class MyThread(Thread):
    """
    A simple test thread for a ReusableBarrier.
    
    Note: Uses `xrange`, which is a Python 2 feature.
    """
    
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier

    def run(self):
        
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + \
            " after barrier, in step " + str(i) + "\n",

