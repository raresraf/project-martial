"""
This module implements a device simulation for a concurrent system.

It features a two-phase semaphore-based reusable barrier and a leader-based
setup for synchronization. However, it contains a critical flaw: the absence
of any locking during script execution, making it highly susceptible to race
conditions.
"""


from threading import Event, Thread

from threading import *
 
class ReusableBarrier():
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a two-phase protocol with two semaphores to ensure
    that threads from a previous waiting cycle do not interfere with threads in
    the current cycle.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters are held in a list to be mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
 
    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                # The last thread to arrive releases the semaphore for all others.
                for i in range(self.num_threads):
                    threads_sem.release()        
                # Reset the counter for the next use.
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
                                                 
 
class Device(object):
    """
    Represents a single device in the simulated network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for all devices.

        The device with ID 0 acts as a leader, creating the barrier instance.
        All other devices then take a reference to the leader's barrier.
        """
        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        else:
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for the device's lifecycle."""

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device.

        This loop is critically flawed as it performs no locking during data
        access and modification, leading to race conditions.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # 1. Wait for the signal to start processing for the current timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            # CRITICAL FLAW: There is no locking mechanism in this loop.
            # When multiple device threads run concurrently, they can read and
            # write to shared neighbor data without synchronization, causing
            # race conditions and data corruption.
            for (script, location) in self.device.scripts:
                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    # Unsafe concurrent writes to shared data.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

            # 2. Wait at the barrier after all (unsafe) processing is done.
            self.device.barrier.wait()
            
