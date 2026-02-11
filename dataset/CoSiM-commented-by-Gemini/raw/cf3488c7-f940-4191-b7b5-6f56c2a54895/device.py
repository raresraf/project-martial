"""
This module defines a device simulation for a concurrent system using Python 2.

It includes a custom implementation of a two-phase reusable barrier using
semaphores and employs a simple, coarse-grained global locking mechanism to
synchronize script execution.
"""


from threading import Thread, Lock, Semaphore, Event


class ReusableBarrier(object):
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
        """Causes the calling thread to wait at the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the two-phase barrier synchronization.

        Args:
            count_threads (list): The counter for the current phase.
            threads_sem (Semaphore): The semaphore to signal on for this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases the semaphore for all others.
                for _ in range(self.num_threads):
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
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barr = None
        self.lock = None
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.

        A leader device (the one at index 0) creates a shared barrier and a
        single global lock, then provides them to all devices in the network.
        """
        root_node = 0
        
        if devices[root_node].barr is None:


            if devices[root_node].device_id == self.device_id:
                barr = ReusableBarrier(len(devices))
                lock = Lock()
                for i in  devices:
                    i.barr = barr
                    i.lock = lock

    def assign_script(self, script, location):
        """Assigns a script to the device for later execution."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Sets the sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for the device's execution lifecycle."""

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device.

        It processes assigned scripts serially for each timepoint, using a
        coarse-grained global lock.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the signal to proceed with the current timepoint.
            self.device.timepoint_done.wait()

            
            # Iterate through assigned scripts and execute them one by one.
            for (script, location) in self.device.scripts:
                # Acquire the single global lock, ensuring serial execution of scripts.
                self.device.lock.acquire()
                script_data = []
                
                # Gather data from neighbors.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the current device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    # Propagate the result to all neighbors and the current device.


                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                # Release the global lock.
                self.device.lock.release()
            
            self.device.timepoint_done.clear()
            # Wait at the barrier for all devices to complete the timepoint.
            self.device.barr.wait()
