"""
This module simulates a network of communicating devices.
It features a properly implemented reusable barrier for synchronization and a
model where each device spawns new threads for each script it needs to execute
in a given time step. Synchronization across devices for specific data
locations is handled by a shared dictionary of locks.
"""


from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier():
    """
    A reusable two-phase barrier implemented with semaphores.
    Allows a group of threads to wait for each other at a synchronization point,
    and can be reused multiple times.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Use a list to hold the count, allowing it to be modified by reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to block. The barrier is passed when all threads have
        called this method.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the two-phase barrier.
        
        Args:
            count_threads (list): A list containing the counter for the current phase.
            threads_sem (Semaphore): The semaphore used for signaling in this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            
            if count_threads[0] == 0:
                # The last thread to arrive releases all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        
        threads_sem.acquire()
        

class Device(object):
    """
    Represents a single device in the network, managing its own data, scripts,
    and a control thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.lock = Lock() # Lock for internal sensor_data access.
        self.locs = []
        self.hashset = {} # Shared dictionary of location-based locks.
        self.bariera = ReusableBarrier(1) # Placeholder barrier.
        
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for all devices, run by a master device.
        The device with ID 0 is designated as the master.
        """
        if self.device_id == 0:
            self.hashset = {}
            # Collect all unique locations from all devices and create a lock for each.
            for device in devices:
                for location in device.sensor_data:
                    self.hashset[location] = Lock()
            
            # Create a shared barrier for all devices.
            self.bariera = ReusableBarrier(len(devices))
            
            # Distribute the shared resources to all devices.
            for device in devices:
                device.bariera = self.bariera
                device.hashset = self.hashset

    def assign_script(self, script, location):
        """Assigns a script to the device or signals the end of a timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safe method to retrieve sensor data for a given location.
        """
        self.lock.acquire()
        aux = self.sensor_data[location] if location in self.sensor_data else None
        self.lock.release()
        return aux

    def set_data(self, location, data):
        """
        Thread-safe method to update sensor data for a given location.
        """
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It spawns worker threads for each script
    and manages synchronization between timepoints.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """
        The main simulation loop for the device.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.


            # Wait for the supervisor to assign all scripts for this timepoint.
            self.device.timepoint_done.wait()
            
            list_threads = []
            # Create and start a new thread for each assigned script.
            for (script, location) in self.device.scripts:
                list_threads.append(ScriptThread(self.device, script,
                location, neighbours))
            
            for i in xrange(len(list_threads)):
                list_threads[i].start()
            
            # Wait for all script threads to complete their execution.
            for i in xrange(len(list_threads)):
                list_threads[i].join()
            
            self.device.timepoint_done.clear()
            # Synchronize with all other devices before starting the next timepoint.
            self.device.bariera.wait()

class ScriptThread(Thread):
    """
    A worker thread responsible for executing a single script.
    """

    def __init__(self, device, script, location, neighbours):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script logic: lock, collect data, compute, update, and unlock.
        """
        # Acquire the shared lock for this specific location to prevent race conditions.
        self.device.hashset[self.location].acquire()
        script_data = []
        
        # Collect data from neighboring devices.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Collect data from the local device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Run the script with the collected data.
            result = self.script.run(script_data)

            # Update the data on all neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Update the data on the local device.
            self.device.set_data(self.location, result)

        # Release the lock for the location.
        self.device.hashset[self.location].release()
