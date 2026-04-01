"""
This module provides a simulation framework for a network of distributed devices.

It includes classes for a `Device`, its control thread (`DeviceThread`), a worker
thread for script execution (`MyWorker`), and a `ReusableBarrier` for synchronization.
The simulation operates in discrete time steps, synchronized across all devices.
"""

from threading import Thread, Lock, Semaphore, Event


class ReusableBarrier(object):
    """A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a two-phase protocol to allow the barrier to be
    used multiple times. The thread count is stored in a list to ensure
    mutability across method calls.
    """
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """Causes a thread to block until all `num_threads` have called wait."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current thread count for the phase.
            threads_sem (Semaphore): The semaphore to block and release threads.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:        
                # When the last thread arrives, it releases all waiting threads.
                for dummy_i in range(self.num_threads):    
                    threads_sem.release()    
                # Resets the counter for reuse.
                count_threads[0] = self.num_threads    
        threads_sem.acquire()    
                                 

class Device(object):
    """Represents a device in the simulated network.

    Manages sensor data, executes assigned scripts, and communicates with
    neighboring devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of the device's sensor readings.
            supervisor (object): A supervisor object to get neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.rbarrier = None

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    @staticmethod
    def send_barrier(devices, barrier):
        """Distributes the shared barrier object to all devices.

        Args:
            devices (list): A list of all devices in the simulation.
            barrier (ReusableBarrier): The shared barrier instance.
        """
        for dev in devices:
            if dev.rbarrier is None and dev is not None:
                dev.rbarrier = barrier


    def setup_devices(self, devices):
        """Initializes and distributes the barrier for synchronization.

        Intended to be called by a single designated device (e.g., device_id 0).

        Args:
            devices (list): All devices in the simulation.
        """
        if self.device_id == 0:
            mybarrier = ReusableBarrier(len(devices))
            self.rbarrier = mybarrier
            self.send_barrier(devices, mybarrier)

    def assign_script(self, script, location):
        """Assigns a script to the device for execution.

        Args:
            script (object): The script to be executed.
            location (int): The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location.

        Args:
            location (int): The sensor location to query.

        Returns:
            The sensor data or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location.
        Note: This method is not thread-safe.

        Args:
            location (int): The sensor location to update.
            data: The new sensor value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class MyWorker(Thread):
    """A worker thread to execute a script in a specific location context."""

    def __init__(self, device, location, neighbours, script):
        """Initializes the worker.

        Args:
            device (Device): The parent device.
            location (int): The location for data aggregation.
            neighbours (list): A list of neighboring devices.
            script (object): The script to execute.
        """
        Thread.__init__(self)
        self.device = device


        self.location = location
        self.neighbours = neighbours
        self.script = script
        self.script_data = []

    def run(self):
        """Gathers data, runs the script, and distributes the result.
        
        Note: Data gathering from neighbors is not protected by locks, which
        could lead to race conditions in a more complex simulation.
        """
    	
        # Gather data from neighboring devices.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                self.script_data.append(data)
        
        # Gather data from the local device.
        data = self.device.get_data(self.location)
        if data is not None:
            self.script_data.append(data)

        if self.script_data != []:
            
            # Execute the script on the aggregated data.
            result = self.script.run(self.script_data)

            
            # Update data on neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Update data on the local device.
            self.device.set_data(self.location, result)

class DeviceThread(Thread):
    """The main control thread for a device."""

    def __init__(self, device):
        """Initializes the device's main thread.
        
        Args:
            device (Device): The device this thread belongs to.
        """

        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the signal to start processing a new timepoint.
            self.device.timepoint_done.wait()

            # Create and start worker threads for each assigned script.
            thrds = []
            for (script, location) in self.device.scripts:
                thrd = MyWorker(self.device, location, neighbours, script)
                thrds.append(thrd)

            for thrd in thrds:
                thrd.start()
            for thrd in thrds:
                thrd.join()

            
            self.device.timepoint_done.clear()
            # Wait at the barrier for all other devices to finish the timepoint.
            self.device.rbarrier.wait()
