"""
Models a device in a distributed simulation using a custom barrier and
a thread-per-script-chunk execution model.

This module defines a set of classes to simulate a network of devices that
perform computations based on scripts. It uses a custom reusable barrier for
synchronization and spawns worker threads to parallelize script execution within
a single device. The locking mechanism is fine-grained, with locks per data
location.

Classes:
    ReusableBarrier: A custom thread barrier for synchronization.
    DeviceThread_Worker: A worker thread that executes a chunk of scripts.
    Device: Represents a single computational node in the network.
    DeviceThread: The main control thread for a Device.
"""

from threading import Event, Thread
from threading import Condition, RLock    

class ReusableBarrier():
    """
    A custom implementation of a reusable barrier for thread synchronization.

    This barrier blocks a set number of threads until all of them have reached
    the barrier, at which point it releases them all and resets for the next use.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  # Condition variable to manage blocking and notification.
 
    def wait(self):
        """
        Causes the calling thread to wait at the barrier.

        The thread will block until the required number of threads have also
        called this method.
        """
        self.cond.acquire()                      
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread has arrived; notify all waiting threads and reset the barrier.
            self.cond.notify_all()              
            self.count_threads = self.num_threads    
        else:
            # Not all threads have arrived yet; wait for notification.
            self.cond.wait()                    
        self.cond.release()                   

class DeviceThread_Worker(Thread):
    """A worker thread responsible for executing a subset of a device's scripts."""
    def __init__(self, device, neighbours, tid, scripts):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.device = device
        self.scripts = scripts
        self.tid = tid 

    def run(self):
        """
        Processes a list of scripts, gathering data and executing them.

        For each script, it gathers data from its own device and its neighbors
        at a specific location, runs the script, and then updates the data
        on the devices based on the result.
        """
        for (script, location) in self.scripts:
            script_data = []
            index = location

            # --- Data Gathering Phase ---
            # Pre-condition: Acquire locks to ensure safe access to sensor data.
            # This implementation uses a fine-grained locking strategy, locking
            # both the specific location and the device for each data access.
            for device in self.neighbours:
                self.device.locks[index].acquire()
                self.device.lock.acquire()

                data = device.get_data(location)

                self.device.lock.release()
                self.device.locks[index].release()

                if data is not None:
                    script_data.append(data)
            
            # Gather data from the current device as well.
            self.device.locks[index].acquire()
            self.device.lock.acquire()

            data = self.device.get_data(location)

            self.device.lock.release()
            self.device.locks[index].release()

            if data is not None:
                script_data.append(data)

            # --- Script Execution and Data Update Phase ---
            if script_data != []:
                result = script.run(script_data)

                self.device.locks[index].acquire()

                # Propagate the result to neighbors based on a condition.
                # Note: This get_data call is outside the device's own lock.
                for dev in self.neighbours:
                    if result > dev.get_data(location):
                        dev.set_data(location, result)
                    
                # Update the current device's data unconditionally.
                self.device.set_data(location, result)

                self.device.locks[index].release()   


class Device(object):
    """
    Represents a single device, managing its data, scripts, and threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.lock = RLock()
        self.barrier = None
        self.devices = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.locations = []
        
        # A list of locks, presumably one for each possible data location.
        self.locks = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs a centralized, one-time setup for all devices.

        This method is intended to be called on a single device (id 0) to
        initialize shared resources like the synchronization barrier and a
        common pool of location-based locks for all devices.

        Args:
            devices (list[Device]): A list of all devices in the simulation.
        """
        self.devices = devices
        if self.device_id == 0:
            # Create and distribute a fixed number of locks to all devices.
            for num in range(0, 1000):
                lock = RLock()
                for i in range (0, len(devices)):
                    devices[i].locks.append(lock)
            
            # Create and distribute a shared barrier.
            barrier = ReusableBarrier(len(devices)) 
            for i in range(0,len(devices)):
                if devices[i].barrier == None:
                    devices[i].barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device.

        Args:
            script (Script): The script to execute.
            location (any): The data location for the script.
        """
        if script is not None:
            self.scripts.append((script, location)) 
            self.timepoint_done.set()
        else:
            # A None script is a signal to end the current timepoint.
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves data for a given location.

        Returns:
            The data value, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data for a given location, protected by a re-entrant lock."""
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()
    
    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a Device's lifecycle."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def divide_in_threads(self, neighbours):
        """
        Divides the assigned scripts into chunks and processes them in parallel.

        Spawns up to 8 worker threads to handle the script workload for the
        current timepoint.

        Args:
            neighbours (list[Device]): The neighboring devices for this context.
        """
        threads = []

        # Determine the number of chunks and the number of worker threads.
        nr = len(self.device.scripts)
        numar = 1 
        if nr > 8:
            numar = nr / 8
            nr = 8

        # Create and start the worker threads, each with a slice of the script list.
        for i in range(0,nr):
            if i == nr - 1:
                t = DeviceThread_Worker(self.device, neighbours, i, self.device.scripts[i * numar : len(self.device.scripts)])
            else:
                t = DeviceThread_Worker(self.device, neighbours, i, self.device.scripts[i * numar : i*numar + numar])
            threads.append(t)

        for i in range(0, nr):
            threads[i].start()
            
        # Block until all worker threads for this device have completed.
        for i in range(0,nr):
            threads[i].join()

    def run(self):
        """
        The main event loop for the device.
        
        Waits for a signal to start a timepoint, processes scripts in parallel,
        and synchronizes with other devices at a barrier.
        """
        while True:
            # Get neighbors from the supervisor. If None, it's a shutdown signal.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to assign scripts.
            self.device.script_received.wait()
            
            # Execute the assigned scripts using a pool of worker threads.
            self.divide_in_threads(neighbours)
            
            # Clear the event to be ready for the next timepoint.
            self.device.script_received.clear()
            
            # --- End of a timepoint ---
            # Wait at the barrier for all other devices to finish their timepoint.
            self.device.barrier.wait()