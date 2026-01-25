
"""
Models a device in a distributed sensor network simulation.

This module defines the core components of a simulated device, including its
state, communication with other devices, and the execution of scripts on its
collected data. It uses threading to simulate concurrent device operation and
synchronization.
"""

from threading import Event, Thread, Condition, Lock


class Device(object):
    """Represents a single device in the simulated network.

    Each device runs in its own thread, processes sensor data, and exchanges
    information with its neighbors under the coordination of a supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local
                sensor readings, keyed by location.
            supervisor (Supervisor): The central supervisor managing the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been received.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that a timepoint's computation is complete.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        # Synchronization barrier for all devices in the simulation.
        self.barr = None
        # Mutual exclusion lock for accessing shared data.
        self.lock = None
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes shared synchronization objects for a list of devices.

        This method is intended to be called once to set up the barrier and lock
        for all participating devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Establishes a common barrier for all devices if not already set.
        if devices[0].barr is None and self.device_id == devices[0].device_id:
                barr = CondBarrier(len(devices))
                for i in devices:
                        i.barr = barr
        # Establishes a common lock for all devices.
        lock = Lock()	
        for d in devices:
                d.lock = lock 
		
		

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        If the script is None, it signals that the current timepoint is done.
        
        Args:
            script: The script object to be executed.
            location: The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.

        Args:
            location: The location identifier for the sensor data.
        
        Returns:
            The sensor data, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location.

        Args:
            location: The location identifier for the sensor data.
            data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
	
    def shutdown(self):
        """Joins the device's thread, effectively shutting it down."""
        self.thread.join()

class CondBarrier():
    """A reusable barrier implementation using a Condition variable.
    
    This barrier blocks a set number of threads until all of them have
    called the wait() method.
    """
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Waits for all threads to reach the barrier.

        When the last thread arrives, it notifies all waiting threads and
        resets the barrier for the next use.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # If this is the last thread to arrive, wake up all others.
        if self.count_threads == 0:
            self.cond.notify_all()
            # Reset the barrier for the next round of synchronization.
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()



class DeviceThread(Thread):
    """The execution thread for a single device."""

    def __init__(self, device):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
		

    def run(self):
        """The main loop for the device thread.

        Continuously waits for a signal to process a timepoint, executes
        assigned scripts on data from itself and its neighbors, and then
        synchronizes with other devices.
        """
        while True:
            # Fetches the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Sentinel for shutdown.
                break



            # Waits for the supervisor to signal that a timepoint is ready for processing.
            self.device.timepoint_done.wait()

            
            
            # Iterates through all scripts assigned to the device for the current timepoint.
            for (script, location) in self.device.scripts:
                # Acquires a lock to ensure exclusive access to shared sensor data.
                self.device.lock.acquire()
                script_data = []
                
                # Gathers data from all neighboring devices for the script's target location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Includes its own data in the dataset for the script.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Executes the script only if there is data to process.
                if script_data != []:
                    
                    # The script computes a new result based on the aggregated data.
                    result = script.run(script_data)

                    


                    # Propagates the new result back to all neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Updates its own data with the new result.
                    self.device.set_data(location, result)
                # Releases the lock after data modification is complete.
                self.device.lock.release()
            
            # Clears the event to prepare for the next timepoint.
            self.device.timepoint_done.clear()
            # Waits at the barrier for all other devices to finish their computations for the current timepoint.
            self.device.barr.wait()
