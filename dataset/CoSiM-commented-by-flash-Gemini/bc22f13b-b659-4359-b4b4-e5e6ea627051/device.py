"""
This module provides a simulation framework for a network of devices.

Each device operates in parallel, running a pool of threads that execute
assigned scripts. The simulation proceeds in synchronized time steps, managed
by a global barrier and a combination of locks and events.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single device in the simulation.

    Each device manages a pool of worker threads, its own sensor data, and a list
    of scripts to be executed. It coordinates with other devices using shared
    synchronization primitives initialized by a single root device (device 0).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): A dictionary of local sensor data, keyed by location.
            supervisor: An object to get neighborhood information.
        """
        self.device_barrier = None
        self.location_locks = None
        self.script_lock = Lock() 
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbour_acquiring_lock = Lock() 
        self.scripts = []
        self.script_taken = [] 
        self.timepoint_done = Event()
        self.threads = []
        self.current_time_neighbours = None 
        self.other_devices = None 
        self.first_device_setup = Event() 

        self.crt_timestamp_neigh_taken = False 
        
        for i in range(8):
            self.threads.append(DeviceThread(self))
            self.threads[i].start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for all devices in the simulation.

        Device 0 is responsible for creating the global device barrier and
        the set of locks for all sensor locations. Other devices wait for
        this setup to complete before starting.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        self.other_devices = devices
        
        if self.device_id == 0:
            self.device_barrier = ReusableBarrierSem(len(devices))
            self.location_locks = []
            for _ in range(150):
                self.location_locks.append(Lock())
            self.first_device_setup.set()
        else:
            # Wait for device 0 to finish setting up shared resources.
            for device in devices:
                if device.device_id == 0:
                    device.first_device_setup.wait()
                    self.device_barrier = device.device_barrier
                    self.location_locks = device.location_locks
                    return

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals the end of a timepoint.

        If `script` is None, it signals that no more scripts will be assigned
        in the current timepoint. This method then participates in the global
        barrier synchronization and releases the worker threads for the next
        timepoint.

        Args:
            script: The script to be executed.
            location (int): The location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_taken.append(False)
        else:
            # A None script marks the end of the script assignment phase.
            if self.device_barrier is None:
                for device in self.other_devices:
                    if device.device_id == 0:
                        self.device_barrier = device.device_barrier
                        self.location_locks = device.location_locks
                        break
            
            # The main thread waits at the barrier for all other devices.
            self.device_barrier.wait() 
            
            # Reset script tracking for the next timepoint.
            for i in range(len(self.script_taken)):
                self.script_taken[i] = False
            
            # Release worker threads to begin the next timepoint.
            self.timepoint_done.set()
            with self.neighbour_acquiring_lock:
                self.crt_timestamp_neigh_taken = False

    def get_data(self, location):
        """
        Gets sensor data for a given location.

        Args:
            location (int): The location of the sensor.

        Returns:
            The data if the location is on this device, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location.

        Args:
            location (int): The location of the sensor.
            data: The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to terminate."""
        for thread in self.threads:
            if thread.isAlive():
                thread.join()


class DeviceThread(Thread):
    """
    A worker thread for a Device.

    It continuously attempts to claim and execute scripts from the device's
    shared script list.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main execution loop of the worker thread."""
        while True:
            # The first thread to acquire the lock gets the neighbors for this timepoint.
            with self.device.neighbour_acquiring_lock:
                if self.device.crt_timestamp_neigh_taken is False:
                    self.device.current_time_neighbours = self.device.supervisor.get_neighbours()
                    self.device.crt_timestamp_neigh_taken = True
            
            neighbours = self.device.current_time_neighbours

            if neighbours is None: # Shutdown signal from the supervisor.
                break
            
            # --- Work Claiming Loop ---
            # Each thread iterates over the full script list to find an unclaimed script.
            for (script, location) in self.device.scripts:
                with self.device.script_lock:
                    script_index = self.device.scripts.index((script, location))
                    if self.device.script_taken[script_index]:
                        continue # Script already claimed by another thread.
                    else:
                        self.device.script_taken[script_index] = True

                # --- Script Execution ---
                # Acquire the global lock for this specific location.
                self.device.location_locks[location].acquire()
                script_data = []
                
                # Gather data from all neighbors.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather local data.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Run the script and update data on all relevant devices.
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
                
                self.device.location_locks[location].release()

            # Wait for the main thread to signal the start of the next timepoint.
            self.device.timepoint_done.wait()
