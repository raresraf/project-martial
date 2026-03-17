"""
This module provides a device simulation framework utilizing a thread pool model with
active polling for tasks. Synchronization is managed through a complex system of shared
locks and events, ensuring both intra-device and inter-device consistency. One device
acts as a setup coordinator to distribute synchronization primitives.
"""

from threading import Event, Thread, Lock
# Assumes the presence of a 'barrier.py' file with a ReusableBarrierSem implementation.
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a device in the simulation, managing a pool of worker threads,
    local sensor data, and shared synchronization objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and immediately starts its pool of worker threads.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's local data store.
            supervisor (object): The central supervisor managing the simulation.
        """
        self.device_barrier = None
        self.location_locks = None
        # Lock to coordinate the claiming of scripts by worker threads.
        self.script_lock = Lock() 
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Lock to ensure neighbors are fetched only once per timepoint.
        self.neighbour_acquiring_lock = Lock() 
        self.scripts = []
        # A list of flags parallel to `scripts` to mark if a script has been "claimed" by a worker.
        self.script_taken = [] 
        # Event to signal the end of a timepoint, releasing worker threads to start the next cycle.
        self.timepoint_done = Event()
        self.threads = []
        # A cache for the list of neighbors for the current timestamp.
        self.current_time_neighbours = None 
        self.other_devices = None 
        # Event to signal that the master device (id 0) has finished setting up shared objects.
        self.first_device_setup = Event() 

        # Flag to check if neighbors have been fetched for the current timestamp.
        self.crt_timestamp_neigh_taken = False 
        
        # Block Logic: Creates and starts a fixed pool of 8 worker threads upon initialization.
        for i in range(8):
            self.threads.append(DeviceThread(self))
            self.threads[i].start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        A centralized setup routine for distributing shared synchronization objects.
        Device 0 creates the objects, and all other devices wait for it to finish
        before copying references to those objects.
        """
        self.other_devices = devices
        
        # Block Logic: The device with ID 0 is responsible for creating shared objects.
        if self.device_id == 0:
            self.device_barrier = ReusableBarrierSem(len(devices))
            self.location_locks = []
            # Creates a list of locks, one for each potential data location.
            for _ in range(150):
                self.location_locks.append(Lock())
            # Signals that the setup is complete.
            self.first_device_setup.set()
        else:
        	# Block Logic: All other devices wait for device 0 to complete setup.
            for device in devices:
                if device.device_id == 0:
                    device.first_device_setup.wait()
                    # They then copy the references to the shared objects.
                    self.device_barrier = device.device_barrier
                    self.location_locks = device.location_locks
                    return

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script triggers the end-of-timepoint
        synchronization logic.
        """
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_taken.append(False)
        else:
            # Pre-condition: All scripts for the current timepoint have been assigned.
            # This is the start of the end-of-timepoint synchronization sequence.
            if self.device_barrier is None:
                for device in self.other_devices:
                    if device.device_id == 0:
                        self.device_barrier = device.device_barrier
                        self.location_locks = device.location_locks
                        break

            # Synchronization Point: All devices wait here before the timepoint is officially declared "done".
            self.device_barrier.wait() 
            
            # Resets the script "claimed" flags for the next timepoint.
            for i in range(len(self.script_taken)):
                self.script_taken[i] = False
            # Releases the worker threads, which are waiting on this event, to begin the next cycle.
            self.timepoint_done.set()
            # Resets the neighbor fetching flag for the next timepoint.
            self.neighbour_acquiring_lock.acquire()
            self.crt_timestamp_neigh_taken = False
            self.neighbour_acquiring_lock.release()

    def get_data(self, location):
        """Safely retrieves data from a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Safely updates data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all worker threads to ensure a clean shutdown."""
        for thread in self.threads:
            if thread.isAlive():
                thread.join()


class DeviceThread(Thread):
    """
    A worker thread that actively polls for available scripts on its parent device,
    "claims" one, and executes it.
    """

    def __init__(self, device):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for the worker. It continuously polls for work, executes it,
        and then waits for the timepoint to end.
        """
        while True:
            # Block Logic: This locked section ensures that the list of neighbors is fetched
            # from the supervisor only once per device per timepoint by the first thread that gets here.
            self.device.neighbour_acquiring_lock.acquire()
            if self.device.crt_timestamp_neigh_taken is False:
                self.device.current_time_neighbours = self.device.supervisor.get_neighbours()
                self.device.crt_timestamp_neigh_taken = True
            self.device.neighbour_acquiring_lock.release()

            neighbours = self.device.current_time_neighbours

            # A `None` neighbor list is the termination signal.
            if neighbours is None:
                break
            
            # Block Logic: This loop represents the worker polling for an available script.
            for (script, location) in self.device.scripts:
                self.device.script_lock.acquire()
                # Pre-condition: The thread checks if the script has already been claimed.
                if self.device.script_taken[self.device.scripts.index((script, location))]:
                    # If claimed, release the lock and poll the next script.
                    self.device.script_lock.release()
                    continue
                else:
                    # If not claimed, claim it by setting the flag to True.
                    self.device.script_taken[self.device.scripts.index((script, location))] = True
                self.device.script_lock.release()

                # Synchronization Point: Acquire the global lock for this specific data location.
                # This ensures that no other thread in the entire system can work on this location.
                self.device.location_locks[location].acquire()
                script_data = []
                
                # Block Logic: Gather data from neighbors and the local device.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Invariant: The script is only run if there is data to process.
                if script_data != []:
                    result = script.run(script_data)
                    
                    # Block Logic: Propagate results back to neighbors and the local device.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                self.device.location_locks[location].release()

            # Synchronization Point: After polling all scripts, the worker waits here until the
            # supervisor signals the end of the timepoint, which unblocks this event.
            self.device.timepoint_done.wait()