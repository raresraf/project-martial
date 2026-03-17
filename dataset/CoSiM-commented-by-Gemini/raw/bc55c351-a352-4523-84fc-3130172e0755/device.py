"""
This module implements a simulation of a distributed device network.

The core components are the `Device` and `DeviceThread` classes, which model
nodes in a sensor network. These devices can run computational scripts on data,
communicate with their neighbors, and synchronize their operations using a
combination of threads, semaphores, barriers, and events. The simulation is
orchestrated by a `supervisor` entity that provides neighbor information.
"""

from threading import Event, Thread, Semaphore
from barrier import ReusableBarrierSem
from Queue import Queue
from copy import deepcopy
from time import sleep
from random import random

class Device(object):
    """
    Represents a single device (or node) in the distributed network simulation.

    Each device has its own sensor data, a set of scripts to execute, and the
    ability to communicate with other devices. It manages a pool of worker
    threads (`DeviceThread`) to perform its tasks concurrently. Synchronization
    across the entire network of devices is achieved using shared barriers and
    semaphores.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's local sensor readings.
        supervisor: An external entity that provides information about the network topology.
        threads (list): A list of worker threads managed by this device.
    """

    
    # The number of worker threads to spawn for each device.
    NR_THREADS = 8

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor: The central supervisor for the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # A list of all other devices in the simulation, including itself.
        self.devices = []
        # An event to signal that scripts have been received and processing can begin.
        self.script_received = Event()
        # A list to store tuples of (script, location) to be executed.
        self.scripts = []
        # A queue to hold scripts for the worker threads to process.
        self.script_queue = Queue()
        # Semaphore to protect access to location-related information.
        self.loc_dev_semaphore = Semaphore(value=1)
        # A dictionary to track the processing state of different locations.
        self.loc_info = {}
        self.script_reset = False
        # A semaphore shared among all devices to protect shared state.
        self.semaphore_devices = Semaphore()
        # A semaphore to ensure atomic updates to device state.
        self.update_semaphore = Semaphore(value=1)
        self.current_script_state = {}
        self.has_neighbours = False
        # Semaphore to control access to the neighbors list.
        self.neighbours_semaphore = Semaphore(value=1)
        self.neighbours = []
        # A barrier for synchronizing all threads of all devices in the network.
        self.barrier_devices = ReusableBarrierSem(0)
        # A barrier for synchronizing all worker threads within this single device.
        self.barrier_threads = ReusableBarrierSem(Device.NR_THREADS)
        # Semaphores for controlling access to the script queue.
        self.queue_semaphore = Semaphore(value=1)
        self.queue_init_semaphore = Semaphore(value=1)
        self.threads = []

        
        # Creates and starts the pool of worker threads for this device.
        for count in range(0, Device.NR_THREADS):
            self.threads.append(DeviceThread(self))
            self.threads[count].start()


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id


    def setup_devices(self, devices):
        """
        Sets up the network of devices with shared synchronization primitives.

        This method injects a shared barrier and a shared semaphore into all
        devices in the simulation, allowing them to synchronize as a group.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        # Creates a barrier that will wait for all threads from all devices.
        same_barrier = ReusableBarrierSem(len(devices) * Device.NR_THREADS)
        
        semaphore_devices = Semaphore(value=1)

        self.devices = devices

        
        # Injects the shared synchronization objects into each device.
        for device in devices:
            device.barrier_devices = same_barrier
            device.semaphore_devices = semaphore_devices


    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location.

        If a script is provided, it's added to the list of scripts to run.
        If the script is None, it signals that all scripts have been assigned
        and the device can start processing.

        Args:
            script: The script object to be executed.
            location: The location context for the script.
        """
        
        if script is not None:
            self.scripts.append((script, location))

            
            # Propagates the new location to all devices in the network.
            for device in self.devices:
                device.add_location(location)
        else:
            
            # Signals that script assignment is complete.
            self.script_received.set()


    def add_location(self, location):
        """
        Adds a location to the device's internal state tracking.

        Initializes the state for the given location to False (unprocessed).

        Args:
            location: The location to add.
        """
        self.update_semaphore.acquire()

        
        if location in self.loc_info:
            self.loc_info[location] = False
        else:
            self.loc_info.update({location : False})

        self.update_semaphore.release()


    def check_location(self, location):
        """
        Atomically checks and marks a location as being processed.

        This method acts as a distributed lock. It ensures that only one
        thread across all devices can process a script for a given location
        at any one time.

        Args:
            location: The location to check and lock.

        Returns:
            bool: True if the lock was acquired, False otherwise.
        """
        self.semaphore_devices.acquire()

        res = False

        
        # Pre-condition: Checks if the location is currently free.
        if self.current_script_state[location] == False:
            res = True
            
            # Marks the location as processed on all devices to enforce the lock.
            for device in self.devices:
                device.current_script_state[location] = True

        self.semaphore_devices.release()

        return res


    def free_location(self, location):
        """
        Releases the lock on a location, marking it as free.

        Args:
            location: The location to free.
        """
        self.semaphore_devices.acquire()
        for device in self.devices:
            device.current_script_state[location] = False

        self.semaphore_devices.release()


    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location: The location for which to retrieve data.

        Returns:
            The sensor data, or None if the location is not found.
        """
        result = None

        if location in self.sensor_data:
            result = self.sensor_data[location]

        return result


    def set_data(self, location, data):
        """
        Updates the sensor data for a specific location.

        Args:
            location: The location to update.
            data: The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data


    def get_current_neighbours(self):
        """
        Fetches and caches the list of neighboring devices.

        Retrieves neighbor information from the supervisor once per cycle.

        Returns:
            list: A list of neighboring Device objects.
        """
        self.neighbours_semaphore.acquire()
        # Invariant: Fetches neighbours only if they haven't been fetched in the current cycle.
        if self.has_neighbours == False:
            self.neighbours = self.supervisor.get_neighbours()
            self.has_neighbours = True

        self.neighbours_semaphore.release()

        return self.neighbours


    def reset_neigbours(self):
        """Resets the neighbor cache, forcing a refresh in the next cycle."""
        self.has_neighbours = False


    def init_queue(self):
        """
        Initializes the script queue for the current processing cycle.

        This is done once per cycle by one thread, protected by a semaphore.
        """
        self.queue_init_semaphore.acquire()
        # Block Logic: Ensures the queue is populated only if it's empty.
        if self.script_queue.empty() == True:
            for (script, location) in self.scripts:
                self.script_queue.put((script, location))

        self.queue_init_semaphore.release()

    
    def again(self):
        """Resets the script reset flag for the next cycle."""
        self.script_reset = False

    
    def reset_script_state(self):
        """
        Resets the script processing state for all locations.

        This is done at the beginning of a cycle to allow scripts to be
        re-evaluated. A deepcopy is used to create a fresh state map.
        """
        self.loc_dev_semaphore.acquire()

        if self.script_reset == False:
            self.current_script_state = deepcopy(self.loc_info)
            self.script_reset = True

        self.loc_dev_semaphore.release()


    def shutdown(self):
        """Waits for all worker threads to complete."""
        for count in range(0, Device.NR_THREADS):
            self.threads[count].join()


class DeviceThread(Thread):
    """
    A worker thread for a Device.

    This thread executes the main logic loop for processing scripts. It
    synchronizes with other threads and devices to collaboratively work through
    the assigned scripts.
    """

    def __init__(self, device):
        """
        Initializes the worker thread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """The main execution loop for the worker thread."""

        # Block Logic: The main loop continues as long as the supervisor provides neighbors.
        # A `None` value for neighbors is the shutdown signal.
        while True:
            
            # Fetches neighbors for the current cycle. This also acts as a synchronization point.
            neighbours = self.device.get_current_neighbours()
            if neighbours is None:
                break

            
            # Waits until all scripts for the cycle have been assigned.
            self.device.script_received.wait()

            
            # One thread will populate the script queue for everyone.
            self.device.init_queue()

            
            # === GLOBAL SYNC BARRIER 1 ===
            # Waits for all threads across all devices to be ready before starting.
            self.device.barrier_devices.wait()

            
            # Resets the state of all locations to 'unprocessed'.
            self.device.reset_script_state()

            
            # === GLOBAL SYNC BARRIER 2 ===
            # Waits for all threads to acknowledge the state reset.
            self.device.barrier_devices.wait()

            
            # Prepares for the next cycle's script assignment.
            self.device.script_received.clear()

            # Block Logic: This inner loop processes scripts from the shared queue.
            while True:
                
                # Acquires a lock to safely access the script queue.
                self.device.queue_semaphore.acquire()

                
                if self.device.script_queue.empty():
                    self.device.queue_semaphore.release()
                    break

                
                (script, location) = self.device.script_queue.get()

                
                # Pre-condition: Checks if another thread is already processing this location.
                if self.device.check_location(location) == False:
                    
                    last_script = self.device.script_queue.empty()

                    
                    # Inline: If the location is locked, put the script back at the end of the queue to try again later.
                    self.device.script_queue.put((script, location))

                    self.device.queue_semaphore.release()

                    
                    # If this was the last script, a small random delay helps prevent live-lock.
                    if last_script:
                        
                        sleep(random() * 0.3)
                    continue

                self.device.queue_semaphore.release()

                script_data = []

                
                # Block Logic: Gathers data from all neighboring devices for the target location.
                for device in neighbours:

                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                
                # Gathers data from the local device as well.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                
                # Block Logic: Executes the script if any data was gathered.
                if script_data != []:
                    result = script.run(script_data)

                    
                    # Propagates the result to all neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)

                    
                    # Updates the local device's data with the result.
                    self.device.set_data(location, result)

                
                # Releases the lock on the location so other scripts might use its result.
                self.device.free_location(location)

            
            # === GLOBAL SYNC BARRIER 3 ===
            # Waits for all threads in the network to finish processing the script queue.
            self.device.barrier_devices.wait()

            
            # Resets the neighbor cache for the next cycle.
            self.device.reset_neigbours()

            
            # Resets the script state for the next cycle.
            self.device.again()

            
            # === LOCAL SYNC BARRIER ===
            # Synchronizes all threads within this device before the next global sync.
            self.device.barrier_threads.wait()

            
            # === GLOBAL SYNC BARRIER 4 ===
            # Final barrier to ensure the entire network is synchronized before the next full cycle begins.
            self.device.barrier_devices.wait()
