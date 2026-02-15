"""
This module defines a distributed device simulation framework featuring a
controller-worker thread architecture with dynamic thread creation and a
multi-level synchronization strategy.
"""

from threading import Event, Thread, Lock, Condition

class Device(object):
    """
    Represents a device node in the simulation.
    
    Each device has a main controller thread (`DeviceThread`) that orchestrates
    the work for each time step. It holds the device's data and shared
    synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device object.
        
        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): The local data store for the device.
            supervisor (Supervisor): The central object managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # Signals the controller to start a new step.
        self.thread = DeviceThread(self) # The main controller thread for this device.
        self.thread.start()
        self.locations = []
        self.sync_data_lock = Lock() # A per-device lock for data access.
        self.sync_location_lock = {} # A globally shared dictionary of per-location locks.
        self.cores = 8
        self.barrier = None # A globally shared barrier for time step synchronization.

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes global synchronization objects.
        
        Intended to be called by a master device (ID 0). It creates a shared
        barrier and a shared dictionary of location-based locks for all devices.
        """
        if self.device_id == 0:
            locations_number = self.get_locations_number(devices)
            for location in range(locations_number):
                self.sync_location_lock[location] = Lock()
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                device.barrier = self.barrier
                device.sync_location_lock = self.sync_location_lock

    def assign_script(self, script, location):
        """
        Assigns a script to the device's list of tasks for the current step.
        
        A `None` script signals the supervisor is done assigning scripts,
        triggering the `timepoint_done` event for the controller thread.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Note: The script_received event is set but not used for waiting.
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from the device's local sensor data."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data in the device's local sensor data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's controller thread to terminate."""
        self.thread.join()

    def get_locations_number(self, devices):
        """Calculates the number of unique data locations across all devices."""
        for device in devices:
            for location in device.sensor_data:
                if location not in self.locations:
                    self.locations.append(location)
        return len(self.locations)


class DeviceThread(Thread):
    """
    The main controller thread for a single Device. It orchestrates the execution
    of scripts for each time step by dynamically creating worker threads.
    """

    def __init__(self, device):
        """Initializes the controller thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop, executed in discrete time steps."""
        while True:
            # Get the list of neighbors for this time step.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # End of simulation.
                break
            
            # Wait for the supervisor to signal that all scripts for this step are assigned.
            self.device.timepoint_done.wait()
            
            # --- Dynamic Worker Thread Creation ---
            my_threads = []
            num_threads = self.device.cores
            if not my_threads:
                for i in range(num_threads):
                    my_threads.append(MyThread(self))
            
            # Distribute scripts among the new worker threads.
            index = 0
            for (script, location) in self.device.scripts:
                my_threads[index % num_threads].assign_script(script, location)
                index += 1
            
            # Start and wait for all worker threads for this step to complete.
            for i in range(num_threads):
                my_threads[i].set_neighbours(neighbours)
                my_threads[i].start()
            for i in range(num_threads):
                my_threads[i].join()
            
            # Reset the event for the next time step.
            self.device.timepoint_done.clear()
            
            # Synchronize with all other devices before starting the next step.
            self.device.barrier.wait()

class MyThread(Thread):
    """A short-lived worker thread created by DeviceThread to run scripts."""
    
    def __init__(self, parent_device_thread):
        """
        Initializes the worker thread.
        
        Args:
            parent_device_thread (DeviceThread): The controller thread that spawned this worker.
        """
        Thread.__init__(self)
        self.parent = parent_device_thread
        self.scripts = []
        self.neighbours = []

    def set_neighbours(self, neighbours):
        """Sets the list of neighbors for the current time step."""
        self.neighbours = neighbours

    def assign_script(self, script, location):
        """Adds a script to this thread's list of tasks."""
        self.scripts.append((script, location))

    def run(self):
        """Executes the assigned scripts."""
        for (script, location) in self.scripts:
            # Acquire a global lock for this location to serialize access across all devices.
            self.parent.device.sync_location_lock[location].acquire()
            script_data = []
            
            # Gather data from neighbors.
            for device in self.neighbours:
                device.sync_data_lock.acquire()
                data = device.get_data(location)
                device.sync_data_lock.release()
                if data is not None:
                    script_data.append(data)
            
            # Gather data from the local device.
            self.parent.device.sync_data_lock.acquire()
            data = self.parent.device.get_data(location)
            self.parent.device.sync_data_lock.release()
            if data is not None:
                script_data.append(data)

            # If data was found, run the script and write back the results.
            if script_data:
                result = script.run(script_data)
                
                for device in self.neighbours:
                    device.sync_data_lock.acquire()
                    device.set_data(location, result)
                    device.sync_data_lock.release()
                
                self.parent.device.sync_data_lock.acquire()
                self.parent.device.set_data(location, result)
                self.parent.device.sync_data_lock.release()
                
            # Release the global location lock.
            self.parent.device.sync_location_lock[location].release()


class ReusableBarrierCond(object):
    """A custom reusable barrier implemented using a Condition variable."""
    
    def __init__(self, num_threads):
        """Initializes the barrier for a specific number of threads."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks until all threads have reached the barrier."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread has arrived, notify all and reset for next use.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
