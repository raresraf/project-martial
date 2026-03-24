"""
This module provides a simulation framework for a network of devices that
process data in synchronized time steps. It models devices with a limited
number of processing cores and uses a barrier for synchronization.

Key components:
- Device: Represents a device node with sensor data and assigned scripts. It
  manages data access with a per-location locking mechanism.
- DeviceCore: A thread representing a single processing core on a device. It
  executes a script by gathering data from its parent device and neighbors.
- DeviceThread: The main control loop for a device, managing a pool of
  DeviceCore threads and synchronizing with other devices at each time step
  using a barrier.
"""


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single device in the network simulation.

    Each device has an ID, local sensor data, and can be assigned scripts
    to execute. It coordinates with a supervisor to understand the network
    topology and synchronizes with other devices using a shared barrier.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): A dictionary of the device's local sensor data,
                                keyed by location.
            supervisor: The supervisor object managing the network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.start_event = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        # A dictionary of locks, one for each data location, to control access.
        self.data_lock = {}

        
        for data in sensor_data:
            self.data_lock[data] = Lock()

        self.barrier = None

    def __str__(self):
        """String representation of the Device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared barrier for all devices in the simulation.
        """
        

        
        # The first device to call this method creates the barrier.
        if self.barrier == None:
            self.barrier = ReusableBarrierSem(len(devices))
            # All other devices will share this same barrier instance.
            for dev in devices:
                dev.barrier = self.barrier

        # Signal that the setup is complete and all devices can start their main loops.
        self.start_event.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution.
        """
        

        
        if script is not None:
            # A script of None is a signal to end the timepoint.
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

        # Notify the device's main thread that a script (or timepoint end) has arrived.
        self.script_received.set()

    def get_data(self, location):
        """
        Gets data for a specific location, acquiring a lock for it.

        IMPORTANT: This method acquires a lock that must be released by a
        corresponding call to `set_data`. This creates a critical section
        managed by the caller.

        Returns:
            The data at the given location, or None if the location is not found.
        """
        

        
        if location in self.sensor_data:
            self.data_lock[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Sets data for a specific location and releases the lock for it.

        This method is intended to be called after `get_data` to complete
        a read-modify-write cycle and release the lock.
        """
        

        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_lock[location].release()

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        
        self.thread.join()


class DeviceCore(Thread):
    """
    A thread representing a single execution core on a device.
    It runs a script using data from its device and neighbors.
    """
    def __init__(self, device, location, script, neighbours):
        """Initializes the DeviceCore thread."""
        
        Thread.__init__(self, name="Device core %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script.
        
        It gathers data from its own device and neighbors, which involves
        acquiring locks via `get_data`. It then runs the script and writes the
        result back, which releases the locks via `set_data`.
        """
        script_data = []
        
        # Gather data from neighboring devices.
        for device in self.neighbours:
            if self.device.device_id != device.device_id:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
        
        # Gather data from the parent device itself.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            # Run the script on the collected data.
            result = self.script.run(script_data)

            
            # Propagate results back to neighbors.
            for device in self.neighbours:
                if self.device.device_id != device.device_id:
                    device.set_data(self.location, result)
            
            # Propagate result to the parent device.
            self.device.set_data(self.location, result)

class DeviceThread(Thread):
    """
    The main control thread for a device, managing its lifecycle and script execution.
    """
    

    def __init__(self, device):
        """Initializes the main thread for a device."""
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main device loop.
        
        Waits for a global start signal, then enters a loop synchronized by a barrier.
        In each loop (timepoint), it waits for scripts, executes them using a pool
        of 'core' threads, and then waits at the barrier for all other devices.
        """
        # Wait until the `setup_devices` method has been called and the simulation is ready to start.
        self.device.start_event.wait()

        while True:
            # Synchronize with all other devices at the start of a timepoint.
            self.device.barrier.wait()

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals termination by returning None.
                break

            
            # Wait for the supervisor to assign all scripts for this timepoint.
            # The signal is the `timepoint_done` event being set.
            while not self.device.timepoint_done.is_set():
                self.device.script_received.wait()

            
            # Simulate a device with 8 cores.
            used_cores = 0
            free_core = list(range(8))
            threads = {}

            
            
            # This loop manages the execution of scripts on the 8 simulated cores.
            # It starts threads for scripts and reuses cores as threads finish.
            
            

            for (script, location) in self.device.scripts:
                if used_cores < 8:
                    dev_core = DeviceCore(self.device, location, script, neighbours)
                    dev_core.start()
                    threads[free_core.pop()] = dev_core
                    used_cores = used_cores + 1

                else:
                    # If all cores are busy, wait for one to become free.
                    for thread in threads:
                        if not threads[thread].isAlive():
                            threads[thread].join()
                            free_core.append(thread)
                            used_cores = used_cores - 1

            # Wait for all remaining script threads to complete.
            for thread in threads:
                threads[thread].join()

            
            # Reset events for the next timepoint.
            self.device.timepoint_done.clear()
            if self.device.script_received.is_set():
                self.device.script_received.clear()
