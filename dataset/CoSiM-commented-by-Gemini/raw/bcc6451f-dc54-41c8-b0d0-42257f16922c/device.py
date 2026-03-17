"""
This module simulates a network of devices that process data in discrete timepoints.

The architecture uses a designated master device (ID 0) to initialize and
distribute synchronization primitives—a shared barrier (`ReusableBarrierCond`)
and a dictionary of location-based locks—to all other devices. Each device
runs a main `DeviceThread` which, for each timepoint, partitions the assigned
scripts into 8 static workloads and spawns a fixed pool of 8 `MyThread`
workers to execute them. This creates a two-level threading model for workload
management.
"""

from threading import Event, Thread, Lock, Condition

class Device(object):
    """
    Represents a device node in the simulation.

    Each device contains a main orchestrator thread (`DeviceThread`) and relies on
    shared state (barrier, locks) provided by a master device at startup.

    Attributes:
        device_id (int): The unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's local sensor readings.
        supervisor: The central entity managing the network topology.
        dict (dict): A shared dictionary mapping locations to Lock objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor: The central supervisor for the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = ReusableBarrierCond(0)
        self.dict = {}

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.

        This method is designed to be run by a single master device (ID 0). It
        creates a barrier for all devices and a lock for each unique data
        location, then distributes these objects to all devices in the simulation.
        """
        
        # Block Logic: Only the device with ID 0 can coordinate this setup.
        if self.device_id == 0:
            idroot = 0
            
            # Create a barrier that waits for all devices' main threads.
            self.barrier = ReusableBarrierCond(len(devices))
            for j in xrange(len(devices)):
                
                if devices[j].device_id == 0:
                    idroot = j
                
                # Discover all unique locations and create a lock for each.
                for location in devices[j].sensor_data:
                    self.dict[location] = Lock()
            
            # Distribute the shared barrier and dictionary of locks to all devices.
            for k in xrange(len(devices)):
                
                devices[k].barrier = devices[idroot].barrier
                
                for j in xrange(len(self.dict)):
                    devices[k].dict[j] = self.dict[j]

        

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current timepoint.

        Args:
            script: The script object to execute.
            location (int): The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script marks the end of script assignment for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location index.
        
        Returns:
            The data for the location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data at a given location.

        Args:
            location (int): The location index.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the main device thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main orchestrator thread for a Device."""

    def __init__(self, device):
        """
        Initializes the orchestrator thread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main control loop, executing timepoint by timepoint."""
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            
            # === SYNC BARRIER 1 ===
            # Waits for all devices to be ready before starting the timepoint.
            self.device.barrier.wait()
            # A None value for neighbors is the shutdown signal.
            if neighbours is None:
                break
            
            # Waits until all scripts for the current timepoint are assigned.
            self.device.timepoint_done.wait()
            if self.device.scripts is None:
                break
            
            threadsnew = []
            
            # Block Logic: Statically partition the assigned scripts into 8 lists.
            # Each list will be processed by one worker thread.
            for j in xrange(8):
                lis = []
                k = 0
                for (script, loc) in self.device.scripts:
                    if k % 8 == j:
                        lis.append((script, loc))
                    k = k + 1


                threadsnew.append(MyThread(self.device, neighbours, lis))
            
            # Starts the pool of worker threads.
            for thread in threadsnew:
                thread.start()
            # Waits for all worker threads to complete their assigned scripts.
            for thread in threadsnew:
                thread.join()
            
            # Resets the event for the next timepoint.
            self.device.timepoint_done.clear()

class MyThread(Thread):
    """
    A worker thread that executes a pre-assigned list of scripts.
    """
    def __init__(self, device, neighbours, lis):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.lis = lis

    def run(self):
        """
        Processes each script in the assigned list sequentially.
        """
        # Block Logic: Iterates through the list of (script, location) tuples.
        for (script, location) in self.lis:
            # Acquires a lock for the specific location to ensure exclusive access.
            self.device.dict[location].acquire()
            script_data = []
            
            # Block Logic: Gathers data to be used as input for the script.
            for device in self.neighbours:
                # BUG: This line incorrectly retrieves data from the parent device
                # (`self.device`) in every iteration, instead of the neighbor (`device`).
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gathers data from the parent device itself.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                # Executes the script with the gathered data.
                result = script.run(script_data)
                
                
                # Distributes the result back to neighbors and the parent.
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)
            
            # Releases the lock for the location.
            self.device.dict[location].release()

class ReusableBarrierCond():
    """
    A simple reusable barrier implementation using a Condition variable.
    
    Attributes:
        num_threads (int): The number of threads that must wait at the barrier.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
        
        

    def wait(self):
        """
        Causes a thread to wait until num_threads have called this method.
        """
        self.cond.acquire()
        
        self.count_threads -= 1
        # Block Logic: The last thread to arrive resets the barrier and notifies all others.
        if self.count_threads == 0:
            self.cond.notify_all()
            
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
            
        self.cond.release()
        
