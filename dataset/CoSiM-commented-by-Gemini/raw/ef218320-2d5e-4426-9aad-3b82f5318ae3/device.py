"""
This module implements a multi-threaded simulation of a network of devices.
Each device can execute scripts that process sensor data from itself and its
neighbors. The simulation uses various synchronization primitives to coordinate
the actions of the devices.
"""

from threading import Event, Thread, RLock, Lock, Semaphore, Condition


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device runs in its own thread and can be assigned scripts to execute.
    It communicates with a supervisor to get information about its neighbors
    and uses a barrier to synchronize with other devices at the end of a timepoint.
    """
    
    barrier = None

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device object.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: A dictionary representing the device's sensor data.
            supervisor: The supervisor object that manages the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.run_script = RLock()
        self.scripts_sem = Semaphore(8) # Limits concurrent script executions

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the synchronization barrier for all devices.
        This method should be called once by one of the devices.
        """
        if Device.barrier == None and self.device_id == 0:
            Device.barrier = ReusableBarrier(len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.
        This method is thread-safe.
        """
        self.run_script.acquire()
        self.script_received.set()
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
        self.run_script.release()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main thread of execution for a Device."""

    def __init__(self, device):
        """
        Initializes the device thread.
        
        Args:
            device: The Device object that this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread.
        It waits for scripts, executes them, and synchronizes with other devices.
        """
        while True:
            # Get the neighbors for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            self.device.script_received.wait()

            # Execute the assigned scripts in parallel using MyThread.
            self.device.run_script.acquire()
            dictionar = {}
            i = 0
            for (script, location) in self.device.scripts:
                self.device.scripts_sem.acquire()
                thread = MyThread(self.device, neighbours, location, script)
                dictionar[i] = thread
                dictionar[i].start()
                i = i + 1
            self.device.run_script.release()
            for idx in range(0, len(dictionar)):
                dictionar[idx].join()

            # Synchronize with other devices before proceeding to the next timepoint.
            Device.barrier.wait()
            self.device.timepoint_done.wait()



class MyThread(Thread):
    """A thread for executing a single script on a device."""
    
    lockForLocations = {}

    def __init__(self, device, neighbours, location, script):
        """
        Initializes a script execution thread.

        Args:
            device: The parent device.
            neighbours: A list of neighboring devices.
            location: The sensor data location to be processed.
            script: The script to be executed.
        """
        Thread.__init__(self)
        self.location = location
        self.script = script
        self.device = device
        self.neighbours = neighbours

        # Create a lock for each location to ensure thread-safe data access.
        if location not in MyThread.lockForLocations:
            MyThread.lockForLocations[location] = Lock()

    def run(self):
        """
        Executes the script.
        It gathers data from the device and its neighbors, runs the script,
        and then updates the data on all involved devices.
        """
        MyThread.lockForLocations[self.location].acquire()
        script_data = []
        
        # Gather data from neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the current device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Run the script on the collected data.
            result = self.script.run(script_data)

            # Update data on neighbors and the current device.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)
        MyThread.lockForLocations[self.location].release()
        self.device.scripts_sem.release()

class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing multiple threads.
    Allows a set of threads to all block until all threads have reached the barrier.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier.

        Args:
            num_threads: The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Causes a thread to wait at the barrier.
        When the last thread arrives, all waiting threads are released.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread has arrived, notify all others.
            self.cond.notify_all()
            # Reset the barrier for reuse.
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
