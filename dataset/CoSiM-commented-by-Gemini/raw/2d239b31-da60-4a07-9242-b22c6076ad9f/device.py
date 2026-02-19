"""
@file device.py
@brief A distributed device simulation featuring a dual-locking strategy and dynamic thread creation for script execution.
@details This module defines a device simulation where a master device (ID 0) sets up shared
synchronization primitives (a barrier and a dictionary of location-based locks). Each device's
main thread dynamically creates a new worker thread (`RunScript`) for each script to be executed.
The concurrency control is notably complex, employing both a shared, location-specific lock and
an individual lock for each device.
"""

from threading import Event, Thread, Semaphore, Lock, RLock
from reusable_barrier import ReusableBarrier # Assumes a correct ReusableBarrier implementation exists.
import multiprocessing

class Device(object):
    """
    @brief Represents a single device in the simulation.
    @details Manages its own state and a main control thread. It relies on a master device (ID 0)
    for the initialization of shared synchronization objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.results = {}
        # A per-device lock, its purpose is unclear given the more specific location locks.
        self.lock = None
        # A shared dictionary mapping data locations to RLock objects.
        self.dislocksdict = None
        # A shared barrier for synchronizing all devices.
        self.barrier = None
        self.sem = Semaphore(1)
        # A semaphore used to sequence the setup process.
        self.sem2 = Semaphore(0)
        self.all_devices = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """@brief Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources. The device with ID 0 acts as the master.
        @details This method uses a semaphore (`sem2`) to ensure that the master device (ID 0)
        initializes the shared barrier and lock dictionary before other devices attempt to use them.
        """
        loc = []
        for d in devices:
            for l in d.sensor_data:
                loc.append(l)
        
        # Block Logic: Master device (ID 0) initializes shared resources.
        if self.device_id == 0:
            self.sem2.release() # Allow one other device to proceed with setup.
            self.barrier = ReusableBarrier(len(devices))
            # The distributed lock dictionary is created here.
            self.dislocksdict = {}
            for k in list(set(loc)):
                self.dislocksdict[k] = RLock()
            self.lock = Lock() # Master creates its own per-device lock.

        # All devices wait here until the resource initialization has progressed.
        self.sem2.acquire()

        # Sequentially, each device gets the shared resources from the master and enables the next device.
        for d in devices:
            if d.barrier == None:
                d.barrier = self.barrier
                d.sem2.release()
                d.dislocksdict = self.dislocksdict
                d.lock = Lock() # Each device creates its own per-device lock.

    def assign_script(self, script, location):
        """@brief Assigns a script to be executed."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data from a specific location.
        @note Locking is handled externally by the calling thread (`RunScript`).
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        @brief Updates sensor data at a specific location.
        @note Locking is handled externally by the calling thread (`RunScript`).
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """@brief Shuts down the device by joining its main thread."""
        self.thread.join()

class RunScript(Thread):
    """
    @brief A worker thread created to execute a single script.
    @details Implements a complex dual-locking strategy. It acquires a global lock for the data's
    location, and then acquires a personal lock on each device it interacts with.
    """
    def __init__(self, script, location, neighbours, device):
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        # Acquire a re-entrant lock for the specific data location. This ensures that
        # only one script thread can operate on this location at a time, system-wide.
        self.device.dislocksdict[self.location].acquire()
        
        script_data = []
        # Block Logic: Gather data from neighbor devices.
        for device in self.neighbours:
            # Acquire the specific device's personal lock before accessing its data.
            device.lock.acquire()
            data = device.get_data(self.location)
            device.lock.release()
            if data is not None:
                script_data.append(data)
        
        # Gather data from the local device, also using its personal lock.
        self.device.lock.acquire()
        data = self.device.get_data(self.location)
        self.device.lock.release()
        if data is not None:
            script_data.append(data)

        if script_data:
            result = self.script.run(script_data)
            
            # Block Logic: Broadcast the result, again using per-device locks.
            for device in self.neighbours:
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()
        
        # Release the location-specific lock.
        self.device.dislocksdict[self.location].release()


class DeviceThread(Thread):
    """
    @brief The main control thread for a Device instance.
    @details This thread spawns a new `RunScript` worker for each assigned script.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """@brief Main execution loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for the signal that all scripts for the timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Synchronize all devices before starting script execution.
            self.device.barrier.wait()
            
            script_threads = []
            # Block Logic: Spawn one thread per script. This is inefficient due to thread creation overhead.
            for (script, location) in self.device.scripts:
                script_threads.append(RunScript(script, location, neighbours, self.device))
            
            for t in script_threads:
                t.start()
            for t in script_threads:
                t.join() # Wait for all scripts in this timepoint to complete.
            
            # Synchronize all devices to signal the end of the computation phase.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
