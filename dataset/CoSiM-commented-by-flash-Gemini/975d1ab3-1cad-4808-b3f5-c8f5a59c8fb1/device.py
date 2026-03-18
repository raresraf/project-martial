

"""
This module implements components for a distributed simulation or sensor network,
featuring device-specific script processing, inter-device communication, and
synchronization mechanisms. It models how individual devices operate, process
tasks, and coordinate their actions in a multi-threaded environment.

Key components:
- `ReusableBarrier`: A thread synchronization barrier that can be reset and reused.
- `Device`: Represents a single device in the network, managing its data, scripts, and interactions with a supervisor and neighbors.
- `DeviceThread`: The main thread of execution for a `Device`, responsible for orchestrating script processing and communication.
- `ScriptThread`: A worker thread that executes individual scripts on behalf of a `Device`.
"""

from threading import Condition, Event, RLock, Thread

class ReusableBarrier(object):
    """
    Implements a reusable thread synchronization barrier.
    All participating threads wait at the barrier until the specified number of threads
    (`num_threads`) have arrived. Once all threads arrive, they are released, and the
    barrier can be reset for reuse.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.
        
        Args:
            num_threads (int): The number of threads that must reach the barrier before they are all released.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        # Condition variable used for synchronizing threads at the barrier.
        self.cond = Condition()

    def reinit(self):
        """
        Reinitializes the barrier. This method is called when a thread needs to signal
        that it is leaving the barrier, adjusting the `num_threads` count.
        It then immediately calls `wait()` to synchronize with other remaining threads.
        """
        self.cond.acquire()
        self.num_threads -= 1
        self.cond.release()
        self.wait()

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have arrived. Once all threads arrive, they are all released simultaneously.
        The barrier then resets its internal counter to `num_threads` for reuse.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all() # Release all waiting threads.
            self.count_threads = self.num_threads # Reset barrier for reuse.
        else:
            self.cond.wait() # Wait until all threads arrive.
        self.cond.release()

class Device(object):
    """
    Represents a single device in the distributed simulation network.
    Each device has an ID, sensor data, interacts with a supervisor, and can process scripts.
    It manages its own threads for script execution and coordinates with other devices
    using shared locks and a reusable barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the sensor data collected by the device.
            supervisor (Supervisor): The supervisor object responsible for managing devices and scripts.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that new scripts have been received by the device.
        self.script_received = Event()
        # Event to signal the device's thread to start processing.
        self.start = Event()
        self.scripts = [] # List of all scripts assigned to this device.
        self.scripts_to_process = [] # Queue of scripts waiting to be processed in the current timepoint.
        # Event to signal that all scripts for the current timepoint have been processed.
        self.timepoint_done = Event()
        self.nr_script_threats = 0 # Counter for active script execution threads.

        # The main thread responsible for this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.script_threats = [] # List of threads currently executing scripts.
        self.barrier_devices = None # Reusable barrier for synchronizing all devices.
        self.neighbours = None # List of neighboring devices.
        self.cors = 8 # Maximum number of concurrent script threads this device can run.
        self.lock = None # Shared RLock for accessing common device resources across all devices.
        self.lock_self = None # Shared RLock for device-specific setup operations to prevent race conditions.
        self.results = {} # Dictionary to store results from executed scripts.
        self.results_lock = None # Shared RLock for thread-safe access to results.

    def __str__(self):
        """
        Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization primitives (locks, barrier) across all devices.
        This method ensures that these primitives are initialized only once and shared
        among all devices in the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Initializes scripts to be processed in the current timepoint.
        for script in self.scripts:
            self.lock.acquire()
            self.scripts_to_process.append(script)
            self.lock.release()

        # Initializes a shared lock for device-specific setup operations if not already set.
        if not self.lock_self:
            lock = RLock()
            for device in devices:
                device.lock_self = lock

        self.lock_self.acquire()
        # Initializes a shared lock for common device resources if not already set.
        if not self.lock:
            rlock = RLock()
            for device in devices:
                device.lock = rlock
        self.lock_self.release()

        self.lock_self.acquire()
        # Initializes a shared lock for accessing results if not already set.
        if not self.results_lock:
            results_lock = RLock()
            for device in devices:
                device.results_lock = results_lock
        self.lock_self.release()

        self.lock_self.acquire()
        # Initializes a shared reusable barrier for all devices if not already set,
        # and signals all device threads to start.
        if not self.barrier_devices:
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier_devices = barrier
                device.start.set() # Signals each device's thread to begin execution.
        self.lock_self.release()

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific data location.
        
        Args:
            script (Script): The script object to be executed.
            location: The data location the script operates on.
        """
        if script is not None:
            self.lock.acquire()
            self.scripts.append((script, location))
            self.scripts_to_process.append((script, location))
            self.script_received.set() # Signals that a new script is available.
            self.lock.release()
        else:
            self.lock.acquire()
            self.timepoint_done.set() # Signals that script assignments for the current timepoint are complete.
            self.script_received.set() # Also signals script received to unblock any waiting threads.
            self.lock.release()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.
        
        Args:
            location: The specific location for which to retrieve data.
            
        Returns:
            Any: The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else:
            data = None
        return data

    def set_data(self, location, data):
        """
        Sets (updates) sensor data for a given location.
        
        Args:
            location: The specific location for which to set data.
            data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
        
    def shutdown(self):
        """
        Waits for the device's main thread to complete its execution.
        """
        self.thread.join()

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.device.neighbours = None

    def run(self):
        
        self.device.start.wait()
        while True:
            self.device.scripts_to_process = []
            for script in self.device.scripts:
                self.device.scripts_to_process.append(script)

            
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                self.device.barrier_devices.reinit()
                break

            self.device.results = {}
            while True:
                if not self.device.timepoint_done.is_set():
                    self.device.script_received.wait()
                    self.device.script_received.clear()

                
                if len(self.device.scripts_to_process) == 0:
                    if self.device.timepoint_done.is_set():
                        break

                
                while len(self.device.scripts_to_process):
                    list_threats = []
                    self.device.script_threats = []
                    self.device.nr_script_threats = 0
                    
                    while len(self.device.scripts_to_process) and self.device.nr_script_threats < self.device.cors:
                        script, location = self.device.scripts_to_process.pop(0)
                        list_threats.append((script, location))
                        self.device.nr_script_threats += 1

                    for script, location in list_threats:
                        script_data = []
                        
                        neighbours = self.device.neighbours
                        for device in neighbours:


                            device.lock_self.acquire()
                            data = device.get_data(location)
                            device.lock_self.release()
                            if data is not None:
                                script_data.append(data)
                        
                        self.device.lock_self.acquire()
                        data = self.device.get_data(location)
                        self.device.lock_self.release()
                        if data is not None:
                            script_data.append(data)

                        thread_script_d = ScriptThread(self.device, script, location, script_data)

                        self.device.script_threats.append(thread_script_d)
                        thread_script_d.start()

                    for thread in self.device.script_threats:
                        thread.join()

            
            for location, result in self.device.results.iteritems():
                
                for device in self.device.neighbours:
                    device.lock_self.acquire()
                    device.set_data(location, result)
                    device.lock_self.release()
                
                self.device.lock_self.acquire()
                self.device.set_data(location, result)
                self.device.lock_self.release()

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            self.device.barrier_devices.wait()

class ScriptThread(Thread):
    

    def __init__(self, device, script, location, script_data):
        
        Thread.__init__(self, name="Device Script Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.script_data = script_data

    def run(self):
        
        if self.script_data != []:
            
            result = self.script.run(self.script_data)
            
            self.device.results_lock.acquire()
            self.device.results[self.location] = result
            self.device.results_lock.release()
        self.device.nr_script_threats -= 1
