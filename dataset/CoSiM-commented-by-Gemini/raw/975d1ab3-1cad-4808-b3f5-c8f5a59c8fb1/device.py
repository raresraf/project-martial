"""
Models a distributed network of devices that process sensor data concurrently.

This script implements a device simulation with a complex synchronization and
threading model. It uses a custom condition-based barrier and attempts to
cooperatively initialize shared locks among devices. Script processing is done
in batches by dynamically created threads.

NOTE: This implementation contains complex and potentially unsafe synchronization
patterns, including a convoluted setup phase and a data gathering logic that may
be prone to deadlocks.
"""

from threading import Condition, Event, RLock, Thread

class ReusableBarrier(object):
    """A reusable barrier implemented using a Condition variable.

    This barrier is designed to block a set of threads until all of them have
    called the wait() method, at which point they are all released. It is
    intended to be reusable for iterative algorithms.

    Attributes:
        num_threads (int): The number of threads to synchronize.
        count_threads (int): The current count of waiting threads.
        cond (Condition): The condition variable used for waiting and notification.
    """
    
    def __init__(self, num_threads):
        """Initializes the ReusableBarrier."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def reinit(self):
        """Intended to re-initialize the barrier for a smaller group of threads.

        NOTE: This method's logic is suspect. It decrements the thread count and
        then immediately waits, which is likely to cause a deadlock as other
        threads are not involved in this call.
        """
        self.cond.acquire()
        self.num_threads -= 1
        self.cond.release()
        self.wait()

    def wait(self):
        """Causes a thread to wait at the barrier until all threads arrive."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """Represents a single device in the distributed sensor network.

    This device implementation uses a complex, cooperative lock initialization
    scheme and spawns new threads for batches of scripts.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor readings.
        supervisor (Supervisor): An object for retrieving neighbor information.
        thread (DeviceThread): The main thread of execution for this device.
        barrier_devices (ReusableBarrier): A shared barrier for synchronization.
        lock (RLock): A shared lock for script lists.
        lock_self (RLock): A shared lock for device-specific data access.
        results_lock (RLock): A shared lock for the results dictionary.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.start = Event()
        self.scripts = []
        self.scripts_to_process = []
        self.timepoint_done = Event()
        self.nr_script_threats = 0


        self.thread = DeviceThread(self)
        self.thread.start()
        self.script_threats = []
        self.barrier_devices = None
        self.neighbours = None
        self.cors = 8
        self.lock = None
        self.lock_self = None
        self.results = {}
        self.results_lock = None

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and shares synchronization objects among devices.

        This method uses a complex, racy pattern to ensure that shared locks
        and the barrier are initialized only once by one of the devices.
        """
        for script in self.scripts:
            self.lock.acquire()
            self.scripts_to_process.append(script)
            self.lock.release()

        if not self.lock_self:
            lock = RLock()
            for device in devices:
                device.lock_self = lock

        self.lock_self.acquire()
        if not self.lock:
            rlock = RLock()
            for device in devices:
                device.lock = rlock
        self.lock_self.release()

        self.lock_self.acquire()
        if not self.results_lock:
            results_lock = RLock()
            for device in devices:
                device.results_lock = results_lock
        self.lock_self.release()

        self.lock_self.acquire()
        if not self.barrier_devices:
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier_devices = barrier
                device.start.set()
        self.lock_self.release()



    def assign_script(self, script, location):
        """Assigns a script to be executed by the device in a thread-safe manner."""
        if script is not None:
            self.lock.acquire()
            self.scripts.append((script, location))
            self.scripts_to_process.append((script, location))
            self.script_received.set()
            self.lock.release()
        else:
            self.lock.acquire()
            self.timepoint_done.set()
            self.script_received.set()
            self.lock.release()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else:
            data = None
        return data

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data
        
    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main execution thread for a Device.

    This thread orchestrates the processing of scripts in batches, spawning
    worker threads (`ScriptThread`) and managing the device's lifecycle through
    simulation timepoints.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.device.neighbours = None

    def run(self):
        """The main control loop for the device.

        It processes scripts in batches, gathers data from neighbors (in a
        potentially deadlock-prone manner), runs the scripts, propagates results,
        and synchronizes with other devices at a barrier.
        """
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
    """A worker thread to execute a single script and store its result.

    Attributes:
        device (Device): The parent device.
        location (str): The location context for the script.
        script (Script): The script object to be executed.
        script_data (list): The collected data to be used as input for the script.
    """

    def __init__(self, device, script, location, script_data):
        """Initializes the ScriptThread."""
        Thread.__init__(self, name="Device Script Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.script_data = script_data

    def run(self):
        """Executes the script and stores the result in the parent device's
        results dictionary in a thread-safe manner.
        """
        if self.script_data != []:
            
            result = self.script.run(self.script_data)
            
            self.device.results_lock.acquire()
            self.device.results[self.location] = result
            self.device.results_lock.release()
        self.device.nr_script_threats -= 1
