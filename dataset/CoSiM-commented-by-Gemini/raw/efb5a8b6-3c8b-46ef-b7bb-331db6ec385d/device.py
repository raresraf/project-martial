"""
This module implements a multi-threaded device simulation using a worker
thread pool pattern. Each device has a pool of ScriptRunner threads to execute
scripts, and synchronization is managed through Events, Locks, and a
reusable barrier.
"""

from threading import Event, Thread, Lock, Condition

class ReusableBarrierCond(object):
    """
    A reusable barrier implementation using a Condition variable.
    It allows a set of threads to wait for each other to reach a certain point.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Causes the calling thread to wait at the barrier. When the last thread
        arrives, all threads are released and the barrier is reset.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a device in the simulation. It manages its own thread and a
    pool of script runners.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.
        Args:
            device_id, sensor_data, supervisor: Standard device parameters.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.new_script = Event()
        self.new_script_received = None
        self.barrier = None
        self.script_lock = Lock()
        self.lock_dict = {}

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for all devices, including the barrier
        and the dictionary of locks for locations.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                device.barrier = self.barrier
                for loc in device.sensor_data:
                    if loc not in self.lock_dict:
                        self.lock_dict[loc] = Lock()
            
            for device in devices:
                device.lock_dict = self.lock_dict

    def assign_script(self, script, location):
        """Assigns a new script to the device."""
        self.script_lock.acquire()
        self.new_script_received = (script, location)
        self.new_script.set()

    def get_data(self, location):
        """Gets sensor data for a given location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Sets sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a device, which manages a pool of ScriptRunner threads.
    """

    def __init__(self, device):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script_runners = []
        self.neighbours = []
        self.script = None
        self.location = None

        self.new_script = Event()
        self.script_lock = Lock()
        self.wait_for_data = Event()

    def run(self):
        """
        The main loop of the device thread. It sets up a pool of script
        runners and coordinates their execution for each timepoint.
        """
        while True:
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break

            # Create a pool of worker threads (ScriptRunners).
            self.script_runners = []
            new_scr = self.new_script
            scr_lock = self.script_lock
            wait_data = self.wait_for_data
            for _ in range(8):
                script_runner = ScriptRunner(self, new_scr, scr_lock, wait_data)
                self.script_runners.append(script_runner)
                script_runner.start()

            # Assign scripts from the initial list to the workers.
            for (script, location) in self.device.scripts:
                self.script = script
                self.location = location
                self.new_script.set()
                self.wait_for_data.wait()
                self.wait_for_data.clear()

            # Process newly assigned scripts for the current timepoint.
            while True:
                self.device.new_script.wait()
                self.device.new_script.clear()
                self.device.script_lock.release()

                self.script = self.device.new_script_received[0]
                self.location = self.device.new_script_received[1]

                if self.script is None:
                    break

                self.new_script.set()
                self.wait_for_data.wait()
                self.wait_for_data.clear()
                self.device.scripts.append((self.script, self.location))

            # Signal the end of scripts for this timepoint to the workers.
            self.script = None
            self.location = None
            self.neighbours = None
            for script_runner in self.script_runners:
                self.new_script.set()
                self.wait_for_data.wait()
                self.wait_for_data.clear()

            for script_runner in self.script_runners:
                script_runner.join()

            # Wait for all devices to finish the timepoint.
            self.device.barrier.wait()

class ScriptRunner(Thread):
    """
    A worker thread that executes a single script.
    """

    def __init__(self, device_thread, new_script, script_lock, wait_for_data):
        """Initializes the script runner."""
        Thread.__init__(self)
        self.device_thread = device_thread
        self.new_script = new_script
        self.script_lock = script_lock
        self.wait_for_data = wait_for_data

    def run(self):
        """
        The main loop for the script runner. It waits for a script, executes it,
        and repeats until it receives a shutdown signal (None).
        """
        while True:
            self.script_lock.acquire()
            self.new_script.wait()
            self.new_script.clear()
            
            script = self.device_thread.script
            location = self.device_thread.location
            neighbours = self.device_thread.neighbours
            
            self.wait_for_data.set()
            self.script_lock.release()

            if neighbours is None or location is None or script is None:
                break

            if neighbours == []:
                continue

            # Acquire the lock for the specific location before processing data.
            self.device_thread.device.lock_dict[location].acquire()

            # Gather data, run the script, and update data.
            script_data = []
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)

                for device in neighbours:
                    device.set_data(location, result)
                
                self.device_thread.device.set_data(location, result)

            self.device_thread.device.lock_dict[location].release()
