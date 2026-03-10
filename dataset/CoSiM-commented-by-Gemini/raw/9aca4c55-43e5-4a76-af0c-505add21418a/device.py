
"""
Models a distributed network of devices that process sensor data concurrently.

This script implements a sophisticated device simulation featuring a robust,
multi-phase setup for synchronization primitives and a manually managed pool of
persistent worker threads (cores).

NOTE: While the setup phase is robust, the `DeviceCore` worker thread contains
a nested lock acquisition pattern that is highly prone to deadlocks.
"""

from threading import Event, Thread, Condition, Lock
from Queue import Queue

class ReusableBarrier(object):
    """A reusable barrier implemented using a Condition variable.

    This barrier blocks a set of threads until all have arrived, and is reusable
    for iterative, multi-timepoint simulations.
    """
    
    def __init__(self, num_threads):
        """Initializes the ReusableBarrier."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

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

    This device participates in a complex, multi-phase setup to establish shared
    synchronization objects and uses a queue to manage scripts for its cores.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.barrier = None
        self.locations_mutex = None
        self.can_begin = Event()
        self.locks_computed = Event()
        self.timepoint_done = Event()
        self.simulation_end = Event()
        self.lock = Lock()
        self.scripts_queue = Queue()
        self.scripts = []
        self.locations = []
        self.devices = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initiates the multi-phase setup, run by the master device (ID 0)."""
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            self.devices = devices
            self.locations_mutex = Lock()

            for device in devices:
                device.locations_mutex = self.locations_mutex
                device.locations = self.locations
                device.barrier = self.barrier
                device.can_begin.set()
            self.can_begin.set()

    def assign_script(self, script, location):
        """Adds a script to the device's list and processing queue."""
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_queue.put((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """The main orchestration thread for a single Device.

    It manages a pool of persistent `DeviceCore` worker threads and orchestrates
    a complex, two-phase setup of shared location-based locks.
    """
    def __init__(self, device):
        """Initializes the main thread and its pool of worker cores."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.current_neighbours = []
        self.cores = [DeviceCore(self, i, self.device.simulation_end) for i in xrange(0, 8)]

    def run(self):
        """The main control loop for the device.

        Handles a robust two-phase setup for shared locks and then enters the
        main simulation loop, where it dispatches scripts to available cores.
        """
        self.device.can_begin.wait()

        # Phase 1 of setup: Cooperatively build a list of all unique locations.
        self.device.locations_mutex.acquire()
        for location in self.device.sensor_data.keys():
            if location not in self.device.locations:
                self.device.locations.append(location)
        self.device.locations_mutex.release()
        self.device.barrier.wait()

        # Phase 2 of setup: Master device creates and shares locks for each location.
        if self.device.device_id == 0:
            self.device.locations_locks = [Lock() for _ in xrange(0, len(self.device.locations))]
            for device in self.device.devices:
                device.locations_locks = self.device.locations_locks
        self.device.barrier.wait()

        for core in self.cores:
            core.start()

        while True:
            # Repopulate queue for the current timepoint.
            while not self.device.scripts_queue.empty():
                self.device.scripts_queue.get()
            for script in self.device.scripts:
                self.device.scripts_queue.put(script)

            self.current_neighbours = self.device.supervisor.get_neighbours()
            if self.current_neighbours is None:
                self.device.simulation_end.set()
                for core in self.cores:
                    core.got_script.set()
                for core in self.cores:
                    core.join()
                break

            # Dispatch scripts from the queue to available cores.
            while not self.device.timepoint_done.isSet() or not self.device.scripts_queue.empty():
                if not self.device.scripts_queue.empty():
                    script, location = self.device.scripts_queue.get()
                    core_found = False
                    while not core_found:
                        for core in self.cores:
                            if core.running is False:
                                core_found = True
                                core.script = script
                                core.location = location
                                core.neighbours = self.current_neighbours
                                core.running = True
                                core.got_script.set()
                                break
            self.device.barrier.wait()
            self.device.timepoint_done.clear()

class DeviceCore(Thread):
    """A persistent worker thread that executes assigned scripts.

    NOTE: The `run` method contains a nested lock acquisition pattern that is
    highly susceptible to deadlocks.
    """
    def __init__(self, device_thread, core_id, simulation_end):
        """Initializes the worker core."""
        Thread.__init__(self, name="Device Core %d" % core_id)
        self.device_thread = device_thread
        self.core_id = core_id
        self.neighbours = []
        self.got_script = Event()
        self.running = False
        self.simulation_end = simulation_end

    def run(self):
        """The main loop for the worker thread."""
        while True:
            self.got_script.wait()
            if self.simulation_end.isSet():
                break

            # BUG: Acquires a location-specific lock, then tries to acquire
            # device-specific locks, a classic deadlock pattern.
            self.device_thread.device.locations_locks[self.location].acquire()
            script_data = []
            for neighbour in self.neighbours:
                neighbour.lock.acquire()
                data = neighbour.get_data(self.location)
                neighbour.lock.release()
                if data is not None:
                    script_data.append(data)

            self.device_thread.device.lock.acquire()
            data = self.device_thread.device.get_data(self.location)
            self.device_thread.device.lock.release()
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = self.script.run(script_data)
                self.device_thread.device.lock.acquire()
                self.device_thread.device.set_data(self.location, result)
                self.device_thread.device.lock.release()

                for neighbour in self.neighbours:
                    neighbour.lock.acquire()
                    neighbour.set_data(self.location, result)
                    neighbour.lock.release()

            self.device_thread.device.locations_locks[self.location].release()
            self.running = False
            self.got_script.clear()
            if self.simulation_end.isSet():
                break
