"""
This module provides a complex, multi-threaded device simulation framework.

Each `Device` manages a dynamic pool of `DeviceThread` workers that process tasks
from a shared queue. The synchronization is custom and intricate, featuring:
- A queue-based peer-to-peer barrier (`sync_devices`).
- A manual barrier implementation for worker threads at the end of each timepoint,
  orchestrated with a Lock, a counter, and an Event.
- A hazardous locking pattern where locks are acquired on other devices and
  released in a separate function call (`get_data_lock`/`set_data_unlock`),
  making the system difficult to reason about and prone to deadlocks.
"""


from threading import Event, Thread, Lock, RLock
from Queue import Queue


class Device(object):
    """
    Represents a device, managing a pool of worker threads and their tasks.
    """
    no_cores = 8

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device, its locks, queues, and worker threads."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.sensor_data_locks = {}
        self.supervisor = supervisor
        self.devices_other = []

        self.scripts = []
        self.script_queue = Queue()
        self.scripts_lock = Lock()
        # A queue used as a "virtual socket" for a custom synchronization barrier.
        self.virt_socket = Queue()

        # --- Complex timepoint synchronization state ---
        self.start_lock = Lock()
        self.start_is_at = True # Flag to select one worker to be the "master" for a timepoint.
        self.end_event = Event() # Signals all workers that a timepoint has fully completed.
        self.counter = 1 # Used by workers to manually implement a barrier.

        self.neighbours = []

        # Create a pool of worker threads, but only start one initially.
        self.threads = [DeviceThread(self) for _ in range(Device.no_cores)]
        self.active_threads = 1
        self.threads[0].start()

    def __start_thread(self):
        """Starts a new worker thread from the pool if needed and available."""
        if self.active_threads >= Device.no_cores:
            return

        no_thr = len(self.scripts)
        if no_thr > self.active_threads:
            self.threads[self.active_threads].start()
            self.active_threads += 1

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes cross-device information and location-specific locks."""
        self.devices_other = [dev for dev in devices if dev != self]
        for loc in self.sensor_data:
            self.sensor_data_locks[loc] = RLock()

    def sync_send(self):
        """Helper for the custom barrier; sends a signal to this device's socket."""
        self.virt_socket.put(None)

    def sync_devices(self):
        """
        A custom barrier implementation. Each device signals all other devices
        and then waits for a signal from every other device.
        """
        for dev in self.devices_other:
            dev.sync_send()
        for _ in self.devices_other:
            _ = self.virt_socket.get()

    def assign_script(self, script, location):
        """Adds a script to the queue and starts more workers if necessary."""
        if script is not None:
            with self.scripts_lock:
                self.script_queue.put((script, location))
                self.scripts.append((script, location))
                self.__start_thread()
        else:
            # A 'None' script is a sentinel; put one for each active worker
            # to signal the end of the task queue for this timepoint.
            with self.scripts_lock:
                for _ in range(self.active_threads):
                    self.script_queue.put(None)

    def get_data(self, location):
        """Thread-safely gets data from a specific sensor location."""
        if location in self.sensor_data:
            with self.sensor_data_locks[location]:
                return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Thread-safely sets data at a specific sensor location."""
        if location in self.sensor_data:
            with self.sensor_data_locks[location]:
                self.sensor_data[location] = data

    def get_data_lock(self, location):
        """
        WARNING: Unsafe locking pattern. Acquires a lock and returns data.
        The caller is responsible for releasing the lock later, which is fragile.
        """
        if location in self.sensor_data:
            self.sensor_data_locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data_unlock(self, location, data):
        """
        WARNING: Counterpart to the unsafe get_data_lock. Sets data and releases a lock
        that was acquired elsewhere.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.sensor_data_locks[location].release()

    def timepoint_init(self):
        """Initializes state for the beginning of a new simulation timepoint."""
        with self.scripts_lock:
            for script in self.scripts:
                self.script_queue.put(script)
        self.neighbours = self.supervisor.get_neighbours()
        if self.neighbours is not None:
            self.neighbours = list(self.neighbours)
            self.neighbours.append(self)
            self.neighbours.sort(key=lambda x: x.device_id)

    def shutdown(self):
        """Joins all worker threads to shut down the device."""
        for thr in self.threads:
            if thr.isAlive():
                thr.join()

class DeviceThread(Thread):
    """
    A worker thread that processes scripts from its parent device's queue.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main worker loop. Contains highly complex synchronization logic.
        """
        while True:
            # --- Timepoint Initialization Phase ---
            # One worker thread becomes the "master" for this timepoint.
            with self.device.start_lock:
                if self.device.start_is_at:
                    self.device.start_is_at = False
                    self.device.timepoint_init()
                    self.device.end_event.clear()

            neighbours = self.device.neighbours
            if neighbours is None:
                break # Termination signal.

            # --- Task Execution Phase ---
            # Each worker pulls tasks from the shared queue until it gets a 'None' sentinel.
            while True:
                pair = self.device.script_queue.get()
                if pair is None:
                    self.device.script_queue.task_done()
                    break
                script, location = pair

                # This block uses the unsafe get_data_lock/set_data_unlock pattern.
                # It acquires locks on neighbor devices.
                script_data = [dev.get_data_lock(location) for dev in neighbours if dev.get_data(location) is not None]

                if script_data:
                    result = script.run(script_data)
                    # It then sets data and releases the locks on those neighbors.
                    for device in neighbours:
                        device.set_data_unlock(location, result)

                self.device.script_queue.task_done()

            # Wait until all tasks in the queue for this timepoint are done.
            self.device.script_queue.join()

            # --- End-of-Timepoint Synchronization Phase ---
            # Workers manually implement a barrier here.
            with self.device.start_lock:
                if not self.device.start_is_at:
                    # The first worker to finish the tasks initiates the cross-device sync.
                    self.device.start_is_at = True
                    self.device.sync_devices()
                    self.device.counter = self.device.active_threads - 1
                else:
                    self.device.counter -= 1
                
                # The very last worker across the pool sets the event to release all workers.
                if self.device.counter == 0:
                    self.device.end_event.set()
            
            # All workers block here until the last one sets the event.
            self.device.end_event.wait()
