"""
Models a device in a distributed sensor network simulation.

This version uses a complex work-sharing model where the main device thread
processes one task itself and distributes the rest to a pool of worker threads
using a shared list as a work queue.
"""


from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device node in the simulated network.

    Manages scripts, sensor data, and the core logic for processing a work item.
    It relies on its `DeviceThread` to manage the execution and distribution of tasks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance and starts its main thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.scripts_lock = Lock()
        self.timepoint_done = Event()
        self.barrier = None
        self.location_locks = {}
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up and distributes shared resources (barrier, locks) to all devices.

        This setup is centralized; it's intended to be called once to configure
        all device instances with the same shared objects.
        """
        barrier = ReusableBarrierCond(len(devices))
        for device in devices:
            device.barrier = barrier

        location_locks = {}

        for device in devices:
            for location in device.sensor_data:
                if location not in location_locks:
                    location_locks[location] = Lock()

        for device in devices:
            device.location_locks = location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to the device, protecting the script list with a lock.
        Signals `script_received` for the main thread and `timepoint_done` for a None script.
        """
        if script is not None:
            self.scripts_lock.acquire()
            self.scripts.append((script, location))
            self.scripts_lock.release()
            self.script_received.set()
        else:
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data. Not internally thread-safe."""
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data. Not internally thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def process_work(self, script, location, neighbours):
        """
        Contains the core logic for executing a single script.

        This method is called by either the main `DeviceThread` or a `Worker` thread.
        It handles locking, data gathering, script execution, and result broadcasting.
        """
        self.location_locks[location].acquire()

        script_data = []

        # Gather data from neighbours.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Gather data from self.
        data = self.get_data(location)


        if data is not None:
            script_data.append(data)

        if script_data:
            result = script.run(script_data)

            for device in neighbours:
                device.set_data(location, result)

            self.set_data(location, result)

        self.location_locks[location].release()

    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main, complex control thread for a Device.

    Manages a pool of worker threads and implements an unusual work-sharing
    strategy where it performs one task itself and delegates the rest.
    """

    def __init__(self, device):
        """Initializes the main thread and its worker pool."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main time-stepped simulation loop."""
        # --- Worker Pool Setup ---
        work_lock = Lock()
        work_pool_empty = Event()
        work_pool_empty.set()
        work_pool = []  # A simple list used as a shared work queue.
        workers = []
        workers_number = 7
        work_available = Semaphore(0)
        own_work = None

        for worker_id in range(1, workers_number + 1):
            workers.append(Worker(worker_id, work_pool, work_available, work_pool_empty, work_lock, self.device))
            workers[worker_id-1].start()

        while True:
            scripts_ran = []
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is not None:
                neighbours = set(neighbours)
                if self.device in neighbours:
                    neighbours.remove(self.device)

            if neighbours is None:
                # Shutdown sequence.
                for i in range(0,7):
                    work_available.release() # Unblock all workers.
                
                for worker in workers:
                    worker.join()
                break

            self.device.barrier.wait() # First barrier sync.

            while True:
                self.device.script_received.wait()

                self.device.scripts_lock.acquire()

                # Distribute work: one for self, rest for workers.
                for (script, location) in self.device.scripts:
                    if script in scripts_ran:
                        continue
                    scripts_ran.append(script)
                    
                    if own_work is None:
                        own_work = (script, location, neighbours)
                    else:
                        work_lock.acquire()
                        work_pool.append((script, location, neighbours))
                        work_pool_empty.clear()
                        work_available.release()
                        work_lock.release()

                self.device.scripts_lock.release()

                # Check for end of timepoint.
                if self.device.timepoint_done.is_set() and len(scripts_ran) == len(self.device.scripts):
                    
                    # Process this thread's own work item.
                    if own_work is not None:
                        script, location, neighbours = own_work
                        own_work = None
                        self.device.process_work(script, location, neighbours)

                    # Wait for the worker pool to finish.
                    work_pool_empty.wait()
                    
                    for worker in workers:
                        worker.work_done.wait()

                    self.device.timepoint_done.clear()
                    self.device.barrier.wait() # Second barrier sync.
                    break


class Worker(Thread):
    """A worker thread that consumes tasks from a shared list (`work_pool`)."""

    def __init__(self, worker_id, work_pool, work_available, work_pool_empty, work_lock, device):
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.work_pool = work_pool
        self.work_available = work_available
        self.work_pool_empty = work_pool_empty
        self.work_lock = work_lock
        self.device = device
        self.work_done = Event()
        self.work_done.set()

    def run(self):
        """The main loop for the worker. Pops a task and executes it."""
        while True:
            self.work_available.acquire() # Wait for a task.
            self.work_lock.acquire()
            self.work_done.clear()

            if not self.work_pool: # Check for shutdown signal.
                self.work_lock.release()
                return

            # Pop a work item from the shared list.
            script, location, neighbours = self.work_pool.pop(0)

            if not self.work_pool:
                self.work_pool_empty.set()

            self.work_lock.release()

            # Execute the work.
            self.device.process_work(script, location, neighbours)

            self.work_done.set() # Signal completion.