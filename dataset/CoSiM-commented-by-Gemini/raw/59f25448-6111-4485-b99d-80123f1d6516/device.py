"""
Models a distributed device network using a producer-consumer architecture.

Each `Device` has a main orchestrator thread (`DeviceThread`) that acts as a
producer, and a pool of `MyWorker` threads that act as consumers. Communication
is handled via a `Queue.Queue`. The system uses a global barrier for
synchronization between devices, but its usage is unconventional.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue

class ReusableBarrier(object):
    """A reusable barrier implemented with a Condition variable."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Blocks the calling thread until the required number of threads arrive.
        Once all threads are waiting, they are all released and the barrier resets.
        """
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                self.cond.notify_all()
                self.count_threads = self.num_threads
            else:
                self.cond.wait()

class Device(object):
    """
    Represents a device node, which owns a work queue and manages a producer
    and a pool of consumers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main orchestrator thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.queue = Queue()
        # An event to signal that the centralized setup is complete.
        self.setup = Event()
        self.threads = []
        # A list of shared locks for data locations.
        self.locations_lock = []
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources for the entire network.
        """
        if self.device_id == 0:
            # Master device creates a global barrier for all devices.
            barrier = ReusableBarrier(len(devices))
            # It also creates a fixed-size list of 25 locks.
            for _ in range(25):
                lock = Lock()
                self.locations_lock.append(lock)

            # Distribute shared resources and signal devices to start their workers.
            for device in devices:
                device.barrier = barrier
                device.locations_lock = self.locations_lock
                device.setup.set()

    def assign_script(self, script, location):
        """Assigns a script to this device's to-do list for the timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main orchestrator/producer thread for a device.

    It manages the lifecycle of a worker pool and the flow of work for each
    timepoint, synchronized by global barriers.
    """

    def __init__(self, device):
        """Initializes the orchestrator thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main execution loop for managing timepoints."""
        # Wait until the master device has finished setting up shared resources.
        self.device.setup.wait()

        # Create and start this device's internal pool of worker threads.
        for _ in range(8):
            thread = MyWorker(self.device)
            thread.start()
            self.device.threads.append(thread)

        while True:
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # --- Shutdown Sequence ---
                # Place a poison pill for each worker thread on the queue.
                for _ in range(len(self.device.threads)):
                    self.device.queue.put(None)
                # Join all worker threads.
                for thread in self.device.threads:
                    thread.join()
                break
            
            # Wait for supervisor to finish assigning all scripts for the timepoint.
            self.device.timepoint_done.wait()

            # --- Timepoint Synchronization and Production ---
            # First global sync: All devices wait here before producing work.
            self.device.barrier.wait()
            
            # Producer phase: Place all work items on the queue.
            # The work item includes neighbors, as they may change each timepoint.
            for (script, location) in self.device.scripts:
                self.device.queue.put((neighbours, location, script))

            self.device.timepoint_done.clear()
            
            # Second global sync: All devices wait here after producing work,
            # but without explicitly waiting for their workers to finish.
            self.device.barrier.wait()

class MyWorker(Thread):
    """A consumer thread that executes tasks from the device's work queue."""
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

    def run(self):
        """Main loop: get work from queue, execute, repeat until poison pill."""
        while True:
            elem = self.device.queue.get()
            
            # `None` is the termination signal.
            if elem is None:
                break
            
            # Acquire the lock for this specific location index.
            self.device.locations_lock[elem[1]].acquire()
            try:
                script_data = []
                data = None

                # Data gathering from neighbors. This loop overwrites `data` on each
                # iteration, so only the data from the last neighbor is stored.
                for device in elem[0]:
                    data = device.get_data(elem[1])
                if data is not None:
                    script_data.append(data)
                
                # Gather data from the local device.
                data = self.device.get_data(elem[1])
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Execute script and disseminate results.
                    result = elem[2].run(script_data)
                    for device in elem[0]:
                        device.set_data(elem[1], result)
                    self.device.set_data(elem[1], result)
            finally:
                self.device.locations_lock[elem[1]].release()

            self.device.queue.task_done()
