
"""
@brief A device simulation using a persistent, queue-based worker pool.
@file device.py

This module implements a distributed device simulation where each device manages
a persistent pool of worker threads. A master device (device 0) is responsible
for creating and distributing shared synchronization objects: a global barrier
and a list of semaphores that act as location-specific locks.

The main thread for each device (`DeviceThread`) dispatches scripts to its
worker threads via a queue and uses `queue.join()` to wait for the completion
of all tasks in a time step before synchronizing at the global barrier.

WARNING: This implementation contains severe concurrency and design flaws.
1.  **Critical Race Condition**: The `WorkerThread` uses a manual lock to check
    if the queue is empty before getting an item. This is a classic race
    condition. A thread can see the queue is not empty, be preempted, have
    another thread take the last item, and then the first thread will block
    forever on `queue.get()`. `queue.get()` is already thread-safe and should
    be called directly without the external lock and check.
2.  **Hardcoded Limits**: The number of location locks is hardcoded to
    `MAX_LOCATIONS = 100`, making the system brittle.
3.  **Uncertain Barrier**: The code relies on an imported `ReusableBarrier` whose
    implementation is not provided. If it's similar to previous flawed versions,
    it likely introduces deadlock risks.
"""

from threading import Semaphore, Lock, Event, Thread
from Queue import Queue
from barrier import ReusableBarrier

class Device(object):
    """
    Represents a device node in the simulation, which owns a pool of workers.
    """
    MAX_LOCATIONS = 100
    CONST_ONE = 1

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.semaphore = [] # This will hold the list of location "locks".
        self.thread = DeviceThread(self)
        self.barrier = ReusableBarrier(Device.CONST_ONE)

    def __str__(self):
        return "Device %d" % self.device_id

    def get_distributed_objs(self, barrier, semaphore):
        """
        Callback used by the master device to provide shared objects.
        This correctly starts the main thread only after setup is complete.
        """
        self.barrier = barrier
        self.semaphore = semaphore
        self.thread.start()

    def setup_devices(self, devices):
        """
        Initializes shared resources using Device 0 as the master.
        """
        if self.device_id == 0:
            # Master device creates the shared barrier.
            barrier = ReusableBarrier(len(devices))
            # Master creates a fixed-size list of semaphores to act as locks.
            semaphore = []
            i = Device.MAX_LOCATIONS
            while i > 0:
                semaphore.append(Semaphore(value=Device.CONST_ONE))
                i = i - 1

            # Distribute the shared objects to all devices (including self).
            for device in devices:
                device.get_distributed_objs(barrier, semaphore)

    def assign_script(self, script, location):
        """Adds a script to the device's workload for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. Manages a worker pool and the
    simulation lifecycle for this device.
    """
    THREADS_TO_START = 8
    MAX_SCRIPTS = 100

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = []
        self.neighbours = []
        self.queue = Queue(maxsize=DeviceThread.MAX_SCRIPTS)

    def run(self):
        """Initializes the worker pool and runs the main simulation loop."""
        lock = Lock()
        # Create and start a persistent pool of worker threads.
        for i in range(DeviceThread.THREADS_TO_START):
            self.threads.append(WorkerThread(self, i, self.device, self.queue, lock))
            # Daemon threads will not block program exit, a crude shutdown mechanism.
            self.threads[i].setDaemon(True)
        for thread in self.threads:
            thread.start()

        while True:
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break # End of simulation.

            # Wait for supervisor to assign all scripts.
            self.device.timepoint_done.wait()

            # Enqueue all scripts for the workers to process.
            for script in self.device.scripts:
                self.queue.put(script)
            
            # This correctly blocks until all items in the queue are processed.
            self.queue.join()

            self.device.timepoint_done.clear()

            # Wait at the global barrier for all other devices to finish.
            self.device.barrier.wait()

        # A graceful shutdown would signal workers to exit their loops here.
        for thread in self.threads:
            thread.join()


class WorkerThread(Thread):
    """
    A persistent worker thread that processes scripts from a queue.
    """
    def __init__(self, master, worker_id, device, queue, lock):
        Thread.__init__(self, name="Worker Thread %d %d" % (worker_id, device.device_id))
        self.master = master # The parent DeviceThread
        self.device = device
        self.queue = queue
        self.lock = lock

    def run(self):
        """Continuously fetches and executes scripts from the queue."""
        while True:
            # --- CRITICAL FLAW ---
            # This lock and empty check are incorrect and create a race condition.
            # `queue.get()` is already thread-safe and should be called directly.
            self.lock.acquire()
            value = self.queue.empty()
            if value is False:
                (script, location) = self.queue.get()
            self.lock.release()

            if value is False:
                script_data = []

                # Use the semaphore for the given location as a lock.
                self.device.semaphore[location].acquire()

                # Gather data from neighbors.
                for device in self.master.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from self.
                data = self.master.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # If data exists, execute the script and update values.
                if script_data != []:
                    result = script.run(script_data)
                    for device in self.master.neighbours:
                        device.set_data(location, result)
                    self.master.device.set_data(location, result)

                self.device.semaphore[location].release()
                
                # Signal that the queue item has been fully processed.
                self.queue.task_done()

            # Check for the simulation end condition.
            if self.master.neighbours is None:
                break
