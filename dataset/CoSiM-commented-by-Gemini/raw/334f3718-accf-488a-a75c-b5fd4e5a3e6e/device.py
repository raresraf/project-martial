"""
@file device.py
@brief A device simulation using a master-worker thread pattern.

This module implements a simulation where each `Device` has a master thread
(`DeviceThread`) that manages a pool of persistent `Worker` threads. The master
dispatches batches of work to the workers and uses a complex system of
Condition variables and Locks to synchronize their execution within a timepoint.
"""


from threading import Thread, Event, Lock, Condition, Semaphore




class ReusableBarrier(object):
    """A standard two-phase reusable barrier for synchronizing multiple threads."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads reach the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes a single phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device, which acts as a container for a master thread
    and its associated worker pool.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.loc_locks = []  # Shared list of location-specific locks.
        self.condition = Condition()
        self.barrier = None  # Shared barrier for inter-device synchronization.
        self.neighbours = None
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for the simulation, coordinated by device 0.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            # The lock list is pre-sized, assuming locations are integer indices.
            self.loc_locks = [None] * 100
            for device in devices:
                device.barrier = self.barrier
                device.loc_locks = self.loc_locks

    def assign_script(self, script, location):
        """Assigns a script and ensures a lock for its location exists."""
        if script is not None:
            self.scripts.append((script, location))
            # Lazily initialize locks as new locations are encountered.
            if self.loc_locks[location] is None:
                self.loc_locks[location] = Lock()

            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data  \
            else None



    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's master thread."""
        self.thread.join()


class Worker(Thread):
    """
    A persistent worker thread that executes batches of scripts.

    Workers are managed by a `DeviceThread` and block on a condition variable
    until they are assigned work.
    """
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device
        self.conditon = Condition() # Main condition to wait for work.
        self.working = True # Flag for graceful shutdown.
        self.scripts = []
        self.work_cond = Condition() # Condition to signal work completion.
        self.work_lock = Lock()
        self.is_working = False # State flag.

    def set_scripts(self, scripts):
        """
        Assigns a new batch of scripts to the worker and wakes it up.
        """
        self.scripts = scripts

        self.work_lock.acquire()
        self.is_working = True

        self.conditon.acquire()
        self.work_lock.release()

        self.conditon.notify_all()
        self.conditon.release()

    def kill(self):
        """Signals the worker thread to terminate its main loop."""
        self.working = False
        self.conditon.acquire()
        self.conditon.notify()
        self.conditon.release()

    def run(self):
        """
        The main loop for the worker. Waits for work, executes it, and signals completion.
        """
        while True:
            # Block until woken up by the master thread via `set_scripts`.
            self.conditon.acquire()
            self.conditon.wait()
            self.conditon.release()

            if not self.working:
                break # Exit loop if killed.

            # Process the assigned batch of scripts.
            for (script, location) in self.scripts:
                self.device.loc_locks[location].acquire()
                script_data = []
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                if script_data:
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
                self.device.loc_locks[location].release()

            # Signal to the master thread that this batch of work is complete.
            self.work_lock.acquire()
            self.is_working = False
            self.work_cond.acquire()
            self.work_cond.notify()
            self.work_cond.release()
            self.work_lock.release()


class DeviceThread(Thread):
    """
    The master thread for a single device. It manages a pool of `Worker` threads,
    dispatching work to them for each timepoint.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    @staticmethod
    def chunkify(lst, size):
        """Splits a list into a specified number of chunks."""
        return [lst[i::size] for i in range(size)]

    def run(self):
        """
        Main loop for the master thread. Orchestrates the workers and synchronization.
        """
        list_of_workers = []
        for i in range(8):
            list_of_workers.append(Worker(self.device))
        
        for worker in list_of_workers:
            worker.start()

        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                # Shutdown sequence for all workers.
                for worker in list_of_workers:
                    worker.kill()
                    if worker.isAlive():
                        worker.join()
                break
            
            # Wait for supervisor to signal that scripts are ready for this timepoint.
            self.device.timepoint_done.wait()

            # Dispatch work to the worker pool.
            list_of_scripts = self.chunkify(self.device.scripts, 8)
            for i in range(8):
                list_of_workers[i].set_scripts(list_of_scripts[i])

            # Block Logic: A complex synchronization mechanism to wait for all workers to finish.
            for worker in list_of_workers:
                worker.work_lock.acquire()
                if worker.is_working:
                    worker.work_cond.acquire()
                    worker.work_lock.release()
                    # Wait for the worker to signal its completion.
                    worker.work_cond.wait()
                    worker.work_cond.release()
                else:
                    worker.work_lock.release()

            self.device.timepoint_done.clear()
            # Synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()          
