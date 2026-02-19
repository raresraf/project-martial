"""
@file device.py
@brief Implements a device simulation using a producer-consumer pattern.

This module defines a simulation framework where each device uses a dedicated
producer thread (`DeviceThread`) to dispatch script-execution jobs to a queue. A pool
of consumer threads (`ThreadsForEachDevice`) processes these jobs in parallel.
Synchronization between devices is handled by a reusable barrier.
"""

from threading import Lock, Thread, Semaphore, Event
from Queue import Queue


class ReusableBarrier(object):
    """A standard two-phase reusable barrier for thread synchronization."""

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all participating threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one of the two barrier phases."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a device in the simulation, managing its state and producer thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device, its locks, and its main producer thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None

        # Eagerly create locks for all sensor data locations.
        self.locks = {}
        for spot in sensor_data:
            self.locks[spot] = Lock()
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared barrier for inter-device synchronization.

        Functional Utility: Device 0 acts as a coordinator, creating and distributing
        a single ReusableBarrier instance to all other devices in the simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to the device for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data for a given location and acquires its lock.

        @warning This method implements an unconventional locking pattern. It acquires a
        lock that is not released within this method. The caller is responsible for
        ensuring that `set_data()` is called on the same location to release the lock,
        otherwise a deadlock will occur.
        """
        for loc in self.sensor_data:
            if loc == location:
                self.locks[loc].acquire()
                return self.sensor_data[loc]

        return None

    def set_data(self, location, data):
        """
        Updates data for a given location and releases its lock.
        
        @see get_data for the corresponding lock acquisition.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """Waits for the device's producer thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Acts as the "producer" for a single device. It populates a job queue with
    scripts to be executed by a pool of consumer threads.
    """

    def __init__(self, device):
        """Initializes the producer thread and its associated consumer pool."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.dev_threads = ThreadsForEachDevice(8)

    def run(self):
        """The main loop for the producer thread."""

        self.dev_threads.device = self.device

        while True:
            # Get neighbors for the upcoming timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the supervisor to signal that scripts are ready.
            self.device.timepoint_done.wait()

            # Block Logic: Producer places all script jobs into the queue.
            for (script, location) in self.device.scripts:
                self.dev_threads.jobs_to_be_done.put(
                    (neighbours, script, location))

            self.device.timepoint_done.clear()

            # Block Logic: Wait for the consumer threads to process all jobs in the queue.
            self.dev_threads.jobs_to_be_done.join()
            
            # Block Logic: Synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()

        # --- Shutdown sequence ---
        self.dev_threads.jobs_to_be_done.join()
        
        # Send "poison pill" tasks to terminate the consumer threads.
        for _ in range(len(self.dev_threads.threads)):
            self.dev_threads.jobs_to_be_done.put((None, None, None))

        for d_th in self.dev_threads.threads:
            d_th.join()


class ThreadsForEachDevice(object):
    """
    Manages a pool of consumer threads that execute script jobs from a queue.
    """

    def __init__(self, number_of_threads):
        self.device = None
        self.jobs_to_be_done = Queue(number_of_threads)
        self.threads = []
        self.create_threads(number_of_threads)
        self.start_threads()

    def create_threads(self, number_of_threads):
        """Creates the worker threads for the pool."""
        for _ in range(number_of_threads):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

    def start_threads(self):
        """Starts all worker threads in the pool."""
        for i_th in self.threads:
            i_th.start()

    def execute(self):
        """The main loop for a consumer thread."""
        while True:
            # Block until a job is available from the producer.
            neighbours, script, location = self.jobs_to_be_done.get()
            
            # Check for the "poison pill" to terminate the thread.
            if neighbours is None and script is None:
                self.jobs_to_be_done.task_done()
                return

            data_for_script = []
            
            # Aggregate data from neighbors, acquiring locks via get_data.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        data_for_script.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                data_for_script.append(data)

            if data_for_script:
                # Run the script.
                scripted_data = script.run(data_for_script)

                # Distribute results, releasing locks via set_data.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, scripted_data)
                self.device.set_data(location, scripted_data)

            # Signal that this job is complete.
            self.jobs_to_be_done.task_done()
