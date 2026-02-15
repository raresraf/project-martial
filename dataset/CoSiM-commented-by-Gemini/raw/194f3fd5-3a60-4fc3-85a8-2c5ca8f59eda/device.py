"""
This module defines a distributed device simulation framework using a controller
thread that dynamically creates a pool of worker threads for each time step.

The architecture uses a task queue for distributing work and a custom,
semaphore-based reusable barrier for synchronization between devices.
"""

from threading import Event, Semaphore, Lock, Thread
from Queue import Queue


class Device(object):
    """
    Represents a device node, managed by a controller thread (`DeviceThread`).
    
    This class holds the device's state, data, and references to shared
    synchronization objects like barriers and locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device object and starts its controller thread.
        
        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The local data store for this device.
            supervisor (Supervisor): The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.all_scripts_received = Event() # Signals script assignment is done.
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects and calculates
        the number of threads per device.
        
        This method uses a complex calculation to distribute a fixed total number
        of threads amongst the available devices. The first device to run this
        creates and shares the barrier and location locks.
        """
        TOTAL_THREADS = 32
        
        # Calculate the number of worker threads for this device.
        self.NUM_THREADS = TOTAL_THREADS / len(devices) + 1
        
        lower = 0
        for device in devices:
            if device.device_id < self.device_id:
                lower += 1
        if lower < TOTAL_THREADS % len(devices):
            self.NUM_THREADS += 1

        if lower == 0:
            # Master device (first to execute) creates and distributes shared objects.
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.on_time_point_barrier(barrier)
            
            location_lock = {}
            for device in devices:
                for location in device.sensor_data:
                    if location not in location_lock:
                        location_lock[location] = Lock()
            for device in devices:
                device.on_location_lock_dictionary(location_lock)

    def on_time_point_barrier(self, barrier):
        """Callback to receive the shared barrier object."""
        self.barrier = barrier

    def on_location_lock_dictionary(self, location_lock):
        """Callback to receive the shared dictionary of location locks."""
        self.location_lock = location_lock

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of
        assignments for the current step.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.all_scripts_received.set()

    def get_data(self, location):
        """
        Retrieves data from the device's local sensor data.
        Note: This method is not thread-safe; locking is handled by the caller.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data, source=None):
        """
        Sets data in the device's local sensor data.
        Note: This method is not thread-safe; locking is handled by the caller.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's controller thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main controller thread for a device, orchestrating the work for each
    time step by creating a new pool of `ScriptRunner` threads.
    """

    def __init__(self, device):
        """Initializes the controller thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            # Reset event for the new time step.
            self.device.all_scripts_received.clear()
            
            # Get neighbors from the supervisor. If None, simulation is over.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for supervisor to signal that all scripts for this step are assigned.
            self.device.all_scripts_received.wait()

            # --- Producer/Consumer Setup for this Time Step ---
            # Create a queue and populate it with all assigned scripts.
            q = Queue()
            for job in self.device.scripts:
                q.put(job)

            # Dynamically create a new pool of worker threads.
            for t in range(self.device.NUM_THREADS):
                runner = ScriptRunner(q, neighbours, self.device)
                runner.start()
            
            # Wait for the queue to be empty, signifying all tasks are done.
            q.join()
            
            # Synchronize with all other devices before starting the next step.
            self.device.barrier.wait()

class ScriptRunner(Thread):
    """A worker thread that consumes and executes scripts from a shared queue."""

    def __init__(self, queue, neighbours, device):
        """
        Initializes the worker thread.
        
        Args:
            queue (Queue): The queue from which to fetch tasks.
            neighbours (list): A list of neighboring Device objects.
            device (Device): The parent device.
        """
        Thread.__init__(self)
        self.queue = queue
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        Pulls a single task from the queue and executes it.
        
        @note This implementation is flawed. Due to the lack of a `while`
              loop, each `ScriptRunner` thread will only process one script
              from the queue and then terminate.
        """
        try:
            (script, location) = self.queue.get_nowait()
            
            # Acquire a global lock for this location to serialize access.
            self.device.location_lock[location].acquire()

            script_data = []
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Run the script and distribute the results.
                result = script.run(script_data)
                for device in self.neighbours:
                    device.set_data(location, result, self.device.device_id)
                self.device.set_data(location, result)

            # Release the global location lock.
            self.device.location_lock[location].release()
            
            # Signal that this task is done.
            self.queue.task_done()
        except:
            # The `try...except` block may hide Queue.Empty exceptions if
            # more threads are created than there are tasks.
            pass

class ReusableBarrier():
    """A custom reusable barrier implemented using semaphores and a lock."""
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks until all participating threads have called this method.
        Uses a two-phase system to ensure reusability.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one of the two barrier phases."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
