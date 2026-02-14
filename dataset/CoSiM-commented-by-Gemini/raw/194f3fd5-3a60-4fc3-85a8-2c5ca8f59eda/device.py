"""
This module defines a device simulation framework where each device's workload
for a given time step is processed by a dynamically created pool of worker
threads consuming tasks from a queue.

The architecture is composed of four main classes:
- ReusableBarrier: A semaphore-based barrier for synchronization.
- Device: Represents a node in the network.
- DeviceThread: The main control thread for a device, which orchestrates time
  steps and manages the dynamic creation of worker threads.
- ScriptRunner: A short-lived worker thread that consumes one task from a
  queue and then terminates.
"""
from threading import Event, Semaphore, Lock, Thread
from Queue import Queue


class ReusableBarrier():
    """
    A reusable barrier implemented with semaphores and a lock.

    This uses a two-phase signaling mechanism to ensure that threads from one
    barrier wait cycle do not proceed before all threads have been released,
    making it safe for repeated use in a loop.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters for each phase of the barrier.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.count_lock = Lock()
        # Semaphores to block threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes the calling thread to wait at the barrier."""
        # Phase 1: Threads arrive and are blocked.
        self.phase(self.count_threads1, self.threads_sem1, is_phase1=True)
        # Phase 2: Ensures all threads from phase 1 are released before reset.
        self.phase(self.count_threads2, self.threads_sem2, is_phase1=False)

    def phase(self, count, threads_sem, is_phase1):
        with self.count_lock:
            # Atomically decrement the counter for the current phase.
            if is_phase1:
                self.count_threads1 -= 1
                if self.count_threads1 == 0:
                    # Last thread arrived, release all threads in this phase.
                    for _ in range(self.num_threads):
                        threads_sem.release()
                    self.count_threads2 = self.num_threads # Reset for next phase.
            else:
                self.count_threads2 -= 1
                if self.count_threads2 == 0:
                    for _ in range(self.num_threads):
                        threads_sem.release()
                    self.count_threads1 = self.num_threads # Reset for next use.
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device in the simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.all_scripts_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization objects and calculates thread distribution.
        """
        TOTAL_THREADS = 32
        # Distribute a total of 32 worker threads as evenly as possible.
        num_devices = len(devices)
        base_threads = TOTAL_THREADS // num_devices
        remainder_threads = TOTAL_THREADS % num_devices

        my_thread_count = base_threads
        if self.device_id < remainder_threads:
            my_thread_count += 1
        self.NUM_THREADS = my_thread_count

        # The root device (id 0) creates and distributes shared objects.
        if self.device_id == 0:
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
        """Assigns a script to be processed in the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.all_scripts_received.set()

    def get_data(self, location):
        """Gets data from a location (not intrinsically thread-safe)."""
        return self.sensor_data.get(location)

    def set_data(self, location, data, source=None):
        """Sets data at a location (not intrinsically thread-safe)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It dynamically creates a new pool
    of worker threads (`ScriptRunner`) for each time step.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main lifecycle loop of the device."""
        while True:
            self.device.all_scripts_received.clear()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for supervisor to signal that all scripts are assigned.
            self.device.all_scripts_received.wait()

            # --- Dynamic Worker and Queue Creation ---
            q = Queue()
            for job in self.device.scripts:
                q.put(job)

            # Create a new set of worker threads for this time step.
            workers = [ScriptRunner(q, neighbours, self.device) for _ in range(self.device.NUM_THREADS)]
            for worker in workers:
                worker.start()

            # Wait for all tasks in the queue to be completed.
            q.join()
            
            # --- Global Synchronization ---
            self.device.barrier.wait()


class ScriptRunner(Thread):
    """
    A short-lived worker thread that processes tasks from a queue.
    
    NOTE: The implementation is not robust. It only attempts to get one item
    from the queue and then exits, which can lead to an incorrect number of
    active threads if some threads start after others have already taken work.
    A `while True` loop with a blocking `get()` would be more conventional.
    """
    def __init__(self, queue, neighbours, device):
        Thread.__init__(self)
        self.queue = queue
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """Executes a single task from the queue."""
        try:
            (script, location) = self.queue.get_nowait()
            
            # Use the global lock for this location to ensure serial access.
            with self.device.location_lock[location]:
                script_data = []
                # Gather data from neighbors and self.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute script and propagate results.
                if script_data:
                    result = script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(location, result, self.device.device_id)
                    self.device.set_data(location, result)
            
            self.queue.task_done()
        except:
            # If get_nowait() fails because queue is empty, thread just exits.
            pass
