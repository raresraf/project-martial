"""
This module simulates a device in a distributed network.

The threading model is notable: for each simulation timepoint, the main device
thread (`DeviceThread`) creates a new, temporary pool of worker threads to
process all assigned scripts for that timepoint. Synchronization is handled by a
custom two-phase semaphore barrier and a dictionary of shared locks for data locations.
"""

from threading import Event, Semaphore, Lock, Thread
from Queue import Queue, Empty


class ReusableBarrier():
    """
    A reusable barrier implemented using two Semaphores for two-phase synchronization.
    This prevents threads from one barrier cycle from mixing with threads from the
    next cycle.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes the calling thread to wait at the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase to ensure no thread races ahead."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """Represents a single device in the network."""
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # NOTE: This event appears to be unused.
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # --- Shared synchronization objects (to be populated by setup_devices) ---
        self.barrier = None
        self.location_lock = None
        self.NUM_THREADS = 8

    def __str__(self):
        return "Device %d" % self.device_id

    def is_master_thread(self, devices):
        """Elects the device with the lowest ID as the 'master' for setup."""
        for device in devices:
            if device.device_id < self.device_id:
                return False
        return True

    def setup_devices(self, devices):
        """
        Initializes and distributes shared barrier and lock objects.

        Warning: This setup logic is fragile. It relies on one specific device
        (the one with the lowest ID) to act as a master to create the shared
        objects for everyone.
        """
        if self.is_master_thread(devices):
            barrier = ReusableBarrier(len(devices))
            location_lock = {}
            self.set_barrier_lock(devices, barrier, location_lock)

    def set_barrier_lock(self, devices, barrier, location_lock):
        """Recursively-style method to assign shared objects to all devices."""
        for device in devices:
            device.barrier = barrier
            for location in device.sensor_data:
                if location not in location_lock:
                    location_lock[location] = Lock()
            device.location_lock = location_lock

    def assign_script(self, script, location):
        """Assigns a script. A `None` script signals the end of assignment."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a location. Locking is handled externally."""
        return self.sensor_data.get(location)

    def set_data(self, location, data, source=None):
        """Updates sensor data for a location. Locking is handled externally."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main device thread to shut down."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It creates a new pool of worker
    threads for each timepoint.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_scripts(self, queue, neighbours):
        """
        The target function for worker threads. It processes one task from the queue.

        Warning: The `try...except: pass` block is dangerous as it will silently
        ignore any exceptions (e.g., if the queue is empty), causing the thread
        to terminate prematurely without completing its work.
        """
        try:
            (script, location) = queue.get_nowait()
            # Acquire a global lock for this location.
            lock_location = self.device.location_lock.get(location)
            if lock_location:
                with lock_location:
                    # --- Data Aggregation and Processing ---
                    script_data = []
                    for device in neighbours:
                        script_data.append(device.get_data(location))
                    script_data.append(self.device.get_data(location))
                    
                    # Filter out None values if a device doesn't have the location.
                    valid_data = [d for d in script_data if d is not None]
                    
                    if valid_data:
                        result = script.run(valid_data)
                        # Propagate result to all relevant devices.
                        for device in neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)
            queue.task_done()
        except Empty:
            # Queue is empty, thread has no work to do.
            pass
        except Exception:
            # All other exceptions are silently ignored.
            pass

    def start_threads(self, threadlist):
        """Helper to start a list of threads."""
        for thread in threadlist:
            thread.start()

    def join_threads(self, threadlist):
        """Helper to join a list of threads."""
        for thread in threadlist:
            thread.join()

    def run(self):
        """The main lifecycle loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the supervisor to signal that scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()

            # --- Per-Timepoint Thread Pool Creation ---
            # A new queue and new threads are created for every single timepoint.
            queue = Queue()
            for (script, location) in self.device.scripts:
                queue.put((script, location))
            
            self.device.scripts = [] # Clear scripts for next round.

            threadlist = []
            for _ in range(self.device.NUM_THREADS):
                thread = Thread(target=self.run_scripts, args=(queue, neighbours))
                threadlist.append(thread)
            
            # Start and wait for all temporary worker threads to complete.
            self.start_threads(threadlist)
            self.join_threads(threadlist)
            
            # This call is redundant because the threads have already been joined.
            queue.join()

            # --- Synchronization ---
            self.device.timepoint_done.clear()
            # Wait at the global barrier for all other devices to finish.
            self.device.barrier.wait()
