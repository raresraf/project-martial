"""
Models a distributed system of devices executing computational scripts concurrently.

This variant of the device simulation framework features a sophisticated, if
complex, approach to synchronization. It uses a two-phase semaphore-based
barrier, a per-script threading model, and a globally shared set of
per-location locks. The locking strategy serializes all work on a given
"location" during script execution to prevent races.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A reusable, two-phase synchronization barrier implemented using Semaphores.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters are wrapped in a list to be mutable (pass-by-reference).
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread arrives, release all waiting threads for this phase.
                n_threads = self.num_threads
                while n_threads > 0:
                    threads_sem.release()
                    n_threads -= 1
                # Reset counter for the next use.
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device in the simulation. Manages its state, scripts,
    and the main control thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.devices = []
        self.barrier = None
        self.workers = []
        # A dictionary of per-location locks, shared across all devices.
        keys = range(60)
        self.loc_barrier = {key: None for key in keys}
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes the shared time-step barrier."""
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            # Ensure all devices share the same barrier instance.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script to the device and sets up the shared per-location lock.
        """
        if script is not None:
            self.scripts.append((script, location))
            # --- Shared Lock Initialization ---
            # This complex logic ensures that all devices use the exact same Lock
            # instance for any given location ID.
            if self.loc_barrier[location] is None:
                # Check if another device already created a lock for this location.
                for device in self.devices:
                    if device.loc_barrier[location] is not None:
                        # If so, copy the reference to that lock.
                        self.loc_barrier[location] = device.loc_barrier[location]
                        break


            # If no lock existed anywhere, create a new one.
            if self.loc_barrier[location] is None:
                self.loc_barrier[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. This read is not individually locked."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This write is not individually locked."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class ScriptWorker(Thread):
    """A worker thread that executes one script, handling all locking for its location."""
    def __init__(self, device, neighbours, script, location):
        
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Acquires the location-wide lock, gathers data, runs script, updates data,
        and releases the lock.
        """
        # Acquire the global lock for this specific location. This serializes all
        # work on this location across the entire system.
        self.device.loc_barrier[self.location].acquire()
        script_data = []
        # Gather data from all neighbors for this location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = self.script.run(script_data)
            # Update data on all involved devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        # Release the location-wide lock.
        self.device.loc_barrier[self.location].release()


class DeviceThread(Thread):
    """The main control thread for a single Device."""

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown.
                break

            # Wait for supervisor to signal that all scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()



            # Create and start a worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                thread = ScriptWorker(self.device, neighbours, script, location)
                self.device.workers.append(thread)

            for worker in self.device.workers:
                worker.start()

            # Wait for all worker threads to complete.
            for worker in self.device.workers:
                worker.join()
            
            # Clean up for the next time step.
            self.device.workers = []
            self.device.timepoint_done.clear()
            # Synchronize with all other devices before the next time step.
            self.device.barrier.wait()
