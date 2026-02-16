"""
This module implements a distributed device simulation with several severe
concurrency flaws.

It defines a `Device` class that uses a `DeviceThread` to manage its work.
The `DeviceThread` attempts to manually manage a pool of `DeviceCore` worker
threads. The locking protocol is fundamentally broken, with locks being
acquired in one method and released in another without guarantees, creating a
high risk of deadlock. The setup of shared resources is also not thread-safe.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrierSem(object):
    """
    A correct, two-phase reusable barrier for thread synchronization.
    This is included here as it is imported by the original code.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks a thread until all threads have reached the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase to prevent premature re-entry."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a device in the simulation, managing its own data and locks.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Event to ensure setup_devices completes before the main loop runs.
        self.start_event = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        # Each device creates a dictionary of locks for its own sensor locations.
        # These locks are NOT shared between devices.
        self.data_lock = {}
        for data in sensor_data:
            self.data_lock[data] = Lock()

        self.barrier = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        A decentralized and race-prone method to set up the shared barrier.
        The first device to call this will create the barrier and distribute it.
        """
        if self.barrier is None:
            self.barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                dev.barrier = self.barrier
        # Signal that setup is complete and the main thread can proceed.
        self.start_event.set()

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
        self.script_received.set()

    def get_data(self, location):
        """
        Acquires a lock and returns data for a location.
        
        DANGER: This method acquires a lock but does not release it. It relies
        on a corresponding `set_data` call to release the lock, which is a
        very brittle and unsafe locking pattern. A failure after this call
        and before `set_data` will cause a permanent deadlock.
        """
        if location in self.sensor_data:
            self.data_lock[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Sets data for a location and releases a lock.
        
        DANGER: This method releases a lock it did not acquire. It is the
        counterpart to the flawed `get_data` method.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_lock[location].release()

    def shutdown(self):
        self.thread.join()


class DeviceCore(Thread):
    """
    A short-lived worker thread that executes a single script. It uses the
    flawed get/set locking protocol from the Device class.
    """
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self, name="Device core %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Gathers data by acquiring locks on multiple devices, runs a script,
        and then disseminates data by releasing those locks.
        """
        script_data = []
        # Block Logic: Acquire a lock on each neighbor and self for the given location.
        for device in self.neighbours:
            if self.device.device_id != device.device_id:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: If data was gathered, run script and release all locks.
        if script_data != []:
            result = self.script.run(script_data)
            # Calling set_data releases the locks that were acquired by get_data.
            for device in self.neighbours:
                if self.device.device_id != device.device_id:
                    device.set_data(self.location, result)
            self.device.set_data(self.location, result)

class DeviceThread(Thread):
    """
    The main control thread for a device, which attempts to manually manage a
    pool of 8 worker threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main loop, containing flawed logic for thread pool management."""
        self.device.start_event.wait()

        while True:
            # The barrier is at the start of the loop, which is an unusual pattern.
            self.device.barrier.wait()

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Inefficiently waits for the timepoint to be done.
            while not self.device.timepoint_done.is_set():
                self.device.script_received.wait()
            
            # --- Flawed Manual Thread Pool Logic ---
            # This section attempts to limit execution to 8 concurrent threads but
            # will fail to schedule all scripts if the first 8 are long-running.
            used_cores = 0
            free_core = list(range(8))
            threads = {}

            for (script, location) in self.device.scripts:
                if used_cores < 8:
                    dev_core = DeviceCore(self.device, location, script, neighbours)
                    dev_core.start()
                    threads[free_core.pop()] = dev_core
                    used_cores = used_cores + 1
                else:
                    # This loop only checks for finished threads once per new script.
                    # If all threads are busy, it will fail to schedule the script.
                    for thread in threads:
                        if not threads[thread].isAlive():
                            threads[thread].join()
                            free_core.append(thread)
                            used_cores = used_cores - 1

            # Wait for any remaining running threads to complete.
            for thread in threads:
                threads[thread].join()

            self.device.timepoint_done.clear()
            if self.device.script_received.is_set():
                self.device.script_received.clear()
