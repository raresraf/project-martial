
"""
@brief A device simulation using a Singleton for state management.
@file device.py

This module implements a distributed device simulation where shared resources
(a barrier and location-based locks) are managed by a Singleton object. Each
device's main thread dynamically creates a new set of worker threads for each
time step to execute computational scripts.

WARNING: This implementation contains severe performance and concurrency flaws.
1.  **Inefficient Threading**: In `DeviceThread.run`, a new list of threads is
    created, started, and joined for every single time step. This introduces
    massive overhead and defeats the purpose of using threads for concurrency.
2.  **Flawed `ReusableBarrier`**: The barrier implementation holds a lock while
    releasing waiting threads, a classic anti-pattern that can cause deadlocks.
3.  **Race Condition in Singleton**: The `Singleton.get_lock` method has a race
    condition. If two threads request a lock for the same new location at the
    same time, they can both enter the `if` block, leading to one lock being
    overwritten. This part should be protected by a lock.
"""

from threading import Thread, Event, Semaphore, Lock

class ReusableBarrier(object):
    """
    A flawed implementation of a reusable two-phase barrier using semaphores.

    WARNING: This implementation is not safe. It holds `count_lock` while
    releasing the semaphores, which is an anti-pattern that can serialize
    thread wakeup and create deadlocks.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Use a list for pass-by-reference.
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all threads reach the barrier. Consists of two phases."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread wakes up others and resets the counter.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Singleton(object):
    """
    A Singleton class to manage shared resources for the simulation.
    It holds the global barrier and the dictionary of location locks.
    """

    class RealSingleton(object):
        """The actual implementation class that holds the shared state."""
        barrier = None
        locks = None

        def initialize(self, devices):
            """Initializes the shared barrier and lock dictionary."""
            self.barrier = ReusableBarrier(devices)
            self.locks = {}

        def get_lock(self, location):
            """
            Lazily initializes and returns a lock for a given location.
            
            WARNING: This method is not thread-safe due to a race condition
            on the check-then-create logic.
            """
            if location not in self.locks:
                self.locks[location] = Lock()
            return self.locks[location]

    # The single, shared instance.
    __instance = None

    def __init__(self, numberOfDevices):
        """
        Initializes the Singleton instance on the first call.
        Subsequent calls have no effect.
        """
        if Singleton.__instance is None:
            Singleton.__instance = Singleton.RealSingleton()
            Singleton.__instance.initialize(numberOfDevices)

    def __getattr__(self, attr):
        """Delegate attribute access to the real instance."""
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """Delegate attribute setting to the real instance."""
        return setattr(self.__instance, attr, value)

    def get_instance(self):
        return self.__instance

    def get_lock(self, location):
        return self.__instance.get_lock(location)

class Device(object):
    """Represents a device node in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.singleton = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared Singleton for all devices.
        """
        self.singleton = Singleton(len(devices))

    def assign_script(self, script, location):
        """Adds a script to the device's workload for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        return (self.sensor_data[location] if location in self.sensor_data
                else None)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device.
    
    This thread's `run` loop is highly inefficient as it spawns new threads
    for every time step.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run_script(self, location, neighbours, script):
        """
        The target function for worker threads. Executes a single script.
        This function correctly uses a lock to protect the critical section.
        """
        script_data = []
        with self.device.singleton.get_lock(location):
            # Gather data from neighbor devices.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Gather data from the current device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # If data is available, run the script and propagate results.
            if script_data != []:
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)


    def run(self):
        """Main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for supervisor to assign all scripts for this time step.
            self.device.timepoint_done.wait()

            # --- INEFFICIENCY WARNING ---
            # A new list of threads is created for every time step. This is
            # extremely inefficient. A persistent thread pool would be better.
            threads = [Thread(target=self.run_script, args=(
                l, neighbours, s)) for (s, l) in self.device.scripts]

            # Start and wait for all of the step's worker threads to complete.
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # Reset the event and wait at the global barrier for all other devices.
            self.device.timepoint_done.clear()
            self.device.singleton.barrier.wait()
