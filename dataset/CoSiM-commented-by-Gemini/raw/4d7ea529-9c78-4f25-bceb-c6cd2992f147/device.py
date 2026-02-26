"""
This module implements a device simulation framework where each device operates
within a single master thread (`DeviceThread`).

Unlike other versions that use a worker pool, this implementation consolidates
the task execution and synchronization logic into the `DeviceThread`.
Synchronization between devices is handled by a two-phase, semaphore-based
reusable barrier. The locking strategy is particularly complex, involving a
lazily populated shared dictionary of location-specific locks and additional
device-specific locks, which may be prone to deadlocks.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    A reusable barrier implemented using two semaphores and a lock.

    This is a classic two-phase barrier. All threads must enter and be released
    from the first phase before any can enter the second, ensuring no thread
    can lap another.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads

        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first synchronization phase of the barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread has arrived, release all threads from phase 1.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next cycle.
        self.threads_sem1.acquire()

    def phase2(self):
        """The second synchronization phase of the barrier."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread has arrived, release all threads from phase 2.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next cycle.
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a device node, acting primarily as a data container.
    
    The active logic is encapsulated within the `DeviceThread`. This class holds
    the device's state and configuration.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.script_received = Event()
        self.locations_map = dict() # Unused variable.
        self.data_lock = Lock()     # A per-device lock for setting data.
        self.barrier = None
        self.locations_locks = None # A shared dictionary for location locks.

        self.thread = DeviceThread(self)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources for the simulation.

        Run by device 0, this method creates and shares the main barrier and the
        lock dictionary. Crucially, it starts each device's `DeviceThread` only
        after these shared resources have been assigned.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            self.locations_locks = dict()

        for device in devices:
            if device.device_id != self.device_id:
                if self.device_id == 0:
                    device.locations_locks = self.locations_locks
                    device.barrier = self.barrier

        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script or signals the end of script assignment."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data. Not internally thread-safe."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data. Not internally thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread of execution for a device.

    This thread is both the controller and the executor. It waits for scripts,
    executes them sequentially, and then synchronizes with other devices.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Shutdown signal.
                break

            # Wait for the supervisor to signal all scripts have been assigned.
            self.device.script_received.wait()
            
            # Execute all assigned scripts sequentially within this single thread.
            for (script, location) in self.device.scripts:
                
                # Lazily initialize a lock for the location if not already present.
                if location not in self.device.locations_locks:
                    self.device.locations_locks[location] = Lock()

                # --- Potentially Unsafe Locking ---
                # This section contains a complex locking pattern that may be
                # prone to deadlocks in a multi-device scenario.
                self.device.locations_locks[location].acquire()

                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    result = script.run(script_data)

                    # While holding the location lock, it acquires a lock on each
                    # neighbor to update its data. This can cause A->B, B->A deadlocks.
                    for device in neighbours:
                        device.data_lock.acquire()
                        device.set_data(location, result)
                        device.data_lock.release()
                    
                    self.device.data_lock.acquire()
                    self.device.set_data(location, result)
                    self.device.data_lock.release()

                self.device.locations_locks[location].release()

            # Wait for all other devices to finish their time step.
            self.device.barrier.wait()
            self.device.script_received.clear()
            self.device.scripts = [] # Clear scripts for the next round.