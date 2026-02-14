"""
This module defines a device simulation framework using a single-threaded
execution model for scripts on each device.

The architecture consists of three primary classes:
- ReusableBarrier: A semaphore-based barrier for global synchronization.
- Device: Represents a node in the network, managed by a single control thread.
- DeviceThread: The control and execution thread for a device. It processes
  all assigned scripts serially within each time step.

Note: The `MyThread` class at the end of the file appears to be unused
leftover code from a different context.
"""
from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier implemented with semaphores and a lock.

    This uses a two-phase signaling mechanism to ensure that threads from one
    barrier wait cycle do not proceed before all threads have been released,
    making it safe for repeated use in a loop.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes the calling thread to wait at the barrier."""
        self._phase(self.count_threads1, self.threads_sem1, is_phase1=True)
        self._phase(self.count_threads2, self.threads_sem2, is_phase1=False)

    def _phase(self, count, threads_sem, is_phase1):
        with self.count_lock:
            if is_phase1:
                self.count_threads1 -= 1
                if self.count_threads1 == 0:
                    for _ in range(self.num_threads): threads_sem.release()
                    self.count_threads2 = self.num_threads
            else:
                self.count_threads2 -= 1
                if self.count_threads2 == 0:
                    for _ in range(self.num_threads): threads_sem.release()
                    self.count_threads1 = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device in the simulation.

    Each device has a single master thread (`DeviceThread`) that serially
    executes all scripts assigned to it for a given time step.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.data_lock = Lock()
        self.barrier = None
        self.location_locks = {}
        self.thread = DeviceThread(self)

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global synchronization objects for all devices.

        The root device (device_id == 0) creates and distributes the shared
        barrier and location lock dictionary.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id != self.device_id:
                    device.barrier = self.barrier
                    device.location_locks = self.location_locks
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be processed in the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of script assignment for this step.
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data from a location (not intrinsically thread-safe)."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a location (not intrinsically thread-safe)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control and execution thread for a Device.

    It waits for all scripts for a time step to be assigned, then executes
    them serially before synchronizing with other devices at a global barrier.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main lifecycle loop of the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the supervisor to signal that script assignment is done.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # --- Serial Script Execution ---
            for (script, location) in self.device.scripts:
                # Lazily create a lock for a location if it's new.
                if location not in self.device.location_locks:
                    self.device.location_locks[location] = Lock()

                # Acquire the global lock for this location to prevent other
                # devices from working on the same data simultaneously.
                with self.device.location_locks[location]:
                    script_data = []
                    # Gather data from neighbors and self.
                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data:
                        result = script.run(script_data)
                        
                        # The use of a second, per-device data_lock here is
                        # redundant and creates a high risk of deadlock. The
                        # outer location_lock is sufficient.
                        for device in neighbours:
                            with device.data_lock:
                                device.set_data(location, result)
                        with self.device.data_lock:
                            self.device.set_data(location, result)

            # --- Global Synchronization ---
            # Wait for all other devices to finish their time step.
            self.device.barrier.wait()