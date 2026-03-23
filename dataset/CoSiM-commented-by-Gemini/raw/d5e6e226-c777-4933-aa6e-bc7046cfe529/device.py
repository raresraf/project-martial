"""
This module contains another implementation of a distributed device simulation,
combining a multi-threaded device model with centralized setup and a mix of
synchronization primitives.
"""

from threading import Lock, Event, Thread, Condition

class ReusableBarrier():
    """
    A reusable barrier implementation using a Condition variable.
    WARNING: This is a single-phase barrier, which is not a correct implementation
    for general-purpose reusable barriers and can be prone to race conditions
    (the 'lapper' problem). A two-phase barrier is required for robustness.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks until all `num_threads` have called this method."""
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                self.cond.notify_all()
                self.count_threads = self.num_threads
            else:
                self.cond.wait()


class Device(object):
    """
    Represents a device in the simulation, managing multiple worker threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.gotneighbours = Event()
        self.zavor = Lock()  # Lock for neighbor discovery
        self.threads = []
        self.neighbours = []
        self.nthreads = 8
        self.barrier = ReusableBarrier(1) # This barrier is overwritten during setup.
        self.lockforlocation = {}
        self.num_locations = supervisor.supervisor.testcase.num_locations
        
        for i in range(self.nthreads):
            self.threads.append(DeviceThread(self, i))
        for thread in self.threads:
            thread.start()

    def __str__(self):
        return f"Device {self.device_id}"

    @staticmethod
    def setup_devices(devices):
        """
        A static-like method to initialize and distribute shared synchronization
        objects (barrier and location locks) to all devices in the simulation.
        """
        num_all_threads = devices[0].nthreads * len(devices)
        barrier = ReusableBarrier(num_all_threads)
        lockforlocation = {i: Lock() for i in range(devices[0].num_locations)}

        for device in devices:
            device.barrier = barrier
            device.lockforlocation = lockforlocation

    def assign_script(self, script, location):
        """Assigns a script to the device for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that all scripts are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets sensor data for a location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data for a location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all worker threads to shut down the device."""
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread within a device.
    """
    def __init__(self, device, id_thread):
        Thread.__init__(self, name=f"Device Thread {device.device_id}-{id_thread}")
        self.device = device
        self.id_thread = id_thread

    def run(self):
        """
        The main simulation loop for the thread.
        """
        while True:
            # Ensure neighbor discovery is done only once per device per time step.
            with self.device.zavor:
                if not self.device.gotneighbours.is_set():
                    self.device.neighbours = self.device.supervisor.get_neighbours()
                    self.device.gotneighbours.set()

            if self.device.neighbours is None:
                break  # End of simulation.

            # Wait for all scripts for the timepoint to be assigned.
            self.device.timepoint_done.wait()

            # Statically partition the scripts among the device's threads.
            # NOTE: The stride `self.device.nthreads + 1` is likely a bug
            # and should probably be `self.device.nthreads`.
            for i in range(self.id_thread, len(self.device.scripts), self.device.nthreads + 1):
                script, location = self.device.scripts[i]
                with self.device.lockforlocation[location]:
                    # Gather data from self and neighbors.
                    script_data = [dev.get_data(location) for dev in self.device.neighbours if dev.get_data(location) is not None]
                    local_data = self.device.get_data(location)
                    if local_data is not None:
                        script_data.append(local_data)

                    if script_data:
                        # Run script and propagate results.
                        result = script.run(script_data)
                        for dev in self.device.neighbours:
                            dev.set_data(location, result)
                        self.device.set_data(location, result)

            # Synchronize all threads from all devices after script execution.
            self.device.barrier.wait()

            # The first thread of each device handles cleanup for the next cycle.
            if self.id_thread == 0:
                self.device.timepoint_done.clear()
                self.device.gotneighbours.clear()

            # Second barrier wait to ensure cleanup is done before the next loop.
            self.device.barrier.wait()
