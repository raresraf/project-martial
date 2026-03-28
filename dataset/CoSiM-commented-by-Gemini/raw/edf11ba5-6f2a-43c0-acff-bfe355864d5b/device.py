"""
This module implements a device simulation using a thread-per-script execution model.

A central `DeviceThread` orchestrates the work for each time step. When triggered,
it creates and starts a new `Worker` thread for every assigned script, waits for
all of them to complete, and then synchronizes with other devices using a barrier.
The implementation contains a critical data race due to non-thread-safe access
to sensor data.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a device node in the simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.lock_locations = []
        # The barrier is initialized empty and configured later in setup_devices.
        self.barrier = ReusableBarrierSem(0)
        self.thread = DeviceThread(self)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs global, centralized setup from the coordinator device (ID 0).

        This method creates and distributes a shared barrier and a list of
        location-based locks to all devices. It also starts the main thread
        for every device in the simulation.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        barrier = ReusableBarrierSem(len(devices))

        if self.device_id == 0:
            # Determine the total number of unique locations to size the lock list.
            nr_locations = 0
            for i in range(len(devices)):
                for location in devices[i].sensor_data.keys():
                    if location > nr_locations:
                        nr_locations = location
            nr_locations += 1

            # Create a lock for each location.
            for i in range(nr_locations):
                self.lock_locations.append(Lock())

            # Distribute the barrier and locks to all devices.
            for i in range(len(devices)):
                devices[i].barrier = barrier
                for j in range(nr_locations):
                    devices[i].lock_locations.append(self.lock_locations[j])
                
                # Start the main thread for each device.
                devices[i].thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A None script signals that all scripts
        for the timepoint have been assigned.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves data for a given location.
        WARNING: This method is not thread-safe.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets data for a given location.
        WARNING: This method is not thread-safe.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """The main orchestration thread for a device."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation loop: waits for scripts, spawns workers, and synchronizes.
        """
        workers = []
        while True:
            # Get neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Supervisor signals termination.

            # Wait for the `assign_script` method to signal all scripts are received.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Block Logic: Spawn a new worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                workers.append(Worker(self.device, script,
                                        location, neighbours))

            # Start all worker threads to run in parallel.
            for i in range(len(workers)):
                workers[i].start()

            # Wait for all worker threads to complete their execution.
            for i in range(len(workers)):
                workers[i].join()

            # Clean up the worker list for the next timepoint.
            workers = []
            
            # All work for this timepoint is done; wait at the barrier for other devices.
            self.device.barrier.wait()


class Worker(Thread):
    """A single-use worker thread for executing one script."""
    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def solve_script(self, script, location, neighbours):
        """Contains the logic for executing the script."""
        # Pre-condition: Acquire the specific lock for this location.
        self.device.lock_locations[location].acquire()

        script_data = []
        # Block Logic: Gather data from neighbors and the local device.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # Invariant: Only run the script if there is data to process.
        if script_data:
            result = script.run(script_data)
            # Block Logic: Propagate the result to all involved devices.
            for device in neighbours:
                device.set_data(location, result)
            self.device.set_data(location, result)

        # Post-condition: Release the lock.
        self.device.lock_locations[location].release()

    def run(self):
        """The thread's entry point."""
        self.solve_script(self.script, self.location, self.neighbours)