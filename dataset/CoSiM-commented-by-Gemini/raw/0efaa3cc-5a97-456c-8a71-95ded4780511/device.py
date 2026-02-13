"""
This module implements a distributed device simulation using a static work
partitioning model. Each device operates a pool of worker threads, and the
set of scripts for a given timepoint is statically divided among these threads.

Key architectural features:
- A two-level barrier system: one barrier for synchronizing all devices at the
  end of a timepoint, and another for synchronizing the worker threads within
  a single device.
- Location-based semaphores are used to ensure thread-safe access to sensor
  data that might be shared or accessed concurrently.
- Static load balancing: The list of scripts is divided among a device's
  worker threads using a fixed stride, rather than a dynamic work queue.
"""

from threading import Event, Thread, Semaphore
from reusable_barrier import TimePointsBarrier, ClassicBarrier

class Device(object):
    """
    Represents a device node in the simulation.

    Each device manages its own sensor data, communicates with a supervisor to get
    information about its neighbors, and executes scripts using a pool of
    dedicated worker threads (`DeviceThread`).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor readings.
            supervisor (Supervisor): An external object for orchestration.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.threads = []
        self.neighbours = []
        self.num_threads = 8

        # A semaphore for each unique location to protect data access.
        self.locations_semaphore = None
        # A barrier to synchronize all devices at the end of a timepoint.
        self.devices_barrier = None
        # A barrier to synchronize worker threads within this device.
        self.neighbours_barrier = None
        # A list of all devices in the simulation.
        self.all_devices = None

    def set_neighbours(self, new_neighbours):
        """Sets the list of neighboring devices for the current timepoint."""
        self.neighbours = new_neighbours

    def set_devices_barrier(self, barrier):
        """Assigns the global, inter-device synchronization barrier."""
        self.devices_barrier = barrier

    def set_locations_semaphore(self, locations_semaphore):
        """Assigns the list of location-specific semaphores."""
        self.locations_semaphore = locations_semaphore

    def get_locations(self, location_list):
        """Appends this device's locations to a provided list."""
        for location in self.sensor_data:
            location_list.append(location)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources for all devices.

        Device 0 acts as the coordinator, creating a shared barrier and a set of
        semaphores for all unique data locations. These are then distributed to
        all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Pre-condition: Device 0 is the coordinator for this one-time setup.
        if self.device_id == 0:
            barrier = ClassicBarrier(len(devices))
            locations = []
            locations_semaphore = []
            # Gather all unique locations from all devices.
            for device in devices:
                device.get_locations(locations)
            locations = sorted(list(set(locations)))

            # Create one semaphore for each unique location.
            for i in range(0, len(locations)):
                locations_semaphore.append(Semaphore(value=1))

            # Distribute the shared barrier and semaphores to all devices.
            for device in devices:
                device.set_devices_barrier(barrier)
                device.set_locations_semaphore(locations_semaphore)

        self.all_devices = devices
        # This barrier synchronizes the worker threads within this device.
        self.neighbours_barrier = TimePointsBarrier(self.num_threads, self)

        # Create and start the pool of worker threads for this device.
        for i in range(self.num_threads):
            current_thread = DeviceThread(self, i)
            current_thread.start()
            self.threads.append(current_thread)

    def assign_script(self, script, location):
        """Assigns a script to the device for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A 'None' script signals the end of script assignments for this timepoint.
            self.timepoint_done.set()

    def has_data(self, location):
        """Checks if the device has data for a given location."""
        return location in self.sensor_data

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to complete."""
        for i in range(self.num_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread that executes a statically assigned subset of scripts.
    """

    def __init__(self, device, thread_id):
        """
        Initializes a worker thread.

        Args:
            device (Device): The parent device.
            thread_id (int): A unique ID (0 to num_threads-1) for this thread.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main execution loop for the worker thread."""
        # Invariant: The loop continues until the simulation ends.
        while True:
            # The internal barrier synchronizes all worker threads within the device
            # after the supervisor has set the neighbors.
            self.device.neighbours_barrier.wait()

            # A 'None' neighbor list signals the end of the simulation.
            if self.device.neighbours is None:
                break

            # Wait for the supervisor to signal that all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()

            if len(self.device.neighbours) != 0:
                devices_with_date = []
                
                # Static work partitioning: each thread processes a subset of the scripts
                # using a strided loop.
                for index in range(
                        self.thread_id,
                        len(self.device.scripts),
                        self.device.num_threads):
                    (script, location) = self.device.scripts[index]
                    
                    script_data = []
                    # Acquire the semaphore for the location to ensure exclusive access.
                    self.device.locations_semaphore[location].acquire()

                    # --- Critical Section for the location ---
                    # Gather data from neighbors.
                    for device in self.device.neighbours:
                        if device.has_data(location):
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                                devices_with_date.append(device)

                    # Gather data from the local device.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                        devices_with_date.append(self.device)

                    if script_data:
                        result = script.run(script_data)
                        # Broadcast the result back to all devices that provided data.
                        for device in devices_with_date:
                            device.set_data(location, result)
                        devices_with_date = []

                    # Release the semaphore for the location.
                    self.device.locations_semaphore[location].release()
