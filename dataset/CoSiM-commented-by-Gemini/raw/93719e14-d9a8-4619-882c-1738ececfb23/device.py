"""
This module provides another implementation for simulating a distributed network of
devices. This version uses a different threading and synchronization model compared
to other implementations. A master device (device 0) is responsible for setting up
shared synchronization objects (barriers and semaphores). Work is distributed
statically among threads.
"""
from threading import Event, Thread, Semaphore
from reusable_barrier import TimePointsBarrier, ClassicBarrier

class Device(object):
    """
    Represents a device in a simulated distributed sensor network.

    Each device manages its own sensor data and a pool of threads to execute
    scripts. It synchronizes with other devices using shared barriers and semaphores
    that are set up by a designated master device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): A dictionary of the device's local sensor readings,
                                keyed by location.
            supervisor: The central supervisor object managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals when a new script is assigned.
        self.scripts = [] # A list of (script, location) tuples for the current timepoint.
        self.timepoint_done = Event() # Signals that script assignment for a timepoint is complete.
        self.threads = [] # A pool of DeviceThread workers.
        self.neighbours = [] # A list of neighboring devices.
        self.num_threads = 8 # The number of worker threads per device.

        # Shared synchronization objects, initialized by setup_devices.
        self.locations_semaphore = None # A list of semaphores, one for each data location.
        self.devices_barrier = None # A barrier for all devices in the simulation.
        self.neighbours_barrier = None # A barrier for the internal threads of this device.
        self.all_devices = None

    def set_neighbours(self, new_neighbours):
        """Assigns a new list of neighbors to this device."""
        self.neighbours = new_neighbours


    def set_devices_barrier(self, barrier):
        """Sets the shared barrier for all devices."""
        self.devices_barrier = barrier


    def set_locations_semaphore(self, locations_semaphore):
        """Sets the shared list of semaphores for data locations."""
        self.locations_semaphore = locations_semaphore

    def get_locations(self, location_list):
        """Appends this device's known locations to the provided list."""
        for location in self.sensor_data:
            location_list.append(location)


    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the simulation environment, workers, and synchronization objects.

        Device 0 acts as a coordinator to create and distribute the shared barrier
        and location semaphores to all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: The device with ID 0 is responsible for global setup.
        if self.device_id == 0:
            # Create a barrier for all devices to synchronize at the end of a timepoint.
            barrier = ClassicBarrier(len(devices))
            
            # Collect all unique data locations from all devices.
            locations = []
            locations_semaphore = []
            for device in devices:
                device.get_locations(locations)
            locations = sorted(list(set(locations)))

            # Create a semaphore for each unique location to protect data access.
            for i in range(0, len(locations)):
                locations_semaphore.append(Semaphore(value=1))

            # Distribute the shared synchronization objects to all devices.
            for device in devices:
                device.set_devices_barrier(barrier)
                device.set_locations_semaphore(locations_semaphore)

        self.all_devices = devices
        # This barrier seems to be for synchronizing the device's own worker threads.
        self.neighbours_barrier = TimePointsBarrier(self.num_threads, self)

        # Create and start this device's pool of worker threads.
        for i in range(self.num_threads):
            current_thread = DeviceThread(self, i)
            current_thread.start()
            self.threads.append(current_thread)

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals the end of script assignment.

        Args:
            script: The script to be executed.
            location: The location the script operates on. If `script` is None,
                      it signals that the timepoint processing can begin.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def has_data(self, location):
        """Checks if the device has data for a specific location."""
        if location in self.sensor_data:
            return True
        return False
        
    def get_data(self, location):
        """Gets data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        """Sets data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to complete."""
        for i in range(self.num_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    ""

    A worker thread that executes a subset of the device's assigned scripts.
    Work is distributed statically among the threads based on their ID.
    """

    def __init__(self, device, thread_id):
        """
        Initializes a DeviceThread worker.

        Args:
            device (Device): The parent device.
            thread_id (int): The unique ID for this thread within the device's pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            # Internal barrier for all threads of this device.
            self.device.neighbours_barrier.wait()
            
            # The supervisor signals the end of the simulation by setting neighbours to None.
            if self.device.neighbours is None:
                break
            
            # Wait until the supervisor has finished assigning all scripts for the timepoint.
            self.device.timepoint_done.wait()

            if len(self.device.neighbours) != 0:
                devices_with_date = []
                
                # --- Static Work Distribution ---
                # Each thread processes a stride of the script list.
                for index in range(
                        self.thread_id,
                        len(self.device.scripts),
                        self.device.num_threads):
                    (script, location) = self.device.scripts[index]
                    
                    script_data = []
                    # --- Critical Section ---
                    # Acquire the semaphore for the data location to ensure exclusive access.
                    self.device.locations_semaphore[location].acquire()

                    # Pre-condition: Gather data from all neighboring devices.
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

                    # Invariant: Execute the script and update data on all involved devices.
                    if script_data != []:
                        result = script.run(script_data)
                        for device in devices_with_date:
                            device.set_data(location, result)
                        devices_with_date = []

                    self.device.locations_semaphore[location].release()
                    # --- End Critical Section ---