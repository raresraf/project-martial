"""
This module defines a device simulation framework using a fixed pool of
worker threads within each device and a multi-level barrier system for
synchronization.

The architecture consists of three primary classes:
- ReusableBarrier: A Condition-based barrier for synchronizing threads.
- Device: Represents a node in the network, containing a fixed pool of
  worker threads.
- DeviceThread: The worker thread. All logic for synchronization and script
  execution is handled within this class.
"""

from threading import Event, Thread, Lock, Condition

class ReusableBarrier(object):
    """
    A reusable barrier implementation using a Condition variable.

    Threads block on `wait()` until the required number of threads have
    arrived. The barrier then releases all threads and resets for future use.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Causes the calling thread to wait at the barrier."""
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                self.cond.notify_all()
                self.count_threads = self.num_threads
            else:
                self.cond.wait()


class Device(object):
    """
    Represents a single device in the simulation.

    Each device manages a fixed-size pool of worker threads (`DeviceThread`)
    that persist for the lifetime of the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.thread_number = 8
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # --- State and Script Management ---
        self.neighbours_list = []
        self.neighbours_list_collected = False
        self.scripts = []
        
        # --- Synchronization Primitives ---
        self.thread_lock = Lock()  # For protecting device-level state like neighbor list.
        self.thread_barrier = ReusableBarrier(self.thread_number) # Local barrier for this device's threads.
        self.locations_list = {}   # Shared dict of locks for data locations.
        self.global_barrier = None # Global barrier for all threads in the simulation.
        self.setup_event = Event()      # Signals that global setup is done.
        self.timepoint_done = Event() # Signals that script assignment is done.

        self.threads = [DeviceThread(self, i) for i in range(self.thread_number)]
        for thread in self.threads:
            thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global synchronization objects for all devices.

        The root device (id 0) creates and distributes the shared global barrier
        and the dictionary of location locks.
        """
        if self.device_id == 0:
            # Discover all unique locations to create a lock for each.
            for device in devices:
                for loc in device.sensor_data:
                    self.locations_list.update({loc:Lock()})
            
            # Global barrier must be sized for all threads from all devices.
            max_threads = self.thread_number * len(devices)
            self.global_barrier = ReusableBarrier(max_threads)
            self.setup_event.set()
        else:
            # Non-root devices wait for the root device to complete setup.
            main_device = next(dev for dev in devices if dev.device_id == 0)
            main_device.setup_event.wait()
            # Copy references to the shared objects.
            self.global_barrier = main_device.global_barrier
            self.locations_list = main_device.locations_list
            self.setup_event.set()

    def assign_script(self, script, location):
        """Assigns a script to be processed in the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of script assignment.
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data from a location (not intrinsically thread-safe)."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a location (not intrinsically thread-safe)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all worker threads."""
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A persistent worker thread within a Device. It handles a statically
    partitioned subset of the device's scripts for each time step.
    """

    def __init__(self, device, my_id):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.my_id = my_id # The ID of this thread within its device (0-7).

    def run(self):
        """The main lifecycle loop of the worker thread."""
        # Wait until the root device has finished setting up shared resources.
        self.device.setup_event.wait()

        while True:
            # --- Neighbor List Fetching (Critical Section) ---
            # Only one thread per device fetches the neighbor list per time step.
            with self.device.thread_lock:
                if self.device.neighbours_list_collected is False:
                    self.device.neighbours_list_collected = True
                    self.device.neighbours_list = self.device.supervisor.get_neighbours()
            
            if self.device.neighbours_list is None:
                break # End of simulation.

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()

            # --- Static Work Distribution ---
            # Process a subset of scripts based on this thread's ID.
            for i in xrange(self.my_id, len(self.device.scripts), self.device.thread_number):
                (script, location) = self.device.scripts[i]
                
                # Use the global lock for this location to ensure serial access.
                with self.device.locations_list[location]:
                    script_data = []
                    # Gather data.
                    for device in self.device.neighbours_list:
                        data = device.get_data(location)
                        if data is not None: script_data.append(data)
                    data = self.device.get_data(location)
                    if data is not None: script_data.append(data)

                    # Execute and propagate results.
                    if script_data:
                        result = script.run(script_data)
                        for device in self.device.neighbours_list:
                            device.set_data(location, result)
                        self.device.set_data(location, result)

            # --- Multi-Level Barrier Synchronization ---
            # 1. Global Barrier: All threads from all devices must arrive here.
            self.device.global_barrier.wait()

            # 2. Local Cleanup: Reset flags for the next time step.
            # Only one thread per device performs the cleanup.
            with self.device.thread_lock:
                if self.device.neighbours_list_collected is True:
                    self.device.neighbours_list_collected = False
                    self.device.timepoint_done.clear()

            # 3. Local Barrier: All threads of this device wait here to ensure
            # cleanup is done before any thread starts the next time step.
            self.device.thread_barrier.wait()
