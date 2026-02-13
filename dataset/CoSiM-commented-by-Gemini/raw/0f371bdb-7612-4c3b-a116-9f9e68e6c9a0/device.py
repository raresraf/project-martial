"""
This module implements a distributed device simulation with a sophisticated
synchronization and work distribution model.

Key architectural features:
- Leader Election: A simple leader (the device with the lowest ID) is chosen to
  initialize and manage resources shared across all devices.
- Custom Work Queue: Instead of a standard library Queue, this implementation uses
  a list (`scripts`) protected by a `threading.Condition` variable to create a
  producer-consumer work queue.
- Two-Level Barrier System: It employs two distinct barriers: one for synchronizing
  worker threads within a single device, and a global barrier for synchronizing
  all devices at the end of a timepoint.
- Specialized Worker Thread: Within each device's thread pool, one thread (ID 0)
  has special responsibilities for setup and teardown of a timepoint.
"""

from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a device node in a distributed simulation environment.

    This device manages a pool of worker threads (`DeviceThread`) that consume tasks
    from a custom, shared work queue. It participates in a global synchronization
    scheme coordinated by a leader device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (Supervisor): The central controller of the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # A dictionary of locks for each data location, shared across all devices.
        self.data_lock = None

        # The `scripts` list acts as a shared work queue.
        self.scripts = []
        self.devices = []
        self.neighbours = []

        # `first_script` and `last_script` are indices to manage the custom work queue.
        self.first_script = -1
        self.last_script = -1
        # A condition variable to synchronize access to the `scripts` list.
        self.script_available = Condition()

        self.timepoint_done = Event()
        
        # A simple leader election mechanism; the device with the lowest ID becomes leader.
        self.leader_id = self.device_id
        # A global barrier to synchronize all devices between timepoints.
        self.timepoint_sync_barrier = None
        self.timepoint_barrier_set = Event()

        # An internal barrier to synchronize the worker threads of this device.
        self.barrier = ReusableBarrierCond(8)

        self.threads = []
        for i in xrange(8):
            self.threads.append(DeviceThread(self, i))

        for i in xrange(8):
            self.threads[i].start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def set_neighbours(self):
        """Fetches the list of neighbors from the supervisor for the current timepoint."""
        self.neighbours = self.supervisor.get_neighbours()

    def setup_devices(self, devices):
        """
        Performs setup, including leader election and shared resource initialization.
        
        Args:
            devices (list): A list of all devices in the simulation.
        """
        self.devices = devices
        # Elect the device with the minimum ID as the leader.
        for dev in self.devices:
            dev_id = dev.device_id
            if dev_id < self.leader_id:
                self.leader_id = dev_id

        # The leader device initializes and distributes the shared resources.
        if self.leader_id == self.device_id:
            self.data_lock = {}
            self.timepoint_sync_barrier = ReusableBarrierCond(len(devices))
            for dev in devices:
                dev.timepoint_sync_barrier = self.timepoint_sync_barrier
                dev.data_lock = self.data_lock
                dev.timepoint_barrier_set.set()
        else:
            # Non-leader devices wait for the leader to finish setup.
            self.timepoint_barrier_set.wait()


    def assign_script(self, script, location):
        """
        Assigns a script to the device, acting as the producer for the work queue.
        
        Args:
            script (Script): The script to be executed.
            location (str): The location context for the script.
        """
        if script is not None:
            # The condition lock protects the shared `scripts` list.
            self.script_available.acquire()

            # Lazily create locks for new locations.
            if location not in self.data_lock:
                self.data_lock[location] = Lock()

            self.last_script = self.last_script + 1
            self.scripts.append((script, location))
            # Notify one waiting worker thread that a new script is available.
            self.script_available.notify()

            self.script_available.release()
        else:
            # A 'None' script signals the end of assignments for this timepoint.
            self.script_available.acquire()

            self.timepoint_done.set()
            # Notify all worker threads to wake up and check the `timepoint_done` event.
            self.script_available.notify_all()

            self.script_available.release()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def end_timepoint(self):
        """Called by the coordinator thread to end the current timepoint."""
        # Synchronize all devices globally.
        self.timepoint_sync_barrier.wait()
        # Reset state for the next timepoint.
        self.first_script = -1
        self.timepoint_done.clear()

    def shutdown(self):
        """Waits for all worker threads of this device to complete."""
        for i in xrange(8):
            self.threads[i].join()


class DeviceThread(Thread):
    """

    A worker thread that consumes and executes scripts from a custom shared queue.
    """

    def __init__(self, device, my_id):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent device.
            my_id (int): The unique ID for this thread within the device's pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.my_id = my_id

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            # The thread with ID 0 is the coordinator for this device's timepoint.
            if self.my_id == 0:
                self.device.set_neighbours()

            # Phase 1: All worker threads sync here after neighbors are set.
            self.device.barrier.wait()
            neighbours = self.device.neighbours

            if neighbours is None:
                break # End of simulation signal.

            # This loop is the consumer part of the producer-consumer pattern.
            while True:
                script = None
                location = None

                # Acquire the condition lock to safely access the work queue.
                self.device.script_available.acquire()
                # If there are no scripts to process...
                if self.device.first_script == self.device.last_script:
                    # ...wait until notified, or until the timepoint is marked as done.
                    while not self.device.timepoint_done.is_set():
                        self.device.script_available.wait()
                        # After waking up, check if new work has arrived.
                        if self.device.first_script < self.device.last_script:
                            self.device.first_script += 1
                            (script, location) = self.device.scripts[self.device.first_script]
                            break # Exit the inner wait loop.
                else:
                    # There are scripts in the queue, take one.
                    self.device.first_script += 1
                    (script, location) = self.device.scripts[self.device.first_script]
                self.device.script_available.release()

                if location is None:
                    break # No more scripts for this timepoint.

                # Use the location-specific lock to ensure exclusive data access.
                with self.device.data_lock[location]:
                    # --- Critical Section ---
                    script_data = []
                    
                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data:
                        result = script.run(script_data)
                        
                        for device in neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)

            # Phase 2: Sync after the script execution loop is finished.
            self.device.barrier.wait()
            # The coordinator thread finalizes the global timepoint.
            if self.my_id == 0:
                self.device.end_timepoint()
            # Phase 3: Sync again to ensure all threads start the next timepoint together.
            self.device.barrier.wait()
