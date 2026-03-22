"""
This module implements a highly complex distributed device simulation.

Key architectural features:
- Each `Device` object manages its own internal pool of worker threads (`DeviceThread`).
- A single, global `Barrier` is created to synchronize *every worker thread from
  every device* simultaneously.
- The control flow within each worker thread is orchestrated by a sequence of
  these global barriers, forcing all threads to execute steps in lockstep.
- A "master" worker thread (thread_id == 0) within each device is designated
  to handle setup tasks like fetching data from the supervisor.
- A global dictionary of location-specific locks is shared among all devices.
- A dangerous nested locking pattern exists where a global location lock is held
  while a per-device lock is acquired, creating a deadlock risk.

Note: This script depends on a local `barrier.py` and uses Python 2 syntax.
"""

from threading import Event, Thread, Lock
from barrier import Barrier

class Device(object):
    """
    Represents a device that manages its own pool of worker threads.

    It participates in a collective setup of global synchronization objects
    (a single large barrier and a dictionary of locks for all locations).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbours = []
        self.scripts = []
        self.temp_scripts = [] # Used for work-stealing by its threads.

        self._thread_list = []
        self.timepoint_done = Event()
        self.device_lock = Lock()
        self.script_list_lock = Lock()

        # --- Globally Shared Objects ---
        self.locations_locks = {}
        self.device_thread_barrier = None
        
        self.thread_number = 8

        # Create this device's pool of worker threads. They are started later.
        for thread_id in xrange(self.thread_number):
            thread = DeviceThread(self, thread_id)
            self._thread_list.append(thread)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs collective setup and starts all worker threads.

        Initializes a single global barrier for all worker threads across all
        devices and a global dictionary of locks for all sensor locations.
        """
        # Create a single barrier for all worker threads in the simulation.
        if self.device_thread_barrier is None:
            self.device_thread_barrier = Barrier(len(devices) * self.thread_number)
            for dev in devices:
                dev.device_thread_barrier = self.device_thread_barrier

        # Create a single dictionary of locks for all locations, shared globally.
        max_location = -1
        if not self.locations_locks:
            for dev in devices:
                for key in dev.sensor_data:
                    if key > max_location:
                        max_location = key
            for i in xrange(max_location + 1):
                self.locations_locks[i] = Lock()
            for dev in devices:
                dev.locations_locks = self.locations_locks

        # Start this device's worker threads.
        for thread in self._thread_list:
            thread.start()

    def assign_script(self, script, location):
        """Assigns a script for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set() # Signal that all scripts are assigned.

    def get_data(self, location):
        """Non-thread-safe method to get data. Synchronization is external."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Non-thread-safe method to set data. Synchronization is external."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all of the device's worker threads."""
        for thread in self._thread_list:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread within a Device's pool.

    Its execution is tightly synchronized with all other worker threads in the
    simulation through a series of global barriers.
    """
    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device %d, Thread %d" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main execution loop, orchestrated by global barriers."""
        while True:
            # BARRIER 1: All worker threads sync at the start of a time step.
            self.device.device_thread_barrier.wait()
            
            # The 'master' thread (id 0) of each device fetches neighborhood data.
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # BARRIER 2: Wait for all masters to get neighbor data.
            self.device.device_thread_barrier.wait()
            
            if self.device.neighbours is None:
                break # Supervisor signals shutdown.

            # The 'master' thread of each device waits for scripts and copies them.
            if self.thread_id == 0:
                self.device.timepoint_done.wait()
                self.device.timepoint_done.clear()
                self.device.temp_scripts = list(self.device.scripts)

            # BARRIER 3: Wait for all masters to copy scripts.
            self.device.device_thread_barrier.wait()

            # --- Work-Stealing Phase ---
            done_iter = False
            while not done_iter:
                item = ()
                
                # Each thread tries to pop a script from its device's list.
                with self.device.script_list_lock:
                    if len(self.device.temp_scripts) > 0:
                        item = self.device.temp_scripts.pop(0)
                    else:
                        done_iter = True
                
                if not item:
                    continue

                script, location = item

                # Acquire the global lock for this location.
                with self.device.locations_locks[location]:
                    # --- Data Gathering ---
                    script_data = []
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    # --- Computation and Data Update ---
                    if script_data:
                        result = script.run(script_data)

                        # DANGEROUS: Nested locking. This thread holds the global
                        # location lock and now tries to acquire a per-device lock.
                        # This can cause deadlocks.
                        for device in self.device.neighbours:
                            with device.device_lock:
                                device.set_data(location, result)
                        
                        with self.device.device_lock:
                            self.device.set_data(location, result)
