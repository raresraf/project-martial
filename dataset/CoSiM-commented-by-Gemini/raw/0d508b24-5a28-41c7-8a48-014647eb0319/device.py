# -*- coding: utf-8 -*-
"""
This module implements a distributed device simulation with a unique
architecture featuring a three-phase barrier and a batched thread-per-task model.

-   **Setup**: A leader-based protocol where the first device in a list creates
    shared resources (locks, barrier) and the last device starts all the main
    threads.
-   **Concurrency**: Each device creates a new thread for each assigned script,
    but it does so in batches of 8, waiting for each batch to complete before
    creating the next.
-   **Synchronization**: A three-phase barrier is used within each timepoint to
    synchronize all devices at multiple stages.
-   **Locking**: Per-location locks are created lazily in a shared object, but
    the creation is not thread-safe, and looking up a lock requires iterating
    through a list.

Classes:
    Device: A node in the network.
    DeviceThread: The main control loop that manages batched execution.
    SharedObjects: A simple container for shared state.
    ScriptThread: A worker thread that executes a single script.
"""

from threading import Event, Thread, Lock
# Assumes ReusableBarrier is defined in a 'reusable_barrier' module.
from reusable_barrier import ReusableBarrier


class Device(object):
    """Represents a device in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.script_received = Event()
        self.timepoint_done = Event()
        
        # --- Synchronization and Shared Objects ---
        # lock and lock2 appear to be global, while shared_obj holds per-location locks
        self.lock = None
        self.lock2 = None
        self.shared_obj = None
        self.reusable_barrier = None
        
        self.script_threads = []
        self.thread = DeviceThread(self)
        # The thread is NOT started here; it's started by the last device in setup_devices.

    def __str__(self):
        return "Device {}".format(self.device_id)

    def setup_devices(self, devices):
        """
        A peculiar setup protocol where the first device is the leader for creating
        resources and the last device is the trigger for starting the simulation.
        """
        # The first device acts as the leader to create shared objects.
        if self == devices[0]:
            lock = Lock()
            lock2 = Lock()
            barrier = ReusableBarrier(len(devices))
            shared_obj = SharedObjects()
            # Distribute the same object references to all devices.
            for device in devices:
                device.lock = lock
                device.lock2 = lock2
                device.reusable_barrier = barrier
                device.shared_obj = shared_obj

        # The last device in the list is responsible for starting all threads.
        if self == devices[-1]:
            for device in devices:
                device.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to the device's list of work for the timepoint."""
        # This lock protects the scripts list.
        with self.lock2:
            if script is not None:
                self.scripts.append((script, location))
            else:
                # Signal that all scripts for this timepoint have been assigned.
                self.timepoint_done.set()
                self.script_received.set()

    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Thread-safely sets data using a shared global lock."""
        with self.lock2:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """The main control loop, executing scripts in batches of 8."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop with a three-phase barrier."""
        position = 0
        while True:
            # --- PHASE 1: Sync before getting neighbors ---
            self.device.reusable_barrier.wait()

            with self.device.lock: # Unnecessary contention just to get neighbors
                neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Shutdown signal
                break

            # --- PHASE 2: Sync after getting neighbors ---
            self.device.reusable_barrier.wait()
            
            # Wait for supervisor to assign all scripts.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # --- Batched Thread-per-Task Execution ---
            for (script, location) in self.device.scripts:
                # Lazy, non-atomic creation of per-location locks.
                with self.device.lock:
                    if not any(loc == location for loc, _ in self.device.shared_obj.locations_lock):
                        self.device.shared_obj.locations_lock.append((location, Lock()))
                
                # Create a thread for the script.
                script_thread = ScriptThread(self.device, script, location, neighbours, position)
                self.device.script_threads.append(script_thread)
                position += 1
                
                # If a batch of 8 is ready, execute it.
                if position == 8:
                    position = 0
                    for st in self.device.script_threads:
                        st.start()
                    for st in self.device.script_threads:
                        st.join()
                    self.device.script_threads = []

            # Execute any remaining threads that didn't form a full batch.
            for script_thread in self.device.script_threads:
                script_thread.start()
            for script_thread in self.device.script_threads:
                script_thread.join()
            self.device.script_threads = []

            # --- PHASE 3: Sync after work is complete ---
            self.device.reusable_barrier.wait()
            
            # Redundant wait, as joins have already confirmed completion.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            self.device.scripts = []


class SharedObjects(object):
    """A simple container for objects to be shared across all devices."""
    def __init__(self):
        # A list of (location, Lock) tuples.
        self.locations_lock = []


class ScriptThread(Thread):
    """A worker thread that executes a single script."""

    def __init__(self, device, script, location, neighbours, position):
        name = "Device Thread {}, Script {}".format(device.device_id, script)
        Thread.__init__(self, name=name)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.position = position

    def run(self):
        """Executes the script logic after finding and acquiring the correct lock."""
        script_data = []
        lock = None
        
        # Inefficiently search for the lock for this location. A dict would be O(1).
        for (location, a_lock) in self.device.shared_obj.locations_lock:
            if location == self.location:
                lock = a_lock
                break
        
        if lock:
            with lock:
                # --- Standard gather-run-update logic ---
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    result = self.script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    self.device.set_data(self.location, result)