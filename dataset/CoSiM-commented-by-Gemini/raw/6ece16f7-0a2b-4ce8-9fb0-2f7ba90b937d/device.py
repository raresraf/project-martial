
"""
This module implements a distributed device simulation using a single global
barrier for all threads and a work-stealing model for task distribution.

Architectural Overview:
- `ReentrantBarrier`: A custom, non-reusable barrier based on `Condition`
  which is used for synchronization. WARNING: This implementation is flawed and
  prone to deadlocks on reuse.
- `Device`: Manages its state and a pool of worker threads (`DeviceThread`).
  It contributes to a set of globally shared resources (barrier, locks).
- `DeviceThread`: The only thread class. All threads from all devices are
  instances of this class.
- Synchronization Model:
  - A single global barrier synchronizes all worker threads from all devices at
    the beginning of each step.
  - A dynamic "master election" pattern is used where the first thread to
    acquire a lock fetches shared data (neighbors) for its device.
  - A work-stealing pattern where threads within a device pull scripts from a
    shared list.
"""

from threading import Event, Thread, Lock, Condition

class ReentrantBarrier(object):
    """
    A custom barrier implementation using a Condition variable. Its name is
    misleading as it is not a reentrant lock.

    WARNING: This barrier is NOT safely reusable. A race condition can occur
    if threads loop and call `wait()` again before all threads from the
    previous `wait()` have woken up and released the condition's underlying
    lock. This can lead to deadlocks.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks until all threads reach the barrier."""
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                self.cond.notify_all()
                self.count_threads = self.num_threads
            else:
                self.cond.wait()

class Device(object):
    """
    Represents a device, managing its state and a pool of worker threads.
    Contributes to and uses globally shared synchronization primitives.
    """
    # Class-level attributes shared by all Device instances.
    barrier = None
    devices_lock = Lock()
    locations = [] # A global list of locks for data locations.
    nrloc = 0

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.sensor_data_lock = Lock() # Lock for the device's local sensor data.

        self.supervisor = supervisor
        self.gen_lock = Lock()

        # Primitives for managing script execution within the device.
        self.script_lock = Lock()
        self.script_event = Event()
        self.scripts = [] # Master list of scripts for the step.
        self.working_scripts = [] # The work-stealing queue.

        # State flags for coordination among the device's threads.
        self.neighbour_request = False
        self.neighbours = None
        self.timepoint_done = False
        self.reinit_barrier = None # Internal barrier for the device's threads.

        self.threads_num = 8
        self.threads = [DeviceThread(self, i) for i in xrange(self.threads_num)]

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the globally shared barrier and location locks.
        """
        with self.gen_lock:
            self.reinit_barrier = ReentrantBarrier(self.threads_num)

        with Device.devices_lock:
            # Initialize a global list of locks for all possible locations.
            Device.nrloc = max(Device.nrloc, (max(self.sensor_data.keys()) + 1))
            while Device.nrloc > len(Device.locations):
                Device.locations.append(Lock())
            
            # Create a single barrier for ALL threads across ALL devices.
            if Device.barrier is None:
                Device.barrier = ReentrantBarrier(len(devices) * self.threads_num)
        
        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        """Assigns a script to this device's workload for the current step."""
        with self.script_lock:
            if script is not None:
                self.scripts.append((script, location))
                self.working_scripts.append((script, location))
            else:
                self.timepoint_done = True
            self.script_event.set() # Notify waiting workers of new scripts.

    def get_data(self, location):
        """Safely retrieves data from the local sensor dictionary."""
        with self.sensor_data_lock:
            return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Safely sets data in the local sensor dictionary."""
        with self.sensor_data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all of this device's threads to complete."""
        for thread in self.threads:
            thread.join()

class DeviceThread(Thread):
    """A worker thread for a device."""
    def __init__(self, device, thread_nr):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.t_num = thread_nr

    def run_script(self, script, location):
        """
        Executes a single script, synchronizing on the specific data location.
        """
        with Device.locations[location]: # Fine-grained lock on location.
            script_data = []
            # Gather data from all neighbors and self.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                result = script.run(script_data)
                # Update data for all involved devices.
                for device in self.device.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)

    def run(self):
        """
        Main execution loop for the worker thread.
        """
        while True:
            # 1. Global Sync: All threads from all devices wait here.
            Device.barrier.wait()
            
            # 2. Reset Step: The first thread to acquire the lock resets state.
            with self.device.script_lock:
                if not self.device.working_scripts:
                    self.device.working_scripts = list(self.device.scripts)
                    self.device.timepoint_done = False
                    self.device.neighbour_request = False
            
            # 3. Internal Sync: All threads of this device wait here.
            self.device.reinit_barrier.wait()
            
            # 4. Master Election: One thread gets neighbors for this device.
            with Device.devices_lock:
                if not self.device.neighbour_request:
                    self.device.neighbours = self.device.supervisor.get_neighbours()
                    self.device.neighbour_request = True

            if self.device.neighbours is None:
                break # Shutdown signal.

            # 5. Work Stealing Loop
            while True:
                script, location = None, None
                with self.device.script_lock:
                    if self.device.working_scripts:
                        (script, location) = self.device.working_scripts.pop()
                    elif self.device.timepoint_done:
                        break # Step is complete, exit work loop.
                    else:
                        # No work and step not done, so wait for a script.
                        self.device.script_event.clear()
                
                if script:
                    self.run_script(script, location)
                elif not self.device.timepoint_done:
                    self.device.script_event.wait() # Block until new script is assigned.
