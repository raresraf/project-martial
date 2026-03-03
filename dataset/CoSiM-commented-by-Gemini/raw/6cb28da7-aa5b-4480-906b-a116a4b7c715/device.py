"""
This module implements a distributed device simulation using a complex,
multi-level synchronization scheme.

The architecture is defined by:
- A custom `Barrier` class implemented with `threading.Condition`. NOTE: This
  barrier implementation is not safely reusable and is prone to race conditions.
- A `Device` class that manages a pool of eight internal `DeviceThread` workers.
- A master/worker pattern within each device's threads, where one "first"
  thread handles inter-device communication and synchronization.
- A work-stealing model where worker threads atomically claim script indexes
  to execute in parallel.
- Two levels of barriers: one for internal thread synchronization within a
  device, and another for synchronizing all devices together.
"""

from threading import Event, Thread, Condition, Lock


class Barrier(object):
    """
    A custom barrier implementation using a Condition variable.

    WARNING: This barrier is NOT safely reusable. A race condition can occur
    if threads loop and call `wait()` again before all threads from the
    previous `wait()` have woken up and released the condition's underlying
    lock. This can lead to deadlocks in the simulation.
    """

    def __init__(self, num_threads=0):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have called `wait`.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # The last thread arrives, wakes up all waiting threads.
            self.cond.notify_all()
            # Resets the counter for the (unsafe) next use.
            self.count_threads = self.num_threads
        else:
            # Not the last thread, so wait to be notified.
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a device in the simulation, managing its state and worker threads.

    It uses class-level attributes to share a global barrier and data locks
    among all device instances.
    """
    
    # Class-level attributes shared by all Device instances.
    bariera_devices = Barrier()
    locks = []

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its internal pool of 8 worker threads."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        self.scripts = []
        self.locations = []
        self.nr_scripturi = 0
        self.script_crt = 0  # Shared index for work-stealing.

        self.timepoint_done = Event()
        self.neighbours = []
        self.event_neighbours = Event()
        self.lock_script = Lock()
        self.bar_thr = Barrier(8) # Internal barrier for this device's threads.

        # Create one "master" thread and 7 "worker" threads.
        self.thread = DeviceThread(self, 1) # first = 1 denotes the master.
        self.thread.start()
        self.threads = []
        for _ in range(7):
            tthread = DeviceThread(self, 0) # first = 0 for workers.
            self.threads.append(tthread)
            tthread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared (class-level) barrier and data locks.
        """
        Device.bariera_devices = Barrier(len(devices))
        if not Device.locks:
            for _ in range(self.supervisor.supervisor.testcase.num_locations):
                Device.locks.append(Lock())

    def assign_script(self, script, location):
        """Assigns a script to the device's workload."""
        if script is not None:
            self.scripts.append(script)
            self.locations.append(location)
            self.nr_scripturi += 1
        else:
            # A None script signals that all assignments are complete for this step.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. Access is not synchronized here."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data. Access is not synchronized here."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all internal threads to complete."""
        self.thread.join()
        for tthread in self.threads:
            tthread.join()


class DeviceThread(Thread):
    """
    A worker thread for a device. One thread is designated as the "first"
    (or master) and has additional responsibilities.
    """

    def __init__(self, device, first):
        """
        Initializes the thread.
        Args:
            device (Device): The parent device.
            first (int): 1 if this is the master thread, 0 otherwise.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.first = first

    def run(self):
        """Main execution loop for the thread."""
        while True:
            # --- Master thread's setup phase ---
            if self.first == 1:
                # The master thread gets neighbors and resets script counter.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.script_crt = 0
                self.device.event_neighbours.set()

            # All threads wait until the master has fetched the neighbors.
            self.device.event_neighbours.wait()

            if self.device.neighbours is None:
                break # Shutdown signal.

            # Wait for supervisor to signal that script assignment is done.
            self.device.timepoint_done.wait()

            # --- Work-stealing loop ---
            while True:
                # Atomically get the index of the next script to execute.
                with self.device.lock_script:
                    index = self.device.script_crt
                    self.device.script_crt += 1
                
                # If the index is out of bounds, all scripts have been claimed.
                if index >= self.device.nr_scripturi:
                    break

                location = self.device.locations[index]
                script = self.device.scripts[index]

                # --- Script execution with location-based locking ---
                with Device.locks[location]:
                    script_data = []
                    # Gather data from neighbors and self.
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    # Run script and update data on all involved devices.
                    if script_data:
                        result = script.run(script_data)
                        for device in self.device.neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)

            # --- Synchronization at end of time step ---
            # 1. All threads within a device sync at their internal barrier.
            self.device.bar_thr.wait()
            
            # 2. Master thread resets events for the next step.
            if self.first == 1:
                self.device.event_neighbours.clear()
                self.device.timepoint_done.clear()
            
            # 3. Another internal sync to prevent race on event clearing.
            self.device.bar_thr.wait()
            
            # 4. Master thread syncs with all other devices at the global barrier.
            if self.first == 1:
                Device.bariera_devices.wait()
