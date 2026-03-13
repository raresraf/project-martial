"""
This module defines a simulated Device for a distributed sensor network,
featuring a custom two-phase semaphore-based barrier for synchronization.

Each Device runs a main control thread (`DeviceThread`) that, for each
simulation timepoint, spawns a new worker thread (`MyScriptThread`) for every
assigned script. It waits for all workers to complete before synchronizing
with other devices to end the timepoint.
"""

from threading import Event, Semaphore, Lock, Thread

class ReusableBarrierSem(object):
    """
    A reusable barrier implemented using two Semaphores for two-phase synchronization.

    This prevents threads from one barrier cycle from mixing with threads from the
    next cycle. It's a classic way to build a reusable barrier from simpler primitives.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters for each phase, protected by a single lock.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        # Semaphores to block threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes the calling thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread to arrive: release all waiting threads for this phase.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use.
        self.threads_sem1.acquire() # All threads will block here until the last one arrives.

    def phase2(self):
        """Second phase to ensure no thread races ahead to the next cycle."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread to re-enter: release all threads for the second phase.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use.
        self.threads_sem2.acquire()

class Device(object):
    """Represents a single device in the network."""
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals that script assignment has begun/ended.
        self.scripts = []
        self.my_lock = Lock() # A general-purpose lock for this device instance.
        self.barrier = ReusableBarrierSem(0) # Placeholder for the shared barrier.
        self.timepoint_done = Event() # Signals that all scripts for a timepoint are assigned.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Distributes a shared barrier instance to all devices."""
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        # This implementation assumes device 0 will set up the barrier for all others.
        self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """Assigns a script. A `None` script signals the end of assignment."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Sentinel script: signal that all scripts are received for this timepoint.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location. Does not lock here."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates the sensor data for a given location. Does not lock here."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main device thread to shut down."""
        self.thread.join()

class MyScriptThread(Thread):
    """A worker thread to execute a single script."""
    def __init__(self, script, location, device, neighbours):
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """Aggregates data, runs the script, and propagates the result under lock."""
        script_data = []

        # --- Data Aggregation (No Locking) ---
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = self.script.run(script_data)

            # --- Data Propagation (Explicit Locking) ---
            # The worker thread explicitly locks each device it needs to modify.
            for device in self.neighbours:
                with device.my_lock:
                    device.set_data(self.location, result)
            
            with self.device.my_lock:
                self.device.set_data(self.location, result)

class DeviceThread(Thread):
    """The main control thread for a device."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main lifecycle loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # End of simulation.

            # Wait for all devices to reach the start of the timepoint.
            self.device.barrier.wait()
            # Wait for the supervisor to signal that scripts have been assigned.
            self.device.script_received.wait()
            
            script_threads = []
            # Spawn a new thread for each assigned script.
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script, location, self.device, neighbours))
            
            # Start and then join all worker threads, waiting for them to complete.
            for thread in script_threads:
                thread.start()
            for thread in script_threads:
                thread.join()
            
            # Clear the script list for the next round.
            self.device.scripts = []

            # Wait for the timepoint_done signal (may be redundant as it's set with script_received).
            self.device.timepoint_done.wait()
            # Wait for all devices to finish their work for this timepoint.
            self.device.barrier.wait()
            # Clear events for the next cycle.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
