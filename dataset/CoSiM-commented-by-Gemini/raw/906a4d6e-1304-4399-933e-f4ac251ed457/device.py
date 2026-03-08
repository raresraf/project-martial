# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices using a 'thread-per-script'
model, with several significant logical and synchronization flaws.
"""

from threading import Event, Thread, Condition, Lock


class Device(object):
    """
    Represents a device in the simulation. It creates a control thread and manages
    shared state like locks and barriers.

    NOTE: The setup logic for shared resources contains race conditions.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.scripts_done = Event()
        self.my_lock = Lock() # A per-device lock that is used ineffectually by workers.

        self.locations = None
        self.barrier = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for all devices.

        NOTE: This method has race conditions. Non-leader devices can attempt to
        modify the shared `locations` dictionary concurrently.
        """
        if self.device_id == 0: # Device 0 is the leader.
            self.locations = {}
            self.barrier = ReusableBarrier(len(devices))
            for loc in self.sensor_data:
                if loc not in self.locations:
                    self.locations[loc] = Lock()
        else: # Non-leader devices.
            self.locations = devices[0].locations
            self.barrier = devices[0].get_barrier()
            # RACE CONDITION: Multiple threads could execute this block at once.
            for loc in self.sensor_data:
                if loc not in self.locations:
                    self.locations[loc] = Lock()

        # The main control thread is only started after setup.
        self.thread = DeviceThread(self, self.barrier, self.locations)
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()

    def get_barrier(self):
        """Returns the shared barrier instance."""
        return self.barrier


class DeviceThread(Thread):
    """The main control thread for a device, spawning workers for each script."""
    def __init__(self, device, barrier, locations):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.locations = locations

    def run(self):
        """Main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to assign all scripts.
            self.device.scripts_done.wait()
            self.device.scripts_done.clear()

            # --- Thread-per-Script Model ---
            # Spawn a new worker thread for every assigned script.
            workers = []
            for (script, location) in self.device.scripts:
                w = Worker(self.device, neighbours, script, location, self.locations)
                workers.append(w)
                w.start()

            # Wait for all workers for this time step to complete.
            for w in workers:
                w.join()

            # Synchronize with all other devices before the next time step.
            self.barrier.wait()


class Worker(Thread):
    """A short-lived worker thread that executes a single script."""
    def __init__(self, device, neighbours, script, location, locations):
        Thread.__init__(self, name="Worker for Device %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.locations = locations

    def run(self):
        """Executes the script logic."""
        # Acquire the global lock for this data location. This is the only
        # lock that provides meaningful, correct synchronization.
        self.locations[self.location].acquire()
        
        script_data = []
        
        # --- Redundant and Ineffectual Locking ---
        # The `my_lock` calls here do not protect the data, as the get_data
        # method itself is not synchronized with this lock.
        for device in self.neighbours:
            device.my_lock.acquire()
            data = device.get_data(self.location)
            device.my_lock.release()
            if data is not None:
                script_data.append(data)

        self.device.my_lock.acquire()
        data = self.device.get_data(self.location)
        self.device.my_lock.release()
        if data is not None:
            script_data.append(data)

        if script_data:
            result = self.script.run(script_data)

            # The same redundant locking is used here for setting data.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()
            
        # Release the global location lock.
        self.locations[self.location].release()


class ReusableBarrier():
    """
    A single-phase barrier implementation using a Condition variable.

    NOTE: This implementation is NOT a correct reusable barrier. It is prone
    to the "lost wakeup" problem and can easily lead to deadlocks in practice.
    A correct implementation requires two distinct phases or stages.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread notifies all waiting threads and resets the counter.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Wait to be notified by the last thread.
            self.cond.wait()
        self.cond.release()
