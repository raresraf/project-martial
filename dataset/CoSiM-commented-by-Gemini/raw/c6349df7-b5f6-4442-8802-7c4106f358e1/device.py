"""
@file raw/c6349df7-b5f6-4442-8802-7c4106f358e1/device.py
@brief Implements a distributed device simulation with a flawed synchronization model.

This module defines a "process-then-synchronize" simulation. Each device
processes all assigned scripts for a time step in parallel worker threads.
After all local work is complete, all devices synchronize at a global barrier
before proceeding to the next time step.

Concurrency is managed via two locking strategies:
1. A global dictionary of locks, shared among all devices, ensures that only one
   worker thread across the entire system can operate on a specific data
   "location" at a time.
2. A per-device lock (`my_lock`) protects each device's internal data during
   read/write operations by other devices' workers.

@warning The `ReusableBarrier` class is not actually reusable. It is a classic
         example of a single-phase barrier that is subject to race conditions
         where fast threads can loop around and re-enter the barrier before slow
         threads have left, breaking the synchronization.
"""

from threading import Event, Thread, Condition, Lock

class Device(object):
    """
    Represents a node in the distributed network. It manages its own sensor
    data, assigned scripts, and participates in global synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.scripts_done = Event() # Signals that script assignment for a step is complete.
        self.my_lock = Lock() # Protects this device's internal data structures.
        
        # Shared state, initialized in `setup_devices`.
        self.locations = None
        self.barrier = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared synchronization objects (locks and barrier).
        This method relies on a "master" device (id=0) to create the objects
        and then shares references to them with all other devices.
        """
        if self.device_id is 0:
            self.locations = {} # The globally shared dictionary of location-to-lock mappings.
            self.barrier = ReusableBarrier(len(devices));
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass
                else:
                    self.locations[loc] = Lock()
        
        else:
            # All other devices get a reference to the master's objects.
            self.locations = devices[0].locations
            self.barrier = devices[0].get_barrier()
            # This part seems to have a logic error, it creates new locks in the shared
            # dictionary, which might be unintentional.
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass
                else:
                    self.locations[loc] = Lock()

        # Each device has its own master thread.
        self.thread = DeviceThread(self, self.barrier, self.locations)
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be run for a specific location."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A `None` script is a sentinel that signals all scripts for the step are assigned.
            self.scripts_done.set()

    def get_data(self, location):
        """Returns the sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates the sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device thread."""
        self.thread.join()

    def get_barrier(self):
        """Returns a reference to the shared barrier object."""
        return self.barrier

class DeviceThread(Thread):
    """The main control thread for a single Device."""

    def __init__(self, device, barrier, locations):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.locations = locations

    def run(self):
        """
        Main simulation loop for the device.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # 1. Wait until all scripts for the current time step have been assigned.
            self.device.scripts_done.wait()
            self.device.scripts_done.clear()
            
            # 2. Create and start a new worker thread for each assigned script.
            workers = []
            for (script, location) in self.device.scripts:
                w = Worker(self.device, neighbours, script, location, self.locations)
                workers.append(w)
                w.start()

            # 3. Wait for all local worker threads to complete their processing.
            for w in workers:
                w.join()

            # 4. Synchronize with all other devices at the global barrier.
            self.barrier.wait()

class Worker(Thread):
    """
    A short-lived thread that executes a single script on data from a single
    location.
    """
    def __init__(self, device, neighbours, script, location, locations):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.locations = locations

    def run(self):
        """Executes the script logic with two levels of locking."""
        # --- Global Lock ---
        # Acquire the lock for this specific location to ensure no other worker
        # (on this or any other device) can process this location concurrently.
        self.locations[self.location].acquire()
        
        script_data = []
        
        # Aggregate data from neighbors, using a per-device lock for each read.
        for device in self.neighbours:
            device.my_lock.acquire()
            data = device.get_data(self.location)
            device.my_lock.release()
            if data is not None:
                script_data.append(data)
        
        # Aggregate data from the local device.
        self.device.my_lock.acquire()
        data = self.device.get_data(self.location)
        self.device.my_lock.release()
        if data is not None:
            script_data.append(data)

        if script_data:
            # Run the script on the aggregated data.
            result = self.script.run(script_data)

            # Broadcast the result back to neighbors, locking each device.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            # Update the local device's data.
            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()
            
        # Release the global location lock.
        self.locations[self.location].release()

class ReusableBarrier():
    """
    A simple barrier implementation using a Condition variable.
    @warning This is NOT a reusable barrier. It is prone to race conditions
             where fast threads can enter the next wait cycle before all slow
             threads have exited the previous one, breaking synchronization.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
 
    def wait(self):
        """Blocks until all `num_threads` have called this method."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread arrives, wakes up all waiting threads.
            self.cond.notify_all()
            # Resets counter for the next (broken) cycle.
            self.count_threads = self.num_threads
        else:
            # Wait to be notified by the last thread.
            self.cond.wait()
        self.cond.release()
