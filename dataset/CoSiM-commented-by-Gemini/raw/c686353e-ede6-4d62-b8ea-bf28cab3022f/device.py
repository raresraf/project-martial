"""
@file raw/c686353e-ede6-4d62-b8ea-bf28cab3022f/device.py
@brief Implements a distributed device simulation using a fixed worker pool
       and a "process-then-synchronize" model.

This module simulates a network of devices that process sensor data. Each device
maintains a fixed-size pool of worker threads. For each time step, scripts are
distributed among these workers in a round-robin fashion.

The simulation follows a "process-then-synchronize" pattern:
1. All devices wait for their scripts for the current time step to be assigned.
2. Each device processes its scripts using its local worker pool.
3. After all local work is complete, all devices synchronize at a global barrier
   before beginning the next time step.

Concurrency is managed by a globally shared dictionary that maps each data
"location" to a unique `Lock`. This ensures that only one worker thread in the
entire system can process a given location at any time.

@warning This implementation has potential race conditions:
         1. The `assign_script` method that dynamically creates locks in the
            shared dictionary is not atomic. Two threads could simultaneously
            check for a location, find it absent, and attempt to create a lock.
         2. The `Worker` thread reads from neighbor devices (`get_data`) without
            any per-device lock, which could be unsafe if dictionary access
            were not protected by the GIL in CPython.
"""

from threading import Event, Thread, Lock
# This module is external and not provided, but is assumed to contain a
# correct reusable barrier implementation.
import cond_barrier


class Device(object):
    """
    Represents a device node in the simulation. It manages sensor data,
    scripts, and participates in global synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = None # Initialized in setup_devices
        # These attributes are shared across all devices after setup.
        self.dict_location = {}
        self.barrier = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes the shared synchronization objects.
        A "master" device (id=0) creates the barrier and the shared lock
        dictionary, and all other devices receive a reference to them.
        """
        if self.device_id == 0:
            num_threads = len(devices)
            # Barrier is from the external `cond_barrier` module.
            self.barrier = cond_barrier.ReusableBarrierCond(num_threads)
            for device in devices:
                device.barrier = self.barrier
                device.dict_location = self.dict_location
        
        # The main thread for the device is started after setup.
        if not self.thread:
            self.thread = DeviceThread(self)
            self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to a location. Dynamically creates a lock for the
        location if it's the first time it's seen.
        """
        # Potential race condition here if two threads call this for the same
        # new location at the same time.
        if location not in self.dict_location:
            self.dict_location[location] = Lock()

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script is the sentinel for end of assignments.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        if self.thread:
            self.thread.join()


class Worker(Thread):
    """
    A stateful worker thread that executes a list of assigned scripts.
    """

    def __init__(self, worker_id, neighbours, device, dict_location):
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.worker_id = worker_id
        self.neighbours = neighbours
        self.device = device
        self.dict_location = dict_location # Shared dictionary of location locks.
        self.scripts = []
        self.location = []

    def addwork(self, script, location):
        """Adds a script and its location to this worker's task list."""
        self.scripts.append(script)
        self.location.append(location)

    def run(self):
        """
        Processes each assigned script sequentially, acquiring the appropriate
        location-specific lock for each one.
        """
        i = 0
        for script in self.scripts:
            # Acquire the global lock for this location to ensure mutual exclusion.
            self.dict_location[self.location[i]].acquire()
            
            script_data = []
            
            # Aggregate data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location[i])
                if data is not None:
                    script_data.append(data)
            
            # Aggregate data from the local device.
            data = self.device.get_data(self.location[i])
            if data is not None:
                script_data.append(data)

            if script_data:
                # Execute the script.
                result = script.run(script_data)

                # Broadcast the result to all neighbors and the local device.
                for device in self.neighbours:
                    device.set_data(self.location[i], result)
                self.device.set_data(self.location[i], result)
            
            # Release the global lock for this location.
            self.dict_location[self.location[i]].release()
            i += 1

class DeviceThread(Thread):
    """The main control thread for a single Device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # 1. Wait for signal that all scripts for this step are assigned.
            self.device.timepoint_done.wait()

            # 2. Create a fixed pool of workers for this time step.
            num_threads = 8
            workers = []
            for i in range(num_threads):
                lock_loc = self.device.dict_location
                workers.append(Worker(i, neighbours, self.device, lock_loc))

            # 3. Distribute scripts to workers in a round-robin fashion.
            nr_worker = 0
            for (script, location) in self.device.scripts:
                workers[nr_worker].addwork(script, location)
                nr_worker = (nr_worker + 1) % num_threads

            # 4. Start all workers and wait for their completion.
            for i in range(num_threads):
                workers[i].start()
            for i in range(num_threads):
                workers[i].join()

            self.device.timepoint_done.clear()
            
            # 5. Wait at the global barrier for all other devices.
            self.device.barrier.wait()
