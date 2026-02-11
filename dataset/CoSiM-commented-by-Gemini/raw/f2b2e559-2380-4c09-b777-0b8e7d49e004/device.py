"""
This module implements a device simulation featuring an internal thread pool
and a dynamic, counter-based work distribution mechanism.

Each device manages a pool of 8 worker threads. Synchronization is handled by
a two-level barrier system: a per-device barrier for its internal workers and a
global barrier for all workers across the entire simulation.
"""

import cond_barrier
from threading import Event, Thread, Lock


class Device(object):
    """
    Represents a device that manages an internal pool of worker threads.

    It coordinates with a leader device (ID 0) to set up a global barrier and
    a shared dictionary of location-based locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.threads = []

        self.neighbourhood = None
        self.map_locks = {}
        self.threads_barrier = None
        self.barrier = None
        # A shared counter for the internal worker threads to pull tasks.
        self.counter = 0
        # A lock to protect the shared counter.
        self.threads_lock = Lock()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes per-device and global synchronization primitives.

        The leader device (ID 0) creates a global barrier and a shared lock map.
        Each device, including the leader, creates its own internal barrier and
        a pool of 8 worker threads.
        """
        if self.device_id == 0:
            num_threads = len(devices)
            
            # The global barrier's size is the total number of worker threads.
            self.barrier = cond_barrier.ReusableBarrier(num_threads * 8)

            for device in devices:
                device.barrier = self.barrier
                device.map_locks = self.map_locks

        
        # Each device has its own barrier for its internal pool of threads.
        self.threads_barrier = cond_barrier.ReusableBarrier(8)
        for i in range(8):
            self.threads.append(DeviceThread(self, i, self.threads_barrier))

        
        for thread in self.threads:
            thread.start()


    def assign_script(self, script, location):
        """
        Assigns a script and lazily initializes a shared lock for its location.
        """

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

        
        if location not in self.map_locks:
            self.map_locks[location] = Lock()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all of the device's worker threads to terminate."""

        
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    An internal worker thread for a single device.

    Uses a shared counter to dynamically pull tasks from the device's script list.
    """

    def __init__(self, device, id, barrier):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


        self.id = id
        self.thread_barrier = barrier

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            
            # The first worker thread (id 0) acts as the leader for this device,
            # fetching the neighbor list for all other workers in its pool.
            if self.id == 0:
                
                self.device.neighbourhood = self.device.supervisor.get_neighbours()

            
            # 1. Internal Barrier: Synchronize all 8 workers within this device.
            self.thread_barrier.wait()

            if self.device.neighbourhood is None:
                break 

            
            # 2. Wait for the signal to start processing scripts for the timepoint.
            self.device.timepoint_done.wait()

            
            # This loop implements a dynamic work queue. Each worker repeatedly
            # tries to grab the next available script from the shared list.
            while True:
                
                # Use a lock to safely access the shared script counter.
                with self.device.threads_lock:
                    if self.device.counter == len(self.device.scripts):
                        break
                    (script, location) = self.device.scripts[self.device.counter]
                    self.device.counter = self.device.counter + 1
                


                # Process the grabbed script.
                self.device.map_locks[location].acquire()
                script_data = []

                for device in self.device.neighbourhood:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    for device in self.device.neighbourhood:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)



                self.device.map_locks[location].release()

            
            # 3. Global Barrier: Wait for ALL threads from ALL devices to finish.
            self.device.barrier.wait()
            # The leader thread resets the shared counter and event for the next cycle.
            if self.id == 0:
                
                
                self.device.counter = 0
                self.device.timepoint_done.clear()
