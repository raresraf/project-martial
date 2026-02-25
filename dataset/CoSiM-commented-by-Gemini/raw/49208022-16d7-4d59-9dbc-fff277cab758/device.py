"""
Models a device in a distributed simulation using a manual dispatcher
and per-thread work queues.

This module defines a simulation where each device has a pool of worker threads.
A main device object dispatches incoming scripts to individual worker threads,
each of which has its own private work queue. A global barrier synchronizes all
worker threads across all devices at the end of a time step.

Classes:
    Device: Manages the device state and dispatches scripts to its worker threads.
    DeviceThread: A worker thread with its own script queue that executes tasks.
"""

from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierCond
from Queue import Queue

THREAD_NR = 8

class Device(object):
    """
    Represents a device and its pool of worker threads.

    This class acts as a dispatcher, assigning incoming scripts to available
    worker threads from its pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.setup_finished = Event()
        self.dataLock = Lock() # A coarse-grained lock for all sensor data access.
        self.shared_lock = Lock()
        
        # A queue holding available worker thread objects.
        self.thread_queue = Queue(0)
        
        # A barrier to synchronize neighbor-fetching within this device's thread pool.
        self.wait_get_neighbours = ReusableBarrierCond(THREAD_NR)
        self.thread_pool = []
        self.neighbours = []

        # Create and start the pool of worker threads.
        for i in range(0, THREAD_NR):
            thread = DeviceThread(self, i)
            self.thread_pool.append(thread)
            self.thread_queue.put(thread) # Add the thread to the pool of available workers.
            thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs centralized setup of shared resources (barrier and location locks).
        """
        if self.device_id == 0:
            # A single barrier is shared by all worker threads of all devices.
            self.reusable_barrier = ReusableBarrierCond(len(devices) * THREAD_NR)
            self.location_locks = {}
            for device in devices:
                if device.device_id != self.device_id:
                    device.set_location_locks(self.location_locks)
                    device.set_barrier(self.reusable_barrier)

            self.setup_finished.set()

    def set_barrier(self, reusable_barrier):
        """Sets the shared barrier for this device."""
        self.reusable_barrier = reusable_barrier
        self.setup_finished.set()

    def set_location_locks(self, location_locks):
        """Sets the shared location locks for this device."""
        self.location_locks = location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to an available worker thread.

        NOTE: The logic here is flawed. When `script` is `None`, it re-dispatches
        all previously seen scripts before sending the termination signal.
        """
        if script is not None:
            self.scripts.append((script, location))
            if location not in self.location_locks:
                self.location_locks[location] = Lock()

            # Get an available worker from the pool and give it the script.
            thread = self.thread_queue.get()
            thread.give_script(script, location)

            
        else:
            # This block appears to be a bug, re-queueing all scripts.
            for (s, l) in self.scripts:
                thread = self.thread_queue.get()
                thread.give_script(s, l)

            # Send a termination signal (poison pill) to each worker.
            for thread in self.thread_pool:
                thread.give_script(None, None)


    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its worker threads."""
        for i in range(THREAD_NR):
            self.thread_pool[i].join()


class DeviceThread(Thread):
    """A worker thread that has its own queue to receive scripts."""

    def __init__(self, device, ID):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id = ID
        self.script_queue = Queue(0) # Each thread has a private work queue.

    def give_script(self, script, location):
        """Adds a script to this thread's personal work queue."""
        self.script_queue.put((script, location))

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            
            self.device.setup_finished.wait()

            # Use a barrier to have only one thread (id 0) fetch neighbors.
            if self.id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            self.device.wait_get_neighbours.wait()

            if self.device.neighbours is None:
                break # Shutdown signal.

            # This inner loop processes all scripts assigned for the current timepoint.
            while True:
                (script, location) = self.script_queue.get()

                if script is None:
                    break # End of timepoint signal (poison pill).

                self.device.location_locks[location].acquire()

                script_data = []
                
                # Gather data, using a coarse-grained lock on each device.
                for device in self.device.neighbours:
                    device.dataLock.acquire()
                    data = device.get_data(location)
                    device.dataLock.release()

                    if data is not None:
                        script_data.append(data)
                
                self.device.dataLock.acquire()
                data = self.device.get_data(location)
                self.device.dataLock.release()
                
                if data is not None:
                   script_data.append(data)

                self.device.location_locks[location].release()

                # Execute script and propagate results, using mixed locking.
                if script_data != []:
                    
                    result = script.run(script_data)
                    
                    self.device.location_locks[location].acquire()

                    for device in self.device.neighbours:
                        device.dataLock.acquire()
                        device.set_data(location, result)
                        device.dataLock.release()
                    
                    self.device.dataLock.acquire()
                    self.device.set_data(location, result)
                    self.device.dataLock.release()
                    self.device.location_locks[location].release()
               
                # Return self to the pool of available workers.
                self.device.thread_queue.put(self)

            # Wait at the global barrier for all other workers in the simulation to finish.
            self.device.reusable_barrier.wait()