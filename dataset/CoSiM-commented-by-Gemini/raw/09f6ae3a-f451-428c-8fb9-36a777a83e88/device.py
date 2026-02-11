"""
This module defines a simulated distributed device network using a complex
producer-consumer model with a persistent thread pool within each device.
Synchronization is handled by a mix of local and global barriers, semaphores,
and events, making the logic highly intricate and difficult to reason about.
"""

from threading import Event, Thread, Semaphore, RLock
# Assumes 'barrier' is a custom module providing a ReusableBarrier class.
from barrier import Barrier


class Device(object):
    """
    Represents a device that manages a local pool of worker threads that
    consume tasks from a shared queue.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # --- State and Task Queues ---
        self.scripts = [] # Persistent list of scripts for the current step.
        self.scripts_queue = [] # The active work queue for worker threads.
        self.threads = []
        self.cores_no = 8
        self.neighbours = []

        # --- Synchronization Primitives ---
        self.timepoint_done = Event() # Supervisor signals this when script assignment is done.
        self.queue_lock = RLock() # Lock for accessing the script queues.
        self.location_locks = {} # Fine-grained locks for data locations.
        self.queue_sem = Semaphore(value=0) # Signals that a task is in the queue.
        self.timepoint_barrier = Barrier() # The main global barrier.
        self.neighbours_barrier = Barrier(self.cores_no) # Local barrier for neighbor setup.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects. This logic appears
        convoluted, with each device attempting to set up locks before they are
        overwritten by the master device (device 0).
        """
        # All devices initialize a local dictionary of locks.
        for dev in devices:
            for location in dev.sensor_data.keys():
                if location not in self.location_locks:
                    self.location_locks[location] = RLock()
        
        # Device 0's objects become the single shared instance for all devices.
        self.timepoint_barrier.set_num_threads(len(devices) * self.cores_no)
        self.timepoint_barrier = devices[0].timepoint_barrier
        self.location_locks = devices[0].location_locks
        
        # Each device starts its own pool of persistent worker threads.
        for i in xrange(self.cores_no):
            self.threads.append(DeviceThread(self, i))
            self.threads[i].start()

    def assign_script(self, script, location):
        """
        Adds a script to the queue. Called by the supervisor.
        """
        if script is not None:
            with self.queue_lock:
                self.timepoint_done.clear()
                self.scripts.append((script, location))
                self.scripts_queue.append((script, location))
            self.queue_sem.release() # Signal that a task is ready.
        else:
            # Supervisor signals end of assignments for this time step.
            with self.queue_lock:
                self.timepoint_done.set()
            # Wake up all workers to check the termination condition.
            for _ in xrange(self.cores_no):
                self.queue_sem.release()

    def recreate_queue(self):
        """
        Repopulates the work queue from the persistent script list for the new time step.
        """
        with self.queue_lock:
            for script in self.scripts:
                self.scripts_queue.append(script)
                self.queue_sem.release() # Signal workers for each re-added task.

    # ... (get_data, set_data, shutdown methods) ...
    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        for i in xrange(self.cores_no):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A persistent worker thread within a device's thread pool.
    """
    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device %d Thread %d" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run_script(self, script, location):
        """Helper function to execute a single script under its location lock."""
        with self.device.location_locks[location]:
            # Gather data from self and neighbors.
            script_data = [d for dev in self.device.neighbours if (d := dev.get_data(location)) is not None]
            local_data = self.device.get_data(location)
            if local_data is not None:
                script_data.append(local_data)

            if script_data:
                result = script.run(script_data)
                # Propagate results.
                for dev in self.device.neighbours:
                    dev.set_data(location, result)
                self.device.set_data(location, result)

    def run(self):
        """The main worker loop."""
        while True:
            # Thread 0 is the leader for setup tasks in each time step.
            if self.thread_id == 0:
                self.device.recreate_queue()
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # Local barrier: All threads in this device wait until neighbors are fetched.
            self.device.neighbours_barrier.wait()

            if self.device.neighbours is None:
                break # End of simulation.

            # --- Main Work-Consuming Loop ---
            while True:
                # Wait for a task to be available in the queue.
                self.device.queue_sem.acquire()
                
                # Lock the queue to check state and pop a task.
                self.device.queue_lock.acquire()
                
                # Complex break condition: The supervisor has signaled completion AND the queue is empty.
                if self.device.timepoint_done.is_set() and not self.device.scripts_queue:
                    self.device.queue_lock.release()
                    break # Exit inner loop and proceed to global barrier.
                
                (script, location) = self.device.scripts_queue.pop(0)
                self.device.queue_lock.release()

                self.run_script(script, location)
            
            # --- Global Synchronization ---
            # All threads from all devices wait here to finish the time step.
            self.device.timepoint_barrier.wait()
