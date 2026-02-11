"""
This module defines a highly complex, three-level threading model for a
simulated device network. It combines a global barrier, a master thread per
device, and a worker pool within each master.

NOTE: This implementation is exceptionally complex and contains several flaws.
- The `ReusableBarrierCond` is a textbook broken barrier with race conditions.
- The `setup_devices` method uses an inefficient busy-wait loop.
- The internal state machine and signaling between the master and worker threads
  using a mix of Semaphores, Events, and shared flags is fragile and hard to reason about.
"""

from threading import Event, Thread, Lock, Semaphore, Condition

class ReusableBarrierCond(object):
    """
    A custom, but flawed, reusable barrier using a Condition variable.
    WARNING: This implementation is susceptible to race conditions and may deadlock.
    A correct implementation requires two distinct phases/turns.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """Represents a single device in the network."""
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.loc_lock = [] # Fine-grained locks for data locations.
        self.crt_nb_scripts = 0
        self.crt_script = 0
        # --- Synchronization Primitives ---
        self.script_sem = Semaphore(value=1) # Used to protect the script counter.
        self.done_processing = Semaphore(value=0) # Signals from workers to master.
        self.timepoint_done = Event() # Signal from supervisor to workers.
        self.wait_for_next_timepoint = Event() # Signal from master to supervisor.
        self.wait_for_next_timepoint.set()
        # Each device contains and starts its own master thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a global barrier and location locks.
        This centralized setup is performed by device 0.
        """
        my_barrier = ReusableBarrierCond(len(devices))
        if self.device_id == 0:
            for i in range(len(devices)):
                devices[i].barr = my_barrier
            # Create and distribute location locks.
            self.loc_lock = [Lock() for _ in range(100)]
            for dev in devices:
                if dev.device_id != 0:
                    dev.loc_lock = self.loc_lock
        else:
            # Inefficient busy-wait for setup to complete.
            while not hasattr(self, 'barr'):
                continue

    def assign_script(self, script, location):
        """Assigns a script; called by the supervisor."""
        self.wait_for_next_timepoint.wait() # Blocks supervisor until device is ready.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
            self.crt_nb_scripts += 1
        else:
            # All scripts assigned; unblock workers for this time step.
            self.wait_for_next_timepoint.clear()
            self.timepoint_done.set()
    
    # ... (get_data, set_data, shutdown) ...
    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class MyWorker(Thread):
    """A worker thread that executes scripts for a single device."""
    def __init__(self, device_thread, worker_barrier, my_id):
        Thread.__init__(self, name="Worker Thread %d" % my_id)
        self.my_dev = device_thread
        self.worker_bar = worker_barrier
        self.my_id = my_id

    def run(self):
        """Main worker loop."""
        while True:
            # Wait for supervisor to signal that scripts are ready for processing.
            self.my_dev.device.timepoint_done.wait()
            neighbours = self.my_dev.neighbours
            if neighbours is None:
                break # End of simulation.
            
            # --- Complex State Machine to signal completion to Master ---
            if self.my_dev.inner_state == 1:
                self.worker_bar.wait() # Barrier 1 for workers.
                if self.my_id == 0: # Leader worker signals master.
                    self.my_dev.device.timepoint_done.clear()
                    self.my_dev.device.done_processing.release() # Unblock master.
                    self.my_dev.inner_state = 0
                self.worker_bar.wait() # Barrier 2 for workers.
                continue

            # --- Task Pulling Loop ---
            while True:
                # Use a semaphore as a lock for the shared script counter.
                self.my_dev.device.script_sem.acquire()
                if self.my_dev.device.crt_script == self.my_dev.device.crt_nb_scripts:
                    self.my_dev.device.script_sem.release()
                    if self.my_id == 0:
                        self.my_dev.inner_state = 1 # Mark local work as done.
                    break # No more scripts for this time step.
                
                my_script = self.my_dev.device.scripts[self.my_dev.device.crt_script]
                self.my_dev.device.crt_script += 1
                self.my_dev.device.script_sem.release()
                
                # --- Script Execution ---
                self.my_dev.device.loc_lock[my_script[1]].acquire()
                try:
                    # Gather data, run script, propagate results...
                    script_data = [d for d in [dev.get_data(my_script[1]) for dev in neighbours] if d is not None]
                    local_data = self.my_dev.device.get_data(my_script[1])
                    if local_data is not None:
                        script_data.append(local_data)
                    
                    if script_data:
                        result = my_script[0].run(script_data)
                        for device in neighbours:
                            device.set_data(my_script[1], result)
                        self.my_dev.device.set_data(my_script[1], result)
                finally:
                    self.my_dev.device.loc_lock[my_script[1]].release()


class DeviceThread(Thread):
    """The master thread within a device, coordinating its worker pool."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.inner_state = 0 # State for communicating with workers.
        self.thread_p = []
        self.w_bar = ReusableBarrierCond(8) # An internal barrier for workers.
        self.neighbours = []
        
        # Create and start the worker pool.
        for i in range(8):
            self.thread_p.append(MyWorker(self, self.w_bar, i))
            self.thread_p[i].start()

    def run(self):
        """Main master loop."""
        while True:
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                self.device.timepoint_done.set() # Wake workers for termination.
                break
            
            # Block until workers signal completion of the *previous* step.
            self.device.done_processing.acquire()
            
            # Reset state for the new time step.
            self.device.crt_script = 0
            
            # Signal to supervisor that this device is ready for new scripts.
            self.device.wait_for_next_timepoint.set()
            
            # Wait at the global barrier for all other devices to be ready.
            self.device.barr.wait()

        # Join all worker threads upon simulation end.
        for i in range(8):
            self.thread_p[i].join()
