
"""
@brief A non-functional device simulation with multiple severe design flaws.
@file device.py

This module attempts to implement a distributed device simulation. It uses static
(class-level) variables for a shared barrier and locks, and uses a highly
inefficient threading model where a new thread is created for each individual
script in every time step.

WARNING: SEVERE ARCHITECTURAL FLAWS - THIS CODE IS NON-FUNCTIONAL.
1.  **Buggy Barrier**: The `ReusableBarrierCond` is a textbook example of an
    incorrect barrier implementation using a Condition variable. It is not safe
    and is prone to race conditions and deadlocks.
2.  **Race-Condition in Setup**: The `setup_devices` method initializes the shared
    `barrier` and `unique` lock list using non-atomic checks (`if ... == None`),
    which will fail if multiple threads call it concurrently.
3.  **Extremely Inefficient Threading**: In `DeviceThread`, the code creates a
    new thread for every single script, every time step. This introduces
    massive performance overhead and defeats the purpose of threading for this task.
4.  **Unsafe Locking**: `ScriptThread` uses manual `acquire()` and `release()`
    calls, which is unsafe. If an exception occurred between these calls, the
    lock would never be released, causing a deadlock.
"""

from threading import Condition, Lock, Event, Thread

class ReusableBarrierCond(object):
    """
    A buggy and fundamentally flawed implementation of a reusable barrier.
    
    WARNING: This implementation is not safe. It is susceptible to race
    conditions (e.g., lost wakeups) and is a classic example of how NOT to
    implement a barrier with Condition variables.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Causes a thread to block until all threads have called wait."""
        self.cond.acquire()     
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread arrives: notify all others and reset the counter.
            self.cond.notify_all()      
            self.count_threads = self.num_threads
        else:
            # Not the last thread: wait to be notified.
            self.cond.wait()    
        self.cond.release()     

class Device(object):
    """
    Represents a device node, using flawed static variables for shared state.
    """
    # Class-level (static) variables shared across ALL device instances.
    barrier = None
    unique = [] # Globally shared list of locks.

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared state. This method is NOT thread-safe.
        
        WARNING: The checks `if Device.barrier == None` and `if len(...)`
        create a race condition. If multiple threads execute this concurrently,
        the shared state will be corrupted.
        """
        if Device.barrier is None:
            Device.barrier = ReusableBarrierCond(len(devices))

        # This check is also not atomic.
        if len(Device.unique) != self.supervisor.supervisor.testcase.num_locations:
            for _ in range(self.supervisor.supervisor.testcase.num_locations):
                Device.unique.append(Lock())

    def assign_script(self, script, location):
        """Adds a script to the device's workload for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()

class ScriptThread(Thread):
    """A short-lived worker thread that executes a single script."""
    def __init__(self, device, scripts, neighbours):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts # This will be a list with one (script, loc) tuple.
        self.neighbours = neighbours

    def run(self):
        """Executes the single script assigned to this thread."""
        for (script, location) in self.scripts:
            # Unsafe manual lock management. A `with` statement should be used.
            Device.unique[location].acquire()

            script_data = []
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            # Gather data from self.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # If data exists, execute the script and update values.
            if script_data:
                result = script.run(script_data)
                for device in self.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)

            Device.unique[location].release()

class DeviceThread(Thread):
    """The main control thread for a device."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for supervisor to assign all scripts for this time step.
            self.device.timepoint_done.wait()

            # --- INEFFICIENCY WARNING ---
            # The following logic creates one new thread per script, every time step.
            # This is extremely inefficient.
            threads = []
            # This complex logic just splits the scripts into a list of single-item lists.
            divide_scripts = [[script] for script in self.device.scripts]

            for script_list in divide_scripts:
                threads.append(ScriptThread(self.device, script_list, neighbours))

            # Start and wait for all of this step's worker threads to complete.
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            Device.barrier.wait() 
            self.device.timepoint_done.clear()
