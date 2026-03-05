"""
This module implements a device simulation with a flawed concurrency model.

It defines a `Device` class and a `DeviceThread` that spawns `ScriptRunner`
threads to execute tasks. This version attempts to manage shared locks and
barriers, and to run worker threads in batches.

NOTE: This script contains several severe bugs:
1. The barrier and lock setup in the `Device` class is racy and not guaranteed
   to work correctly under concurrent initialization.
2. The `get_data` and `set_data` methods are not thread-safe.
3. The main loop in `DeviceThread` has a fundamental design flaw. It iterates
   through scripts, but on each iteration, it attempts to manage the execution
   of *all* previously created threads for the current time step. This leads to
   incorrect serialization and re-joining of already finished threads.
4. The batching logic (`n % 7`) is likely a typo for `n % 8`.
"""

from threading import Event, Thread, Lock
# Assumes the presence of a 'barrier.py' file with a ReusableBarrierSem class.
import barrier
# Assumes the presence of a 'runner.py' file with a ScriptRunner class.
# NOTE: ScriptRunner is also defined in this file, which may cause import ambiguity.
# For clarity, this documentation will refer to the locally defined ScriptRunner.

class Device(object):
    """Represents a device in the simulation, managing data, scripts, and threads."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barr = None # The shared barrier object.
        self.devices = []
        self.runners = [] # A list to hold worker threads for the current step.
        # A list of locks, indexed by location.
        self.locks = [None] * 50
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources.

        BUG: The check `if self.barr is None` is not atomic and can lead to
        race conditions if multiple devices call this method concurrently.
        """
        if self.barr is None:
            barr = barrier.ReusableBarrierSem(len(devices))
            self.barr = barr
            for dev in devices:
                if dev.barr is None:
                    dev.barr = barr
        
        # This just copies the devices list locally.
        for dev in devices:
            if dev is not None:
                self.devices.append(dev)

    def assign_script(self, script, location):
        """
        Assigns a script and attempts to set up a shared lock for its location.

        BUG: The lock setup is racy and incorrect. It iterates through other
        devices to "find" a lock. If multiple devices do this at the same time
        for a new location, they can fail to find a lock and each create their
        own, providing no mutual exclusion between devices.
        """
        if script is not None:
            self.scripts.append((script, location))
            
            # Flawed attempt to find or create a shared lock.
            lock_found = False
            if self.locks[location] is None:
                for device in self.devices:
                    if device.locks[location] is not None:
                        self.locks[location] = device.locks[location]
                        lock_found = True
                        break
                if not lock_found:
                    self.locks[location] = Lock()
            self.script_received.set()
        else:
            # Signal that all scripts for the time step have been received.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data. Not thread-safe.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates sensor data. Not thread-safe.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device.

    BUG: The run loop contains a major logical flaw that incorrectly mixes
    the execution and locking of scripts for different locations.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal.
            
            # Wait for all scripts for the current step to be assigned.
            self.device.timepoint_done.wait()

            # --- BROKEN LOGIC BLOCK ---
            # The following loop is fundamentally flawed. It iterates through scripts,
            # but on each pass, it re-evaluates the entire `self.device.runners` list,
            # which grows on each iteration. This leads to re-joining finished threads
            # and incorrect serialization.
            for (script, location) in self.device.scripts:
                # A new runner is created for each script.
                run = ScriptRunner(self.device, script, location, neighbours)
                self.device.runners.append(run)

                n = len(self.device.runners)
                x = n / 8
                # This is likely a typo and should be n % 8.
                r = n % 7
                
                # This lock serializes the execution management for different locations,
                # which should be parallel.
                self.device.locks[location].acquire()
                
                # The code attempts to start and join threads in batches. However, since
                # this happens inside the script loop, it re-processes already-joined
                # threads from previous iterations of this same time step.
                for i in xrange(0, x):
                    for j in xrange(0, 8):
                        self.device.runners[i * 8 + j].start()
                
                if n >= 8:
                    for i in xrange(len(self.device.runners) - r, len(self.device.runners)):
                        self.device.runners[i].start()
                else:
                    for i in xrange(0, n):
                        self.device.runners[i].start()
                
                # This join block will try to join threads that may have already
                # been joined in a previous iteration of this outer loop.
                for i in xrange(0, n):
                    self.device.runners[i].join()
                
                self.device.locks[location].release()
                
                # Resetting the runners list here means only the last script's
                # runner is actually guaranteed to run and be joined properly.
                self.device.runners = []

            self.device.timepoint_done.clear()
            self.device.barr.wait() # Wait at the global barrier.


class ScriptRunner(Thread):
    """A worker thread to execute a single script."""

    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """Gathers data, runs the script, and propagates results."""
        script_data = []
        
        # Unsafe reads from neighbors and self.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = self.script.run(script_data)
            
            # Unsafe writes to neighbors and self.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
