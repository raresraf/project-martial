"""
@file device.py
@brief Defines a device model with a flawed, two-phase "compute-then-write" mechanism.

This file implements a simulation device that uses a pool of `MyThread` workers
to execute scripts in batches. It attempts a two-phase update protocol,
separating script computation from result write-back using two barrier waits.

@warning The implementation has severe synchronization flaws. The initial data
         gathering is not locked, and the locking used during the write-back
         phase is ineffective as each device uses its own local lock, failing
         to prevent race conditions between devices.
"""

from threading import *
from barrier import *
from my_thread import * # Assumed to be in a separate file, but defined below.

class MyThread(Thread):
    """
    A worker thread that executes a batch of scripts serially.

    It runs a list of tasks, computes their results, and stores them internally
    for the main device thread to process later.
    """

    def __init__(self, id):
        Thread.__init__(self, name="Device Thread %s" % id)
        self.tasks = []
        self.results = []

    def add_task(self, script, location, script_data):
        """Adds a script execution task to this thread's batch."""
        self.tasks.append((script, location, script_data))

    def clear(self):
        """Clears the tasks and results from the previous run."""
        self.tasks = []
        self.results = []

    def run(self):
        """Executes all tasks in the batch and stores the results."""
        for (script, location, script_data) in self.tasks:
            self.results.append((script, location, script.run(script_data)))


class Device(object):
    """
    Represents a single device in the simulation.
    """
    # A class-level barrier shared by all device instances.
    barrier = ReusableBarrier(0)

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # Each device has its own independent lock, which is ineffective for
        # synchronizing with other devices.
        self.lock = Lock()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes the shared barrier for all devices."""
        if self.device_id == 0:
            Device.barrier = ReusableBarrier(len(devices))

    def assign_script(self, script, location):
        """Assigns a script to be executed."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that script assignment is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. This access is not synchronized."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This access is not synchronized by this method."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating the flawed two-phase update.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = []

    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to finish assigning scripts.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # --- Start Compute Phase ---
            self.threads = []
            for i in xrange(8):
                self.threads.append(MyThread("{}-{}".format(self.device.device_id, i)))
            
            # Block Logic: Distribute scripts to the 8 worker threads in a round-robin fashion.
            i = 0
            for (script, location) in self.device.scripts:
                script_data = []
                # Gather data from neighbors and self (UNSYNCHRONIZED).
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                         script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                if script_data != []:
                    self.threads[i].add_task(script, location, script_data)
                    i = (i+1) % 8

            # Start and join worker threads.
            for i in xrange(8):
                if self.threads[i].tasks != []:
                    self.threads[i].start()

            for i in xrange(8):
                if self.threads[i].tasks != []:
                    self.threads[i].join()
            
            # --- First Barrier ---
            # All devices wait here, ensuring computation is complete everywhere.
            Device.barrier.wait()

            # --- Start Write Phase ---
            # Block Logic: Write back results. The locking here is incorrect as each
            # device uses its own lock, providing no protection against other devices.
            for i in xrange(8):
                if self.threads[i].results != []:
                    for (script, location, result) in self.threads[i].results:
                        for device in neighbours:
                            device.lock.acquire()
                            device.set_data(location, result)
                            device.lock.release()
                        
                        self.device.lock.acquire()
                        self.device.set_data(location, result)
                        self.device.lock.release()

            # --- Second Barrier ---
            # Wait for all devices to finish the flawed write-back phase.
            Device.barrier.wait()