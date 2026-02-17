
"""
Models a distributed system of devices using a semaphore-throttled thread pool.

This module provides a simulation framework for a network of devices that
process sensor data. It is written for Python 2. Key features include a custom
reusable barrier built with a Condition object, and a Semaphore to limit the
number of concurrently executing script threads to a fixed-size pool. The main
device loop uses a two-phase barrier synchronization.
"""

from threading import Thread, Lock, Event, Condition, Semaphore

class ReusableBarrier():
    """A reusable barrier implementation using a Condition variable.

    This barrier blocks threads calling `wait()` until a specified number of
    threads have all called `wait()`. It then releases them all and resets
    for subsequent use.
    """
    
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Causes a thread to wait at the barrier."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread to arrive notifies all waiting threads and resets the counter.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Threads wait for the last thread to signal.
            self.cond.wait()
        self.cond.release()

class Device(object):
    """Represents a single device in the simulation.

    Manages sensor data, script execution, and synchronization. It uses a
    Semaphore to limit concurrent worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_event = Event() # Signals that setup is complete.

        self.lock_location = []
        self.lock_n = Lock()
        self.barrier = None

        self.thread_script = []
        self.num_thread = 0
        # This semaphore limits the number of concurrent script threads to 8.
        self.sem = Semaphore(value=8)

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources like the barrier and locks.

        Intended to be run by a single master device (id 0).
        
        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # Hardcoded to create 25 location locks.
            for _ in xrange(25):
                self.lock_location.append(Lock())

            for dev in devices:
                dev.barrier = barrier
                dev.lock_location = self.lock_location
                dev.setup_event.set() # Signal to each device that setup is done.

    def assign_script(self, script, location):
        """Assigns a script to be executed at a specific location."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of assignments for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in \
            self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

    def shutdown_script(self):
        """Waits for all running script threads to complete."""
        for i in xrange(self.num_thread):
            self.thread_script[i].join()
        
        # Clears the list of script threads.
        for i in xrange(self.num_thread):
            del self.thread_script[-1]

        self.num_thread = 0

class NewThreadScript(Thread):
    """A worker thread to execute a single script.

    Its concurrency is limited by a Semaphore in the parent Device.
    """
    def __init__(self, parent, neighbours, location, script):
        """Initializes the script-executing thread."""
        Thread.__init__(self)
        self.neighbours = neighbours
        self.parent = parent
        self.location = location
        self.script = script

    def run(self):
        """Acquires a lock, runs the script, and releases the semaphore."""
        # Acquire a location-specific lock to prevent race conditions.
        with self.parent.lock_location[self.location]:
            script_data = []
            
            # Aggregate data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Aggregate data from the parent device.
            data = self.parent.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = self.script.run(script_data)

                # Disseminate the result back to all involved devices.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.parent.set_data(self.location, result)
        
        # Release the semaphore to allow another script thread to run.
        self.parent.sem.release()

class DeviceThread(Thread):
    """The main control thread for a Device.

    Orchestrates the device's lifecycle, using a two-phase barrier scheme.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main execution loop of the device."""
        
        # Wait until the master device has finished setting up shared resources.
        self.device.setup_event.wait()

        while True:
            
            with self.device.lock_n:
                neighbours = self.device.supervisor.get_neighbours()
                # A None value for neighbours signals the simulation's end.
                if neighbours is None:
                    break

            # Wait for the supervisor to finish assigning scripts for the timepoint.
            self.device.timepoint_done.wait()

            # For each script, acquire the semaphore and start a new worker thread.
            # This limits the number of active workers to the semaphore's value (8).
            for (script, location) in self.device.scripts:
                self.device.sem.acquire()
                self.device.thread_script.append(NewThreadScript \
                    (self.device, neighbours, location, script))

                self.device.num_thread = self.device.num_thread + 1
                self.device.thread_script[-1].start()

            # --- Start of Two-Phase Barrier Synchronization ---

            # Phase 1: Wait for all devices to finish spawning their worker threads.
            self.device.barrier.wait()
            
            # Wait for all local worker threads to complete their execution.
            self.device.shutdown_script()
            
            # Reset the timepoint event for the next cycle.
            self.device.timepoint_done.clear()
            
            # Phase 2: Wait for all devices to finish their timepoint's work before
            # looping to the next timepoint.
            self.device.barrier.wait()
