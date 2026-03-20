"""
This module implements a distributed device simulation framework.

It features a custom two-phase, semaphore-based reusable barrier for synchronization,
and a multi-threaded device model where a main `DeviceThread` spawns `DeviceSubThread`
workers to execute scripts. Concurrency is managed via a shared array of
location-based locks and a global device lock.
"""

from threading import Event, Thread, Semaphore, Lock



class ReusableBarrier(object):
    """A reusable, two-phase barrier implemented with semaphores.

    This barrier uses mutable list elements as counters to work around Python 2's
    lack of a `nonlocal` keyword, allowing the `phase` method to modify the
    counter variables.
    """

    def __init__(self, num_threads):
        """Initializes the barrier."""
        self.num_threads = num_threads
        # Counters are wrapped in a list to be mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread arrived, release all waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for reuse.
                count_threads[0] = self.num_threads
        threads_sem.acquire()




class Device(object):
    """Represents a device in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.lock = Lock()
        self.locationlock = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared synchronization objects for all devices.

        This method is intended to be run by a single master device (device 0).
        It creates and distributes a global barrier and a list of location-based locks.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # Assumes locations are integer indices from 0 to 99.
            locationlock = [Lock() for _ in xrange(100)]
            for device in devices:
                device.locationlock = locationlock
                device.set_barrier(barrier)
        else:
            pass

    def set_barrier(self, barrier):
        """Assigns the global barrier to this device."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """Assigns a script to be executed in the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
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
        self.thread.join()




class DeviceThread(Thread):
    """The main control thread for a single device."""

    def __init__(self, device):
        """Initializes the control thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop of the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # End of simulation

            # Wait until all scripts for the time step have been assigned.
            self.device.timepoint_done.wait()
            subthreads = []

            # Spawn a sub-thread for each script.
            for (script, location) in self.device.scripts:
                subthread = DeviceSubThread(self, neighbours, script, location)
                subthreads.append(subthread)
                subthread.start()
            
            # Wait for all sub-threads to complete.
            for subthread in subthreads:
                subthread.join()
            
            self.device.timepoint_done.clear()
            
            # Synchronize with all other devices before the next time step.
            self.device.barrier.wait()


class DeviceSubThread(Thread):
    """A worker thread that executes a single script."""
    
    def __init__(self, devicethread, neighbours, script, location):
        """Initializes the worker sub-thread."""
        Thread.__init__(self, name="Device SubThread %d"
            % devicethread.device.device_id)
        self.neighbours = neighbours
        self.devicethread = devicethread
        self.script = script
        self.location = location

    def run(self):
        """The execution logic for the script."""
        # Acquire a lock specific to the data location.
        self.devicethread.device.locationlock[self.location].acquire()
        script_data = []
        
        # Gather data from neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the local device.
        data = self.devicethread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Run the script and distribute the results.
            result = self.script.run(script_data)
            for device in self.neighbours:
                # This global lock could be a performance bottleneck.
                with device.lock:
                    device.set_data(self.location, result)
            
            with self.devicethread.device.lock:
                self.devicethread.device.set_data(self.location, result)
        
        # Release the location-specific lock.
        self.devicethread.device.locationlock[self.location].release()
