"""
This module contains a fifth variant of the distributed device simulation.
Key features include a self-implemented two-phase reusable barrier, a design where
one device (device 0) acts as a centralized owner of synchronization objects, and
a complex thread management model that manually batches thread-per-task execution.
"""
from threading import Event, Thread, Lock, Semaphore, RLock


class ReusableBarrier(object):
    """
    A custom, reusable barrier implementation using two semaphores.

    Threads are synchronized in two phases to ensure that no thread can start a new
    cycle of the barrier before all threads have completed the previous cycle.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a fixed number of threads.

        Args:
            num_threads (int): The number of threads that will synchronize on this barrier.
        """
        self.num_threads = num_threads
        # Use lists for counters to make them mutable and shared across threads.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the two-phase barrier synchronization.

        The last thread to arrive resets the counter and releases all other threads.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread releases the semaphore for all other waiting threads.
                i = 0
                while i < self.num_threads:
                    threads_sem.release()
                    i += 1
                # Reset the counter for the next use of the barrier.
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a device in the simulation. This implementation relies on a master
    device (device 0) to manage all synchronization objects.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.devices = []
        self.script_received = Event()
        self.timepoint_done = Event() # Note: Set but not waited upon in this version.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device list and initializes sync objects on the master device.

        Device 0 creates and holds the single barrier and lock dictionary for all devices.
        """
        for device in devices:
            self.devices.append(device)
        # Assumes device at index 0 is the master.
        self.devices[0].barrier = ReusableBarrier(len(self.devices))
        self.devices[0].locations_lock = {}

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts have been assigned.
            self.timepoint_done.set()
            # This event is the one actually used to start the processing phase.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It manages the execution of scripts
    by creating a new thread for each script and running them in manual batches.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.num_threads = 8 # The size of the manual execution batches.

    def run(self):
        """The main simulation loop, executed once per timepoint."""
        while True:
            threads = []
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.
            
            # Wait for the signal that all scripts for the timepoint have been assigned.
            self.device.script_received.wait()
            
            # --- Thread-per-Task Creation ---
            # A new thread is created for every script to be executed.
            for (script, location) in self.device.scripts:
                thread = MyThread(self, script, location, neighbours)
                threads.append(thread)
            
            # --- Manual Batch Execution ---
            # The created threads are run in fixed-size batches. This is a highly
            # inefficient way to manage concurrent tasks.
            rounds = len(self.device.scripts) / self.num_threads
            leftovers = len(self.device.scripts) % self.num_threads
            
            while rounds > 0:
                for j in xrange(self.num_threads):
                    threads[j].start()
                for j in xrange(self.num_threads):
                    threads[j].join()
                for j in xrange(self.num_threads):
                    threads.pop(0)
                rounds -= 1
            
            # Process the remaining threads that didn't form a full batch.
            for j in xrange(leftovers):
                threads[j].start()
            for j in xrange(leftovers):
                threads[j].join()
            for j in xrange(leftovers):
                threads.pop(0)

            del threads[:]
            
            # Synchronize with all other devices at the master barrier.
            self.device.devices[0].barrier.wait()
            
            # Reset the event for the next timepoint.
            self.device.script_received.clear()


class MyThread(Thread):
    """
    A temporary worker thread created to execute a single script.
    """
    def __init__(self, device_thread, script, location, neighbours):
        Thread.__init__(self)
        self.device_thread = device_thread
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes a single script. It lazily initializes locks on the master device,
        which is not a thread-safe operation.
        """
        # --- Non-Thread-Safe Lazy Lock Initialization ---
        # If two threads for the same new location execute this simultaneously,
        # they can race to create the lock.
        if self.location not in\
                self.device_thread.device.devices[0].locations_lock:
            self.device_thread.device.devices[0].locations_lock[self.location]\ = RLock()
        
        # All threads acquire locks from the central dictionary on the master device.
        with self.device_thread.device.devices[0].locations_lock[self.location]:
            # --- Critical Section ---
            script_data = []
            
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from the local device.
            data = self.device_thread.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            if script_data != []:
                # Execute the script and update data on all involved devices.
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device_thread.device.set_data(self.location, result)
            # --- End Critical Section ---