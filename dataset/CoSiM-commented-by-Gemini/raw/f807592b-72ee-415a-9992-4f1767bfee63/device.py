
"""
This module provides a third variant of a simulated distributed device system.

The architecture is a time-stepped simulation where devices perform computations
in parallel. Synchronization between devices is managed by a `ReusableBarrier`
implemented with a `Condition` variable. Unlike previous versions, this
implementation uses a "spawn-and-join" model for parallelism: for each time step,
the main device thread spawns a new worker thread (`ScriptThread`) for each
computational task and waits for it to complete. Data gathering is performed by
the main thread before spawning workers, and result dissemination happens after
all workers have finished.
"""

from threading import Condition, Event, RLock, Thread

class ReusableBarrier(object):
    """
    A reusable barrier implemented using a `threading.Condition`.

    This barrier blocks threads at a `wait()` call until a specified number of
    threads have all reached the barrier.

    Note: The `reinit` method has unusual logic that may not behave as expected
    for a typical reusable barrier, as it modifies the thread count mid-simulation.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier.

        Args:
            num_threads (int): The number of threads to wait for.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def reinit(self):
        """
        Reduces the number of threads required by the barrier.

        This is intended to be called when a device thread is shutting down.
        """
        self.cond.acquire()
        self.num_threads -= 1
        self.cond.release()
        self.wait()

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have called `wait`.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread to arrive notifies all waiting threads and resets the counter.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Threads wait for the last thread to arrive.
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a device node in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.start = Event()
        self.scripts = []
        self.scripts_to_process = []
        self.timepoint_done = Event()
        self.nr_script_threats = 0
        self.thread = DeviceThread(self)
        self.thread.start()
        self.script_threats = []
        self.barrier_devices = None
        self.neighbours = None
        # Simulates the number of cores available for running script threads.
        self.cors = 8
        self.lock = None
        self.lock_self = None
        self.results = {}
        self.results_lock = None

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (locks and barrier)
        across all devices in the simulation. This method uses a series of
        locks to ensure that shared objects are instantiated only once.
        """
        for script in self.scripts:
            self.lock.acquire()
            self.scripts_to_process.append(script)
            self.lock.release()

        # Initialize and distribute a shared lock for device-specific operations.
        if not self.lock_self:
            lock = RLock()
            for device in devices:
                device.lock_self = lock

        # Initialize and distribute a shared lock for script list operations.
        self.lock_self.acquire()
        if not self.lock:
            rlock = RLock()
            for device in devices:
                device.lock = rlock
        self.lock_self.release()

        # Initialize and distribute a shared lock for the results dictionary.
        self.lock_self.acquire()
        if not self.results_lock:
            results_lock = RLock()
            for device in devices:
                device.results_lock = results_lock
        self.lock_self.release()

        # Initialize and distribute the shared barrier for all devices.
        self.lock_self.acquire()
        if not self.barrier_devices:
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier_devices = barrier
                device.start.set()
        self.lock_self.release()



    def assign_script(self, script, location):
        """Assigns a script to the device for a future timepoint."""
        if script is not None:
            self.lock.acquire()
            self.scripts.append((script, location))
            self.scripts_to_process.append((script, location))
            self.script_received.set()
            self.lock.release()
        else:
            # A None script signals the end of script assignments for this timepoint.
            self.lock.acquire()
            self.timepoint_done.set()
            self.script_received.set()
            self.lock.release()

    def get_data(self, location):
        """Safely retrieves data from the device's sensor data."""
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else:
            data = None
        return data

    def set_data(self, location, data):
        """Safely updates the device's sensor data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data
        
    def shutdown(self):
        """Joins the main device thread to shut down the device."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, managing the simulation lifecycle.
    """

    def __init__(self, device):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.device.neighbours = None

    def run(self):
        """The main simulation loop for the device."""
        self.device.start.wait()
        while True:
            # At the start of each timepoint, reset the list of scripts to process.
            self.device.scripts_to_process = []
            for script in self.device.scripts:
                self.device.scripts_to_process.append(script)

            # Get the list of neighbors for this timepoint from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                # If no neighbors, the simulation is ending for this device.
                self.device.barrier_devices.reinit()
                break

            self.device.results = {}
            # Loop to process all scripts for the current timepoint.
            while True:
                if not self.device.timepoint_done.is_set():
                    self.device.script_received.wait()
                    self.device.script_received.clear()

                if len(self.device.scripts_to_process) == 0:
                    if self.device.timepoint_done.is_set():
                        break

                # Process scripts in batches based on the number of 'cors'.
                while len(self.device.scripts_to_process):
                    list_threats = []
                    self.device.script_threats = []
                    self.device.nr_script_threats = 0
                    
                    # Create a batch of script execution tasks.
                    while len(self.device.scripts_to_process) and self.device.nr_script_threats < self.device.cors:
                        script, location = self.device.scripts_to_process.pop(0)
                        list_threats.append((script, location))
                        self.device.nr_script_threats += 1

                    # For each task, gather data and spawn a worker thread.
                    for script, location in list_threats:
                        script_data = []
                        
                        # Gather data from all neighboring devices.
                        neighbours = self.device.neighbours
                        for device in neighbours:
                            device.lock_self.acquire()
                            data = device.get_data(location)
                            device.lock_self.release()
                            if data is not None:
                                script_data.append(data)
                        
                        # Gather data from the local device.
                        self.device.lock_self.acquire()
                        data = self.device.get_data(location)
                        self.device.lock_self.release()
                        if data is not None:
                            script_data.append(data)

                        # Spawn and start a new thread for the script.
                        thread_script_d = ScriptThread(self.device, script, location, script_data)
                        self.device.script_threats.append(thread_script_d)
                        thread_script_d.start()

                    # Wait for the current batch of threads to complete.
                    for thread in self.device.script_threats:
                        thread.join()

            # After all scripts are run, disseminate the results.
            for location, result in self.device.results.iteritems():
                
                # Update data on neighboring devices.
                for device in self.device.neighbours:
                    device.lock_self.acquire()
                    device.set_data(location, result)
                    device.lock_self.release()
                
                # Update data on the local device.
                self.device.lock_self.acquire()
                self.device.set_data(location, result)
                self.device.lock_self.release()

            # Wait for the timepoint completion signal.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Wait at the barrier for all other devices to complete the timepoint.
            self.device.barrier_devices.wait()

class ScriptThread(Thread):
    """
    A short-lived thread responsible for executing a single script.
    """

    def __init__(self, device, script, location, script_data):
        """
        Initializes the script thread.

        Args:
            device (Device): The parent device.
            script (object): The script to be executed.
            location (any): The location associated with the data.
            script_data (list): The pre-gathered data for the script.
        """
        Thread.__init__(self, name="Device Script Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.script_data = script_data

    def run(self):
        """Executes the script and stores the result."""
        if self.script_data != []:
            
            result = self.script.run(self.script_data)
            
            # Atomically add the result to the parent device's results dictionary.
            self.device.results_lock.acquire()
            self.device.results[self.location] = result
            self.device.results_lock.release()
        
        # Decrement the count of active script threads.
        self.device.nr_script_threats -= 1
