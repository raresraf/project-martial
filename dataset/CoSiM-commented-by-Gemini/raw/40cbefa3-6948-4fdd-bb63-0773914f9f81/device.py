"""
Models a distributed sensor network simulation using a "thread-per-task" model.

This script simulates a network of devices that process sensor data in discrete,
synchronized time steps. For each computational task within a time step, a new,
short-lived `ScriptThread` is spawned to handle the full cycle of data
aggregation, computation, and result dissemination.

Classes:
    ReusableBarrier: A custom two-phase reusable barrier for synchronizing
                     all device threads between time steps.
    Device: Represents a node in the network, holding its state and data.
    ScriptThread: A short-lived thread created to execute a single script.
    DeviceThread: The main orchestrator thread for a Device, which spawns and
                  manages the `ScriptThread` instances for a given time step.
"""


from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """A reusable barrier for synchronizing a fixed number of threads.
    
    This implementation uses a two-phase protocol with semaphores to ensure that
    threads wait for each other at the barrier and can reuse the barrier in a loop
    without race conditions.
    """
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        # Counters are stored in a list to be mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all `num_threads` have called this method."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier protocol."""
        with self.count_lock:
            count_threads[0] -= 1
            # The last thread to arrive resets the counter and releases all threads.
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """Represents a single device or node in the simulated network."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device and starts its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()
        # This will hold a list of locks, one for each unique data location.
        self.location_locks = None

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources across all devices.

        The master device (id 0) discovers all unique data locations across the
        entire network, creates a dedicated lock for each, and distributes the
        list of locks and a shared barrier to all other devices.
        """
        # This setup is performed only by the master device (id 0).
        if 0 == self.device_id:
            # Create a barrier for all devices.
            self.barrier = ReusableBarrier(len(devices))
            
            # Discover all unique locations from all devices' sensor data.
            locations = []
            for device in devices:
                for location in device.sensor_data:
                    if location not in locations:
                        locations.append(location)
            
            # Create a lock for each unique location.
            self.location_locks = []
            for _ in range(len(locations)):
                self.location_locks.append(Lock())
            
            # Distribute the shared barrier and locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """Assigns a computational script for the upcoming time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all assignments for this time step are done.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()


class ScriptThread(Thread):
    """A short-lived thread that executes one script for one location."""

    def __init__(self, device, script, location, neighbours):
        """Initializes the thread with all context needed for the task."""
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the full task: lock, aggregate data, compute, and disseminate results.
        """
        # Use a `with` statement for exception-safe locking of the data location.
        with self.device.location_locks[self.location]:
            script_data = []
            
            # 1. Aggregate data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Aggregate data from self.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # 2. Run the computation if there is data.
            if script_data != []:
                result = self.script.run(script_data)
                
                # 3. Disseminate the result to all neighbors and self.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """The main orchestrator thread for a Device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            # Get neighbors for this time step. If None, terminate.
            vecini = self.device.supervisor.get_neighbours()
            if vecini is None:
                break
            
            # 1. Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            threads = []
            
            # 2. Spawn a new ScriptThread for each assigned script.
            if len(vecini) != 0:
                for (script, locatie) in self.device.scripts:
                    thread = ScriptThread(self.device, script, locatie, vecini)
                    threads.append(thread)
                    thread.start()

                # 3. Wait for all spawned script threads to complete.
                for thread in threads:
                    thread.join()
            
            # Reset for the next time step.
            self.device.timepoint_done.clear()
            
            # 4. Synchronize with all other devices before starting the next time step.
            self.device.barrier.wait()
