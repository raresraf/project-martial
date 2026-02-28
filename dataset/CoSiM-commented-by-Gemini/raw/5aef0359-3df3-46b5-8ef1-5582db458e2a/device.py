


"""
Models a distributed device network using a "thread-per-task" architecture.

In this Bulk Synchronous Parallel (BSP) simulation, each `Device` has a main
`DeviceThread`. For each timepoint, this thread spawns a new, short-lived
`ScriptThread` for every assigned task. It waits for all tasks to complete by
joining these threads before synchronizing with other devices at a global barrier.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    A reusable barrier implemented with two semaphores for two-phase synchronization.

    This prevents race conditions where fast threads could loop around and re-enter
    the barrier before slow threads have exited. The use of a list for the counter
    is an idiom for creating a mutable integer in Python 2.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
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
        """Executes one phase of the two-phase barrier."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device node in the network. It holds state and is
    managed by a main `DeviceThread`.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main orchestrator thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        # Shared resources to be populated by `setup_devices`.
        self.barrier = None
        self.location_locks = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources for the entire network.

        The master device (ID 0) creates a global barrier and a list of shared
        locks (one for each unique data location) and distributes them to all
        other devices.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            
            # Identify all unique data locations to create a lock for each.
            locations = []
            for device in devices:
                for location in device.sensor_data:
                    if location not in locations:
                        locations.append(location)
            
            self.location_locks = []
            for _ in range(len(locations)):
                self.location_locks.append(Lock())
            
            # Propagate shared resources to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """Assigns a script to this device for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # `None` script signals that all work has been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class ScriptThread(Thread):
    """
    A short-lived worker thread responsible for executing a single script.
    """

    def __init__(self, device, script, location, neighbours):
        """Initializes the worker with all context needed for its task."""
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """Executes the script atomically for its location."""
        # Use a `with` statement on the shared location lock to ensure atomicity.
        with self.device.location_locks[self.location]:
            script_data = []
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from self.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Execute script and disseminate results if data was found.
            if script_data:
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    The main orchestrator thread for a device. It manages the "thread-per-task"
    execution for each timepoint.
    """

    def __init__(self, device):
        """Initializes the orchestrator thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main loop for managing timepoints."""
        while True:
            # Get neighbors for the current timepoint. `vecini` is Romanian for "neighbors".
            vecini = self.device.supervisor.get_neighbours()
            if vecini is None:
                break # Shutdown signal.
            
            # Wait until the supervisor has assigned all scripts for this timepoint.
            self.device.timepoint_done.wait()
            
            # --- Intra-device Work Execution ---
            threads = []
            if vecini:
                # Spawn a new thread for each assigned script.
                for (script, locatie) in self.device.scripts:
                    thread = ScriptThread(self.device, script, locatie, vecini)
                    threads.append(thread)
                    thread.start()
                # Wait for all spawned script threads to complete.
                for thread in threads:
                    thread.join()
            
            # Clear the event to prepare for the next timepoint.
            self.device.timepoint_done.clear()
            
            # --- Inter-device Synchronization ---
            # Wait at the global barrier for all other devices to finish their work.
            self.device.barrier.wait()
