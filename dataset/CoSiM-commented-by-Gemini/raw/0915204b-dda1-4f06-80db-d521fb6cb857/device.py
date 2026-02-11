"""
This module defines a simulated distributed device network using a Bulk
Synchronous Parallel (BSP) model. Each device has a master thread that spawns
worker threads for each task in a time step.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    A correct, two-phase reusable barrier for synchronizing multiple threads.
    It uses a pair of semaphores to ensure that no thread begins a new phase
    until all threads have completed the previous one.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters are stored in a list to make them mutable inside methods.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Waits at the barrier; composed of two distinct synchronization phases."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases the others for this phase.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for the next use of the barrier.
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """Represents a single device in the network."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.location_locks = None
        # Each device has one master thread that orchestrates its work.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.
        This centralized setup is performed by device 0.
        """
        if 0 == self.device_id:
            # Create a single global barrier for all device master threads.
            self.barrier = ReusableBarrier(len(devices))
            
            # Discover all unique locations across all devices.
            locations = []
            for device in devices:
                for location in device.sensor_data:
                    if location not in locations:
                        locations.append(location)
            
            # Create a fine-grained lock for each unique location.
            self.location_locks = [Lock() for _ in range(len(locations))]
            
            # Distribute the shared barrier and locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        Assigns a script. A `None` script signals the end of assignments for the step.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal the master thread that all scripts are received.
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class ScriptThread(Thread):
    """
    A short-lived worker thread to execute one script for a given location.
    """
    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script under a location-specific lock.
        """
        # Use a `with` statement for safe lock acquisition and release.
        with self.device.location_locks[self.location]:
            script_data = []
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            # Gather data from the local device.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            if script_data:
                # Run the script and propagate the result.
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    The master thread for a device, implementing the BSP logic.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main time-stepped loop for the device."""
        while True:
            # `vecini` is Romanian for "neighbors".
            vecini = self.device.supervisor.get_neighbours()
            if vecini is None:
                break # End of simulation.
            
            # 1. Wait for the supervisor to signal that scripts are ready.
            self.device.timepoint_done.wait()
            
            threads = []
            # 2. Spawn a new worker thread for each script.
            if len(vecini) != 0:
                for (script, locatie) in self.device.scripts:
                    thread = ScriptThread(self.device, script, locatie, vecini)
                    threads.append(thread)
                    thread.start()
                
                # 3. Wait for all local worker threads to complete.
                for thread in threads:
                    thread.join()
            
            # 4. Clear the event for the next time step.
            self.device.timepoint_done.clear()
            
            # 5. Wait at the global barrier for all other devices to finish.
            self.device.barrier.wait()
