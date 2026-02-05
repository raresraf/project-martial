"""
This module implements a distributed device simulation using a custom, two-phase
reusable barrier for synchronization. The barrier is based on semaphores.
NOTE: This implementation lacks any explicit locking for data access, making the
get_data and set_data methods prone to race conditions.
"""

from threading import Thread, Event, Semaphore, Lock

class ReusableBarrier(object):
    """
    A custom implementation of a reusable barrier using a two-phase protocol
    with semaphores. This design prevents threads that are released from one barrier
    wait from re-entering and interfering with threads still exiting the same wait.
    """
    def __init__(self, num_threads):
        """
        Initializes the two-phase barrier.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        # Counters for each phase, wrapped in a list to be mutable across instances.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Semaphores for each phase, acting as the gates.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks the calling thread until all threads have reached this point.
        Executes two synchronization phases to ensure safe reuse of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single synchronization phase.

        Args:
            count_threads (list): The counter for the current phase.
            threads_sem (Semaphore): The semaphore used for gating in this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # The last thread to arrive is responsible for opening the gate.
            if count_threads[0] == 0:
                # Release the semaphore N times, once for each waiting thread.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # All threads, including the last one, will block here until the gate is opened.
        threads_sem.acquire()


class Device(object):
    """
    Represents a device in the simulation. Each device runs in its own thread
    and communicates with others, with synchronization managed by the ReusableBarrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and its control thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barrier = None
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes the shared ReusableBarrier to all devices.
        This setup logic appears to allow any device to initialize the barrier.

        Args:
            devices (list): The list of all devices in the simulation.
        """
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            # Attempt to propagate the barrier to all other devices.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
        # Store a local copy of all devices.
        for device in devices:
            if device is not None:
                self.devices.append(device)



    def assign_script(self, script, location):
        """
        Assigns a script to be executed. A `None` script signals that the
        script assignment for the current timepoint is complete.

        Args:
            script (Script): The script to be executed.
            location (str): The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data.
        WARNING: This method is not thread-safe and is subject to race conditions
        as it does not use any locks.

        Args:
            location (str): The data location.
        
        Returns:
            The sensor data, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data.
        WARNING: This method is not thread-safe and is subject to race conditions
        as it does not use any locks.

        Args:
            location (str): The data location.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's control thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, handling script execution and synchronization
    for each timepoint of the simulation.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the supervisor to signal that all scripts for this step are assigned.
            self.device.timepoint_done.wait()

            # Execute all assigned scripts.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Gather data from neighbors. This is a potential race condition source.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    result = script.run(script_data)
                    
                    # Distribute results to neighbors. This is a potential race condition source.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
            
            self.device.timepoint_done.clear()
            # Invariant: All threads must synchronize at the barrier before the next timepoint.
            self.device.barrier.wait()