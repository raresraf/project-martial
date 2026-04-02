"""
This module provides a framework for a discrete-time simulation of a network of devices.

It defines the core components for a distributed system where devices can execute
computational scripts, exchange data with their neighbors, and synchronize their
operations in time steps. The simulation is managed through a combination of
threading, locks, and a custom reusable barrier.
"""


from threading import Event, Semaphore, Lock, Thread



class ReusableBarrierSem(object):
    """
    A reusable barrier implemented using Semaphores for thread synchronization.

    This barrier allows a specified number of threads to wait for each other to
    reach a certain point of execution before any of them are allowed to continue.
    It is "reusable" as it automatically resets after all threads have passed,
    making it suitable for use in loops. It uses a two-phase signaling
    mechanism (two semaphores) to prevent race conditions where faster threads
    from a subsequent iteration could pass the barrier before slower threads from
    the current iteration have completed.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads

        self.counter_lock = Lock()

        self.threads_sem1 = Semaphore(0)

        self.threads_sem2 = Semaphore(0)
    def wait(self):
        """
        Causes a thread to wait at the barrier until all participating threads
        have called this method.
        """
        self.phase1()
        self.phase2()
    def phase1(self):
        """First phase of the two-phase barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    def phase2(self):
        """Second phase of the two-phase barrier wait."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a single device (or node) in the simulated network.

    Each device runs in its own thread, maintains its own sensor data, and can
    execute scripts. It communicates with a supervisor to discover its neighbors
    and synchronizes with other devices using a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: A dictionary representing the device's local data.
            supervisor: An object responsible for providing network topology (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []

        self.my_lock = Lock()
        self.barrier = ReusableBarrierSem(0)
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier among a list of devices.

        The device with ID 0 creates the barrier, and all other devices
        reference it. This ensures all devices synchronize on the same object.

        Args:
            devices: A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device in the current time step.

        Args:
            script: The script object to execute.
            location: The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()


    def get_data(self, location):
        """
        Retrieves data from a specific location on the device.

        Args:
            location: The key for the desired data in the sensor_data dictionary.

        Returns:
            The data if the location exists, otherwise None.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates data at a specific location on the device.

        Args:
            location: The key for the data to be updated.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()



class MyScriptThread(Thread):
    """
    A thread to execute a single script on a device and its neighbors.

    This thread is responsible for gathering data from a device and its neighbors,
    running a script with that data as input, and then writing the result
    back to all involved devices.
    """

    def __init__(self, script, location, device, neighbours):
        """
        Initializes the script execution thread.

        Args:
            script: The script to run.
            location: The data location the script targets.
            device: The primary device executing the script.
            neighbours: A list of the primary device's neighbors.
        """
        Thread.__init__(self)


        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        The main logic for the script thread.

        It gathers data from the local device and its neighbors, runs the script,
        and then distributes the result back to all of them, using locks to
        ensure thread-safe updates to device data.
        """
        script_data = []

        # Gather data from all neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Gather data from the local device itself.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)



        if script_data != []:

            # Execute the script with the aggregated data.
            result = self.script.run(script_data)


            # Atomically update the data on all neighbor devices with the result.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()
            
            # Atomically update the data on the local device.
            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()

class DeviceThread(Thread):
    """The main execution thread for a single Device."""

    def __init__(self, device):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop for the device.

        This loop represents the progression of time steps. In each step, the
        device synchronizes with all others, executes its assigned scripts, and
        then synchronizes again before beginning the next step.
        """
        while True:
            # Pre-condition: At the start of a time step, discover neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break;

            # Invariant: All devices wait here until every device is ready to start the step.
            self.device.barrier.wait()


            self.device.script_received.wait()
            script_threads = []

            # Spawn a new thread for each assigned script.
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            for thread in script_threads:
                thread.start()
            for thread in script_threads:
                thread.join()


            self.device.timepoint_done.wait()
            # Invariant: All devices wait here until every device has finished its computation for the step.
            self.device.barrier.wait()
            self.device.script_received.clear()
