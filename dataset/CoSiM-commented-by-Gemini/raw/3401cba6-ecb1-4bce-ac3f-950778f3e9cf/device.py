"""
A simulation of a distributed system of devices, likely for a sensor network 
or a multi-agent system. This module defines the behavior of individual devices,
their interaction with each other, and the synchronization mechanisms required
for their concurrent operation.
"""

from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """
    Represents a single device in the simulated network. Each device runs in its
    own thread, processes scripts, and communicates with its neighbors.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.
        
        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor readings.
            supervisor (obj): An object that manages the network, providing neighbors, etc.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        A collective setup method, intended to be called by a 'master' device.
        It initializes a shared barrier and locks for a group of devices.
        The logic ensures this setup is performed only once by one device in the group.
        """
        
        flag = True
        device_number = len(devices)

        # Pre-condition: Determine if this device is the 'master' for this setup task.
        # The device with the lowest ID in the 'devices' list becomes the master.
        for dev in devices:
            if self.device_id > dev.device_id:
                flag = False

        if flag == True:
            # This block is executed only by the 'master' device.
            barrier = ReusableBarrierSem(device_number)
            map_locations = {}
            tmp = {}
            for dev in devices:
                dev.barrier = barrier
                # Identify unique sensor locations across all devices to create shared locks.
                tmp = list(set(dev.sensor_data) - set(map_locations))
                for i in tmp:
                    map_locations[i] = Lock()
                dev.map_locations = map_locations
                tmp = {}

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device. Called by the supervisor.
        If script is None, it signals the end of a timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location from this device."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location on this device."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device. It orchestrates the script execution
    for each timepoint in the simulation.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.
        
        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        The main loop of the device. It waits for signals from the supervisor,
        executes scripts, and synchronizes with other devices.
        """
        while True:
            # Clear the event for the current timepoint.
            self.device.timepoint_done.clear()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Simulation ends.
            
            # Wait until the supervisor signals that all scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()
            
            script_list = []
            thread_list = []
            index = 0
            for script in self.device.scripts:
                script_list.append(script)

            # Spawn multiple threads to process scripts in parallel.
            # Here, it's hardcoded to 8 threads.
            for i in xrange(8):
                thread = SingleDeviceThread(self.device, script_list, neighbours, index)
                thread.start()
                thread_list.append(thread)

            for i in xrange(len(thread_list)):
                thread_list[i].join()
            
            # Synchronize with other devices at the barrier before proceeding to the next timepoint.
            self.device.barrier.wait()

class SingleDeviceThread(Thread):
    """A worker thread to execute a single script on a device."""
    def __init__(self, device, script_list, neighbours, index):
        Thread.__init__(self)
        self.device = device
        self.script_list = script_list
        self.neighbours = neighbours
        self.index = index

    def run(self):
        """Pops a script from the list and executes it."""
        # This logic has a race condition if multiple threads access pop with the same index.
        # However, as the index is always 0, it seems designed for threads to compete for scripts.
        if self.script_list != []:
            (script, location) = self.script_list.pop(self.index)
            self.compute(script, location)

    def update(self, result, location):
        """Updates the computed result on the local device and its neighbors."""
        for device in self.neighbours:
            device.set_data(location, result)
        self.device.set_data(location, result)

    def collect(self, location, neighbours, script_data):
        """
        Collects data for a given location from the local device and its neighbors.
        This operation is protected by a lock for that specific location.
        """
        self.device.map_locations[location].acquire()
        for device in self.neighbours:
            
            data = device.get_data(location)
            if data is None:
                pass
            else:
                script_data.append(data)

        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

    def compute(self, script, location):
        """
        The core logic for a single script execution. It collects data, runs the
        script, and updates the network with the result.
        """
        script_data = []
        self.collect(location, self.neighbours, script_data)

        if script_data == []:
            pass
        else:
            # Execute the script with the collected data.
            result = script.run(script_data)
            self.update(result, location)

        # Release the lock for the location.
        self.device.map_locations[location].release()

class ReusableBarrierSem():
    """
    A classic reusable barrier implementation using semaphores, allowing a group
    of threads to synchronize at a point, multiple times.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.
        
        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0) # For the first phase
        self.threads_sem2 = Semaphore(0) # For the second phase

    def wait(self):
        """
        Causes a thread to block until all `num_threads` have called this method.
        Implemented as a two-phase barrier to prevent race conditions on reuse.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive unblocks all waiting threads.
                for i in range(self.num_threads):


                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use

        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase to ensure all threads have left phase1 before reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread to arrive unblocks all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use

        self.threads_sem2.acquire()
