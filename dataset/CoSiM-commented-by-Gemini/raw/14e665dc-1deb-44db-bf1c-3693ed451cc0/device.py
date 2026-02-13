"""
This module defines a simulation framework for a network of concurrent devices,
intended for a Python 2 environment.

It provides classes to model devices that operate in parallel on shared data,
coordinated through synchronization primitives like barriers and events. This
version uses a two-phase semaphore-based barrier and a static work
distribution scheme.

Classes:
    ReusableBarrierSem: A two-phase reusable barrier using Semaphores.
    Device: Represents a device, managing its state and worker threads.
    DeviceThread: The worker thread class that executes computational scripts.
"""
from threading import Lock, Event, Thread, Semaphore, Condition


class ReusableBarrierSem():
    """
    A reusable two-phase barrier implemented with Semaphores.

    This barrier ensures that all participating threads wait at a synchronization
    point before any of them are allowed to proceed. It uses two phases to prevent
    threads from one cycle from proceeding into the next cycle prematurely.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that will wait on the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads


        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase1()
        self.phase2()
    def phase1(self):
        """First phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    def phase2(self):
        """Second phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a single device in the simulation network.

    This class manages device-specific data and a pool of worker threads.
    Shared resources like the global barrier and location locks are handled
    at the class level.
    """
    location_locks = []
    barrier = None
    nr_t = 8
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): The local data for the device.
            supervisor: The supervisor object managing the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.neighbours_event = Event()
        self.threads = []
        for i in xrange(Device.nr_t):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(Device.nr_t):
            self.threads[i].start()
    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for the entire device network.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        Device.barrier = ReusableBarrierSem(Device.nr_t * len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script to the device for a given location.

        Args:
            script: The script object to execute.
            location: The location context for the script.
        """
        if location not in [elem[0] for elem in Device.location_locks]:
            Device.location_locks.append((location, Lock()))

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data for a specific location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its worker threads."""
        for i in xrange(Device.nr_t):
            self.threads[i].join()

class DeviceThread(Thread):
    """A worker thread that executes scripts for a device."""
    def __init__(self, device, index):
        """
        Initializes the thread.

        Args:
            device (Device): The parent device.
            index (int): The thread's index within the device's thread pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.index = index
        self.neighbours = None

    def run(self):
        """The main execution loop of the thread."""
        while True:
            # In each device, thread 0 fetches neighbor info and signals others.
            if self.index == 0:
                self.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_event.set()
            else:
                # Other threads wait for neighbor info to be available.
                self.device.neighbours_event.wait()
                self.neighbours = self.device.threads[0].neighbours
            if self.neighbours is None:
                # A None neighbor list is the signal to terminate.
                break

            # Wait for the signal that all scripts for the timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Process scripts using a static partitioning scheme.
            # Each thread processes a subset of scripts based on its index.
            for j in range(self.index, len(self.device.scripts), Device.nr_t):
                location = self.device.scripts[j][1]
                script = self.device.scripts[j][0]

                # Find and acquire the lock for the script's location.
                for i in range(len(Device.location_locks)):
                    if location == Device.location_locks[i][0]:
                        Device.location_locks[i][1].acquire()

                script_data = []
                # Gather data from self and neighbors.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Run script and broadcast the result.
                    result = script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                # Release the location lock.
                for i in range(len(Device.location_locks)):
                    if location == Device.location_locks[i][0]:
                        Device.location_locks[i][1].release()

            # Global barrier to synchronize all threads after script processing.
            Device.barrier.wait()
            # Thread 0 resets the events for the next timepoint.
            if self.index == 0:
                self.device.timepoint_done.clear()
                self.device.neighbours_event.clear()
            # A second barrier wait to ensure events are cleared before the next cycle.
            Device.barrier.wait()
