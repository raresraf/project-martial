"""
This module implements a simulation framework for a network of devices.

Each device operates on a master-worker threading model. A `DeviceThread` acts as the
master, coordinating simulation steps, while a pool of `Worker` threads executes
tasks. Synchronization between devices is handled by a `ReusableBarrierSem`, a
custom two-phase semaphore-based barrier.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """A reusable barrier implemented with semaphores for thread synchronization.

    This barrier uses a two-phase protocol to ensure that threads wait for each
    other at a synchronization point before any of them proceed.
    """

    def __init__(self, num_threads):
        """Initializes the semaphore-based barrier.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread arrived, release all waiting threads for this phase
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread arrived, release all waiting threads for this phase
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """Represents a device in the simulation with a master-worker architecture."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a device, including its master and worker threads.

        Args:
            device_id (int): Unique identifier for the device.
            sensor_data (dict): Sensor data for this device.
            supervisor (object): A supervisor object for simulation control.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.lock = {}
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_done = Event()
        self.terminate = Event()
        self.neighbours = []

        self.barrier = None
        # Barrier for synchronizing the device's internal worker threads
        self.threads_barrier = ReusableBarrierSem(9)  # 8 workers + 1 master
        self.master = DeviceThread(self, self.terminate, self.barrier, self.threads_barrier, \
                                    self.setup_done)
        self.master.start()

        self.threads = []
        for _ in range(8):
            thread = Worker(self.master, self.terminate, self.threads_barrier)
            self.threads.append(thread)
            thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources for all devices, intended to be called by device 0."""
        if self.device_id == 0:
            # Device 0 acts as the master setup node
            self.barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                self.lock[dev] = Lock()
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier
                    dev.lock = self.lock
                    dev.setup_done.set()
            self.setup_done.set()

    def assign_script(self, script, location):
        """Assigns a script to be processed by the device.

        Args:
            script (object): The script to execute.
            location (str): The data location for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of a timepoint
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully shuts down the device and all its threads."""
        self.terminate.set()
        for i in range(8):
            self.threads[i].script_received.set()
            self.threads[i].join()
        self.master.join()


class DeviceThread(Thread):
    """The master thread for a device, coordinating workers and synchronization."""

    def __init__(self, device, terminate, barrier, threads_barrier, setup_done):
        """Initializes the master thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = []
        self.terminate = terminate
        self.barrier = barrier
        self.threads_barrier = threads_barrier
        self.setup_done = setup_done

    def run(self):
        """Main execution loop for the master thread."""
        self.setup_done.wait()
        self.device.barrier.wait()

        while True:
            # Synchronize at the beginning of each time step
            self.device.barrier.wait()

            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break  # End of simulation

            # Wait for the timepoint to be marked as done
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

            # Distribute scripts among worker threads
            scripts = [[] for _ in range(8)]
            for i, script_item in enumerate(self.device.scripts):
                scripts[i % 8].append(script_item)

            for i in range(8):
                self.device.threads[i].scripts = scripts[i]
                self.device.threads[i].script_received.set()

            # Wait for all internal worker threads to finish
            if not self.terminate.is_set():
                self.threads_barrier.wait()


class Worker(Thread):
    """A worker thread that executes scripts for a device."""

    def __init__(self, master, terminate, barrier):
        """Initializes a worker thread."""
        Thread.__init__(self)
        self.master = master
        self.script_received = Event()
        self.terminate = terminate
        self.scripts = []
        self.barrier = barrier

    @staticmethod
    def append_data(device, location, script_data):
        """Safely retrieves and appends data from a device."""
        with device.lock[device]:
            data = device.get_data(location)
        if data is not None:
            script_data.append(data)

    @staticmethod
    def set_data(device, location, result):
        """Safely sets data on a device."""
        with device.lock[device]:
            device.set_data(location, result)

    def run(self):
        """Main execution loop for the worker thread."""
        while True:
            self.script_received.wait()
            self.script_received.clear()

            if self.terminate.is_set():
                break
            
            if self.scripts:
                for (script, location) in self.scripts:
                    # Gather data from neighbors
                    script_data = []
                    if self.master.neighbours:
                        for device in self.master.neighbours:
                            self.append_data(device, location, script_data)

                    # Gather data from self
                    self.append_data(self.master.device, location, script_data)

                    # Execute script and distribute results
                    if script_data:
                        result = script.run(script_data)
                        if self.master.neighbours:
                            for device in self.master.neighbours:
                                self.set_data(device, location, result)
                        self.set_data(self.master.device, location, result)

            # Synchronize with master and other workers of the same device
            self.barrier.wait()
