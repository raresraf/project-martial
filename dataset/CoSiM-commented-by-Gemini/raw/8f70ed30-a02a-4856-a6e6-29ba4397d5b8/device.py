# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices using a manually managed,
fixed-size thread pool.

NOTE: This implementation is overly complex and contains several logical flaws
and race conditions. A standard producer-consumer pattern with a `Queue` would be
a more robust and efficient solution.
"""

from threading import Event, Thread, Lock, Semaphore

class Device(object):
    """
    Represents a device that orchestrates a pool of worker threads.

    This class holds the device's state and shared synchronization primitives, but
    the core work distribution logic is handled by its `DeviceThread`.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device and its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.lock = {}
        self.barrier = None
        self.devices = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for all devices.

        NOTE: The setup logic has race conditions. The 'lock' dictionary shared
        between devices will be the one created by the last device to execute
        this method. A leader-election pattern would be a cleaner approach.
        """
        self.devices = devices
        self.barrier = ReusableBarrierSem(len(self.devices))

        # Redundant work: Every device creates a full dictionary of locks.
        for location in self.sensor_data:
            self.lock[location] = Lock()
        for device in devices:
            for location in device.sensor_data:
                self.lock[location] = Lock()

        # All devices will end up sharing the `barrier` and `lock` from the
        # last device that executes this block.
        for i in xrange(len(self.devices)):
            self.devices[i].barrier = self.barrier
            self.devices[i].lock = self.lock

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is used as a sentinel value.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. Not internally thread-safe."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data. Not internally thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class MyThread(Thread):
    """A worker thread that executes a single script."""

    def __init__(self, my_id, device, neighbours, lock, script, location):
        Thread.__init__(self, name="Thread %d from device %d" % (my_id, device.device_id))
        self.device = device
        self.my_id = my_id
        self.neighbours = neighbours
        self.lock = lock
        self.script = script
        self.location = location

    def run(self):
        """
        Executes the script, ensuring exclusive access to the data location
        via a globally shared lock.
        """
        # The `with` statement correctly acquires and releases the shared lock.
        with self.lock[self.location]:
            script_data = []
            # Aggregate data from neighbors and self.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Run script and disseminate results.
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """The main control thread, which manually manages a pool of worker threads."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.numThreads = 0
        self.listThreads = []

    def run(self):
        """
        Main simulation loop.

        NOTE: The logic for managing the thread pool is inefficient and buggy.
        It involves polling thread status and can lead to incorrect behavior
        if all threads in the pool are busy.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for supervisor to signal that scripts are ready.
            self.device.script_received.wait()

            # --- Manual and Flawed Thread Pool Management ---
            for (script, location) in self.device.scripts:
                if len(self.listThreads) < 8:
                    # If pool is not full, create and start a new worker.
                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.append(thread)
                    thread.start()
                    self.numThreads += 1
                else:
                    # If pool is full, find a finished thread to replace.
                    # This is inefficient polling and has race conditions.
                    index = -1
                    for i in xrange(len(self.listThreads)):
                        if not self.listThreads[i].is_alive():
                            self.listThreads[i].join()
                            index = i
                    
                    # BUG: If all threads are alive, index will be -1, causing
                    # the wrong thread to be removed.
                    self.listThreads.pop(index)

                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.insert(index, thread)
                    self.listThreads[index].start()
                    self.numThreads += 1

            # Wait for all scheduled threads for this timepoint to complete.
            for thread in self.listThreads:
                thread.join()

            # Wait for supervisor to signal the end of the timepoint.
            self.device.timepoint_done.wait()
            
            # Clear state for the next iteration.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            self.device.scripts = []
            self.listThreads = []
            
            # Synchronize with all other devices.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """A reusable barrier for thread synchronization using semaphores."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        self.phase1()
        self.phase2()

    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
