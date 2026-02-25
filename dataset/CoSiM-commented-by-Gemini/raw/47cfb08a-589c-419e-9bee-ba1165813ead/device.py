"""
Models a device in a distributed simulation using a custom two-phase barrier
and a thread-per-task execution model.

This module defines a simulation where devices execute computational scripts.
Synchronization between devices is achieved with a custom-built two-phase barrier.
Within a single device, parallelism is achieved by spawning a new thread for each
assigned script.

Classes:
    ReusableBarrier: A custom two-phase thread barrier for synchronization.
    Device: Represents a single computational node in the network.
    MyThread: A worker thread that executes exactly one script.
    DeviceThread: The main control thread for a Device.
"""

from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    A custom, two-phase implementation of a reusable barrier.

    Threads calling wait() must pass through two distinct synchronization phases,
    each controlled by a separate counter and semaphore. This ensures that no
    thread can start a new 'wait' cycle until all threads have exited the
    previous one.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Use lists for counters to pass them by reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads reach the barrier."""
        # The barrier consists of two distinct synchronization phases.
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases the semaphores for all other threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of the barrier.
                count_threads[0] = self.num_threads
        # All threads will block here until the last thread releases the semaphores.
        threads_sem.acquire()

class Device(object):
    """
    Represents a device that manages its data and executes assigned scripts.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barrier = None
        self.list_thread = []
        self.thread.start()
        # A list to hold locks for specific data locations.
        self.location_lock = [None] * 100

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes shared resources, primarily the synchronization barrier."""
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script to the device and handles lock initialization for the script's location.
        """
        flag = 0
        if script is not None:
            self.scripts.append((script, location))
            
            # This block attempts to lazily initialize and share a lock for the given location
            # among all devices. This is a complex and potentially racy approach.
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        flag = 1
                        break
                if flag == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            # A None script signals that all scripts for the timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates the sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class MyThread(Thread):
    """
    A worker thread designed to execute a single script task.
    """
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script. It locks the location, gathers data, runs the
        script, and propagates the result to itself and its neighbors.
        """
        self.device.location_lock[self.location].acquire()
        script_data = []
        
        # Gather data from neighboring devices.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        # Gather data from the parent device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Run the script on the collected data.
            result = self.script.run(script_data)
            
            # Update the data on all neighboring devices and the parent device.
            for device in self.neighbours:
                device.set_data(self.location, result)
                
            self.device.set_data(self.location, result)
        self.device.location_lock[self.location].release()


class DeviceThread(Thread):
    """
    The main control thread for a device, managing the execution of scripts
    and synchronization with other devices.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            # Get neighbors from the supervisor; a None value is a shutdown signal.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the signal that all scripts for this timepoint have been assigned.
            self.device.timepoint_done.wait()

            # --- Script Execution Phase ---
            # Create a new thread for each assigned script.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            # Start all worker threads for this timepoint.
            for thread_elem in self.device.list_thread:
                thread_elem.start()
            # Wait for all worker threads to complete.
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            self.device.list_thread = []

            # --- Synchronization Phase ---
            self.device.timepoint_done.clear()
            # Wait at the barrier for all other devices to finish the timepoint.
            self.device.barrier.wait()