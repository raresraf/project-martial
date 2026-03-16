"""
This module defines an alternative simulation framework for a network of devices.

It simulates devices that operate in synchronized time steps, process data,
and share results with neighbors. This implementation uses lower-level
synchronization primitives like Semaphores and a custom-built ReusableBarrier
to manage concurrency and time-step progression.

Note: This implementation contains a potential race condition in how worker
threads consume tasks from the shared 'canal' list.
"""

from threading import Event, Thread, Semaphore, Lock


class Device(object):
    """Represents a single device in the simulation.

    This device manages its own state and uses a pool of worker threads to
    execute scripts. It coordinates with other devices using a shared,
    custom-built barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's initial local data, keyed by location.
            supervisor: The simulation's supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        # A dictionary of locks to protect access to sensor data locations.
        self.locks = {}
        for x in sensor_data.keys():
            self.locks[x] = Lock()
        self.supervisor = supervisor
        # A semaphore to signal to worker threads that scripts are available.
        self.script_received = Semaphore(0)
        self.scripts = []
        self.scripts_number = 0
        # A list used as a work queue for scripts. Popping from this list by
        # multiple threads without a lock is a potential race condition.
        self.canal = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.time = 0

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Distributes the shared barrier from device 0 to all other devices."""
        self.devices = devices
        if self.device_id == 0:
            # Device 0 creates and distributes the shared barrier.
            self.barrier = ReusableBarrier(len(devices))
            for device in self.devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to the device for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A 'None' script is a sentinel. All scripts are now moved to the
            # 'canal' (work queue) and the semaphore is released for each one.
            for a in self.scripts:
                self.scripts_number += 1
                self.canal.insert(0, a)
                self.script_received.release()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a device, managing its worker pool and lifecycle."""

    def __init__(self, device):
        """Initializes the main thread and its associated worker threads."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads_number = 16
        # A semaphore to track the completion of tasks by worker threads.
        self.tasks_done = Semaphore(0)
        self.stop = False

        # Create the pool of worker threads.
        self.threads = [Thread(target=self.script_work, args=(i,))
                        for i in range(self.threads_number)]

    def start_threads(self):
        """Starts all worker threads."""
        for thread in self.threads:
            thread.start()

    def stop_threads(self):
        """Joins all worker threads to ensure clean shutdown."""
        for thread in self.threads:
            thread.join()

    def script_work(self, id):
        """The target function for worker threads."""
        while True:
            # Block until a script is made available by the main thread.
            self.device.script_received.acquire()
            # Potential Race Condition: Two threads could pop from the shared list concurrently.
            script, location = self.device.canal.pop()
            # A 'None' script is the signal to terminate the worker thread.
            if script is None:
                break

            script_data = []
            # Gather data from all neighbors that have data for the target location.
            all_data = [(device.get_data(location), device)
                        for device in self.device.neighbours
                        if device.get_data(location)]

            data = self.device.get_data(location)
            if data is not None:
                all_data.append((data, self.device))
            script_data = [x for x, _ in all_data]
            neighbours = [x for _, x in all_data]

            # Block Logic: If data was gathered, run the script and update neighbors.
            if len(script_data) > 1:
                result = script.run(script_data)
                for neighbour in neighbours:
                        # Lock the neighbor's data location to prevent concurrent updates.
                        with neighbour.locks[location]:
                            data = neighbour.get_data(location)
                            # Conditional Update: Only update if the new result is greater.
                            # This implies a max-finding or similar convergent algorithm.
                            if data < result:
                                neighbour.set_data(location, result)
            
            # Signal that this task is complete.
            self.tasks_done.release()

    def run(self):
        """The main simulation loop for the device."""
        self.device.neighbours = self.device.supervisor.get_neighbours()
        self.start_threads()
        # Initial synchronization to ensure all devices have started their threads.
        self.device.barrier.wait()

        while True:
            # Check for shutdown signal from the supervisor.
            if self.device.neighbours is None:
                self.stop = True
                # Send termination signals to all worker threads.
                for i in range(self.threads_number):
                    self.device.canal.insert(0, (None, None))
                    self.device.script_received.release()
                break

            # Wait until the supervisor has assigned all scripts for this time step.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            # Wait for all enqueued tasks to be completed by the worker threads.
            for i in range(self.device.scripts_number):
                self.tasks_done.acquire()
            self.device.scripts_number = 0
            # Synchronize with all other devices before starting the next time step.
            self.device.barrier.wait()
            # Get neighbors for the next time step.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.time += 1

        self.stop_threads()


class ReusableBarrier():
    """A from-scratch implementation of a reusable, two-phase barrier."""

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        # Phase 1: All threads arrive and wait.
        self.phase(self.count_threads1, self.threads_sem1)
        # Phase 2: All threads proceed and reset for the next use.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Implements one phase of the barrier."""
        with self.count_lock:
            count_threads[0] -= 1
            # The last thread to arrive opens the gate for all others.
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # All threads wait here until the gate is opened.
        threads_sem.acquire()
