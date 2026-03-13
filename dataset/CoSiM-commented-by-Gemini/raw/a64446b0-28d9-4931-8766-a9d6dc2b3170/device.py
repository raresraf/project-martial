"""
This module defines a simulated Device using a master-worker threading model
for a distributed sensor network simulation.

The Device class represents a node that receives scripts from a supervisor,
distributes them as tasks to a pool of worker threads, and synchronizes with
other devices at the end of each simulation timepoint.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty


class ReusableBarrier(object):
    """
    A reusable barrier implemented using a Condition variable.

    This allows a set of threads to wait for each other to reach a certain point
    of execution before proceeding. It automatically resets after all threads
    have passed.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads  # Countdown for threads arriving at the barrier.
        self.cond = Condition()  # The underlying condition variable.

    def wait(self):
        """
        Causes the calling thread to wait at the barrier.

        The thread will block until `num_threads` have called this method.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread to arrive: notify all waiting threads and reset the counter.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Not the last thread, so wait to be notified.
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Represents a single device using a master-worker threading model.

    The master thread coordinates with the supervisor and manages the task queue,
    while a pool of worker threads executes the scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor data.
            supervisor (Supervisor): The central supervisor controlling the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []  # List to hold (script, location) tuples for the current timepoint.
        self.neighbours = [] # List of neighboring devices.

        # --- Synchronization Primitives ---
        self.barrier = None  # A global barrier for all devices in the simulation.
        self.locks = []      # A list of shared locks for each data location.
        self.timepoint_done = Event()  # Signals that script assignment for a timepoint is complete.
        self.tasks_ready = Event()     # Signals to worker threads that tasks are in the queue.
        self.tasks = Queue()           # Queue to hold tasks for worker threads.
        self.simulation_ended = False  # Flag to signal worker threads to terminate.

        # --- Threads ---
        self.master = DeviceThreadMaster(self)
        self.master.start()

        self.workers = []
        # Create a pool of 8 worker threads.
        for i in xrange(8):
            worker = DeviceThreadWorker(self, i)
            self.workers.append(worker)
            worker.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device [%d]" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources (barrier, locks) for all devices.
        
        This is typically called once by a single device (e.g., device 0) to
        ensure all devices share the same synchronization objects.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            locks = [Lock() for _ in xrange(24)] # Assuming 24 locations.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed. A `None` script signals the end of assignment.

        Args:
            script (Script): The script object to execute.
            location (str): The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Sentinel script: signal that all scripts for this timepoint are received.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates the sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all master and worker threads to shut down the device."""
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceThreadMaster(Thread):
    """
    The master thread that coordinates the device's lifecycle for each timepoint.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device [%d] Thread Master" % device.device_id)
        self.device = device

    def run(self):
        """
        The main control loop for the device.
        """
        while True:
            # Get current neighbors from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # `None` from supervisor signals the end of the simulation.
            if self.device.neighbours is None:
                self.device.simulation_ended = True
                self.device.tasks_ready.set() # Wake up workers to let them terminate.
                break

            # Wait until the supervisor has finished assigning all scripts for this timepoint.
            self.device.timepoint_done.wait()

            # --- Task Distribution ---
            # Place all assigned scripts into the task queue for the workers.
            for task in self.device.scripts:
                self.device.tasks.put(task)
            self.device.scripts = [] # Clear scripts for the next round.

            # Signal to worker threads that the queue has tasks.
            self.device.tasks_ready.set()

            # Wait for the worker threads to process all tasks in the queue.
            self.device.tasks.join()

            # --- Cleanup for next timepoint ---
            self.device.tasks_ready.clear()
            self.device.timepoint_done.clear()

            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()


class DeviceThreadWorker(Thread):
    """
    A worker thread that executes script tasks from a shared queue.
    """
    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device [%d] Thread Worker [%d]" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        The main loop for a worker thread.
        """
        while not self.device.simulation_ended:
            # Wait until the master signals that tasks are available.
            self.device.tasks_ready.wait()

            # If the simulation has ended, break the loop immediately.
            if self.device.simulation_ended:
                break

            try:
                # Fetch a task from the queue without blocking.
                script, location = self.device.tasks.get(block=False)

                # Acquire the lock for the location to ensure atomic operations.
                self.device.locks[location].acquire()

                # --- Data Aggregation and Processing ---
                script_data = []
                # Collect data from neighbors.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Collect data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Run the script and propagate the result if there is data.
                if len(script_data) > 0:
                    result = script.run(script_data)
                    
                    # Update data on all neighbors and the local device.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                self.device.locks[location].release()
                
                # Signal that this task is complete.
                self.device.tasks.task_done()
            except Empty:
                # This can happen if multiple workers wake up but one clears the queue.
                pass
