"""
This module provides a robust, multi-threaded framework for simulating a
network of communicating devices.

The design separates concerns into distinct classes:
- `Device`: A data-centric class representing a node in the network.
- `WorkPool`: A thread pool that manages a set of worker threads.
- `Worker`: A thread that executes a single computational task.
- `DeviceThread`: A master thread for each device that orchestrates the
  simulation steps, synchronizes with other devices, and dispatches tasks to
  its `WorkPool`.

The simulation proceeds in synchronized time steps, coordinated by a global
barrier.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a device node in the simulation network.

    This class primarily acts as a data container and an entry point for
    configuration. It holds sensor data, assigned scripts, and references to
    its control thread (`DeviceThread`) and thread pool (`WorkPool`).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # This event seems unused in this version.
        self.thread = None
        self.work_pool = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and starts the simulation components for all devices.

        This method, intended to be run by a single master device (id 0),
        creates a shared barrier and a shared set of locks. It then instantiates
        and starts the `WorkPool` and `DeviceThread` for every device in the
        simulation.
        """
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            lock_locations = []

            # Create a single, shared list of locks for all data locations
            # across all devices.
            for device in devices:
                for _ in xrange(len(device.sensor_data)):
                    lock_locations.append(Lock())

            # For each device, create its own thread pool and control thread.
            for device in devices:
                tasks_finish = Event()
                device.work_pool = WorkPool(tasks_finish, lock_locations)

                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals that all scripts for a
        timepoint have been assigned.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is the signal that the script assignment phase is over.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data. Not internally thread-safe."""
        return self.sensor_data[location] if location in self.sensor_data \
                else None

    def set_data(self, location, data):
        """Sets sensor data. Not internally thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's control thread to terminate."""
        self.thread.join()

class WorkPool(object):
    """
    Manages a pool of worker threads to execute tasks concurrently.
    """

    def __init__(self, tasks_finish, lock_locations):
        self.workers = []
        self.tasks = []
        self.current_task_index = 0
        self.lock_get_task = Lock()
        self.work_to_do = Event()
        self.tasks_finish = tasks_finish # Event to signal when all tasks are done.
        self.lock_locations = lock_locations
        self.max_num_workers = 8

        for i in xrange(self.max_num_workers):
            self.workers.append(Worker(self, self.lock_get_task, \
                    self.work_to_do, self.lock_locations))
            self.workers[i].start()

    def set_tasks(self, tasks):
        """
        Provides a new batch of tasks to the pool and wakes up worker threads.
        """
        self.tasks = tasks
        self.current_task_index = 0
        self.work_to_do.set() # Signal workers that there are tasks to process.

    def get_task(self):
        """
        Atomically retrieves the next available task from the list.

        Called by worker threads. When the last task is dispensed, it signals
        that the batch is finished by setting the `tasks_finish` event.
        """
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index]
            self.current_task_index = self.current_task_index + 1

            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear() # No more tasks in this batch.
                self.tasks_finish.set() # Signal completion to DeviceThread.

            return task
        else:
            return None

    def close(self):
        """Shuts down the work pool by terminating all worker threads."""
        self.tasks = []
        self.current_task_index = len(self.tasks)
        self.work_to_do.set() # Wake up workers so they can receive a None task.
        for worker in self.workers:
            worker.join()

class Worker(Thread):
    """
    A worker thread that executes a single task from the WorkPool.
    """

    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        Thread.__init__(self)
        self.work_pool = work_pool
        self.lock_get_task = lock_get_task
        self.work_to_do = work_to_do
        self.lock_locations = lock_locations

    def run(self):
        """Main loop to acquire and execute tasks."""
        while True:
            # Safely acquire a task from the pool.
            self.lock_get_task.acquire()
            self.work_to_do.wait() # Wait for the signal that tasks are available.
            task = self.work_pool.get_task()
            self.lock_get_task.release()

            if task is None:
                # A None task is the signal to shut down.
                break

            script, location, neighbours, self_device = task

            # --- Core Task Logic ---
            self.lock_locations[location].acquire()
            script_data = []

            # Gather data from neighbors.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from the local device.
            data = self_device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Run the script and distribute the results.
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                self_device.set_data(location, result)

            self.lock_locations[location].release()


class DeviceThread(Thread):
    """
    The main control thread that orchestrates a device's simulation step.

    This thread follows a clear synchronization pattern:
    1. Synchronize with all other devices at a global barrier.
    2. Wait for the supervisor to assign all scripts for the time step.
    3. Produce a batch of tasks and hand them to the WorkPool.
    4. Wait for the WorkPool to signal that all tasks are complete.
    5. Loop back to the barrier for the next time step.
    """

    def __init__(self, device, barrier, tasks_finish):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.tasks_finish = tasks_finish

    def run(self):
        while True:
            # 1. Wait for all devices to reach this point.
            self.barrier.wait()

            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # Shutdown signal from the supervisor.
                self.device.work_pool.close()
                break

            # 2. Wait for the signal that script assignment is complete.
            self.device.script_received.wait()

            # 3. Prepare and dispatch tasks to the work pool.
            tasks = []
            for (script, location) in self.device.scripts:
                tasks.append((script, location, neighbours, self.device))

            if tasks != []:
                # Hand off the batch of tasks to the WorkPool.
                self.device.work_pool.set_tasks(tasks)

                # 4. Wait for the WorkPool to confirm all tasks are done.
                self.tasks_finish.wait()
                self.tasks_finish.clear() # Reset for the next step.
            
            # Clear scripts and event for the next simulation step.
            self.device.script_received.clear()
            self.device.scripts = []