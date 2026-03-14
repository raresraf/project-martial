"""
@633114d2-dc94-46ad-9573-c3a169ef4d07/device.py
@brief Implements a distributed simulation or data processing system with a master-worker threading model.

This module defines three core classes:
- `ReusableBarrier`: A custom barrier implementation for synchronizing multiple threads.
- `Device`: Represents a computational node, managing sensor data, scripts, and orchestrating
  master and worker threads for processing and synchronization.
- `DeviceThreadMaster`: The main thread for a `Device` instance, responsible for managing
  timepoints, fetching scripts, and distributing them to worker threads.
- `DeviceThreadWorker`: Executes assigned scripts on specific data locations, handling data
  access and synchronization using locks.

The system uses `threading.Event` for event signaling, `threading.Thread` for concurrency,
`threading.Lock` for protecting shared data access, and `threading.Condition` for the custom barrier.
`Queue` is used for task distribution.

Algorithm:
- Master-Worker Model: Each Device has a master thread that distributes tasks to a pool of worker threads.
- Timepoint synchronization: Devices synchronize at discrete timepoints using a custom reusable barrier.
- Concurrent script execution: `DeviceThreadWorker`s execute scripts in parallel.
- Distributed locking: Location-specific locks ensure data consistency across devices.
- Task Queue: Tasks (scripts) are placed in a queue and processed by available workers.

Time Complexity:
- `__init__`: O(1)
- `setup_devices`: O(D * L) where D is number of devices and L is number of locations.
- `run` (DeviceThreadMaster): O(T * N) where T is timepoints, N is number of tasks.
- `run` (DeviceThreadWorker): O(S * N * L) where S is scripts per worker, N is neighbors, L is locations.
Space Complexity:
- `Device`: O(L) for sensor_data and locks, O(W) for workers.
- `ReusableBarrier`: O(1).
- `DeviceThreadMaster`: O(1).
- `DeviceThreadWorker`: O(1).
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty


class ReusableBarrier(object):
    """
    @brief A reusable barrier synchronization primitive for multiple threads.

    This barrier allows a specified number of threads to wait until all have
    reached a certain point, then releases them all simultaneously. It can be
    reused for subsequent synchronization points.
    """
    

    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrier.

        @param num_threads: The number of threads that must reach the barrier
                            before all waiting threads are released.
        """
        
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Current count of threads yet to reach the barrier.
        self.cond = Condition() # Condition variable to manage waiting and notification.

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all
               `num_threads` have arrived.

        If this thread is the last to arrive, it notifies all other waiting
        threads and resets the barrier for future use.
        """
        
        self.cond.acquire() # Acquire the lock associated with the condition variable.
        self.count_threads -= 1 # Decrement the count of threads yet to reach.
        
        # Block Logic: If this is the last thread to arrive at the barrier.
        if self.count_threads == 0:
            self.cond.notify_all() # Release all waiting threads.
            self.count_threads = self.num_threads # Reset the barrier for reuse.
        else:
            self.cond.wait() # This thread waits until notified by the last thread.
        self.cond.release() # Release the lock.


class Device(object):
    """
    @brief Represents a single computational device (node) in the distributed system.

    Each device manages its sensor data, assigned scripts, and orchestrates
    its processing using a master thread and a pool of worker threads. It also
    handles synchronization with other devices.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: Unique identifier for the device.
        @param sensor_data: Dictionary containing initial sensor data for various locations.
        @param supervisor: Reference to the supervisor object for inter-device communication.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []


        self.neighbours = []

        self.barrier = None
        self.locks = []
        self.timepoint_done = Event()
        self.tasks_ready = Event()
        self.tasks = Queue()
        self.simulation_ended = False

        
        self.master = DeviceThreadMaster(self)
        self.master.start()

        
        self.workers = []
        for i in xrange(8):
            worker = DeviceThreadWorker(self, i)
            self.workers.append(worker)
            worker.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device [<device_id>]".
        """
        
        return "Device [%d]" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared resources (barrier and locks) across all devices.

        This method is designed to be called by a single designated device (device_id 0)
        to initialize global synchronization primitives and distribute them.

        @param devices: A list of all Device instances in the system.
        """
        

        
        # Block Logic: Only the device with ID 0 is responsible for setting up global resources.
        if self.device_id == 0:
            # Initializes a reusable barrier for all devices to synchronize timepoints.
            barrier = ReusableBarrier(len(devices))
            # Initializes a list of locks, one for each data location (assuming 24 locations).
            locks = [Lock() for _ in xrange(24)]
            # Distributes the initialized barrier and locks to all other devices.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the device's script list.
        If no script is provided (None), it signals that all scripts for the current
        timepoint have been assigned, by setting `timepoint_done`.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location relevant to the script.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set() # Signal that all scripts for this timepoint have been assigned.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The identifier for the data location.
        @return The sensor data for the location, or None if the location is not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets (updates) sensor data for a given location.

        @param location: The identifier for the data location.
        @param data: The new data to set for the location.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its master and worker threads.
        """
        
        self.master.join() # Wait for the master thread to finish.
        for worker in self.workers:
            worker.join() # Wait for all worker threads to finish.


class DeviceThreadMaster(Thread):
    """
    @brief The master thread for a Device instance.

    This thread is responsible for advancing simulation timepoints,
    fetching neighbors from the supervisor, distributing scripts (tasks)
    to a queue for worker threads, and coordinating synchronization
    across devices using a barrier.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThreadMaster.

        @param device: The Device instance that this master thread manages.
        """
        
        Thread.__init__(self, name="Device [%d] Thread Master" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThreadMaster.

        Manages timepoint progression, script acquisition, task distribution
        to worker threads, and inter-device synchronization.
        """
        while True:
            # Block Logic: Get the list of neighboring devices from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: If no neighbors are returned (e.g., simulation has ended),
            # signal termination to workers and break the loop.
            if self.device.neighbours is None:
                self.device.simulation_ended = True # Set flag for workers to terminate.
                self.device.tasks_ready.set() # Wake up workers so they can see the termination flag.
                break

            # Block Logic: Wait until scripts for the current timepoint are assigned to this device.
            self.device.timepoint_done.wait()

            # Block Logic: Put all assigned scripts into the task queue for workers to process.
            for task in self.device.scripts:
                self.device.tasks.put(task)

            # Block Logic: Signal that tasks are ready for worker threads to start processing.
            self.device.tasks_ready.set()

            # Block Logic: Wait for all tasks in the queue to be marked as done by worker threads.
            self.device.tasks.join()

            # Block Logic: Reset events for the next timepoint.
            self.device.tasks_ready.clear() # Clear tasks_ready event.
            self.device.timepoint_done.clear() # Clear timepoint_done event.

            # Block Logic: Synchronize with other devices at the barrier, marking the end of the current timepoint.
            self.device.barrier.wait()


class DeviceThreadWorker(Thread):
    """
    @brief A worker thread for a Device, responsible for executing assigned scripts.

    Workers fetch tasks from a shared queue (`self.device.tasks`), execute them,
    and handle data access and propagation, ensuring thread-safe operations
    using location-specific locks.
    """
    

    def __init__(self, device, thread_id):
        """
        @brief Initializes a new DeviceThreadWorker.

        @param device: The parent Device instance this worker belongs to.
        @param thread_id: A unique identifier for this worker thread within its device.
        """
        
        Thread.__init__(self, name="Device [%d] Thread Worker [%d]" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        @brief The main execution loop for the DeviceThreadWorker.

        Continuously fetches tasks from the device's task queue, processes them,
        and ensures thread-safe access to shared data by acquiring and releasing
        appropriate locks.
        """
        
        while not self.device.simulation_ended: # Loop as long as the simulation is not ended.
            # Block Logic: Wait for tasks to be ready in the queue.
            self.device.tasks_ready.wait()

            try:
                # Block Logic: Attempt to get a task (script and location) from the queue without blocking.
                script, location = self.device.tasks.get(block=False)

                # Block Logic: Acquire the global lock for the specific data location to ensure exclusive access.
                self.device.locks[location].acquire()

                script_data = []

                # Block Logic: Collect data from neighboring devices for the specified location.
                # Note: This implementation assumes neighbors' data can be read without explicit
                # locks on their side, or that their internal `get_data` handles its own locking.
                # However, for consistency with `set_data`, it might be safer to acquire neighbor's `lock_data`.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Block Logic: Collect data from the local device for the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If any data was collected, execute the script and propagate results.
                if len(script_data) > 0:
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Propagate the script's result back to neighboring devices.
                    # This assumes neighbors' `set_data` handles its own locking if needed.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Update the local device's data with the script's result.
                    self.device.set_data(location, result)

                # Block Logic: Release the global lock for the data location.
                self.device.locks[location].release()

                # Block Logic: Mark the task as done in the queue, allowing the master thread to track completion.
                self.device.tasks.task_done()
            except Empty:
                # If the queue is empty, it means all current tasks for this timepoint have been processed.
                # The worker will then wait again on `tasks_ready`.
                pass
