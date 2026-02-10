"""
@file device.py
@brief This file defines a simulated device for a distributed sensing environment, using a multi-level threading model.
@details The script implements a device that processes computational scripts based on sensor data.
         It features a main device thread that spawns a new pool of worker threads for each simulation time step.
         Synchronization across devices is managed by a shared barrier. A notable feature of this implementation
         is that worker threads requeue tasks after execution, leading to continuous processing within a time step.
"""


from threading import Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue


class Device(object):
    """
    @brief Represents a single device in the network.
    @details The Device class manages its local data, a queue of scripts to be executed, and synchronization primitives.
             It relies on a coordinator device (ID 0) to initialize shared locks and barriers.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary holding the local sensor data, keyed by location.
        @param supervisor An object for querying global state, like the device's neighbors.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # A queue to hold (script, location) tuples for processing by worker threads.
        self.queue = Queue()
        self.num_threads = 8

        # Shared synchronization objects, initialized by the coordinator device.
        self.location_locks = None
        self.lock = None
        self.barrier = None

        self.thread = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs collective setup for all devices in the simulation.
        @details The device with ID 0 acts as a coordinator, creating and distributing shared
                 synchronization objects (locks and a barrier) to all other devices.
        @param devices A list of all Device objects in the simulation.
        """
        
        # Pre-condition: Device with ID 0 is the coordinator for setting up shared resources.
        if self.device_id == 0:
            self.location_locks = {}
            self.lock = Lock()
            # Barrier to synchronize all devices at the end of a time step.
            self.barrier = ReusableBarrierCond(len(devices))


            for device in devices:
                if device.device_id != 0:
                    device.location_locks = self.location_locks
                    device.lock = self.lock
                    device.barrier = self.barrier
        # Each device starts its own main control thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device's processing queue.
        @param script The script to be executed.
        @param location The location context for the script.
        """
        
        if script is not None:
            # A global lock is used to safely create a new lock for a location if it's the first time we've seen it.
            with self.lock:
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()
            self.queue.put((script, location))
        else:
            # A `None` script acts as a "poison pill" to terminate the worker threads.
            # One is enqueued for each worker.
            for _ in range(self.num_threads):
                self.queue.put((None, None))

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location The location for which to retrieve data.
        @return The sensor data or None if the location is not found.
        """
        return self.sensor_data[
            location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data at a given location.
        @param location The location to update.
        @param data The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its main thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main control thread for a device, responsible for managing worker threads.
    @details In each simulation step, this thread fetches the device's neighbors and spawns a new
             set of WorkerThread instances to process the script queue. This pattern of creating
             and joining threads within a loop is a key feature of this design.
    """
    

    def __init__(self, device):
        
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        @brief The main loop for the device's control thread.
        """
        while True:
            
            # Pre-condition: Get the current set of neighbors for the new time step.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A `None` value for neighbors is the signal to terminate the simulation.
                break

            # Block Logic: Spawns a fresh pool of worker threads for each time step.
            # This is an unconventional design, as threads are typically long-lived.
            worker_threads = [WorkerThread(self.device, neighbours) for _ in
                              range(self.device.num_threads)]
            for thread in worker_threads:
                thread.start()
            for thread in worker_threads:
                thread.join()

            # Invariant: After all worker threads for the current time step have completed,
            # wait at the global barrier to synchronize with all other devices.
            self.device.barrier.wait()


class WorkerThread(Thread):
    """
    @brief A worker thread that executes scripts from the device's queue.
    @details This thread continuously fetches tasks, executes them, and then places them back
             onto the queue, effectively creating a busy-wait loop until a termination
             signal (a `None` script) is received.
    """
    

    def __init__(self, device, neighbours):
        
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours

    def run_script(self, script, location):
        """
        @brief Executes a single script.
        @details Gathers data from the local device and its neighbors, runs the script,
                 and broadcasts the result back to all involved devices.
        @param script The script to run.
        @param location The location context for the data.
        """
        script_data = []
        
        # Block Logic: Aggregate data from all neighbors for the specified location.
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            
            result = script.run(script_data)

            
            # Block Logic: Update the data on the local device and all neighbors with the script's result.
            for device in self.neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)

    def run(self):
        """
        @brief The main processing loop for the worker thread.
        """
        while True:
            # Block Logic: Retrieve a task from the shared queue.
            script, location = self.device.queue.get()
            if script is None:
                # The "poison pill" signals that the thread should terminate.
                return
            # A location-specific lock ensures that only one thread operates on a given location at a time.
            with self.device.location_locks[location]:
                self.run_script(script, location)
            
            # Inline: The task is placed back on the queue after processing. This will cause this
            # thread and others to re-process the same task repeatedly within the same time step
            # until a `None` is dequeued.
            self.device.queue.put((script, location))