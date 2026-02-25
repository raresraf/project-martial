"""
This script implements a distributed device simulation using a producer-consumer
pattern with a thread pool. It's a more complex but robust approach to the
concurrency requirements of the simulation compared to other versions.

The architecture involves a main 'DeviceThread' acting as a producer that places
computational tasks onto a shared queue. A pool of 'WorkerThread' instances
act as consumers, processing tasks from the queue in parallel. Semaphores are
used to synchronize the flow of tasks between the producer and consumers.
"""

import Queue
from threading import Event, Thread, Lock, Semaphore
# Assuming a barrier implementation using Condition variables is in barrier.py
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a device node, encapsulating its data, state, and the
    concurrency primitives required for its operation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and all its associated threading objects."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.devices = []
        self.event_setup = Event()
        self.barrier_device = None
        self.locations_lock = []
        self.data_set_lock = Lock() # Lock to protect writes to self.sensor_data
        self.device_shutdown_order = False

        # Concurrency primitives for the producer-consumer model
        self.work_queue = Queue.Queue()
        self.worker_barrier = ReusableBarrierCond(8) # For worker shutdown sync
        self.data_semaphore = Semaphore(value=0) # Counts tasks available for workers
        self.worker_semaphore = Semaphore(value=0) # Counts completed tasks

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Performs one-time global setup, creating and distributing shared resources."""
        if self.device_id == 0:
            self.barrier_device = ReusableBarrierCond(len(devices))
            for _ in range(25): # Assuming max 25 locations
                self.locations_lock.append(Lock())

            # Distribute shared resources to all devices
            for dev in devices:
                dev.devices = devices
                dev.barrier_device = self.barrier_device
                dev.locations_lock = self.locations_lock
                dev.event_setup.set() # Signal that setup is complete

    def assign_script(self, script, location):
        """Adds a script to the list of tasks for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set() # Signal that all scripts are assigned

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Thread-safely updates sensor data for a given location."""
        with self.data_set_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, acting as a "producer". It creates
    worker threads and then feeds them tasks via a shared queue.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main producer loop."""
        # Wait for the initial global setup to complete.
        self.device.event_setup.wait()

        # Create and start the pool of worker (consumer) threads.
        list_threads = []
        for i in range(8):
            thrd = WorkerThread(self.device, self.device.locations_lock, self.device.work_queue, i)
            list_threads.append(thrd)
        for thrd in list_threads:
            thrd.start()

        script_number = 0
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation

            # Wait for all scripts for this timepoint to be assigned.
            self.device.timepoint_done.wait()

            # Produce tasks: put them on the queue and signal the workers.
            for (script, location) in self.device.scripts:
                tup = (script, location, neighbours)
                self.device.work_queue.put(tup)
                # Release the semaphore to wake up one worker thread.
                self.device.data_semaphore.release()
                script_number += 1

            self.device.timepoint_done.clear()

            # Wait for all other devices to finish producing tasks for this timepoint.
            self.device.barrier_device.wait()

        # --- Shutdown sequence ---
        # 1. Wait for all dispatched tasks to be completed by workers.
        for _ in xrange(script_number):
            self.device.worker_semaphore.acquire()

        # 2. Signal workers to shut down.
        self.device.device_shutdown_order = True

        # 3. Unblock any workers that are still waiting on the semaphore.
        for _ in xrange(8):
            self.device.data_semaphore.release()

        # 4. Join all worker threads.
        for thrd in list_threads:
            thrd.join()

class WorkerThread(Thread):
    """
    A "consumer" thread that takes tasks from a queue and executes them.
    """
    def __init__(self, device, locations_lock, work_queue, worker_id):
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.device = device
        self.locations_lock = locations_lock
        self.work_queue = work_queue
        self.worker_id = worker_id

    def run(self):
        """Main consumer loop."""
        while True:
            # Block until the producer signals that a task is available.
            self.device.data_semaphore.acquire()

            # Check for the shutdown signal after being woken up.
            if self.device.device_shutdown_order is True:
                break

            # Get a task from the queue.
            (script, location, neighbours) = self.work_queue.get()

            # Execute the task.
            with self.locations_lock[location]:
                script_data = []
                # Gather data from neighbors and the local device.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Run the script and broadcast the results.
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
            
            # Signal back to the producer that one task is complete.
            self.device.worker_semaphore.release()
        
        # Synchronize with other workers on the same device before exiting.
        self.device.worker_barrier.wait()
