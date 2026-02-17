"""
Models a distributed network of devices that execute scripts on sensor data.

This script simulates a network of devices that operate in synchronized time steps.
Each device runs a set of scripts that can read data from the device itself and its
neighbors, compute a result, and update the data in the same set of devices.
Concurrency is managed using threads, locks, semaphores, and a reusable barrier.
"""

from threading import Event, Thread, Lock, Semaphore
from reusable_barrier_condition import ReusableBarrier
import multiprocessing
import Queue

class Device(object):
    """
    Represents a single device in the simulated network.

    Each device has a unique ID, local sensor data, and is managed by a dedicated
    supervisor thread. It can be assigned computational scripts to execute.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a device.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor data.
            supervisor (Supervisor): An object that manages the network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.baariera = None
        self.dicti = {}
        self.device_master = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the master-worker relationship and the synchronization barrier.

        One device (ID 0) acts as the master, holding the central barrier. All other
        devices hold a reference to the master.

        Args:
            devices (list): A list of all Device objects in the network.
        """
        # Logic: The device with ID 0 is designated as the master.
        if self.device_id == 0:
            self.device_master = self
            self.baariera = ReusableBarrier(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    self.device_master = device
                    break

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device for a specific location.

        Args:
            script (Script): The script object to be executed.
            location (str): The location key for which the script is relevant.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Pre-condition: A lock is created for a location if one does not already exist.
            # This ensures that data updates for a given location are atomic.
            if not self.device_master.dicti.has_key(location):
                self.device_master.dicti[location] = Lock()
        else:
            # A None script signals the end of a timepoint's script assignments.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device.

        Args:
            location (str): The data location key.

        Returns:
            The data associated with the location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location on this device.

        Args:
            location (str): The data location key.
            data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread."""
        self.thread.join()

class ThreadExecutor(Thread):
    """
    A worker thread that executes scripts for a device.

    It fetches tasks from a queue, gathers data from neighboring devices,
    runs the script, and distributes the results.
    """

    def __init__(self, device_thd):
        """Initializes the worker thread.

        Args:
            device_thd (DeviceThread): The parent device thread that manages this worker.
        """
        Thread.__init__(self)
        self.device_thd = device_thd

    def run(self):
        """The main loop for the worker thread."""
        while True:
            # Block Logic: Waits for a script task to be available in the queue.
            self.device_thd.sem_produce.acquire()
            item = self.device_thd.coada.get()

            # A None item is a signal to terminate the worker thread.
            if item is None:
                break
            
            neighbours, script, location = item

            script_data = []

            # Invariant: Acquires a lock for the specific location to ensure exclusive access
            # during data aggregation and update, preventing race conditions.
            self.device_thd.device.device_master.dicti[location].acquire()

            # Gathers data from all neighbors for the specified location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gathers data from the local device as well.
            data = self.device_thd.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: Only run the script if there is data to process.
            if script_data:
                result = script.run(script_data)
                # Distributes the computed result back to all neighbors.
                for device in neighbours:
                    device.set_data(location, result)
                # Updates the local device's data with the result.
                self.device_thd.device.set_data(location, result)
            
            self.device_thd.device.device_master.dicti[location].release()

class DeviceThread(Thread):
    """
    The main control thread for a single Device.

    Manages a pool of worker threads (ThreadExecutor) and orchestrates the
    device's operation across synchronized time steps.
    """
    def create_workers(self, device_thd):
        """Creates and starts the pool of worker threads."""
        lista_workers = []
        # Heuristic: Creates a number of worker threads based on CPU count.
        for _ in xrange(self.numar_proc):
            aux_t = ThreadExecutor(device_thd)
            lista_workers.append(aux_t)
        for thd in lista_workers:
            thd.start()
        return lista_workers

    def __init__(self, device):
        """Initializes the device's main thread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.numar_proc = multiprocessing.cpu_count() * 6
        self.sem_produce = Semaphore(0)
        self.coada = Queue.Queue(maxsize=0)
        self.lista_workers = self.create_workers(self)

    def run(self):
        """
        The main synchronization loop for the device.

        This loop represents the progression of time in the simulation. In each step,
        it waits for scripts, queues them for execution, and then synchronizes with
        all other devices using a barrier before starting the next step.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            # A None value for neighbors is the signal to shut down the entire device.
            if neighbours is None:
                # Sends termination signals to all worker threads.
                for _ in xrange(self.numar_proc):
                    self.coada.put(None)
                    self.sem_produce.release()

                for item in self.lista_workers:
                    item.join()
                break
            
            # Waits until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Queues all assigned scripts for the worker threads to execute.
            for (script, location) in self.device.scripts:
                item = (neighbours, script, location)
                self.coada.put(item)
                self.sem_produce.release()
            
            # Invariant: All devices wait at the barrier, ensuring that all script
            # executions for the current timepoint are completed before any device
            # proceeds to the next timepoint.
            self.device.device_master.baariera.wait()
            
            # Resets the event for the next timepoint.
            self.device.timepoint_done.clear()

class ReusableBarrier():
    """
    A custom implementation of a reusable barrier for thread synchronization.
    
    Allows a specified number of threads to block until all have called the wait() method,
    at which point they are all released and the barrier resets for re-use.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Causes a thread to block until all participating threads have also called wait()."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # When the last thread arrives, notify all waiting threads.
            self.cond.notify_all()
            # Reset the barrier for the next use.
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class MyThread(Thread):
    """
    An example thread class to demonstrate the usage of the ReusableBarrier.
    
    This class is not part of the main device simulation logic.
    """
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
 
    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",

# Note: The original file had a redundant import of 'threading' at the end,
# which has been omitted in this commented version for clarity.