"""
A simulation framework for a network of communicating devices that employs
intra-device parallelism for script execution.

This script defines a system where each `Device` spawns a pool of worker
threads to process its assigned scripts in parallel during each time-step.
Synchronization is managed globally between devices and locally within each
device's thread pool.

The main components are:
- ReusableBarrier: A synchronization primitive to ensure a group of threads
  waits for each other.
- Device: Represents a node in the network. It holds data and scripts.
- DeviceThread: The main control loop for a Device. It partitions the assigned
  scripts and distributes them to a pool of worker `Instance` threads.
- DeviceThread.Instance: A nested worker thread class that executes a subset
  of a device's scripts. It gathers data, runs a computation, and updates
  data on itself and its neighbors, using a specific `max` aggregation logic.

Key Features:
- A global dictionary stores and shares barrier objects among devices.
- Each device uses a thread pool (`Instance` threads) to parallelize its workload.
- A custom `max` function is used during data dissemination, suggesting an
  aggregation or consensus-finding algorithm (e.g., finding a max value).
"""
from threading import *

MAX_THREADS = 8

class ReusableBarrier(object):
    """
    Implements a reusable barrier using two semaphores for two-phase signaling.

    Attributes:
        num_threads (int): The number of threads to wait for.
        count_threads1, count_threads2 (list): Mutable counters for each phase,
                                               wrapped in a list to be passed
                                               by reference.
        count_lock (Lock): A mutex to protect counter access.
        threads_sem1, threads_sem2 (Semaphore): Semaphores for each phase.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads arrive."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier protocol."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread to arrive unblocks all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

# Global dictionary to store and reuse ReusableBarrier objects.
dictionary = dict()

class Device(object):
    """
    Represents a single device in the distributed system simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.lock = Lock() # Lock for this device's data.
        self.barrier = None # The global barrier for all devices.
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []

        self.timepoint_done = Event()
        self.locks = dict()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for all devices in the simulation.
        
        It uses a global dictionary to ensure all devices share the same
        barrier instance for a given simulation size.
        """
        if not dictionary.has_key(len(devices)):
            dictionary[len(devices)] = ReusableBarrier(len(devices))
        self.barrier = dictionary[len(devices)]

    def assign_script(self, script, location):
        """Assigns a script to be executed by this device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of script assignment for a timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device, managing a pool of worker threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def build_lists(self, lst, number):
        """
        Partitions a list into a specified number of sublists.
        
        This is used to distribute scripts evenly among worker threads.
        
        Args:
            lst (list): The list to partition.
            number (int): The number of partitions to create.
            
        Returns:
            A list of sublists.
        """
        result_list = []
        size = int(len(lst) / number)
        for i in range(number):
            row = size * i
            column = size * (i + 1)
            result_list.append(lst[row: column])
        # Distribute remaining elements one by one.
        for i in range(len(lst) - size * number):
            row = i % number
            column = i + (size * number)
            result_list[row].append(lst[column])
        return result_list

    class Instance(Thread):
        """
        A worker thread that executes a portion of a device's scripts.
        """
        def __init__(self, device, big_list, neighbours):
            Thread.__init__(self, name="Instance")
            self.neighbours = neighbours
            self.big_list = big_list # The assigned subset of scripts.
            self.device = device

        def run(self):
            """Executes each script in the assigned list."""
            for (script, location) in self.big_list:
                script_data = []
                # Gather data from neighbors.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                if script_data != []:
                    result = script.run(script_data)
                    # Disseminate result using a custom aggregation logic.
                    self.all_devices_set_data(location, result)

        def my_max(self, a, b):
            """A simple maximum function."""
            if a >= b:
                return a
            return b

        def all_devices_set_data(self, location, result):
            """
            Updates data on this device and its neighbors using a max function.

            This implies an aggregation strategy where the goal is to propagate
            the maximum value seen across the neighborhood. The lock ensures
            atomic updates.
            """
            for device in self.neighbours:
                with self.device.lock:
                    device.set_data(location, self.my_max(result, device.get_data(location)))
            with self.device.lock:
                result_data = self.my_max(result, self.device.get_data(location))
                self.device.set_data(location, result_data)

    def run(self):
        """
        Main orchestration loop for the device.
        """
        while True:
            threads_scripts = list()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals termination.
                break
            
            # Wait for the supervisor to signal all scripts are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            
            # Partition scripts among worker threads.
            list_of_scripts = self.build_lists(self.device.scripts, MAX_THREADS)
            
            # Create and start worker threads.
            for i in range(MAX_THREADS):
                if list_of_scripts != []:
                    threads_scripts.append(self.Instance(self.device, list_of_scripts[i], neighbours))
            for script_thread in threads_scripts:
                script_thread.start()
            
            # Wait for all local worker threads to complete.
            for script_thread in threads_scripts:
                script_thread.join()
            
            # --- Global Sync Point ---
            # Wait for all other devices to finish their time-step.
            self.device.barrier.wait()
