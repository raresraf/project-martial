"""
Models a distributed system of communicating devices for a sensor network simulation.

This module implements a simulation where multiple devices, each running in its own
thread context, process scripts on shared sensor data. It features a different
threading model where each device's main thread spawns temporary worker threads
for each timepoint. Synchronization is managed by a reusable barrier and a
coarse-grained locking mechanism.
"""

from threading import Thread, Lock, Semaphore, Event

# Global dictionary to store ReusableBarrier instances, acting as a singleton factory.
dictionary = {}


class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a two-phase protocol. A notable implementation detail
    is the use of single-element lists for counters to simulate pass-by-reference
    behavior for integers in the `phase` method.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        # Counters for each phase, wrapped in a list to be mutable across methods.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the two-phase barrier protocol.

        Args:
            count_threads (list): A single-element list holding the current thread count.
            threads_sem (Semaphore): The semaphore to block/release threads for this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread to arrive releases all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for the next use.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

def pseudo_singleton(count):
    """
    A factory function that ensures only one barrier is created per thread count.

    Args:
        count (int): The number of threads the barrier should support.

    Returns:
        ReusableBarrier: The shared barrier instance for the given count.
    """
    global dictionary
    if not dictionary.has_key(count):
        dictionary[count] = ReusableBarrier(count)
    return dictionary[count]

class Device(object):
    """
    Represents a single device in the simulated network.
    
    Each device has a main control thread (`DeviceThread`) that coordinates its
    actions during each timepoint of the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a device and starts its main control thread."""
        self.lock = Lock() # Note: this lock is used by worker threads in a coarse manner.
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self, 0)
        self.thread.start()

    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared barrier for all devices in the simulation.
        
        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.barrier = pseudo_singleton(len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device in the current timepoint.

        Args:
            script (object): The script object to execute.
            location (int): The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script signals the end of assignments for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main device thread to ensure a clean shutdown."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a Device."""

    def listoflists(self, list, number):
        """
        Partitions a list into a specified number of almost-equal-sized sublists.
        
        Args:
            list (list): The list to partition.
            number (int): The number of partitions to create.
        
        Returns:
            list: A list of lists.
        """
        size = int(len(list) / number)
        chunks = []
        for i in xrange(number):
            chunks.append(list[0 + size * i: size * (i + 1)])
        for i in xrange(len(list) - size * number):
            chunks[i % number].append(list[(size * number) + i])
        return chunks

    def __init__(self, device, id):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = id

    class Instance(Thread):
        """
        A temporary worker thread created by DeviceThread to execute a chunk of scripts.
        """
        def __init__(self, device, listfromlist, neighbours):
            Thread.__init__(self, name="Instance")
            self.device = device
            self.listfromlist = listfromlist
            self.neighbours = neighbours

        def set_data_for_all_devices(self, location, result):
            """
            Updates data on the local device and all its neighbors.

            Note: This method uses a single lock from the parent device to protect
            updates on all involved devices, which can be a performance bottleneck.
            """
            for device in self.neighbours:
                self.device.lock.acquire()
                device.set_data(location, max(result, device.get_data(location)))
                self.device.lock.release()
            self.device.lock.acquire()
            self.device.set_data(location, max(result, self.device.get_data(location)))


            self.device.lock.release()

        def run(self):
            script_data = []
            for (script, location) in self.listfromlist:
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                if script_data != []:
                    result = script.run(script_data)
                    self.set_data_for_all_devices(location, result)


    def run(self):
        """
        The main execution loop for the device's control logic.
        
        In each timepoint, it partitions the assigned scripts, spawns temporary
        worker threads (`Instance`) to execute them, waits for them to complete,
        and then synchronizes with all other devices using a global barrier.

        Note: The loop for creating `Instance` threads has a potential bug. It
        always loops 8 times, which can lead to an `IndexError` if the number
        of script chunks (`list_of_scripts`) is less than 8 but greater than 0.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()


            if neighbours is None:
                break
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            list_of_scripts = self.listoflists(self.device.scripts, 8)
            instances = []
            for i in range(8):
                if len(list_of_scripts):
                    instances.append(self.Instance(self.device, list_of_scripts[i], neighbours))
            for index in range(len(instances)):
                instances[index].start()
            for index in range(len(instances)):
                instances[index].join()
            self.device.barrier.wait()
