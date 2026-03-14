
"""
This module simulates a distributed system of devices that execute scripts concurrently.

It uses a thread pool to manage script execution and various synchronization 
primitives, including Semaphores and a Condition variable, to coordinate the 
actions of the devices.
"""

from threading import Thread, Condition, Semaphore
from barrier import Barrier
from threadpool import ThreadPool

class Device(object):
    """
    Represents a single device in the simulated distributed system.

    Each device runs in its own thread (`DeviceThread`) and manages its own
    sensor data and scripts. It uses a `Condition` object to signal and wait for
    events like script assignment and timepoint completion.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's sensor data.
            supervisor: The supervisor managing this device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.data_semaphores = {loc : Semaphore(1) for loc in sensor_data}
        self.scripts = []

        self.new_script = False
        self.timepoint_end = False
        self.cond = Condition()

        self.barrier = None
        self.supervisor = supervisor
        self.thread = DeviceThread(self)

        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the barrier for synchronization if this is the root device.

        Args:
            devices (list): A list of all devices in the system.
        """
        if self.device_id == 0:
            
            self.barrier = Barrier(len(devices))
            for neigh in devices:
                if neigh.device_id != self.device_id:
                    neigh.set_barrier(self.barrier)

    def set_barrier(self, barrier):
        """
        Assigns a barrier to the device.

        Args:
            barrier (Barrier): The barrier object to be used for synchronization.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device and notifies the device's thread.

        Args:
            script: The script to be executed.
            location: The location associated with the script.
        """
        with self.cond:
            if script is not None:
                self.scripts.append((script, location))
                self.new_script = True
            else:
                self.timepoint_end = True
            self.cond.notifyAll()

    def timepoint_ended(self):
        """
        Blocks until a new script is assigned or the timepoint ends.

        Returns:
            bool: True if the timepoint has ended, False if a new script was received.
        """
        with self.cond:
            while not self.new_script and 
                  not self.timepoint_end:
                self.cond.wait()

            if self.new_script:
                self.new_script = False
                return False
            else:
                self.timepoint_end = False
                self.new_script = len(self.scripts) > 0
                return True

    def get_data(self, location):
        """
        Safely retrieves data from a sensor location.

        Acquires a semaphore for the location before reading.

        Args:
            location: The sensor location to read from.

        Returns:
            The data at the location, or None if the location is not found.
        """
        if location in self.sensor_data:
            self.data_semaphores[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Safely sets data at a sensor location.

        Releases the semaphore for the location after writing.

        Args:
            location: The sensor location to write to.
            data: The data to be written.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_semaphores[location].release()

    def shutdown(self):
        """Shuts down the device by joining its thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main thread of execution for a Device.

    This thread manages a pool of worker threads to execute scripts concurrently.
    It synchronizes with other devices using a barrier at the end of each timepoint.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    @staticmethod
    def run_script(own_device, neighbours, script, location):
        """
        Executes a single script, gathering data from neighbors.

        This static method is designed to be called by worker threads from the thread pool.

        Args:
            own_device (Device): The device executing the script.
            neighbours (list): A list of neighboring devices.
            script: The script to execute.
            location: The location for which to get/set data.
        """
        script_data = []

        
        for device in neighbours:
            if device is own_device:
                continue
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        
        data = own_device.get_data(location)


        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data)

            
            for device in neighbours:
                if device is not own_device:
                    device.set_data(location, result)

            
            own_device.set_data(location, result)

    def run(self):
        """
        The main loop for the device thread.

        It continuously waits for scripts, adds them to a thread pool for execution,
        and waits at a barrier for all devices to finish the timepoint.
        """
        
        
        pool_size = 8
        pool = ThreadPool(pool_size)

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            
            offset = 0
            while not self.device.timepoint_ended():
                scripts = self.device.scripts[offset:]
                for (script, location) in scripts:
                    pool.add_task(DeviceThread.run_script, self.device,
                                  neighbours, script, location)

                
                offset = len(scripts)

            
            pool.wait()

            
            self.device.barrier.wait()

        
        pool.terminate()


from Queue import Queue
from threading import Thread

class Worker(Thread):
    """
    A worker thread that consumes tasks from a queue.
    """

    def __init__(self, tasks):
        """
        Initializes the Worker.

        Args:
            tasks (Queue): A queue of tasks to be executed.
        """
        Thread.__init__(self)
        self.tasks = tasks

    def run(self):
        """
        Continuously gets and executes tasks from the queue.
        The thread terminates when it receives a task that raises a ValueError.
        """
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except ValueError:
                return
            finally:
                self.tasks.task_done()

class ThreadPool(object):
    """
    A simple thread pool implementation.
    """

    def __init__(self, num_threads):
        """
        Initializes the ThreadPool.

        Args:
            num_threads (int): The number of worker threads in the pool.
        """
        self.tasks = Queue(num_threads)
        self.workers = [Worker(self.tasks) for _ in range(num_threads)]

        for worker in self.workers:
            worker.start()

    def add_task(self, func, *args, **kargs):
        """
        Adds a task to the thread pool's queue.

        Args:
            func: The function to be executed.
            *args: Positional arguments for the function.
            **kargs: Keyword arguments for the function.
        """
        self.tasks.put((func, args, kargs))

    def wait(self):
        """Blocks until all tasks in the queue are completed."""
        self.tasks.join()

    def terminate(self):
        """
        Terminates all worker threads in the pool.

        This is done by adding a special task for each worker that causes it to exit.
        """
        self.wait()

        def raising_dummy():
            
            raise ValueError

        for _ in range(len(self.workers)):
            self.add_task(raising_dummy)
        for worker in self.workers:
            worker.join()
