"""
Models a network of communicating devices using a complex, multi-layered
threading model for a time-stepped simulation.

This module is another variant of the device simulation. Its architecture is
notably convoluted, employing three levels of threading (`DeviceThread`,
`Scripter`, `ScriptExecutor`) and several complex, and at times unclear,
synchronization mechanisms.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue


class Device(object):
    """
    Represents a single device, holding data and managing a complex lifecycle
    of threads for script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and its numerous synchronization primitives.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding the initial sensor data.
            supervisor: An external entity for managing the network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None
        
        # This lock seems intended to pause script assignment from the supervisor.
        self.script_running = Lock()
        self.timepoint_done = Event()
        
        # A dictionary of locks for each individual data location.
        self.data_locks = dict()
        
        self.queue = Queue()
        
        self.available_threads = 14

        for loc in sensor_data:
            self.data_locks.__setitem__(loc, Lock())

        # A global lock for the device, seemingly to serialize all data access.
        self.can_get_data = Lock()

        self.master = None
        self.script_over = False
        self.alive = True
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the barrier and sets the master device.

        In this implementation, every device creates its own barrier, but then
        points to device 0 as the 'master'. All `wait()` calls are directed
        to `self.master.barrier`, meaning only device 0's barrier is ever used.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.barrier = ReusableBarrier(len(devices))
        self.master = devices[0]

    def assign_script(self, script, location):
        """
        Assigns a script from the supervisor.

        The method appends the script to a list and also puts it on a queue.
        A 'None' script signals the end of assignments for the time step.

        Args:
            script: The script object to be executed.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.script_running.acquire()
            self.scripts.append((script, location))
            self.queue.put_nowait((script, location))
            self.script_received.set()
        else:
            # End of timepoint signaled by a None script.
            self.script_running.acquire()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data under a global device lock.

        @warning This method's locking scheme (`can_get_data`) conflicts with
                 `get_device_data`, which uses per-location locks. This can
                 lead to complex lock interactions or bottlenecks.
        """
        self.can_get_data.acquire()
        return_value = self.sensor_data[location] if location in self.sensor_data else None
        self.can_get_data.release()
        return return_value

    def get_device_data(self, location):
        """
        Retrieves data for a given location using a per-location lock.
        This is the method used by worker threads.
        """
        if location not in self.sensor_data:
            return None
        
        self.data_locks.get(location).acquire()
        new_data = self.sensor_data[location]
        self.data_locks.get(location).release()
        return new_data

    def set_data(self, location, data):
        """Sets data for a given location using a per-location lock."""
        if location in self.sensor_data:
            self.data_locks.get(location).acquire()
            self.sensor_data[location] = data
            self.data_locks.get(location).release()

    def shutdown(self):
        """Shuts down the device by joining its main control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main, highly complex control thread for a device.

    This thread's `run` method contains a convoluted simulation loop that
    spawns and manages intermediate threads (`Scripter`) for each time step.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop, executed for each time step."""
        while True:
            # Acquires a global lock for the entire duration of the time step.
            self.device.can_get_data.acquire()
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # Shutdown signal from the supervisor.
                self.device.master.barrier.wait() # Final synchronization.
                self.device.can_get_data.release()
                return

            # Spawns a new 'Scripter' thread to manage workers for this time step.
            script_instance = Scripter(self.device, neighbours)
            script_instance.start()

            # Waits for the supervisor to signal the end of script assignment.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Signals the Scripter thread that it should terminate its workers.
            self.device.script_over = True
            self.device.script_received.set()

            script_instance.join()

            # @warning The logic here appears flawed. It re-queues all scripts
            #          after the worker threads that were supposed to process
            #          them have already been joined and terminated.
            for (script, location) in self.device.scripts:
                self.device.queue.put_nowait((script, location))

            self.device.script_over = False

            # Synchronize with all other devices via the master's barrier.
            self.device.master.barrier.wait()

            # Release locks to allow the next time step to begin.
            self.device.can_get_data.release()
            self.device.script_running.release()


class Scripter(Thread):
    """
    A middle-manager thread that spawns and manages a pool of worker threads.
    An instance of this class is created for each time step.
    """

    def __init__(self, device, neighbours):
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """Starts a pool of ScriptExecutor workers and manages their lifecycle."""
        list_executors = []

        for iterator in range(1, self.device.available_threads):
            executor = ScriptExecutor(self.device, self.device.queue, self.neighbours, iterator)
            list_executors.append(executor)
            executor.start()

        while True:
            # Waits for a signal, either a new script or the end of the time step.
            self.device.script_received.wait()
            self.device.script_received.clear()

            if self.device.script_over:
                # End-of-timepoint signal received from DeviceThread.
                # Send a "poison pill" to each worker to terminate it.
                for iterator in range(1, self.device.available_threads):
                    self.device.queue.put((None, None))

                for executor in list_executors:
                    executor.join()

                # @note The queue is reset here, but its contents were likely
                #       already processed by the workers.
                self.device.queue = Queue(-1)
                return

            self.device.script_running.release()


class ScriptExecutor(Thread):
    """
    A worker thread that executes a single script.
    """

    def __init__(self, device, queue, neighbours, identifier):
        Thread.__init__(self, name="Script Executor %d" % identifier)
        self.device = device
        self.queue = queue
        self.neighbours = neighbours

    def run(self):
        """
        Performs a distributed read-compute-write cycle for a script.
        The loop terminates when a 'None' script (poison pill) is received.
        """
        while True:
            (script, location) = self.queue.get()
            if script is None:
                return

            script_data = []
            
            # Read phase: Collect data from neighbors and self using per-location locks.
            for device in self.neighbours:
                data = device.get_device_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_device_data(location)
            if data is not None:
                script_data.append(data)

            # Compute and Write phase.
            if script_data != []:
                result = script.run(script_data)
                
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)


class ReusableBarrier:
    """
    A reusable barrier implemented with semaphores and a lock.

    @note The implementation is unnecessarily complex by wrapping the counter
          in a list `[self.num_threads]` to achieve mutability inside the
          `phase` method, which is an unconventional and confusing pattern.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all participating threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread to arrive releases all other waiting threads.
                for iterator in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for reuse.
                count_threads[0] = self.num_threads
        threads_sem.acquire()
