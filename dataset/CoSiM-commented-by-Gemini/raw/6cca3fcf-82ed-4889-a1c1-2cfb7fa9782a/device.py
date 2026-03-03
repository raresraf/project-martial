"""
This module presents a highly complex and inefficient implementation of a
distributed device simulation.

The architecture involves multiple layers of threads being created and destroyed
in every simulation step, which is a significant performance overhead.

Architectural Overview:
- `Device`: The main class for a device.
- `DeviceThread`: The top-level thread for a device. In each simulation step,
  it creates a `Scripter` thread.
- `Scripter`: A manager thread created per-step. It creates a pool of
  `ScriptExecutor` threads.
- `ScriptExecutor`: The worker thread that actually executes the script logic.
- `ReusableBarrier`: A custom two-phase semaphore barrier for synchronization.

The flow is: DeviceThread -> Scripter -> ScriptExecutor pool.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue


class Device(object):
    """
    Represents a device in the simulation.
    It manages a complex hierarchy of threads for executing tasks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None
        
        # Lock to coordinate between assign_script and the DeviceThread
        self.script_running = Lock()
        self.timepoint_done = Event()
        
        # A dictionary of locks for each data location.
        self.data_locks = dict()
        
        self.queue = Queue()
        
        self.available_threads = 14

        for loc in sensor_data:
            self.data_locks.__setitem__(loc, Lock())

        # A coarse-grained lock that protects most of a simulation step.
        self.can_get_data = Lock()

        self.master = None
        self.script_over = False
        self.alive = True
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes the shared barrier and sets the master device."""
        self.barrier = ReusableBarrier(len(devices))
        # All devices reference device 0 as the master to access the barrier.
        self.master = devices[0]

    def assign_script(self, script, location):
        """Assigns a script to be executed."""
        if script is not None:
            self.script_running.acquire()
            self.scripts.append((script, location))
            self.queue.put_nowait((script, location))
            # Signal to the Scripter thread that a new script is ready.
            self.script_received.set()
        else:
            # A None script signals the end of script assignment for this step.
            self.script_running.acquire()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data. This method is protected by a coarse-grained lock,
        suggesting it's intended to be called during a specific phase.
        """
        with self.can_get_data:
            return self.sensor_data.get(location)

    def get_device_data(self, location):
        """
        Retrieves sensor data for a given location, using a fine-grained lock
        specific to that location.
        """
        if location not in self.sensor_data:
            return None
        with self.data_locks.get(location):
            return self.sensor_data[location]

    def set_data(self, location, data):
        """
        Sets sensor data for a given location, using a fine-grained lock.
        """
        if location in self.sensor_data:
            with self.data_locks.get(location):
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. Its primary role is to spawn a
    `Scripter` thread for each simulation step.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main simulation loop."""
        while True:
            # This coarse lock protects the majority of the step, including
            # getting neighbors and running/joining the Scripter.
            with self.can_get_data:
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    # Shutdown logic.
                    self.device.master.barrier.wait()
                    return

                # A new Scripter thread is created for every single time step.
                script_instance = Scripter(self.device, neighbours)
                script_instance.start()

                # Wait for the supervisor to signal all scripts are assigned.
                self.device.timepoint_done.wait()
                self.device.timepoint_done.clear()

                # Signal the Scripter that script assignment is over and it should terminate its workers.
                self.device.script_over = True
                self.device.script_received.set()

                # Wait for the Scripter and all its workers to finish.
                script_instance.join()
                
                # This logic to repopulate the queue seems redundant as the workers
                # that would process it have already been terminated.
                for (script, location) in self.device.scripts:
                    self.device.queue.put_nowait((script, location))
                self.device.script_over = False

                # Synchronize with all other devices.
                self.device.master.barrier.wait()
            
            # Release the lock that was acquired in `assign_script`.
            self.device.script_running.release()

class Scripter(Thread):
    """
    A manager thread that creates and oversees a pool of ScriptExecutor threads.
    An instance of this class is created for each simulation step.
    """
    def __init__(self, device, neighbours):
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """Creates a pool of ScriptExecutors and manages their lifecycle."""
        list_executors = []
        # Create and start the actual worker threads.
        for iterator in range(1, self.device.available_threads):
            executor = ScriptExecutor(self.device, self.device.queue, self.neighbours, iterator)
            list_executors.append(executor)
            executor.start()

        while True:
            # Wait for a signal that a new script has been added.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Check for the signal to end the script execution phase.
            if self.device.script_over:
                # Send a sentinel value to each worker to terminate it.
                for _ in range(1, self.device.available_threads):
                    self.device.queue.put((None, None))

                # Wait for all worker threads to finish.
                for executor in list_executors:
                    executor.join()

                # Reset the queue and exit the Scripter thread.
                self.device.queue = Queue(-1)
                return
            
            # Release the lock acquired in `assign_script`, allowing the next
            # script to be assigned.
            self.device.script_running.release()

class ScriptExecutor(Thread):
    """The actual worker thread that executes a script."""
    def __init__(self, device, queue, neighbours, identifier):
        Thread.__init__(self, name="Script Executor %d" % identifier)
        self.device = device
        self.queue = queue
        self.neighbours = neighbours

    def run(self):
        """Continuously fetches tasks from the queue and executes them."""
        while True:
            # Block until a script is available in the queue.
            (script, location) = self.queue.get()
            # The sentinel value (None, None) signals termination.
            if script is None:
                return

            # Execute script logic.
            script_data = []
            for device in self.neighbours:
                data = device.get_device_data(location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_device_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                result = script.run(script_data)
                for device in self.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)

class ReusableBarrier:
    """A custom, two-phase reusable barrier implemented using Semaphores."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all threads reach the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
