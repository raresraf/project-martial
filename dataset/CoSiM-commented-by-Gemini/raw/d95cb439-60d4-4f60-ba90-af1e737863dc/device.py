"""
This module implements a complex, multi-layered simulation of a distributed
device network. It features a hierarchical threading model where each device
manages a pool of script-executing worker threads.

Key Components:
- Device: The main class representing a node. It manages script queues, data,
  and a complex set of locks for synchronization. It has a concept of a 'master'
  device that holds the central barrier.
- DeviceThread: The top-level thread for a device. It orchestrates the overall
  lifecycle for a single time step and launches a `Scripter` thread.
- Scripter: A dispatcher thread that manages a pool of worker threads.
- ScriptExecutor: The worker thread that dequeues and executes a single script.
- ReusableBarrier: A custom barrier implementation for synchronizing all devices
  between time steps.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue


class Device(object):
    """
    Represents a device node in the simulation, managing scripts, data, and threads.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device with its data, supervisor, and threading primitives.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None
        


        self.script_running = Lock()
        self.timepoint_done = Event()
        
        # Per-location locks for fine-grained data access control.
        self.data_locks = dict()
        
        # A queue to hold scripts waiting for execution by worker threads.
        self.queue = Queue()
        
        self.available_threads = 14 # Simulates a device with 14 cores.

        for loc in sensor_data:
            self.data_locks.__setitem__(loc, Lock())

        # A global lock, seems intended to control access during neighbor data discovery.
        self.can_get_data = Lock()

        self.master = None
        self.script_over = False
        self.alive = True
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for the simulation, owned by a master device.
        """
        

        
        self.barrier = ReusableBarrier(len(devices))

        
        # The first device in the list is designated as the master.
        self.master = devices[0]

    def assign_script(self, script, location):
        """
        Assigns a script to this device, adding it to the execution queue.
        """
        
        if script is not None:
            # A script of None signifies the end of script assignments for the time step.
            
            self.script_running.acquire()
            self.scripts.append((script, location))
            self.queue.put_nowait((script, location))
            self.script_received.set()
        else:
            self.script_running.acquire()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Gets data from a location, protected by a global lock.
        Note: This locking seems coarse and may be a bottleneck.
        """
        

        

        self.can_get_data.acquire()
        return_value = self.sensor_data[location] if location in self.sensor_data else None
        self.can_get_data.release()
        return return_value

    def get_device_data(self, location):
        """
        Gets data from a location, protected by a per-location lock.
        This allows for more concurrent data access than `get_data`.
        """
        

        if location not in self.sensor_data:
            return None

        

        self.data_locks.get(location).acquire()

        new_data = self.sensor_data[location]

        self.data_locks.get(location).release()

        return new_data

    def set_data(self, location, data):
        """
        Sets data for a location, protected by a per-location lock.
        """
        
        if location in self.sensor_data:

            

            self.data_locks.get(location).acquire()
            self.sensor_data[location] = data
            self.data_locks.get(location).release()

    def shutdown(self):
        """Shuts down the device's main thread."""
        
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It manages the high-level
    synchronization and lifecycle for each time step.
    """
    

    def __init__(self, device):
        """Initializes the device's main thread."""
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main device loop. Manages a `Scripter` thread and synchronizes
        with the master barrier at the end of each time step.
        """



        while True:
            self.device.can_get_data.acquire()
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:

                
                # Supervisor signals termination. Wait at barrier one last time.
                self.device.master.barrier.wait()

                self.device.can_get_data.release()
                return

            
            # Launch the Scripter thread to manage the worker pool for this time step.
            script_instance = Scripter(self.device, neighbours)

            script_instance.start()

            
            # Wait for the supervisor to signal that all scripts for this step have been assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Signal the Scripter that it should terminate its worker threads.
            self.device.script_over = True
            self.device.script_received.set()

            
            # Wait for the Scripter and its workers to finish.
            script_instance.join()

            
            # Refill the queue for the next time step.
            for (script, location) in self.device.scripts:
                self.device.queue.put_nowait((script, location))

            self.device.script_over = False

            
            # Wait at the master barrier to synchronize with all other devices.
            self.device.master.barrier.wait()

            self.device.can_get_data.release()
            self.device.script_running.release()


class Scripter(Thread):
    """
    A dispatcher thread that creates and manages a pool of ScriptExecutor workers.
    """
    

    def __init__(self, device, neighbours):
        """Initializes the Scripter thread."""
        


        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        Creates a pool of ScriptExecutor threads and manages their lifecycle.
        """
        

        list_executors = []

        # Create and start a pool of worker threads (executors).
        for iterator in range(1, self.device.available_threads):
            executor = ScriptExecutor(self.device, self.device.queue, self.neighbours, iterator)
            list_executors.append(executor)
            executor.start()

        while True:
            
            # Wait for a new script to be added to the queue or for the timepoint to end.
            self.device.script_received.wait()
            self.device.script_received.clear()

            if self.device.script_over:

                
                
                # The time step is over. Terminate worker threads by sending poison pills.
                

                for iterator in range(1, self.device.available_threads):
                    self.device.queue.put((None, None))

                
                # Wait for all worker threads to finish.
                for executor in list_executors:
                    executor.join()

                
                # Reset the queue for the next time step.
                self.device.queue = Queue(-1)
                return

            self.device.script_running.release()


class ScriptExecutor(Thread):
    """
    A worker thread that executes scripts from a shared queue.
    """
    

    def __init__(self, device, queue, neighbours, identifier):
        """Initializes a worker thread."""
        
        Thread.__init__(self, name="Script Executor %d" % identifier)
        self.device = device
        self.queue = queue
        self.neighbours = neighbours

    def run(self):
        """
        Continuously fetches scripts from the queue and executes them.
        """
        
        while True:
            
            # Block until a script is available in the queue.
            (script, location) = self.queue.get()
            if script is None:
                # A None script is a "poison pill" signaling termination.
                return

            script_data = []
            
            # Gather data from neighbors using the per-location lock method.
            for device in self.neighbours:
                data = device.get_device_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_device_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                # Run the script and propagate the result.
                result = script.run(script_data)
                
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)


class ReusableBarrier:
    """
    A custom reusable barrier using semaphores. It uses a two-phase
    protocol to ensure threads can reuse the barrier in a loop.
    """
    

    def __init__(self, num_threads):
        """Initializes the barrier."""
        
        self.num_threads = num_threads
        # The counts are stored in a list to be mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Waits at the barrier, blocking until all threads arrive."""
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread to arrive releases all others.
                for iterator in range(self.num_threads):
                    threads_sem.release()
                # Reset count for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire()
