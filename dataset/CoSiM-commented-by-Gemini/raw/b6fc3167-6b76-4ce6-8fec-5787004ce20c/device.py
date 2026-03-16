"""
This module implements a simulation of a distributed device network with a multi-layered
threading model. Each device has a primary "manager" thread that, at each synchronized
time step, spawns a new pool of worker threads (`ExecuteSript`) to perform tasks.

The simulation uses a global barrier to synchronize all worker threads across all
devices. It also pre-emptively creates a shared set of locks for all possible data
locations, which is distributed to all devices.

@warning This implementation contains several severe design flaws:
         1. The `ReusableBarrier` is not thread-safe and is prone to deadlocks.
         2. A new pool of worker threads is created and destroyed in every single
            time step, which is a highly inefficient pattern.
         3. The distribution of work to the worker threads via `list.pop()` on a
            shared list is not synchronized, representing a critical race condition.
"""

from threading import Thread, Lock, Condition, Event


class Device(object):
    """
    Represents a device node, which manages a pool of worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the synchronization primitives for all devices.
        A single "root" device (the one with the lowest ID) is responsible for
        creating the shared barrier and lock set.
        """
        if self.is_root_device(devices) == 0:
            set_barriers(devices)

    def assign_script(self, script, location):
        """Assigns a script to be processed in the next time step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def is_root_device(self, devices):
        """
        Checks if this device is the "root" device (has the lowest ID).
        """
        is_root = 0
        for current_device in devices:
            if current_device.device_id < self.device_id:
                is_root = 1
                break
        return is_root

    def get_data(self, loc):
        """Gets sensor data from a specific location."""
        return self.sensor_data[loc] if loc in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data
        else:
            pass

    def shutdown(self):
        """Shuts down the device by joining its manager thread."""
        self.thread.join()

def set_barriers(devices):
    """
    Creates and distributes the shared barrier and lock set to all devices.
    """
    lock_set = {}
    barrier = ReusableBarrier(len(devices))
    for current_device in devices:
        current_device.barrier = barrier
        # Pre-create a lock for every possible location.
        for current_location in current_device.sensor_data:
            lock_set[current_location] = Lock()
        # Every device gets a reference to the same dictionary of locks.
        current_device.lock_set = lock_set

class DeviceThread(Thread):
    """
    A manager thread for a single device. Its main role is to coordinate
    the spawning of worker threads at each time step.
    """
    def __init__(self, device):
        
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        The main time-step loop for the device manager.

        @warning This loop uses a highly inefficient model of creating and joining
                 a new pool of 8 threads for every single time step of the simulation.
        """
        nr_threads = 8
        while True:
            self.device.timepoint_done.clear()
            neigh = self.device.supervisor.get_neighbours()
            
            # --- Synchronization Phase 1 ---
            self.device.barrier.wait()
            if neigh is None: # Shutdown signal
                break
            
            # Wait for supervisor to assign all scripts for this timepoint.
            self.device.timepoint_done.wait()
            
            perform_s = []
            for script in self.device.scripts:
                perform_s.append(script)
            
            # --- Worker Thread Pool Creation ---
            threads = []
            for i in xrange(nr_threads):
                threads.append(ExecuteSript(self.device, neigh, perform_s))
            for i in xrange(nr_threads):
                threads[i].start()
            for i in xrange(nr_threads):
                threads[i].join()
            
            # --- Synchronization Phase 2 ---
            self.device.barrier.wait()

class ReusableBarrier(object):
    """
    An attempted implementation of a reusable barrier using a Condition variable.

    @warning This implementation is NOT thread-safe. It is prone to a classic
             race condition known as the "lost wakeup" problem. A thread might
             pass the barrier, loop, and re-enter `wait()` before all other threads
             have woken up, leading to a deadlock.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
    
    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

    def print_barrier(self):
        "Print this barrier"
        print self.num_threads, self.count_threads

class ExecuteSript(Thread):
    """
    A worker thread that executes a single script.
    """

    def __init__(self, device, neighbours, perform_script):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.perform_script = perform_script

    def run(self):
        """
        Executes one script from the shared list.

        @warning This method contains a critical race condition. Multiple threads
                 are created with a reference to the same `perform_script` list
                 and they all call `.pop()` without any synchronization. This
                 will lead to unpredictable behavior and likely `IndexError`
                 exceptions as threads race to pop from the list.
        """
        if len(self.perform_script) != 0:
            (script, location) = self.perform_script.pop()
            collected = []
            
            # Acquire the lock for the specific location this script targets.
            self.device.lock_set[location].acquire()

            # --- Data Gathering and Processing ---
            for current_neigh in self.neighbours:
                data = current_neigh.get_data(location)
                collected.append(data)
            data = self.device.get_data(location)
            collected.append(data)

            if collected != []:
                result = script.run(collected)
                for current_neigh in self.neighbours:
                    current_neigh.set_data(location, result)
                self.device.set_data(location, result)

            self.device.lock_set[location].release()