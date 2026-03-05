"""
This module implements a device simulation where a master `DeviceThread`
serially gathers data, dispatches computation to worker threads, and then
serially propagates the results.

This architecture differs significantly from other versions by centralizing all
data gathering and propagation within the main `DeviceThread` for a device,
rather than having each worker handle its own I/O. This severely limits
parallelism.

NOTE: This script contains several critical bugs:
1. The `ReusableBarrierSem` is implemented incorrectly and will fail after the
   first `wait()` call, breaking synchronization between devices.
2. The `setup_devices` logic for distributing the barrier is inefficient and racy.
3. The `get_data` and `set_data` methods remain completely thread-unsafe. Since
   they are called by `DeviceThread`s from different devices, this will lead to
   data corruption.
"""

from threading import enumerate, Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    An implementation of a reusable barrier.

    BUG: This barrier is not correctly implemented. It improperly resets the
    counters for the two phases, which will cause it to fail after the first
    full wait cycle. `phase1` should reset `count_threads1`, and `phase2` should
    reset `count_threads2`.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()       
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        self.phase1()
        self.phase2()

    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            # BUG: This should reset count_threads1, not count_threads2.
            self.count_threads2 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            # BUG: This should reset count_threads2, not count_threads1.
            self.count_threads1 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """Represents a device node in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier.

        NOTE: This logic is racy and inefficient. It relies on device 0 running
        first, and other devices iterating to find it.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    self.barrier = device.barrier
                    break

    def assign_script(self, script, location):
        """Assigns a script to the device for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that script assignment is complete for this step.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data. Not thread-safe."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data. Not thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class Node(Thread):
    """A worker thread that only runs the computation part of a script."""
    def __init__(self, script, script_data):
        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        self.result = None
         
    def run(self):
        """Executes the script's run method on pre-fetched data."""
        self.result = self.script.run(self.script_data)

    def join(self):
        """Custom join to return the script and its result."""
        Thread.join(self)
        return (self.script, self.result)


class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating a highly serialized workflow.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal.

            # Wait for supervisor to signal that scripts are ready.
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            thread_list = []
            scripts_result = {}
            scripts_data = {}

            # --- 1. Serial Data Gathering Phase ---
            # The main thread gathers all data for all scripts sequentially.
            for (script, location) in self.device.scripts:
                script_data = []
                # BUG: `get_data` is not thread-safe. Concurrent reads/writes
                # from other devices will cause data races here.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                scripts_data[script] = script_data
                if script_data:
                    # Create a worker thread, passing the pre-fetched data.
                    nod = Node(script, script_data)
                    thread_list.append(nod)

            # --- 2. Parallel Computation Phase ---
            # Start all worker threads.
            for nod in thread_list:
                nod.start()
            
            # Join all threads and collect their results.
            for nod in thread_list:
                key, value = nod.join()
                scripts_result[key] = value

            # --- 3. Serial Data Propagation Phase ---
            # The main thread propagates all results sequentially.
            for (script, location) in self.device.scripts:
                if scripts_data.get(script):
                    result = scripts_result.get(script)
                    # BUG: `set_data` is not thread-safe, leading to races.
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
            
            self.device.scripts = [] # Clear scripts for the next round.
            
            # --- 4. Synchronization Phase ---
            # Wait at the global barrier (which is bugged).
            self.device.barrier.wait()
