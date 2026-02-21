


"""
Defines a distributed device simulation using a thread-per-task model.

Note: This file is named `device.py` but contains the full simulation logic.

This module implements a simulation framework where devices execute tasks
concurrently. The architecture consists of:
- `ReusableBarrierSem`: A custom two-phase barrier for synchronization.
- `Device`: The main class for a node, holding data and state.
- `DeviceThread`: The main control loop that spawns script executors.
- `MyScriptThread`: A short-lived thread that executes one script task.

The synchronization logic in this implementation is unusual and may contain
flaws that could lead to deadlocks or race conditions.
"""

from threading import Event, Semaphore, Lock, Thread


class ReusableBarrierSem(object):
    """A custom, reusable barrier implemented with semaphores.

    This barrier synchronizes a fixed number of threads over two phases to
    ensure it can be safely reused in a loop.
    """
    def __init__(self, num_threads):
        """Initializes the barrier."""
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes the calling thread to wait at the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """The second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """Represents a single device (node) in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.my_lock = Lock()  # A personal lock for the device.
        self.barrier = ReusableBarrierSem(0)  # Placeholder barrier.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the shared barrier for the device group.

        Note: This has a race condition. It assumes device 0 will be the master
        and sets its barrier for all other devices. Correct execution depends
        on the order in which threads call this method.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """Assigns a script to the device for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A `None` script signals the end of assignments for this time step.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data from a specific sensor location (not thread-safe)."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a specific sensor location (not thread-safe)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class MyScriptThread(Thread):
    """A short-lived thread to execute one script."""

    def __init__(self, script, location, device, neighbours):
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """The core logic for executing a script.

        It gathers data, runs the script, and then propagates the result.
        The data propagation step is heavily serialized, as it acquires a
        lock on each device individually before writing the result.
        """
        script_data = []

        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = self.script.run(script_data)

            # Update neighbors one by one with a lock for each.
            for device in self.neighbours:
                with device.my_lock:
                    device.set_data(self.location, result)
            # Update self with a lock.
            with self.device.my_lock:
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """The main control thread for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main, and highly unusual, time step loop.

        The loop structure contains two barrier waits, which is not a standard
        pattern for cyclic barriers and could lead to deadlocks or incorrect
        synchronization.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # 1. First synchronization point.
            self.device.barrier.wait()

            # 2. Wait for scripts to be assigned for this time step.
            self.device.script_received.wait()
            
            # 3. Create and run a new thread for each script.
            script_threads = [
                MyScriptThread(script, location, self.device, neighbours)
                for (script, location) in self.device.scripts
            ]
            for thread in script_threads:
                thread.start()
            for thread in script_threads:
                thread.join()
            
            # 4. Wait for the timepoint "done" signal.
            self.device.timepoint_done.wait()
            
            # 5. Second synchronization point.
            self.device.barrier.wait()
            self.device.script_received.clear()
