"""
This module implements a distributed device simulation using a lock-step
execution model facilitated by a custom, semaphore-based reusable barrier.

The key components are:
- `ReusableBarrierSem`: A custom implementation of a reusable barrier that allows
  a set of threads to wait for each other at a synchronization point multiple
  times.
- `Device`: Represents a node in the network, which runs its logic in a
  dedicated `DeviceThread`.
- `DeviceThread`: The main control loop for a device, orchestrating the
  lock-step execution using two barrier waits per timepoint.
- `MyScriptThread`: A thread dedicated to executing a single computational
  script, including data gathering from neighbors and result propagation.
"""

from threading import Event, Semaphore, Lock, Thread



class ReusableBarrierSem(object):
    """A reusable barrier implemented using two semaphores.

    This barrier enables a group of threads to synchronize at a point once,
    and then again at a second point, allowing it to be safely reused in a loop.
    It uses a two-phase signaling mechanism to prevent threads that have completed
    a wait from looping around and re-entering the barrier before other threads
    have exited it.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)  # For the first phase
        self.threads_sem2 = Semaphore(0)  # For the second phase

    def wait(self):
        """Causes a thread to wait until all threads have called wait."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase of the barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive releases all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use.
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase to ensure safe reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread to arrive releases all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use.
        self.threads_sem2.acquire()

class Device(object):
    """Represents a single device in the simulated network."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.my_lock = Lock()  # A lock specific to this device instance.
        self.barrier = ReusableBarrierSem(0) # Dummy barrier, replaced in setup.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the shared barrier for all devices in the network.

        Device 0 acts as the leader, creating the barrier instance that all
        other devices will then reference.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """Assigns a script for the device to run in the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A `None` script signals that all scripts for the timepoint are assigned.
            self.script_received.set()
            self.timepoint_done.set()


    def get_data(self, location):
        """Gets sensor data for a specific location from this device."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Sets sensor data for a specific location on this device."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the device's thread, effectively stopping the device."""
        self.thread.join()



class MyScriptThread(Thread):
    """A dedicated thread for executing a single script."""

    def __init__(self, script, location, device, neighbours):
        """Initializes the script execution thread."""
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """Gathers data, runs the script, and propagates the results."""
        script_data = []

        # Gather data from neighboring devices.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the parent device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Execute the script's logic.
            result = self.script.run(script_data)

            # Propagate the result back to neighbors and the parent device.
            # Each device's `set_data` call is protected by its own lock.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()

class DeviceThread(Thread):
    """The main control thread for a device, managing its lifecycle."""

    def __init__(self, device):
        """Initializes the device's main thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop, which implements a lock-step execution model."""
        while True:
            # Get neighbors for the upcoming timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break; # Termination signal from supervisor.
            
            # --- PHASE 1: Pre-computation synchronization ---
            # All threads wait here before scripts are assigned for the timepoint.
            self.device.barrier.wait()

            # Wait for the supervisor to finish assigning scripts for this timepoint.
            self.device.script_received.wait()
            script_threads = []
            
            # --- PHASE 2: Script Execution ---
            # Create and start a new thread for each assigned script.
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            for thread in script_threads:
                thread.start()
            # Wait for all script executions for this device to complete.
            for thread in script_threads:
                thread.join()
            
            # --- PHASE 3: Post-computation synchronization ---
            # Wait for the timepoint completion signal (redundant with script_received).
            self.device.timepoint_done.wait()
            # All threads wait here after completing their work for the timepoint.
            self.device.barrier.wait()
            
            # Reset events for the next cycle.
            self.device.script_received.clear()
