"""
@file device.py
@brief Simulates a network of synchronized computational devices.
@details This file implements a model for a distributed sensor network or edge computing
environment. It defines devices that can execute scripts on sensor data. The devices
operate in synchronized timepoints, using a reusable barrier to ensure all nodes
complete a step before proceeding to the next.
"""

from threading import Lock, Thread, Event, Semaphore

class Device(object):
    """
    Represents a single device or node in the distributed network.
    It manages its own state, sensor data, and a dedicated worker thread.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary representing the device's local sensor readings.
        @param supervisor A central object that manages network topology (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that a timepoint's computation is done.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # Shared synchronization primitives, initialized by setup_devices.
        self.barrier = None
        self.location_locks = None

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for a group of devices.
        Uses a leader-follower pattern where device 0 creates the shared barrier
        and lock dictionary, and all other devices reference them.
        @param devices A list of all Device objects in the simulation.
        """
        Device.devices_no = len(devices)
        # Block Logic: Device 0 acts as the leader to initialize shared objects.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            self.location_locks = {}
        # Block Logic: Follower devices reference the leader's shared objects.
        else:
            self.barrier = devices[0].barrier
            self.location_locks = devices[0].location_locks

    def assign_script(self, script, location):
        """
        Assigns a computational script to this device for a specific location.
        @param script The script object to be executed.
        @param location The data location the script will operate on.
        """
        # Block Logic: If a valid script is passed, add it to the execution queue.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        # Block Logic: A null script is a signal to end the current timepoint.
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's worker thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The dedicated worker thread for a Device.
    Manages the device's lifecycle of executing scripts and synchronizing with the network.
    """

    def __init__(self, device):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_scripts(self, script, location, neighbours):
        """
        Executes a single script. This function is intended to be run in its own thread.
        It gathers data from the device and its neighbors, runs the script, and
        broadcasts the result back to all involved devices.
        @param script The script to run.
        @param location The location context for the data.
        @param neighbours A list of neighboring Device objects.
        """
        # Pre-condition: Acquire a lock for the specific data location to prevent race conditions.
        lock_location = self.device.location_locks.get(location)
        # Inline: Lazily create the lock if it doesn't exist.
        if lock_location is None and location is not None:
            self.device.location_locks[location] = Lock()
            lock_location = self.device.location_locks[location]
        lock_location.acquire()
        
        script_data = []
        # Data Aggregation: Collect data from all neighboring devices for the script.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Data Aggregation: Include the device's own data.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        if script_data != []:
            # Execute the computational script.
            result = script.run(script_data)
            # Result Broadcasting: Update the data on all involved devices with the new result.
            for device in neighbours:
                device.set_data(location, result)
            self.device.set_data(location, result)
            
        lock_location.release()

    def run(self):
        """
        The main lifecycle loop of the device thread.
        """
        # Invariant: The thread runs for the entire simulation lifetime.
        while True:
            # At each timepoint, get the current set of neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Synchronization: Wait until the supervisor signals the end of the previous timepoint.
            self.device.timepoint_done.wait()
            
            tlist = []
            # Block Logic: Spawn a new thread for each assigned script to run them in parallel.
            for (script, location) in self.device.scripts:
                thread = Thread(target=self.run_scripts, args=(script, location, neighbours))
                tlist.append(thread)
                thread.start()
            
            # Wait for all script threads for this device to complete.
            for thread in tlist:
                thread.join()
            
            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()
            # Synchronization: Wait at the global barrier for all devices to finish this timepoint.
            self.device.barrier.wait()

class ReusableBarrierSem():
    """
    A reusable barrier implementation using semaphores.
    It forces a fixed number of threads to wait until all of them have reached
    the barrier. It uses a two-phase protocol to allow for safe reuse in loops.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a specific number of threads.
        @param num_threads The number of threads that will synchronize on this barrier.
        """
        self.num_threads = num_threads
        # Counters for each phase.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        # Lock to protect access to the counters.
        self.counter_lock = Lock()
        # Semaphores used as "gates" for each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Main entry point for threads to wait at the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the two-phase barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            # Invariant: The last thread to arrive opens the gate for all waiting threads.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        # All threads wait here until the last one arrives and releases the permits.
        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase, preventing race conditions on barrier reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            # Invariant: The last thread to arrive opens the second gate.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        # All threads wait at the second gate to ensure all have exited phase 1.
        self.threads_sem2.acquire()
