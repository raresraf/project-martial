"""
This module provides a framework for a distributed device simulation.

The architecture is composed of several classes:
- ReusableBarrier: A synchronization primitive to make a number of threads wait
  until all of them have reached a certain point.
- Device: Represents a node in the network. Each device has its own thread
  (`DeviceThread`) and a set of `DeviceCore` threads for parallel processing.
- DeviceThread: The main thread of control for a device, managing its lifecycle,
  synchronization, and distribution of work to its cores.
- DeviceCore: A worker thread that executes a specific computational task (a "script")
  on data from its parent device and its neighbors.

The simulation appears to be time-stepped, synchronized by the ReusableBarrier.
Within each time step, scripts are assigned and executed in parallel by the device cores.
"""


from threading import Event, Thread, Condition, Lock
from Queue import Queue

class ReusableBarrier(object):
    """
    A reusable barrier implementation using a Condition variable.

    This barrier allows a set of threads to wait for each other to reach a
    common execution point. Once all threads have reached the barrier, they are
    all released and the barrier resets for the next use.
    """
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        # Counter for threads that have arrived at the barrier.
        self.count_threads = self.num_threads
        # Condition variable to orchestrate waiting and notification.
        self.cond = Condition()

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have arrived.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread has arrived; notify all waiting threads and reset the counter.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Not all threads have arrived; wait to be notified.
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a single device in the distributed simulation.

    Each device manages its own data, executes assigned scripts, and communicates
    with neighboring devices under the coordination of a supervisor.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device and starts its main thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor  # External component that defines network topology.
        self.barrier = None  # Shared barrier for time-step synchronization.
        self.locations_mutex = None  # Shared lock for accessing the global locations list.
        self.can_begin = Event()  # Signals that initial setup is complete.
        self.locks_computed = Event()
        self.timepoint_done = Event()  # Signals that all scripts for a timepoint are assigned.
        self.simulation_end = Event()  # Signals all threads to terminate.
        self.lock = Lock()  # A per-device lock for its own data.
        self.scripts_queue = Queue()  # Queue of (script, location) tasks for its cores.
        self.scripts = []  # Persistent list of all scripts assigned to this device.
        self.locations = []  # Shared list of all unique data locations in the system.
        self.devices = []  # Shared list of all devices in the system.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs global, one-time setup for the entire simulation.

        This method should be called on a single "master" device (e.g., device 0).
        It initializes and distributes shared synchronization objects to all devices.
        """
        if self.device_id == 0:
            # Create a barrier for all devices to synchronize on.
            self.barrier = ReusableBarrier(len(devices))
            self.devices = devices
            self.locations_mutex = Lock()

            # Distribute shared objects to all other devices.
            for device in devices:
                device.locations_mutex = self.locations_mutex
                device.locations = self.locations
                device.barrier = self.barrier
                # Unblock other device threads to continue their setup.
                device.can_begin.set()

            # Unblock this device's own thread.
            self.can_begin.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be run. A `None` script signals the end of
        script assignment for the current time step.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_queue.put((script, location))
        else:
            # All scripts for this time step have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Safely retrieves data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Safely sets data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to complete."""
        self.thread.join()

class DeviceThread(Thread):
    """
    Main control thread for a single Device. It manages the device's lifecycle
    through the simulation's time steps.
    """
    def __init__(self, device):
        """Initializes the thread and its pool of DeviceCore workers."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.current_neighbours = []
        # Each device has its own pool of core/worker threads.
        self.cores = [DeviceCore(self, i, self.device.simulation_end) for i in xrange(0, 8)]

    def run(self):
        """Main execution loop for the device thread."""
        # Wait for the initial global setup to complete.
        self.device.can_begin.wait()

        # --- Initial Setup Phase ---
        # Atomically register all of this device's data locations into a global list.
        self.device.locations_mutex.acquire()
        for location in self.device.sensor_data.keys():
            if location not in self.device.locations:
                self.device.locations.append(location)
        self.device.locations_mutex.release()

        # Synchronize with all other devices after location registration.
        self.device.barrier.wait()

        # The master device (id 0) creates a set of locks for all unique locations.
        if self.device.device_id == 0:
            self.device.locations_locks = [Lock() for _ in xrange(0, len(self.device.locations))]
            # Distribute the shared locks to all other devices.
            for device in self.device.devices:
                device.locations_locks = self.device.locations_locks

        # Synchronize again to ensure all devices have received the location locks.
        self.device.barrier.wait()

        # Start all core worker threads. They will wait for tasks.
        for core in self.cores:
            core.start()

        # --- Main Simulation Loop ---
        while True:
            # Clear the script queue from any previous (or aborted) time steps.
            while not self.device.scripts_queue.empty():
                self.device.scripts_queue.get()

            # Refill the queue with all scripts assigned to this device.
            for script in self.device.scripts:
                self.device.scripts_queue.put(script)

            # Get the current set of neighbors from the supervisor.
            self.current_neighbours = self.device.supervisor.get_neighbours()

            # A `None` value for neighbors is the signal to terminate the simulation.
            if self.current_neighbours is None:
                self.device.simulation_end.set()
                for core in self.cores:
                    # Unblock any waiting core threads so they can exit.
                    core.got_script.set()
                
                for core in self.cores:
                    core.join()
                break

            # Process scripts until the supervisor signals the timepoint is done
            # and the local queue of scripts is empty.
            while not self.device.timepoint_done.isSet() or not self.device.scripts_queue.empty():
                if not self.device.scripts_queue.empty():
                    script, location = self.device.scripts_queue.get()

                    # Find an idle core to execute the script.
                    core_found = False
                    while not core_found:
                        for core in self.cores:
                            if core.running is False:
                                core_found = True
                                core.script = script
                                core.location = location
                                core.neighbours = self.current_neighbours
                                core.running = True
                                # Signal the core to start processing.
                                core.got_script.set()
                                break
            
            # Synchronize with all other devices to mark the end of the timepoint.
            self.device.barrier.wait()

            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()

class DeviceCore(Thread):
    """A worker thread that executes a single script."""
    def __init__(self, device_thread, core_id, simulation_end):
        """Initializes the core worker thread."""
        Thread.__init__(self, name="Device Core %d" % core_id)
        self.device_thread = device_thread
        self.core_id = core_id
        self.neighbours = []
        self.got_script = Event()  # Event to signal that a new task is ready.
        self.running = False
        self.simulation_end = simulation_end  # Event to signal simulation termination.

    def run(self):
        """Main loop for the worker core."""
        while True:
            # Wait for a script to be assigned.
            self.got_script.wait()

            # Check for simulation end signal.
            if self.simulation_end.isSet():
                break

            # --- Task Execution ---
            # Acquire the lock for the specific location this script targets.
            self.device_thread.device.locations_locks[self.location].acquire()

            # Gather data from all neighbors for the target location.
            script_data = []
            for neighbour in self.neighbours:
                neighbour.lock.acquire()
                data = neighbour.get_data(self.location)
                neighbour.lock.release()
                if data is not None:
                    script_data.append(data)

            # Gather data from the parent device for the target location.
            self.device_thread.device.lock.acquire()
            data = self.device_thread.device.get_data(self.location)
            self.device_thread.device.lock.release()
            if data is not None:
                script_data.append(data)

            if script_data:
                # Execute the script on the collected data.
                result = self.script.run(script_data)

                # Broadcast the result back to the parent device.
                self.device_thread.device.lock.acquire()
                self.device_thread.device.set_data(self.location, result)
                self.device_thread.device.lock.release()

                # Broadcast the result back to all neighbors.
                for neighbour in self.neighbours:
                    neighbour.lock.acquire()
                    neighbour.set_data(self.location, result)
                    neighbour.lock.release()

            # Release the location-specific lock.
            self.device_thread.device.locations_locks[self.location].release()

            # Mark this core as idle and clear the script event.
            self.running = False
            self.got_script.clear()

            # Final check for simulation end signal before waiting again.
            if self.simulation_end.isSet():
                break