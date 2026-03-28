"""
This module implements a more complex simulation of a distributed device network.

It expands on a simpler device model by introducing a worker-pool pattern for each
device. A `Device` object now manages one main `DeviceThread` and a pool of
`WorkerThread` objects. This architecture suggests a design for handling a higher
throughput of script executions per device. The synchronization is significantly
more intricate, involving multiple barriers and events to coordinate the different
stages of a simulation time step across all threads and devices.
"""

import barrier
from threading import Event, Thread, Lock


class Device(object):
    """
    Represents a device node with a dedicated thread pool for script execution.

    This version of the Device class uses a complex set of synchronization primitives
    (Events, Locks, Barriers) to manage a multi-threaded execution environment. It
    coordinates a main thread and several worker threads to process scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device with its thread pool and synchronization objects.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): The device's local data, keyed by location.
            supervisor (object): The central coordinator of the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # --- Synchronization Primitives ---
        # Events for signaling state changes between threads.
        self.script_received = Event()  # Signals that scripts for a timepoint are ready.
        self.timepoint_done = Event()   # A signal from the supervisor that a timepoint has ended.
        self.got_neighbours = Event()   # Signals that the neighbor list has been updated.
        self.initialize = Event()       # Signals that device setup is complete.
        
        # Flags and Locks for managing shared state.
        self.flag_terminate = False     # Global flag to signal all threads to terminate.
        self.put_take_data = Lock()     # Lock for thread-safe access to sensor_data.
        self.index_lock = Lock()        # Lock for thread-safe access to the shared script index 'i'.
        self.i = 0                      # Shared index for pulling scripts from the list.

        # --- Script and Network Data ---
        self.scripts = []
        self.neighbours = None
        self.devices = []
        self.locations_locks = []

        # --- Barriers for multi-stage synchronization ---
        # Note: 'bariera' is Romanian for 'barrier'.
        self.bariera = None             # Main barrier for timepoint synchronization.
        self.barrier_time = None        # A secondary barrier, likely for another sync phase.
        self.barrier_clear_events = None # A third barrier, likely for resetting events.

        # --- Thread Management ---
        self.thread = DeviceThread(self) # Main thread, primarily for supervisor communication.
        self.threads = []                # Pool of worker threads for script execution.

        # The following attributes appear to be unused or part of a legacy design.
        self.script_taken = Event()
        self.assign_script_none = Event()
        self.script_de_orice_fel = Event() # "script_of_any_kind" in Romanian
        self.assign_script_not_none = Event()
        self.bariera_join = None
        self.script_sent = Lock()
        self.script_sent_thread = Lock()
        self.barrier_lock = Lock()
        self.counter = 0
        self.flag_received = Event()
        self.count_threads = []
        self.flag_script_received = False
        self.flag_script_taken = False
        self.flag_assign_script = 2
        self.flag_get_neigbours = False
        self.get_neighbours_lock = Lock()


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs global setup for all devices from the coordinator device (ID 0).

        This method initializes and distributes shared resources like location locks
        and synchronization barriers to all devices in the simulation. It also starts
        the device's own threads.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        self.devices = devices
        self.count_threads = [len(self.devices)]

        # Pre-condition: Global setup is performed only by the coordinator device.
        if self.device_id == 0:
            # Collect all unique locations to create a global set of locks.
            locations = []
            for device in self.devices:
                for key in device.sensor_data.keys():
                    locations.append(key)
            
            maxim = max(locations) if locations else -1
            self.locations_locks = [None] * (maxim + 1)
            for locatie in locations:
                if self.locations_locks[locatie] is None:
                    self.locations_locks[locatie] = Lock()

            # Initialize reusable barriers for multi-stage synchronization.
            self.bariera = barrier.ReusableBarrierCond(len(self.devices))
            num_threads = len(self.devices) * 8 # 1 DeviceThread + 7 WorkerThreads per device
            self.bariera_join = barrier.ReusableBarrierCond(num_threads)
            self.barrier_time = barrier.ReusableBarrierCond(num_threads)
            self.barrier_clear_events = barrier.ReusableBarrierCond(num_threads)

            # Distribute the shared synchronization objects to all devices.
            for device in self.devices:
                device.i = 0
                device.bariera = self.bariera
                device.counter = len(self.devices)
                device.barrier_time = self.barrier_time
                device.barrier_clear_events = self.barrier_clear_events
                device.locations_locks = self.locations_locks

        # Start the main thread and the pool of worker threads for this device.
        self.thread.start()
        for _ in range(7):
            dev = WorkerThread(self)
            dev.start()
            self.threads.append(dev)
        
        self.initialize.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device's script list.

        Args:
            script (object): The script to be executed.
            location: The location context for the script.
        """
        # The lock 'script_sent' seems to be intended to protect the scripts list,
        # but its usage is not consistent across the class. A simple lock on the list
        # would be more conventional.
        self.scripts.append((script, location))
        if script is None:
            # A None script signals the end of script assignments for the timepoint.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Thread-safely retrieves data from a specified location."""
        with self.put_take_data:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Thread-safely sets data at a specified location."""
        with self.put_take_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all of the device's threads to complete."""
        for thread in self.threads:
            thread.join()
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a device, responsible for communication with the supervisor.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main loop: gets neighbor updates from the supervisor and participates
        in script execution.
        """
        while True:
            # Block Logic: Fetches updated neighbor information from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                # A None response signals the end of the simulation.
                self.device.flag_terminate = True
                self.device.got_neighbours.set() # Wake up worker threads to terminate.
                break
            
            self.device.got_neighbours.set() # Signal workers that neighbors are available.
            
            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            
            # --- Script Execution Section (duplicated in WorkerThread) ---
            # This thread also acts as a worker, processing scripts from the shared list.
            while True:
                with self.device.index_lock:
                    if self.device.i >= len(self.device.scripts):
                        break
                    # The modulo logic here is unusual and potentially buggy.
                    # It implies a fixed-size circular buffer of size 8 for scripts.
                    (script, location) = self.device.scripts[self.device.i % 8]
                    self.device.i += 1
                
                if script is not None:
                    lock = self.device.locations_locks[location]
                    with lock:
                        script_data = []
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data:
                            result = script.run(script_data)
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            self.device.set_data(location, result)
                if script is None:
                    break
            
            # --- Multi-Stage Synchronization ---
            # 1. All threads (main + workers) signal they have finished processing scripts.
            self.device.barrier_clear_events.wait() 
            # 2. Reset events for the next timepoint.
            self.device.script_received.clear()
            self.device.got_neighbours.clear()
            # 3. All threads wait at a final barrier before the next timepoint begins.
            self.device.barrier_time.wait()


class WorkerThread(Thread):
    """
    A worker thread in the device's thread pool for executing scripts.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main loop: waits for work and executes scripts."""
        while True:
            # Wait for the main thread to signal that neighbors are ready.
            self.device.got_neighbours.wait()
            if self.device.flag_terminate:
                break
            
            # Wait for the main thread to signal that scripts are ready.
            self.device.script_received.wait()
            
            # --- Script Execution Section (identical to DeviceThread) ---
            # This loop competes with the main thread and other workers to process scripts.
            while True:
                with self.device.index_lock:
                    if self.device.i >= len(self.device.scripts):
                        break
                    (script, location) = self.device.scripts[self.device.i % 8]
                    self.device.i += 1

                if script is not None:
                    lock = self.device.locations_locks[location]
                    with lock:
                        script_data = []
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data:
                            result = script.run(script_data)
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            self.device.set_data(location, result)
                if script is None:
                    break

            # --- Multi-Stage Synchronization ---
            # The worker thread participates in the same barrier synchronization as the main thread.
            self.device.barrier_clear_events.wait()
            self.device.barrier_time.wait()