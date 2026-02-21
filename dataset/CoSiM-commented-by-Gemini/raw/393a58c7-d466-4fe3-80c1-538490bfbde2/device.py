"""
Models a node in a distributed, parallel computation system.

This module provides a framework for simulating a network of devices that operate
in synchronized time steps, following a Bulk Synchronous Parallel (BSP) model.
A master-slave pattern is used for initialization, where one `Device` acts as
the master to create and distribute shared synchronization primitives like
barriers and locks to the other devices in its group. Each device then executes
assigned scripts in parallel, aggregating data from its neighbors, computing a
result, and disseminating it back to the neighborhood before synchronizing at a
global barrier to end the time step.
"""


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """Represents a single device in a distributed network simulation.

    Each device runs its own thread, communicates with neighbors supervised by a
    central entity, and participates in synchronized, parallel computations.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's local data.
        supervisor: An external object responsible for providing neighborhood info.
        is_master (bool): Flag indicating if this device is the master node.
        master_id (int): The ID of the master device for this group.
        barrier (ReusableBarrierSem): A shared barrier for synchronizing time steps.
        data_lock (list): A list of shared locks for fine-grained data access.
        scripts (list): A list of (script, location) tuples to be executed.
        thread (DeviceThread): The main execution thread for this device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Sets up device-specific attributes, synchronization primitives, and starts
        the main `DeviceThread` to handle its lifecycle.
        """
        self.are_locks_ready = Event() 
        self.master_id = None
        self.is_master = True 
        self.barrier = None 
        self.stored_devices = [] 
        self.data_lock = [None] * 100 
        self.master_barrier = Event() 
        self.lock = Lock() 
        self.started_threads = [] 
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Configures the device group, electing a master and sharing resources.

        This method implements the master-slave initialization. The first device
        to enter (or one designated by logic) becomes the master. The master
        creates a shared barrier and data locks. Slave devices wait for the
        master to complete this setup and then receive a reference to these
        shared synchronization objects.
        """
        # Determine if this device is a slave by checking if any other device
        # has already established a master.
        for device in devices:
            if device is not None and device.master_id is not None:
                self.master_id = device.master_id
                self.is_master = False
                break

        if self.is_master is True:
            # If this device is the master, it initializes shared resources.
            self.barrier = ReusableBarrierSem(len(devices))
            self.master_id = self.device_id
            for i in range(100):
                self.data_lock[i] = Lock()
            # Signal that the master has finished setting up shared resources.
            self.are_locks_ready.set()
            self.master_barrier.set()
            for device in devices:
                if device is not None:
                    # Distribute the shared barrier to all devices.
                    device.barrier = self.barrier
                    self.stored_devices.append(device)
        else: 
            # If this device is a slave, it waits for and adopts the master's resources.
            for device in devices:
                if device is not None:
                    if device.device_id == self.master_id:
                        # Wait for the master to signal that setup is complete.
                        device.master_barrier.wait()
                        if self.barrier is None:
                            self.barrier = device.barrier
                    self.stored_devices.append(device)

    def assign_script(self, script, location):
        """Assigns a computational script to the device for the next time step.

        If a script is provided, it is added to the execution queue. The device
        then waits until the master device's shared locks are ready before
        proceeding. If the script is None, it signals that the timepoint is
        complete without computation.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Ensure master's lock setup is complete before accessing locks.
            for device in self.stored_devices:
                if device.device_id == self.master_id:
                    device.are_locks_ready.wait()
            # Adopt the master's data locks.
            for device in self.stored_devices:
                if device.device_id == self.master_id:
                    self.data_lock = device.data_lock
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific location in the device's sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data at a specific location in a thread-safe manner."""
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        """Gracefully shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main execution thread that drives a device's lifecycle.

    This thread operates in discrete, synchronized time steps. In each step,
    it executes assigned scripts and then waits at a barrier for all other
    devices in its group to complete the step.
    """

    def __init__(self, device):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """The main control loop for the device.

        The loop represents the progression of time steps. In each step, it
        spawns ExecutorThreads for assigned scripts, waits for their completion,
        and then synchronizes with all other devices at a global barrier.
        """
        while True:
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit condition for the simulation.

            # Wait for the supervisor to signal the start of a new timepoint.
            self.device.timepoint_done.wait()


            # For each assigned script, create and start an ExecutorThread.
            for (script, location) in self.device.scripts:
                executor = ExecutorThread(self.device, script, neighbours, location)
                self.device.started_threads.append(executor)
                executor.start()

            # Wait for all spawned executor threads to finish their computation.
            for executor in self.device.started_threads:
                executor.join()

            # Clean up and prepare for the next time step.
            del self.device.started_threads[:]
            self.device.timepoint_done.clear()
            # Synchronize with all other devices before starting the next step.
            self.device.barrier.wait()


class ExecutorThread(Thread):
    """A thread that executes a single script on a device.

    This thread handles the core logic of gathering data from neighbors,
    running a script, and broadcasting the result back to the neighborhood.
    """

    def __init__(self, device, script, neighbours, location):
        """Initializes the script executor thread."""
        Thread.__init__(self, name="Executor Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.location = location

    def run(self):
        """Executes the script.

        The execution is wrapped in a location-specific lock to ensure data
        consistency. It aggregates data from the device and its neighbors,
        runs the script on the aggregated data, and writes the result back
        to all devices in the neighborhood.
        """
        # Acquire a lock for the specific data location to prevent race conditions.
        self.device.data_lock[self.location].acquire()

        if self.neighbours is None:
            return

        script_data = []
        
        # Gather data from all neighbors at the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        # Include the device's own data.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            # Run the computation script on the aggregated data.
            result = self.script.run(script_data)
            
            # Broadcast the result to all neighbors.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Update the device's own data.
            self.device.set_data(self.location, result)

        # Release the lock for the data location.
        self.device.data_lock[self.location].release()
