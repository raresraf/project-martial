"""
Defines the core Device entity for a distributed sensor network simulation.

This module contains the Device class, which represents a single node in the
network, and the DeviceThread class, which manages the device's lifecycle and
synchronization across discrete time steps.
"""


from threading import Event, Thread, Semaphore
from barrier import ReusableBarrierSem
from worker import WorkerThread


class Device(object):
    """
    Represents a single device (node) in the simulated sensor network.

    A Device manages its own state, including sensor data, a list of scripts to
    execute, and its connections to neighbors. It orchestrates a pool of
    WorkerThreads to perform computations. The device with device_id=0 has a
    special role in initializing shared resources like synchronization barriers
    and data locks for the entire network.
    """

    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping locations to sensor values,
                                representing the data held by this device.
            supervisor (Supervisor): A reference to the main simulation supervisor.
        """

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.barrier_set = Event()
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()


        self.thread = DeviceThread(self)
        
        self.barrier = None
        
        self.neighbours = []
        
        self.data_locks = []
        
        self.thread_list = []
        
        self.worker_number = 8
        
        self.worker_barrier = ReusableBarrierSem(self.worker_number)
        
        self.script_queue = []
        
        self.script_lock = Semaphore(1)
        
        self.exit_flag = Event()
        
        self.tasks_finished = Event()
        
        self.start_tasks = Event()

    def set_flag(self):
        """Sets the event to indicate that the main barrier is ready."""
        self.barrier_set.set()

    def set_barrier(self, barrier):
        """Receives and sets the main simulation barrier from the coordinator device."""
        self.barrier = barrier

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the network, with device 0 acting as the coordinator.

        The coordinator (device 0) creates a shared barrier for all devices and
        a set of locks for all possible data locations. It distributes these
        shared objects to all other devices in the network. All devices then
        start their main control thread and worker thread pools.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        # Device 0 is the coordinator for setting up shared resources.
        if self.device_id == 0:
            
            # Create a barrier for all devices to synchronize on at each time step.
            self.barrier = ReusableBarrierSem(len(devices))
            location_index = -1
            for dev in devices:
                for k in dev.sensor_data:
                    if k > location_index:
                        location_index = k

            
            # Distribute the barrier and data locks to all devices.
            self.data_locks = {loc : Semaphore(1) for loc in range(location_index+1)}
            for dev in devices:
                dev.set_barrier(self.barrier)
                dev.data_locks = self.data_locks
                dev.set_flag() # Signal that setup is complete.
        else:
            
            # Non-coordinator devices wait until the barrier and locks are set.
            self.barrier_set.wait()
        self.thread.start()

        
        # Start the pool of worker threads for this device.
        for tid in range(self.worker_number):
            thread = WorkerThread(self, tid)
            self.thread_list.append(thread)
            thread.start()

    def assign_script(self, script, location):
        """
        Receives a script from the supervisor to be executed in the next time step.

        Args:
            script (Script): The script object to run. Can be None to signal
                             the end of script assignment for the current timepoint.
            location (int): The data location the script applies to.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is the signal that all scripts for this timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Safely gets data for a given location from this device's sensor data."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Safely sets data for a given location in this device's sensor data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all running threads for a clean shutdown of the device."""
        
        for thread in self.thread_list:
            thread.join()
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    This thread manages the device's participation in the synchronized,
    turn-based simulation. It orchestrates the device's state transitions
    from one timepoint to the next.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop for the device.

        The loop represents the progression of time steps. In each step, it:
        1. Synchronizes with all other devices at a global barrier.
        2. Gets its neighbors for the current time step.
        3. Checks for the simulation end signal.
        4. Waits for the supervisor to finish assigning scripts.
        5. Sets up the work queue and signals its worker threads to start.
        6. Waits for its worker threads to finish processing.
        """
        while True:
            
            # 1. Wait for all devices to reach the start of the time step.
            self.device.barrier.wait()

            
            # 2. Get the list of neighboring devices for this time step.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            
            # 3. Check for the supervisor's signal to terminate.
            if self.device.neighbours is None:
              
                self.device.exit_flag.set()
                self.device.start_tasks.set()
                break

            
            # 4. Wait for the supervisor to signal that all scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            # 5. Prepare the work queue for the worker threads.
            self.device.script_queue = list(self.device.scripts)
            self.device.scripts.clear() # Clear scripts for the next timepoint

            
            # 6. Signal workers to start processing the queue.
            self.device.start_tasks.set()

            
            # 7. Wait for a signal from the workers that they have finished this timepoint.
            self.device.tasks_finished.wait()
            self.device.tasks_finished.clear()