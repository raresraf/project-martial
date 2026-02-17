"""
@file device.py
@brief Simulates a network of devices processing sensor data in synchronized time steps.
@details This script models a distributed system where multiple devices, each running in its
own thread, perform computations based on sensor data. The system uses a two-phase reusable
barrier for synchronization to ensure that all devices complete a time step before any
device begins the next one. Fine-grained locking is used to protect access to shared
data locations.
"""

from threading import Event, Thread
from threading import Lock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier for synchronizing a fixed number of threads.
    It employs a two-phase synchronization mechanism to prevent race conditions
    where fast threads could loop and re-enter the barrier before slow threads have exited.
    """

    
    def __init__(self, num_threads):
        """
        Initializes the barrier for a specified number of threads.
        Two counters and two semaphores are used to manage the two synchronization phases.
        """
        self.num_threads = num_threads
        # Counter for threads arriving at the first phase of the barrier.
        self.count_threads1 = [self.num_threads]
        # Counter for threads arriving at the second phase of the barrier.
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Semaphores act as gates for each phase. They are initialized to 0 (locked).
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier until all participating threads have called this method.
        This is achieved by progressing through two distinct phases.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single synchronization phase.
        The last thread to arrive resets the counter and signals other threads to proceed.
        """
        with self.count_lock:
            # Atomically decrement the counter for the current phase.
            count_threads[0] -= 1

            # Block invariant: If the count is zero, this is the last thread to arrive.
            if count_threads[0] == 0:
                # The last thread opens the gate for all other threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this barrier phase.
                count_threads[0] = self.num_threads
        # All threads, including the last one, will block here until the semaphore is released.
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device in the distributed network simulation.
    Each device manages its own sensor data and executes assigned scripts in a dedicated thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device, its state, and starts its main execution thread.
        """
        # Event to signal that the initial setup of all devices is complete.
        self.done_setup = Event()
        self.device_id = device_id
        self.thread = DeviceThread(self)
        self.thread.start()
        # Event to signal that a new script has been received from the supervisor.
        self.script_received = Event()
        self.sensor_data = sensor_data
        # Limits the number of concurrently running scripts this device can run.
        self.semaphore = Semaphore(value=8)

        self.supervisor = supervisor
        self.scripts = []
        # Event to signal that the supervisor is done assigning work for the current timepoint.
        self.timepoint_done = Event()

        self.nr_thread = 0
        self.lock_timepoint = Lock()
        self.script_list = []
        # A list of locks, providing fine-grained protection for each data location.
        self.lock_index = []

        # The shared reusable barrier for all devices, initialized by device 0.
        self.r_barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs global setup for the simulation. Intended to be called by one device (e.g., device 0).
        It initializes and distributes the shared synchronization objects (barrier, locks) to all devices.
        """

        
        used_devices = len(devices)
        # Block invariant: This setup is performed only once by the primary device.
        if self.device_id is 0:
            r_barrier = ReusableBarrier(used_devices)
            # Create a lock for each potential data location to prevent data races.
            for _ in range(0, 24):
                self.lock_index.append(Lock())

            # Distribute the shared objects to all devices in the simulation.
            for d in range(len(devices)):


                devices[d].lock_index = self.lock_index
                devices[d].r_barrier = r_barrier
                # Signal to each device that setup is complete and it can start its main loop.
                devices[d].done_setup.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device for the current time step.
        Called by the supervisor.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script is the supervisor's signal that all assignments for this time step are done.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def close_scripts(self):
        """
        Waits for all currently running DataScript threads to complete their execution.
        """
        nrThreads = len(self.script_list)
        for i in range(0, nrThreads):
            self.script_list[i].join()

        # Clean up the list of finished threads.
        for i in range(0, nrThreads):
            self.script_list.pop()

        self.nr_thread = 0

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control loop thread for a single device, orchestrating its activity over discrete time steps.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_script(self, neighbours):
        """
        Spawns DataScript threads to execute the scripts assigned for the current time step.
        """
        for (script, location) in self.device.scripts:
            # Limit the number of concurrently running scripts.
            self.device.semaphore.acquire()
            self.device.script_list.append(DataScript\
            (neighbours, location, script, self.device))

            self.device.nr_thread = self.device.nr_thread + 1
            self.device.script_list[len(self.device.script_list)-1].start()

    def run(self):
        """
        The main execution loop. It waits for setup, then enters a loop representing
        synchronized time steps.
        """

        
        # Wait until the initial setup of shared resources is complete.
        self.device.done_setup.wait()

        # Main simulation loop, where each iteration is one time step.
        while True:
            
            

            # Acquire information about neighbors for the current time step from the supervisor.
            with self.device.lock_timepoint as neighbours:
                neighbours = self.device.supervisor.get_neighbours()
                # A None value for neighbours signals the end of the simulation.
                if neighbours is None:
                    break

            
            # Wait for the supervisor to signal that all script assignments are complete for this step.
            self.device.timepoint_done.wait()
            self.run_script(neighbours)

            # --- BARRIER SYNCHRONIZATION: PHASE 1 ---
            # All devices wait here. This ensures that all devices have started their DataScripts
            # for the current time step before any device proceeds to cleanup.
            self.device.r_barrier.wait()
            self.device.timepoint_done.clear()
            self.device.close_scripts()
            # --- BARRIER SYNCHRONIZATION: PHASE 2 ---
            # All devices wait here again. This ensures that all DataScripts on all devices
            # have finished execution for the current time step before the next time step begins.
            self.device.r_barrier.wait()


class DataScript(Thread):
    """
    A worker thread that executes a single data processing script on a specific location.
    It gathers data from neighboring devices, runs the script, and distributes the result.
    """
    def __init__(self, neighbours, location, script, scr_device):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.scr_device = scr_device


    def getdata(self, script_data):
        """Helper to get data from the parent device."""
        data = self.scr_device.get_data(self.location)
        if data is not None:
            script_data.append(data)

    def scriptdata(self, script_data):
        """
        Runs the script on the collected data and distributes the result.
        """
        if script_data != []:
            
            # Execute the computational logic of the script.
            result = self.script.run(script_data)
            
            # Pre-condition: Update the data on all neighboring devices with the new result.
            for device in self.neighbours:
                device.set_data(self.location, result)

            self.scr_device.set_data(self.location, result)
        # Release the semaphore to allow another script on this device to run.
        self.scr_device.semaphore.release()

    def run(self):
        """
        The main logic for the script thread. It ensures exclusive access to a data location,
        gathers data, computes, and updates.
        """
        # Acquire the lock for this specific data location to prevent race conditions.
        with self.scr_device.lock_index[self.location]:
            script_data = []

            # Block logic: Aggregate data from all neighbors for the target location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            # Get data from the local device as well.
            self.getdata(script_data)
            # Execute the script and distribute the results.
            self.scriptdata(script_data)