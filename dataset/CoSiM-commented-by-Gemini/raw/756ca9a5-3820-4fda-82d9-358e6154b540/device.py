"""
This module implements a multi-threaded simulation framework for a network of devices.

It defines a `Device` that can run computational scripts on sensor data,
coordinating with neighboring devices in discrete time steps. The simulation
utilizes multiple layers of threading and synchronization primitives to manage
concurrency both within a single device (multi-core processing) and between
multiple devices in the network.

Classes:
    Device: Represents a single node in the device network.
    InnerThread: A worker thread that executes a single script on sensor data.
    DeviceThread: The main control thread for a Device, managing a pool of InnerThreads.
    ReusableBarrierSem: A synchronization primitive to barrier-synchronize multiple threads.
"""

from threading import Event, Thread, Semaphore, Lock
from Queue import Queue

class Device(object):
    """Represents a device in a distributed sensor network simulation.

    Each device has a unique ID, its own local sensor data, and can be assigned
    scripts to run. It coordinates with a central supervisor and other devices.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary mapping locations to sensor values.
        supervisor: An external object that manages the overall simulation and network topology.
        scripts (list): A list of (script, location) tuples to be executed.
        timepoint_done (Event): An event to signal the start of a new simulation step.
        thread (DeviceThread): The main thread that manages this device's lifecycle.
        time_point_barrier (ReusableBarrierSem): A barrier to synchronize all devices at the end of a time step.
        location_semaphore_dict (dict): A dictionary mapping locations to semaphores for mutual exclusion.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Event to signal that the device is ready to start processing a new time point.
        self.timepoint_done = Event()
        # The main thread for this device's control loop.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Barrier for synchronizing all devices between time points.
        self.time_point_barrier = None
        # Semaphores for ensuring exclusive access to sensor data at a given location.
        self.location_semaphore_dict = None

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for the entire network of devices.
        
        This method is intended to be called on one device (e.g., device_id 0)
        to set up synchronization primitives that are shared across all devices.
        
        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Only the device with ID 0 should perform the setup to avoid redundancy.
        if self.device_id == 0:
            # Create a barrier for all devices to synchronize at the end of a time step.
            barrier = ReusableBarrierSem(len(devices))
            self.time_point_barrier = barrier
            # Distribute the shared barrier to all other devices.
            for dev in devices:
                if dev.time_point_barrier is None:
                    dev.time_point_barrier = barrier
            
            # Create a set of all unique sensor locations across all devices.
            location_set = set()
            for dev in devices:
                for location in dev.sensor_data:
                    location_set.add(location)
            
            # Create a dictionary of semaphores, one for each unique location.
            # This ensures that only one script can access a given location at a time across the entire network.
            loc_dict = {}
            for loc in location_set:
                loc_dict[loc] = Semaphore(1) # Semaphore initialized to 1 for mutex behavior.
            # Distribute the shared location semaphores to all devices.
            for dev in devices:
                dev.location_semaphore_dict = loc_dict

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device.
        
        If the script is None, it's a signal that no more scripts are coming for
        the current time step, so the device can start processing.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal the DeviceThread that it's time to process the assigned scripts.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location from this device."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location on this device."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class InnerThread(Thread):
    """
    A worker thread that executes scripts for a single Device.
    
    It processes scripts from a queue, gathers data from its parent device and
    its neighbors, runs the script, and distributes the results back.
    """
    
    def __init__(self, device, barrier, queue):
        """Initializes the worker thread."""
        Thread.__init__(self)
        self.device = device
        self.barrier = barrier # Barrier to synchronize with other InnerThreads of the same device.
        self.queue = queue # Queue from which to pull scripts.
        self.neighbours = []

    def run(self):
        """The main loop for the worker thread."""
        while True:
            # Block and wait for a script from the queue.
            script = self.queue.get()

            # --- Control Signals ---
            if script[0] == "exit":
                self.barrier.wait() # Ensure all workers synchronize before exiting.
                break
            if script[0] == "done":
                self.barrier.wait() # Signal that one script is done, synchronize.
                continue
            if script[0] == "neighbours":
                self.neighbours = script[1]
                self.barrier.wait() # Synchronize after updating neighbors.
                continue

            # --- Script Execution ---
            script_improved = script[0]
            location = script[1]
            script_data = []
            
            # Acquire the lock for the specific location to ensure data consistency.
            self.device.location_semaphore_dict[location].acquire()
            
            # Gather data from all neighboring devices at the given location.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from the parent device itself.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Only run the script if there is data to process.
            if script_data != []:
                # Execute the script with the collected data.
                result = script_improved.run(script_data)
                
                # Propagate the result to all neighbors.
                for device in self.neighbours:
                    device.set_data(location, result)
                
                # Update the parent device's own data.
                self.device.set_data(location, result)
            
            # Release the lock for the location.
            self.device.location_semaphore_dict[location].release()

class DeviceThread(Thread):
    """
    The main control thread for a Device.
    
    It manages a pool of worker threads (`InnerThread`) and orchestrates the
    device's participation in the simulation's time steps.
    """
    
    def __init__(self, device):
        """Initializes the device's main thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.cores_number = 8 # Simulates a device with 8 cores.
        # A barrier for the internal worker threads.
        self.threads_barrier = ReusableBarrierSem(self.cores_number)
        # A queue to dispatch scripts to the worker threads.
        self.scripts_queue = Queue()
        self.thread_list = []

    def run(self):
        """The main control loop for the device."""
        # --- Initialization ---
        # Create and start the pool of worker threads.
        for _ in range(self.cores_number):
            inner_t = InnerThread(self.device, self.threads_barrier,\
            self.scripts_queue)
            self.thread_list.append(inner_t)
        for thread in self.thread_list:
            thread.start()

        # --- Main Simulation Loop ---
        while True:
            # Get the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # If the supervisor returns None, it's a signal to shut down.
            if neighbours is None:
                # Send an exit signal to all worker threads.
                for _ in range(self.cores_number):
                    self.scripts_queue.put(("exit", None))
                break
            
            # Wait for the `assign_script` method to signal that all scripts for this time step have been received.
            self.device.timepoint_done.wait()
            
            # --- Dispatch Phase ---
            # Dispatch the updated neighbor list to all worker threads.
            for _ in range(self.cores_number):
                self.scripts_queue.put(("neighbours", neighbours))
            
            # Dispatch all assigned scripts to the worker threads.
            for pair in self.device.scripts:
                self.scripts_queue.put(pair)
            
            # Send a "done" signal to indicate the end of script dispatch for this time step.
            for _ in range(self.cores_number):
                self.scripts_queue.put(("done", None))
            
            # --- Cleanup and Synchronization ---
            # Clear the event for the next time step.
            self.device.timepoint_done.clear()
            # Wait at the global barrier for all other devices to finish their time step.
            self.device.time_point_barrier.wait()


        # --- Shutdown ---
        # Join all worker threads to ensure a clean exit.
        for thread in self.thread_list:
            thread.join()

class ReusableBarrierSem(object):
    """
    A reusable barrier implementation using semaphores.
    
    This allows a fixed number of threads to wait for each other to reach a
    certain point before proceeding. It works in two phases to be reusable.
    """
    
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        # Semaphores act as "turnstiles". They are locked (0) until all threads arrive.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the two-phase barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            # The last thread to arrive opens the first semaphore "turnstile".
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release() # Let all waiting threads pass.
                self.count_threads1 = self.num_threads # Reset for the next use.
        # All threads wait here until the last thread releases the semaphore.
        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase, preventing threads from looping back and mixing with late threads."""
        with self.counter_lock:
            self.count_threads2 -= 1
            # The last thread to arrive opens the second semaphore "turnstile".
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for the next use.
        # All threads wait here, ensuring all have passed phase 1 before any can restart.
        self.threads_sem2.acquire()
