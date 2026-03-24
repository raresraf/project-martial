"""
This module simulates a network of interconnected devices that execute scripts
in synchronized time steps. It provides a framework for modeling distributed
computation on sensor data, using a custom multi-threaded architecture.

The key components are:
- Device: Represents a node in the network, with its own data and scripts.
- DeviceThread: The main control loop for a device, managing script execution
  and synchronization.
- RunScripts: A thread responsible for executing a single script, gathering
  data from neighbors, and propagating results.
- ReusableBarrier: A synchronization primitive to ensure all devices complete
  a time step before any device proceeds to the next.
"""


from threading import Thread, Event
from threading import Lock, Semaphore

class ReusableBarrier():
    """
    A custom, reusable barrier implementation for synchronizing multiple threads.

    This barrier ensures that all participating threads will block on the `wait()`
    method until all of them have called it. Once all threads have reached the
    barrier, they are all released. The barrier is "reusable" because it resets
    itself, allowing it to be used multiple times (e.g., in a loop).

    It uses a two-phase protocol to prevent race conditions where fast threads
    could loop around and re-enter the barrier before slow threads have exited it.
    """
    
    
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that will be synchronized.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have arrived.
        """
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """
        The first phase of the barrier wait.
        Threads are blocked on `threads_sem1` until all have arrived.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            # The last thread to arrive releases all other waiting threads.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads      
        self.threads_sem1.acquire()
    
    def phase2(self):
        """
        The second phase of the barrier wait.
        Ensures no thread proceeds until all have passed phase 1, making it reusable.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            # The last thread to arrive releases everyone for the next cycle.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads   
        self.threads_sem2.acquire()

class RunScripts(Thread):
    """
    A thread that executes a given script on a device's data and its neighbors' data.
    """                                     
    
    def __init__(self, device, location, script, neighbours):
        """
        Initializes the script-running thread.

        Args:
            device (Device): The parent device object.
            location (int): The location index for which data is being processed.
            script: The script object to be executed.
            neighbours (list): A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    
    def run(self):
        """
        Main logic for the thread. It acquires a lock for a location,
        gathers data, runs the script, and propagates the results.
        """

        
        # Acquire a lock for the specific location to prevent concurrent data modification.
        self.device.location_lock[self.location].acquire()

        script_data = []
        
        # Gather data from all neighboring devices for the given location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        # Also gather data from the current device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            # Execute the script with the collected data.
            result = self.script.run(script_data)
            
            

            # Propagate the result of the script to all neighbors and the current device.
            for device in self.neighbours:
                device.set_data(self.location, result)
                
            self.device.set_data(self.location, result)

        
        # Release the lock for the location.
        self.device.location_lock[self.location].release()

class Device(object):
    """
    Represents a device in the simulated network.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data, keyed by location.
            supervisor: An object that manages the network topology (e.g., neighbors).
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()


        # Each device runs its own main control loop in a separate thread.
        self.thread = DeviceThread(self)
        self.barrier = None
        self.list_thread = []
        self.thread.start()
        # A list of locks, one for each possible data location.
        self.location_lock = [None] * 200

    def __str__(self):
        """String representation of the Device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the network of devices and the shared synchronization barrier.
        """
        
        
        nr_devices = len(devices)
        
        # The first device to be set up creates the shared barrier.
        if self.barrier is None:
            barrier = ReusableBarrier(nr_devices)


            self.barrier = barrier

            # Propagate the shared barrier to all other devices.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        
        # Store the list of all devices in the network.
        for device in devices:
            if device is not None:
                self.devices.append(device)


    def assign_script(self, script, location):
        """
        Assigns a new script to be executed by this device.
        """
        
        lock_location = False

        if script is None:
            # A None script is a signal to begin the next time step.
            self.timepoint_done.set()



        else:
            
            # Add the script to the list of scripts to be run.
            self.scripts.append((script, location))
            # Set up a shared lock for the script's location if it doesn't exist.
            if self.location_lock[location] is None:

                # Check if any other device has already created a lock for this location.
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        
                        # Use the existing shared lock.
                        self.location_lock[location] = device.location_lock[location]
                        lock_location = True
                        break

                if lock_location is False:
                    # If no lock exists, create a new one for this location.
                    self.location_lock[location] = Lock()

            # Signal the DeviceThread that a script has been received.
            self.script_received.set()
            

    def get_data(self, location):
        """Returns the sensor data for a given location."""
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates the sensor data for a given location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main device thread to shut down the device cleanly."""
        
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a Device, orchestrating its lifecycle.
    """
    

    def __init__(self, device):
        """Initializes the device's main thread."""
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop for the device. It waits for a timepoint, runs scripts,
        and synchronizes with other devices using a barrier.
        """
        
        
        while True:

            
            # Get the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbors is the signal to terminate the thread.
                break

            # Wait until the supervisor signals the start of the next time step.
            self.device.timepoint_done.wait()

            


            # For each script assigned to this device, create a thread to run it.
            for (script, location) in self.device.scripts:
                thread = RunScripts(self.device, location, script, neighbours) 
                self.device.list_thread.append(thread)

            
            # Start all script-running threads for the current time step.
            for thread_elem in self.device.list_thread:
                thread_elem.start()

            # Wait for all script-running threads to complete.
            for thread_elem in self.device.list_thread:
                thread_elem.join()

            # Clean up for the next time step.
            self.device.list_thread = []
            self.device.timepoint_done.clear()
            # Wait at the barrier for all other devices to finish the current time step.
            self.device.barrier.wait()