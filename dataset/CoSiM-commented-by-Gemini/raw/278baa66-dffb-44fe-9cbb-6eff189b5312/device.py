"""
Models a distributed system of devices executing computational scripts concurrently.

This module provides an alternative implementation for a device simulation framework.
Key features include a two-phase, semaphore-based reusable barrier, a per-device
locking mechanism, and a multi-threaded approach where each assigned script runs
in its own thread. The simulation proceeds in synchronized steps, demarcated by
two barrier calls per step.
"""
from threading import Event, Semaphore, Lock, Thread



class ReusableBarrierSem(object):
    """
    A reusable, two-phase synchronization barrier implemented using Semaphores.

    This barrier ensures that threads wait until a specified number of them have
    reached a synchronization point. It uses two separate phases (and two semaphores)
    to prevent race conditions where a fast thread could loop around and re-enter
    the barrier before slow threads from the previous phase have exited.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread arrives, release all threads waiting in phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase of the barrier wait, prevents wraparound races."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread arrives, release all threads waiting in phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a single device in the distributed system simulation.

    Manages local sensor data and orchestrates script execution via its own thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data.
            supervisor (object): A supervisor object managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        
        # A single lock to protect this device's data during writes.
        self.my_lock = Lock()
        self.barrier = ReusableBarrierSem(0)
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Distributes a shared barrier instance to all devices.

        Device 0 creates the barrier, and all other devices get a reference to it.
        
        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of assignments.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that all scripts for this timepoint have been received.
            self.script_received.set()
            # Signal that the timepoint processing is considered complete by the assigner.
            self.timepoint_done.set()


    def get_data(self, location):
        """
        Retrieves sensor data. Note: This read is not protected by a lock.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates sensor data. Note: This write is intended to be protected
        externally by acquiring `my_lock`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()



class MyScriptThread(Thread):
    """
    A dedicated thread to execute a single script.
    """

    def __init__(self, script, location, device, neighbours):
        """
        Initializes the script thread.
        
        Args:
            script (object): The script to execute.
            location (str): The data location associated with the script.
            device (Device): The parent device running this script.
            neighbours (list): A list of neighboring devices.
        """
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """Gathers data, runs the script, and writes back the results."""
        script_data = []

        # --- Data Gathering Phase ---
        # Note: get_data is not locked, relying on simulation-wide synchronization.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            # --- Data Update Phase ---
            # Acquire lock on each device before writing the result.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()

class DeviceThread(Thread):
    """
    The main control thread for a single Device.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break;
            
            # Barrier 1: Ensure all devices have their neighbor lists for the new time step.
            self.device.barrier.wait()

            # Wait until the supervisor has finished assigning all scripts for this step.
            self.device.script_received.wait()
            script_threads = []
            
            # --- Script Execution Phase ---
            # Launch a separate thread for each assigned script.
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            for thread in script_threads:
                thread.start()
            # Wait for all script threads to complete their execution.
            for thread in script_threads:
                thread.join()
            
            
            # Wait for the timepoint_done signal from the script assigner.
            self.device.timepoint_done.wait()
            
            # Barrier 2: Ensure all devices have completed computation before the next step.
            self.device.barrier.wait()
            self.device.script_received.clear()
