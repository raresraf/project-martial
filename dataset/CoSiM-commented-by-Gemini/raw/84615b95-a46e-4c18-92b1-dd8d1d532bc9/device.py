"""
This module implements a simulation framework for a network of distributed devices.

It provides classes for a reusable barrier, a device, and the threads that
drive the device's execution. The framework appears to be designed for
data-parallel computations where devices exchange and process sensor data
in synchronized steps.
"""
from threading import enumerate, Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    A reusable barrier implementation using semaphores for thread synchronization.

    This barrier allows a fixed number of threads to wait for each other to
    reach a certain point of execution before any of them are allowed to proceed.
    It is reusable, meaning it can be used multiple times. It employs a two-phase
    protocol to prevent race conditions between different synchronization rounds.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                               before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads


        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()       
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase of the barrier
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase of the barrier

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have called this method.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive releases all other threads from the first semaphore.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase of the barrier wait, ensuring all threads from phase 1 have exited before reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread to arrive releases all other threads from the second semaphore.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            self.count_threads1 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a single device in the distributed simulation network.

    Each device runs in its own thread, processes data according to assigned
    scripts, and communicates with its neighbors.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor data.
            supervisor (obj): The supervisor object that manages the network topology.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for synchronization among all devices.
        
        Device with ID 0 is responsible for creating and sharing the barrier instance.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    self.barrier = device.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script (obj): The script object to be executed. If None, it signals
                          that no more scripts will be assigned in this step.
            location (any): The data location the script operates on.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that all scripts for the current step have been received.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
    def shutdown(self):
        """Shuts down the device's thread."""
        
        self.thread.join()

class Node(Thread):
    """A worker thread to execute a single script."""

    def __init__(self, script, script_data):
        """
        Initializes the Node thread.

        Args:
            script (obj): The script to execute. Must have a `run` method.
            script_data (list): The data to be passed to the script's `run` method.
        """

        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        self.result = None
         
    def run(self):
        """Executes the script and stores the result."""
        self.result = self.script.run(self.script_data)

    def join(self):
        """Waits for the thread to complete and returns the script and its result."""
        Thread.join(self)
        return (self.script, self.result)


class DeviceThread(Thread):
    """The main execution thread for a Device."""
    

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device.

        In each iteration, it waits for scripts, gathers data from neighbors,
        executes scripts in parallel, updates data, and synchronizes with
        other devices using a barrier.
        """
        

        while True:
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            thread_list=[]
            scripts_result = {}
            scripts_data = {}
            if neighbours is None:
                # Supervisor signals shutdown.
                break

            string = ""
            for neighbour in neighbours:
                string = string + " " + str(neighbour)
            
            # Wait until all scripts for the current simulation step are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            # Prepare script execution by gathering data.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Collect data from neighbors.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Collect local data.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                scripts_data[script] = script_data
                if script_data != []:
                    # Create a worker thread for each script with available data.
                    nod = Node(script,script_data)
                    thread_list.append(nod)

            # Start all worker threads.
            for nod in thread_list:
                
                nod.start()

            # Wait for all worker threads to finish and collect results.
            for nod in thread_list:
                key ,value = nod.join()
                scripts_result[key] = value

            # Update data on this device and its neighbors based on script results.
            for (script, location) in self.device.scripts:
                
                if scripts_data[script] != []:
                    for device in neighbours:
                        device.set_data(location, scripts_result[script])
                        
                    self.device.set_data(location, scripts_result[script])
            
            # Synchronize with all other devices before starting the next step.
            self.device.barrier.wait()
