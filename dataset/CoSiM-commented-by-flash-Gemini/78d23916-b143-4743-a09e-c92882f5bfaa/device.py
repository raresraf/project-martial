

"""
This module simulates a distributed system composed of interconnected devices,
focusing on synchronization using a reusable barrier and concurrent script execution.

Classes:
- `ReusableBarrierSem`: Implements a reusable barrier synchronization primitive using semaphores,
                        allowing multiple threads to wait for each other at a common point.
- `Device`: Represents an individual device in the distributed system, managing its state,
            sensor data, and interactions with other devices and a central supervisor.
- `Node`: A thread-like object responsible for executing a specific script with given data.
- `DeviceThread`: The main operational thread for each Device, handling its lifecycle,
                  script assignment, data collection, script execution, and synchronization.
"""

from threading import enumerate, Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    Implements a reusable barrier using semaphores for thread synchronization.
    This barrier ensures that a specified number of threads (num_threads)
    all reach a certain point in their execution before any of them are allowed to proceed.
    It operates in two phases to allow for reusability without deadlocks.
    """

    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrierSem instance.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any are allowed to proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads


        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()       
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have called `wait()`. Once all threads arrive, they are all released.
        This method executes both phase1 and phase2 of the barrier mechanism.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        First phase of the barrier: threads count down and the last thread releases
        all waiting threads in the first semaphore.

        Pre-condition: All threads entering this phase are ready to synchronize.
        Post-condition: All threads have passed `threads_sem1.acquire()` for this phase.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire()

    def phase2(self):
        """
        Second phase of the barrier: threads count down again, and the last thread releases
        all waiting threads in the second semaphore. This allows the barrier to be reused.

        Pre-condition: All threads have completed phase 1 and are ready for the second synchronization point.
        Post-condition: All threads have passed `threads_sem2.acquire()` for this phase, and the barrier is reset.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            self.count_threads1 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a single computational device in a distributed simulation.

    Each device has a unique ID, sensor data, and interacts with a supervisor
    for coordinating tasks and with other devices indirectly through shared
    synchronization primitives like the barrier. It manages its own thread
    of execution (`DeviceThread`).
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing the sensor data managed by this device,
                                where keys are locations and values are data points.
            supervisor (Supervisor): A reference to the central supervisor managing the devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier across all devices in the simulation.
        The barrier is initialized only once by the device with device_id 0,
        and then propagated to all other devices.

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
        Assigns a script to be executed by this device at a specific location.
        If 'script' is None, it signals that the current timepoint's script assignment is complete.

        Args:
            script (object or None): The script object to assign, or None to signal completion.
            location (str): The sensor data location to which the script pertains.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from this device.

        Args:
            location (str): The identifier of the sensor data location.

        Returns:
            Any or None: The data associated with the location if present, otherwise None.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a specified location on this device.

        Args:
            location (str): The identifier of the sensor data location to update.
            data (Any): The new data value to set for the location.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
    def shutdown(self):
        """
        Initiates the shutdown process for the device by joining its associated thread.
        This ensures that the device's thread completes its execution before the program exits.
        """
        
        self.thread.join()

class Node(Thread):

    def __init__(self, script, script_data):
        """
        Initializes a Node (script execution) thread.

        Args:
            script (object): The script object to execute, which must have a `run` method.
            script_data (list): A list of data points to be passed to the script's `run` method.
        """


        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        self.result = None
    def run(self):
        """
        Executes the assigned script with the provided script data.
        The result of the script execution is stored in `self.result`.
        """
        self.result = self.script.run(self.script_data)

    def join(self):
        """
        Waits for the Node thread to complete and returns its script and result.

        Returns:
            tuple: A tuple containing the script object and the result of its execution.
        """
        Thread.join(self)
        return (self.script, self.result)


class DeviceThread(Thread):
    

    def __init__(self, device):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage and execute.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

        """

        Main execution loop for the device thread.



        This loop continuously:

        1. Fetches neighbors from the supervisor. If no neighbors are returned (indicating shutdown),

           the loop breaks.

        2. Waits for a 'script_received' event, signifying that scripts have been assigned for the current time step.

        3. Clears the 'script_received' event.

        4. Gathers sensor data for each assigned script from the device and its neighbors.

        5. Creates and starts `Node` threads for each assigned script with its collected data.

        6. Joins all `Node` threads, waiting for their completion and collecting results.

        7. Propagates the results of script execution back to the device and its neighbors.

        8. Waits on a shared barrier to synchronize with other devices before the next iteration.

        """        

        while True:
            # Block Logic: Retrieves the current list of neighboring devices from the supervisor.
            # Pre-condition: `self.device.supervisor` is a valid supervisor object.
            neighbours = self.device.supervisor.get_neighbours()
            thread_list=[]
            scripts_result = {}
            scripts_data = {}
            if neighbours is None:
                # Block Logic: If no neighbors are returned (None), it signals the simulation is ending.
                # Post-condition: The device thread terminates its execution.
                break

            string = "" # Debugging variable, not critical to core logic.
            # Block Logic: (Potential Debugging) Concatenates neighbor IDs into a string.
            for neighbour in neighbours:
                string = string + " " + str(neighbour)
            # Block Logic: Waits for the supervisor to signal that scripts have been assigned for the current timepoint.
            # Invariant: The device pauses here until `script_received.set()` is called by the supervisor.
            self.device.script_received.wait()
            # Block Logic: Clears the event to allow it to be waited upon again in the next timepoint.
            self.device.script_received.clear()
            
            # Block Logic: Prepares scripts for execution by gathering data from neighbors and self.
            # Invariant: `scripts_data` maps each script to the sensor data collected for its execution.
            for (script, location) in self.device.scripts:
                script_data = []
                # Block Logic: Collects sensor data from each neighboring device for the current location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collects sensor data from the current device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                scripts_data[script] = script_data
                # Block Logic: If data is available for the script, creates a Node thread for its execution.
                if script_data != []:
                    nod = Node(script,script_data)
                    thread_list.append(nod)
            # Block Logic: Starts all Node threads concurrently to execute scripts.
            for nod in thread_list:
                
                nod.start()
            # Block Logic: Waits for all Node threads to complete and collects their results.
            # Invariant: `scripts_result` maps each script to the output produced by its execution.
            for nod in thread_list:
                key ,value = nod.join()
                scripts_result[key] = value
            # Block Logic: Propagates the results of script execution back to neighbors and the current device.
            for (script, location) in self.device.scripts:
                
                if scripts_data[script] != []:
                    # Block Logic: Updates sensor data on each neighboring device with the script's result.
                    for device in neighbours:
                        device.set_data(location, scripts_result[script])
                        
                    # Block Logic: Updates sensor data on the current device with the script's result.
                    self.device.set_data(location, scripts_result[script])
            
            # Block Logic: Waits on the shared barrier to synchronize with other devices before the next timepoint.
            self.device.barrier.wait()
