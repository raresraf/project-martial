"""
This module implements a device simulation framework that utilizes a master-slave
thread model for distributed script execution and synchronization. It defines:
- Device: Represents a simulated device managing sensor data and script assignments.
- DeviceThread: The main thread for a Device, coordinating script execution and data sharing.
- SlaveList: Manages a pool of 'Slave' worker threads for a DeviceThread.
- Slave: Individual worker threads responsible for executing assigned scripts.

The system uses Events, Locks, and Semaphores for inter-thread communication and data consistency.
"""


from threading import Event, Thread, Lock, Condition, Semaphore


class Device(object):
    """
    Represents a simulated device in a distributed system. Each device manages
    its sensor data, assigns and executes scripts via its `DeviceThread`,
    and interacts with a supervisor. Synchronization is handled through
    shared locks and a leader-follower pattern.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for various locations.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are assigned
        self.scripts = [] # List to store assigned scripts (script, location) tuples
        self.devices = None # Reference to the list of all devices in the simulation


        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing
        self.thread = DeviceThread(self) # The main thread for this device
        self.thread.start() # Start the main device thread
        self.leader = -1 # Stores the device_id of the leader device
        
        self.location_locks = [] # List to store locks for different locations (shared across devices)
        # Block Logic: If this device is the leader (device_id 0), initialize synchronization primitives.
        if device_id == 0:
            
            self.finishedthread = 0 # Counter for threads that have finished their work for a timepoint

            
            self.condition = Condition() # Condition variable for coordinating finished threads

            
            self.can_start = Event() # Event to signal when the leader has completed its setup

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs initial setup for all devices in the simulation.
        This includes identifying the leader device, initializing location-specific locks
        (by the leader), and distributing these locks to all devices.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Inline: Store the list of all devices.
        self.devices = devices

        # Block Logic: Identify the leader device (device with device_id 0).
        for i in xrange(len(self.devices)):
            if devices[i].device_id == 0:
                self.leader = i # Store the index of the leader device.
                break

        # Block Logic: If this device is the leader, initialize shared resources.
        if self.device_id == 0:
            self.can_start.clear() # Clear the event, indicating setup is in progress.

            # Block Logic: Determine the maximum location ID to initialize enough locks.
            maximum = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > maximum:
                        maximum = location

            # Block Logic: Initialize a Lock for each possible location.
            for _ in range(0, maximum + 1):
                self.location_locks.append(Lock())

            # Block Logic: Propagate the initialized location_locks to all other devices.
            for device in devices:
                device.location_locks = self.location_locks

            self.can_start.set() # Signal that the leader has completed its setup.
        else:
            # Block Logic: Non-leader devices wait for the leader to complete setup.
            devices[self.leader].can_start.wait()

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific location.
        If a script is provided, it's added to the device's script list.
        If no script is provided (None), it signals that a timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None to signal timepoint completion.
            location (str): The location associated with the script.
        """
        

        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list
            self.script_received.set() # Signal that a new script is available
        else:
            self.timepoint_done.set() # Signal that processing for the current timepoint is complete

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            Any or None: The sensor data if available for the location, otherwise None.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (str): The location for which to set data.
            data (Any): The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its main device thread.
        """
        
        self.thread.join()

    def finished(self):
        """
        Signals that a device has completed its script execution for the current timepoint
        and synchronizes with the leader device.
        """
        # Block Logic: Acquire the condition variable lock on the leader device.
        self.devices[self.leader].condition.acquire()
        # Increment the count of finished threads on the leader device.
        self.devices[self.leader].finishedthread += 1

        # Block Logic: If not all devices have finished, wait. Otherwise, notify all waiting devices.
        if self.devices[self.leader].finishedthread != len(self.devices):
            # Block Logic: Wait for all other devices to finish their work.
            self.devices[self.leader].condition.wait()
        else:
            self.devices[self.leader].condition.notifyAll() # Notify all waiting devices.
            self.devices[self.leader].finishedthread = 0 # Reset the finished threads counter.

        self.devices[self.leader].condition.release() # Release the condition variable lock.

class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for orchestrating script execution
    and data sharing within the device and with neighbors. It manages a `SlaveList`
    to distribute work to `Slave` threads and synchronizes with other devices.
    """
    

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.slavelist = SlaveList(device) # Initializes a SlaveList to manage worker threads

    def run(self):
        """
        Executes the main logic of the DeviceThread.
        - Continuously fetches neighbor information from the supervisor.
        - If no neighbors are returned (signal for shutdown), it shuts down the `SlaveList` and breaks.
        - Waits for scripts to be assigned for the current timepoint.
        - Distributes assigned scripts to `Slave` threads via the `SlaveList`.
        - Waits for all `Slave` threads to complete their work for the current timepoint.
        - Signals to the leader device that it has finished its work for the timepoint.
        """
        
        # Block Logic: Main loop for continuous processing of timepoints.
        while True:

            # Block Logic: Fetch updated neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: If no neighbors are returned, it's a shutdown signal.
            if neighbours is None:
                self.slavelist.shutdown() # Shut down the slave worker threads.
                break # Exit the main loop

            # Block Logic: Wait for scripts to be assigned for the current timepoint, then clear the event.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Block Logic: Distribute each assigned script to a slave thread for processing.
            for (script, location) in self.device.scripts:
                self.slavelist.do_work(script, location, neighbours)

            # Block Logic: Wait for all slave threads to finish processing their assigned scripts.
            self.slavelist.event_wait()

            # Block Logic: Signal to the leader device that this device has finished its work for the current timepoint.
            self.device.finished()

class SlaveList(object):
    """
    Manages a pool of 'Slave' worker threads for a `DeviceThread`. It distributes
    work (scripts) to available slaves, tracks their readiness, and provides
    synchronization mechanisms for the `DeviceThread`.
    """
    
    def __init__(self, device):
        """
        Initializes a SlaveList.

        Args:
            device (Device): The parent Device instance that this SlaveList serves.
        """

        self.device = device
        self.event = Event() # Event to signal when all slaves are done with current work.
        self.event.set() # Initially set, meaning all slaves are ready.
        self.semaphore = Semaphore(8) # Semaphore to control the number of active slaves (initially 8 available).
        self.lock = Lock() # Lock to protect shared data within SlaveList.
        self.slavelist = [] # List of all Slave thread instances.
        self.readythreads = [] # List of currently ready (idle) Slave threads.
        
        # Block Logic: Create and start 8 Slave worker threads.
        for _ in xrange(8):
            thread = Slave(self, self.device)
            self.slavelist.append(thread)
            self.readythreads.append(thread) # Initially all slaves are ready.
            thread.start()

    def do_work(self, script, location, neighbours):
        """
        Assigns a new script execution task to an available slave thread.
        Blocks if no slave is immediately available.

        Args:
            script (Script): The script object to be executed.
            location (str): The location associated with the script.
            neighbours (list): A list of neighboring Device instances relevant to this script.
        """
        
        # Block Logic: If the event is currently set (all slaves ready), clear it to indicate work is being distributed.
        if self.event.isSet():
            self.event.clear()
        self.semaphore.acquire() # Acquire a semaphore to get an available slave (blocks if none available).
        self.lock.acquire() # Acquire lock to safely access the readythreads list.
        slave = self.readythreads.pop(0) # Get an available slave.
        self.lock.release() # Release lock.
        slave.do_work(script, location, neighbours) # Assign work to the slave.

    def shutdown(self):
        """
        Shuts down all slave threads managed by this SlaveList.
        It signals each slave to terminate and waits for them to join.
        """
        
        # Block Logic: Iterate through all slave threads to signal termination and join them.
        for slave in self.slavelist:
            slave.imdone = True # Set termination flag for the slave.
            slave.semaphore.release() # Release semaphore to unblock the slave if it's waiting.
            slave.join() # Wait for the slave thread to finish.

    def slave_done(self, slave):
        """
        Callback method invoked by a `Slave` thread when it completes its work.
        It marks the slave as ready and signals the `DeviceThread` if all slaves are done.

        Args:
            slave (Slave): The `Slave` thread that has completed its work.
        """
        
        self.lock.acquire() # Acquire lock to safely access the readythreads list and event.
        self.readythreads.append(slave) # Add the slave back to the list of ready threads.

        # Block Logic: If the event is not set (meaning work was distributed), check if all slaves are now ready.
        if self.event.isSet() == False:
            if len(self.readythreads) == 8: # Assuming 8 slaves, check if all have returned.
                self.event.set() # Set the event, signaling that all slaves are ready.

        self.lock.release() # Release lock.
        self.semaphore.release() # Release semaphore, indicating one more slave is available.

    def event_wait(self):
        """
        Blocks until all slave threads have completed their assigned work for the current timepoint.
        """
        
        self.event.wait()



class Slave(Thread):
    """
    A worker thread (`Slave`) that performs script execution for a `Device`.
    It waits for tasks from the `SlaveList`, retrieves data, runs the assigned script,
    updates devices, and then signals its completion to the `SlaveList`.
    """
    
    def __init__(self, slavelist, device):
        """
        Initializes a Slave worker thread.

        Args:
            slavelist (SlaveList): The `SlaveList` managing this worker thread.
            device (Device): The parent Device instance this slave belongs to.
        """
        
        Thread.__init__(self)


        self.slavelist = slavelist # Reference to the managing SlaveList.
        
        self.semaphore = Semaphore(0) # Semaphore to wait for work assignments.
        self.device = device # Reference to the parent Device.
        self.script = None # Placeholder for the assigned script.
        self.location = None # Placeholder for the assigned location.
        self.neighbours = None # Placeholder for the list of neighboring devices.
        self.imdone = False # Flag to signal termination.

    def do_work(self, script, location, neighbours):
        """
        Assigns a new script execution task to this slave thread and
        releases its semaphore to unblock the `run` method.

        Args:
            script (Script): The script object to be executed.
            location (str): The location associated with the script.
            neighbours (list): A list of neighboring Device instances relevant to this script.
        """
        
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.semaphore.release()


    def run(self):
        """
        The main execution loop for the Slave thread.
        It continuously waits for work assignments, retrieves data from devices,
        executes the assigned script, updates data on relevant devices, and
        then signals its completion to the `SlaveList`.
        The worker terminates if the `imdone` flag is set.
        """
        # Block Logic: Main loop for continuous processing of assigned work.
        while True:
            self.semaphore.acquire() # Block Logic: Wait for work to be assigned (semaphore released by `do_work`).
            values = [] # List to store collected data for the script.
            # Block Logic: Check if a termination signal has been received.
            if self.imdone is True:
                break # Exit the worker loop

            # Block Logic: Acquire a lock for the specific location to prevent race conditions during data access.
            self.device.location_locks[self.location].acquire()
            
            # Block Logic: Collect data from the current device itself for the assigned location.
            data = self.device.get_data(self.location)
            if data is not None:
                values.append(data)
            
            # Block Logic: Collect data from neighboring devices for the assigned location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    values.append(data)
            

            # Block Logic: If data was collected, execute the script and update devices.
            if values != []:
                # Inline: Execute the assigned script with collected data.
                result = self.script.run(values)

                # Block Logic: Propagate the script's result to neighboring devices.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                # Block Logic: Update the current device's sensor data with the script's result.
                self.device.set_data(self.location, result)
            # Block Logic: Release the lock for the current location after processing.
            self.device.location_locks[self.location].release()
            self.slavelist.slave_done(self) # Signal to the SlaveList that this slave has completed its work.
