"""
This module simulates a network of distributed devices that execute scripts
in synchronized time steps, coordinated by a barrier.

Key Components:
- Device: Represents a node in the network. A master-worker pattern is used,
  where device 0 is the master, responsible for creating the shared barrier
  and lock dictionary.
- DeviceThread: The main thread for each device. It launches new threads for each
  assigned script within a time step and handles the main synchronization logic.
- ReusableBarrierSem: A custom two-phase semaphore-based barrier to synchronize
  all devices at the end of each time step.
"""


from threading import Lock, Thread, Event, Semaphore


class Device(object):
    """
    Represents a single device in the network.
    
    The device with device_id 0 acts as the master, initializing shared resources
    like the barrier and location locks.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main control thread."""
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.location_locks = None
    def __str__(self):
        """String representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for all devices.
        
        Device 0 creates the shared barrier and lock dictionary. Other devices
        get a reference to these resources from device 0.
        """
        
        Device.devices_no = len(devices)
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            self.location_locks = {}
        else:
            self.barrier = devices[0].barrier
            self.location_locks = devices[0].location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to this device. A 'None' script signals the end of
        the timepoint.
        """
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It manages the execution of scripts
    and synchronization between time steps.
    """
    

    def __init__(self, device):
        """Initializes the main device thread."""
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
    def run_scripts(self, script, location, neighbours):
        """
        The logic for executing a single script. This method is the target
        for the worker threads created in the main `run` loop.
        """
        # Lazily initialize a lock for the location if it doesn't exist.
        # Note: This check-then-set is not atomic and could lead to a race condition
        # if multiple threads try to create the lock for the same new location at once.
        lock_location = self.device.location_locks.get(location)
        if lock_location is None and location is not None:
            self.device.location_locks[location] = Lock()
            lock_location = self.device.location_locks[location]
        lock_location.acquire()
        script_data = []
        
        # Gather data from neighboring devices.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
            
        # Gather data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        if script_data != []:
            
            # Execute the script and propagate the result.
            result = script.run(script_data)

            
            for device in neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)
        lock_location.release()

    def run(self):
        """
        The main device lifecycle loop.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals termination.
                break
            # Wait for the supervisor to signal that all scripts for this time step have been assigned.
            self.device.timepoint_done.wait()
            tlist = []
            # For each assigned script, spawn a new thread to execute it.
            for (script, location) in self.device.scripts:
                thread = Thread(target=self.run_scripts, args=(script, location, neighbours))
                tlist.append(thread)
                thread.start()
            # Wait for all script threads for this timepoint to complete.
            for thread in tlist:
                thread.join()
            self.device.timepoint_done.clear()
            # Synchronize with all other devices before the next time step.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    A custom, reusable barrier for synchronizing multiple threads using semaphores.
    It uses a two-phase protocol to prevent race conditions on reuse.
    """
    

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        

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
        """First phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive releases all other waiting threads for this phase.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase, ensures all threads have exited phase 1 before reset."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread to arrive releases all others for the next cycle.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()
