"""
This module provides a sixth variant of the distributed device simulation.
This implementation uses a master device for setup, a persistent worker thread pool,
and a two-barrier synchronization system. The work distribution from the main thread
to the worker threads is implemented with a list and a semaphore, which is not
a thread-safe pattern.
"""
from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a device in the simulation, coordinated by a master device (ID 0)
    that sets up shared synchronization objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's local sensor data.
            supervisor: The central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals that scripts have been assigned.
        self.scripts = []
        self.timepoint_done = Event() # Note: This event is not used in this version.
        self.thread = DeviceThread(self) # The main control thread.
        self.thread.start()
        
        # Two-barrier system for multi-stage synchronization.
        self.time_bar = None # Main barrier for end-of-timepoint synchronization.
        self.script_bar = None # Barrier for post-script-assignment synchronization.
        
        # `devloc` will hold the shared list of locks for data locations.
        self.devloc = []

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the simulation environment. Device 0 acts as the master.

        The master device creates and distributes two shared barriers and a list
        of locks for all data locations to every device in the simulation.
        """
        # Block Logic: The device with ID 0 is the master for setup.
        if self.device_id == 0:
            # Create two barriers for a two-stage synchronization process.
            self.time_bar = ReusableBarrierSem(len(devices))
            self.script_bar = ReusableBarrierSem(len(devices))

            # Distribute the barriers to all other devices.
            for device in devices:
                device.time_bar = self.time_bar
                device.script_bar = self.script_bar

            # Determine the total number of unique locations to create locks for.
            maxim = 0
            for device in devices:
                if device.sensor_data:
                    loc_list = device.sensor_data.keys()
                    loc_list.sort()
                    if loc_list[-1] > maxim:
                        maxim = loc_list[-1]
            
            # Create a lock for each location.
            while maxim >= 0:
                self.devloc.append(Lock())
                maxim -= 1

            # Distribute the list of locks to all devices.
            for device in devices:
                device.devloc = self.devloc


    def assign_script(self, script, location):
        """
        Assigns a script for the current timepoint or signals the end of assignment.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that script assignment is complete for this device.
            self.script_received.set()
            # Wait until all other devices have also received this signal.
            self.script_bar.wait()



    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main control thread."""
        self.thread.join()



class ParallelScript(Thread):
    """
    A worker thread that processes scripts from a shared, non-thread-safe list.
    """
    def __init__(self, device_thread):
        Thread.__init__(self)
        self.device_thread = device_thread

    def run(self):
        while True:
            # Wait for the main thread to signal that work is available.
            self.device_thread.sem_scripts.acquire()
            
            # --- Non-Thread-Safe Work Queue ---
            # Accessing and modifying this shared list without a lock can lead to race conditions.
            # A `Queue.Queue` should be used here.
            nod = self.device_thread.to_procces[0]
            del self.device_thread.to_procces[0]
            
            # `None` is the sentinel value to terminate the worker thread.
            if nod is None:
                break
            
            neighbours, script, location = nod[0], nod[1], nod[2]

            # Acquire the specific lock for the data location.
            self.device_thread.device.devloc[location].acquire()
            # --- Critical Section ---
            script_data = []
            
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)
                
                for device in neighbours:
                    device.set_data(location, result)
                self.device_thread.device.set_data(location, result)
            # --- End Critical Section ---
            self.device_thread.device.devloc[location].release()


class DeviceThread(Thread):
    """
    The main control thread for a device, managing a persistent pool of worker threads.
    """
    def create_pool(self, device_thread):
        """Creates and starts the persistent pool of worker threads."""
        pool = []
        for _ in xrange(self.numar_procesoare):
            aux_t = ParallelScript(device_thread)
            pool.append(aux_t)
            aux_t.start()
        return pool

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.sem_scripts = Semaphore(0) # Semaphore to signal work to the pool.
        self.numar_procesoare = 8 # "Number of processors" in Romanian.
        self.to_procces = [] # A list used as a non-thread-safe queue.
        self.pool = self.create_pool(self)

    def run(self):
        """The main simulation loop, executed once per timepoint."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # --- Shutdown Sequence ---
                for _ in range(self.numar_procesoare):
                    self.to_procces.append(None)
                    self.sem_scripts.release() # Wake up workers to terminate.
                for item in self.pool:
                    item.join()
                break
            
            # Wait for the signal that script assignment for the timepoint is complete.
            self.device.script_received.wait()
            
            # --- Producer Logic ---
            # Add all script tasks to the shared list for the workers.
            for (script, location) in self.device.scripts:
                nod = (neighbours, script, location)
                self.to_procces.append(nod)
                self.sem_scripts.release() # Signal one worker.

            # Wait at the first barrier after dispatching all work.
            self.device.script_bar.wait()

            # Wait at the second barrier for the end of the timepoint.
            self.device.time_bar.wait()
            
            self.device.script_received.clear()