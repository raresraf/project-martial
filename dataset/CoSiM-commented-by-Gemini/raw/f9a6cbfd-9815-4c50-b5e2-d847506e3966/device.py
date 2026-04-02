"""
This module implements a sixth variant of a distributed device simulation.

The architecture uses a master-worker pattern for initialization, where the device
with ID 0 creates and distributes shared synchronization objects: a global reusable
barrier and a dictionary of location-based locks. The computational model uses an
unbounded "spawn-and-join" approach, where for each time-step, a new thread is
created for every assigned script, executed, and then joined. This model can be
inefficient and lead to high resource consumption if many scripts are processed.
"""


from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier():
    """
    A custom, reusable barrier using a two-phase protocol with semaphores.
    This ensures that threads from a new cycle cannot start until all threads
    from the previous cycle have cleared the barrier.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal logic for a single phase of the barrier."""
        with self.count_lock:
            count_threads[0] -= 1
            
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                
                    threads_sem.release()
                
                count_threads[0] = self.num_threads
        
        threads_sem.acquire()
        

class Device(object):
    """
    Represents a device in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # This lock protects the device's own sensor_data dictionary.
        self.lock = Lock()
        self.locs = []
        # This will hold a reference to the global dictionary of location locks.
        self.hashset = {}
        # This will hold a reference to the global barrier.
        self.bariera = ReusableBarrier(1)
        
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources. The device with ID 0 acts as the master,
        creating a global barrier and a lock for each unique data location,
        then distributing these to all other devices.
        """
        if self.device_id == 0:
            self.hashset = {}
            # Create a dictionary of global locks, one per location.
            for device in devices:
                for location in device.sensor_data:
                    self.hashset[location] = Lock()
            # Create a global barrier for all devices.
            self.bariera = ReusableBarrier(len(devices))
            # Distribute the shared objects.
            for device in devices:
                device.bariera = self.bariera
                device.hashset = self.hashset

    def assign_script(self, script, location):
        """Assigns a script to be run. A None script signals the timepoint end."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safe method to get data from the device's local sensor storage.
        Note: This adds a second layer of locking, as the calling ScriptThread
        will already hold a global location lock.
        """
        self.lock.acquire()
        aux = self.sensor_data[location] if location in self.sensor_data else None
        self.lock.release()
        return aux

    def set_data(self, location, data):
        """
        Thread-safe method to set data in the device's local sensor storage.
        Note: This adds a second layer of locking.
        """
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, which orchestrates the execution of
    scripts for each time-step of the simulation.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """The main simulation loop."""
        while True:
            # Get neighbors for the current timepoint from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the signal that all scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()
            
            # Unbounded spawn-and-join: create a new thread for every script.
            # This can be inefficient if there are many scripts.
            list_threads = []
            for (script, location) in self.device.scripts:
                list_threads.append(ScriptThread(self.device, script,
                location, neighbours))
            
            # Start all worker threads.
            for i in xrange(len(list_threads)):
                list_threads[i].start()
            
            # Wait for all worker threads to complete.
            for i in xrange(len(list_threads)):
                list_threads[i].join()
            
            # Reset for the next timepoint.
            self.device.timepoint_done.clear()
            # Wait at the global barrier for all other devices to finish.
            self.device.bariera.wait()

class ScriptThread(Thread):
    """
    A short-lived thread that executes a single computational script.
    """

    def __init__(self, device, script, location, neighbours):
        """Initializes the ScriptThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script. This involves acquiring a global lock for the data
        location, gathering data, running the script, and disseminating the result.
        """
        # Acquire the global lock for this specific location.
        self.device.hashset[self.location].acquire()
        script_data = []
        
        # Gather data from all neighbors at the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the local device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Run the script if any data was collected.
        if script_data != []:
            
            result = self.script.run(script_data)

            # Update the data on all neighbors and the local device.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        # Release the global lock for the location.
        self.device.hashset[self.location].release()
