"""
Models a distributed system of devices executing computational scripts concurrently.

This module is another variant of a device simulation framework. It utilizes a
two-phase semaphore-based barrier for synchronization. It attempts a fine-grained
locking strategy with both per-device and per-location locks, but this design
is complex and likely to cause deadlocks. The script execution model creates a new
thread for each script, but the logic for joining these threads is flawed.
"""

from threading import * # Using wildcard import is generally discouraged.


class Device(object):
    """
    Represents a single device in the simulation. Each device runs a main control
    loop in its own thread and spawns 'slave' threads to execute scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        # A per-device lock, primarily for writing to its own sensor_data.
        self.lock_data = Lock()
        # A list of per-location locks, shared across all devices.
        self.lock_location = []
        # A shared barrier for synchronizing all devices at the end of a time step.
        self.time_barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (barrier, location locks) to all devices.
        This is expected to be called on a single 'master' device (device_id 0).
        """
        if self.device_id == 0:
            # Create the main time step barrier.
            self.time_barrier = ReusableBarrierSem(len(devices)) 

            for device in devices:
                device.time_barrier = self.time_barrier

            loc_num = 0

            # Find the maximum location ID to determine how many locks are needed.
            for device in devices:
                for location in device.sensor_data:
                    loc_num = max(loc_num, location) 
            # Create a shared list of locks, one for each location.
            for i in range(loc_num + 1):
                self.lock_location.append(Lock()) 

            # Distribute the shared list of location locks to all devices.
            for device in devices:
                device.lock_location = self.lock_location 

    def assign_script(self, script, location):
        """Assigns a script to the device or signals the end of assignments."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that script assignment for this timepoint is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Thread-safely updates sensor data for a given location."""
        with self.lock_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a single Device."""

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

        while True:
            slaves = []
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown.
                break

            # Wait for the supervisor to signal that all scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() 

            
            # Launch a new 'slave' thread for each assigned script.
            for (script, location) in self.device.scripts:
                slave = SlaveThread(script, location, neighbours, self.device) 
                slaves.append(slave)
                slave.start()

            # --- Wait for script threads to complete ---
            # BUG: This loop is flawed. It modifies the list `slaves` while iterating
            # over its original length, resulting in only half the threads being joined.
            for i in range(len(slaves)):
                slaves.pop().join()

            # Synchronize with all other devices to mark the end of the computation step.
            self.device.time_barrier.wait() 

class SlaveThread(Thread):
    """A worker thread to execute a single script."""
    def __init__(self, script, location, neighbours, device):
        

        Thread.__init__(self, name="Slave Thread of Device %d" % device.device_id)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        
        device = self.device
        script = self.script
        location = self.location
        neighbours = self.neighbours
        
        data = device.get_data(location)
        input_data = []
        this_lock = device.lock_location[location] # The shared lock for this specific location.

        if data is not None:
            input_data.append(data) 

        # Acquire the location-specific lock to ensure atomic read/write across devices for this location.
        # NOTE: This complex locking can lead to deadlocks if another script on another device
        # tries to acquire locks in a different order (e.g., this location lock and the per-device `lock_data`).
        with this_lock: 
            # Gather data from all neighbors for this location.
            for neighbour in neighbours:
                temp = neighbour.get_data(location) 

                if temp is not None:
                    input_data.append(temp)

            if input_data != []: 
                result = script.run(input_data) 

                # Update data on all neighbors and the local device.
                for neighbour in neighbours:
                    neighbour.set_data(location, result) 

                device.set_data(location, result) 


class ReusableBarrierSem():
    """
    A reusable, two-phase synchronization barrier implemented using Semaphores.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        self.phase1()
        self.phase2()

    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the counter for the *other* phase. This is a common pattern
                # in two-phase barriers to prepare for the next cycle.
                self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire()
         
    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the counter for the *other* phase.
                self.count_threads1 = self.num_threads
         
        self.threads_sem2.acquire()
