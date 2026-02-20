"""
This module defines a device simulation where each device spawns a new thread
for every script it must execute in a time step.

The synchronization and resource setup (for barriers and locks) is handled
in a decentralized manner, which is prone to race conditions.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier():
    """
    A reusable two-phase barrier implemented with semaphores, for synchronizing
    multiple threads at a specific point in their execution, multiple times.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Use a list to hold the count, allowing it to be modified by reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """Blocks the caller until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes a single phase of the two-phase barrier protocol."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()


class Device(object):
    """
    Represents a device in the network. It attempts to manage shared resources
    like barriers and locks in a decentralized way.
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
        self.devices = []
        self.reusable_barrier = None
        self.thread_list = []
        self.location_lock = [None] * 99 # Pre-allocates a list for location locks.

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources in a decentralized manner.
        @note This method has a race condition. If multiple devices call it
        concurrently, they may create and propagate different barrier objects,
        leading to a deadlock.
        """
        if self.reusable_barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.reusable_barrier = barrier
            for device in devices:
                if device.reusable_barrier is None:
                    device.reusable_barrier = barrier

        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script and handles lazy, decentralized lock initialization.
        @note The lock creation logic is not atomic and can lead to race
        conditions where multiple devices create different lock objects for the
        same location.
        """
        is_location_locked = 0
        if script is not None:
            self.scripts.append((script, location))
            # If this device doesn't have a lock for the location...
            if self.location_lock[location] is None:
                # ...search all other devices to see if they have one.
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        is_location_locked = 1
                        break
                # If no other device had a lock, create a new one.
                if is_location_locked == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set()

        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. Spawns a new worker thread for
    each script in a timepoint.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for supervisor to assign all scripts.
            self.device.timepoint_done.wait()

            # Create a new worker thread for each script.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.thread_list.append(thread)

            # Start all worker threads for this timepoint.
            for thread in self.device.thread_list:
                thread.start()

            # Wait for all worker threads to complete.
            for thread in self.device.thread_list:
                thread.join()

            self.device.thread_list = []

            self.device.timepoint_done.clear()
            # Synchronize with all other devices before the next timepoint.
            self.device.reusable_barrier.wait()

class MyThread(Thread):
    """
    A worker thread that executes a single script.
    """

    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script logic: lock, collect data, compute, update, and unlock.
        """
        self.device.location_lock[self.location].acquire()
        script_data = []
        
        # Collect data from neighbors and the local device.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # If data was found, run the script and update the network.
        if script_data != []:
            
            result = self.script.run(script_data)

            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        self.device.location_lock[self.location].release()
