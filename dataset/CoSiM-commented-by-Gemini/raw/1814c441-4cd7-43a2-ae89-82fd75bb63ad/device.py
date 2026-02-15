"""
This module defines a multi-threaded framework for a distributed device simulation.
It features a custom reusable barrier, a Device class that manages a pool of
worker threads, and a DeviceThread class that performs the computational work.
"""

from threading import Event, Thread, Lock, Condition


class ReusableBarrierCond(object):
    """
    A custom implementation of a reusable barrier using a Condition variable.
    
    This barrier allows a set number of threads to wait for each other to reach
    a common execution point before continuing. It automatically resets after
    all threads have passed.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a specific number of threads.
        
        Args:
            num_threads (int): The number of threads that must call wait()
                               before they are all released.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()  # The condition variable for synchronization.
        

    def wait(self):
        """
        Causes a thread to block until all `num_threads` have called this method.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread has arrived; notify all waiting threads to wake up.
            self.cond.notify_all()
            # Reset the barrier for the next use.
            self.count_threads = self.num_threads
        else:
            # Not all threads have arrived yet; wait to be notified.
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Represents a single device (node) which manages its own pool of worker threads.
    
    Each device contains local data and a list of scripts to execute. It coordinates
    with other devices using a global barrier and manages its internal threads with a
    local barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device object and its internal pool of worker threads.
        
        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): The local data repository for this device.
            supervisor (Supervisor): The central object coordinating the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event() # Signals that all scripts for a timepoint are assigned.
        self.my_lock = Lock()

        self.local_lock = Lock()
        self.setup_device = Event() # Signals that global device setup is complete.
        self.device_barrier = None # A global barrier for all threads across all devices.
        self.local_barrier = ReusableBarrierCond(8) # A local barrier for this device's threads.
        self.location_lock = {} # A shared dictionary of locks for data locations.
        self.neighbours = []
        self.threads = []
        # Create and start the device's internal pool of worker threads.
        for i in range(8):
            self.threads.append(DeviceThread(self, i))
        for thread in self.threads:
            thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global synchronization context for all devices in the simulation.
        
        This method should be called by a master device (e.g., device_id 0). It creates
        a single global barrier and a single shared lock dictionary and distributes
        them to all devices.
        """
        if self.device_id == 0:
            my_barrier = ReusableBarrierCond(len(devices)*8)
            my_location_lock = {}
            for device in devices:
                device.device_barrier = my_barrier
                device.location_lock = my_location_lock
                device.setup_device.set() # Signal that setup is complete.

    def assign_script(self, script, location):
        """
        Assigns a script to this device's list of tasks for the current timepoint.
        
        Args:
            script (Script): The script to execute. If None, it signals that
                             all scripts for the timepoint have been assigned.
            location (any): The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Ensure a lock exists for this location in the shared lock dictionary.
            if location not in self.location_lock:
                self.location_lock[location] = Lock()
        else:
            # A None script signifies the end of script assignment for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Gets data from the device's local sensor data.
        
        Note: This method is not thread-safe by itself. Locking must be
        handled by the caller (i.e., the DeviceThread).
        """
        val = self.sensor_data[location] if location in self.sensor_data else None
        return val

    def set_data(self, location, data):
        """
        Sets data in the device's local sensor data.
        
        Note: This method is not thread-safe by itself. Locking must be
        handled by the caller (i.e., the DeviceThread).
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all of the device's worker threads to complete."""
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread belonging to a Device's internal thread pool.
    
    This thread executes a portion of the device's assigned scripts in a
    coordinated, time-stepped simulation loop.
    """

    def __init__(self, device, thread_id):
        """
        Initializes the worker thread.
        
        Args:
            device (Device): The parent device this thread belongs to.
            thread_id (int): A unique ID (0-7) for this thread within the device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main simulation loop for the worker thread."""
        # Wait until the master device has completed the global setup.
        self.device.setup_device.wait()

        while True:
            # --- Start of Timepoint Synchronization ---
            # All threads from all devices wait here, synchronizing the start of a timepoint.
            self.device.device_barrier.wait()
            index = self.thread_id

            # Only thread 0 of each device is responsible for fetching the neighbor list.
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # All threads within this device wait here until the neighbor list is fetched.
            self.device.local_barrier.wait()

            # If the neighbor list is None, the simulation is over.
            if self.device.neighbours is None:
                break

            # Wait until the supervisor has finished assigning all scripts for this timepoint.
            self.device.timepoint_done.wait()
            
            # --- Start of Computation ---
            # Process a subset of the scripts based on thread_id (static work distribution).
            while index < len(self.device.scripts):
                (script, location) = self.device.scripts[index]
                index += 8 # Move to the next script this thread is responsible for.
                script_data = []

                # Acquire the shared lock for this specific location to ensure data consistency.
                self.device.location_lock[location].acquire()

                # Gather data from all neighbors.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Run the script on the aggregated data.
                    result = script.run(script_data)

                    # Write the result back to all neighbors.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    # Write the result back to the local device.
                    self.device.set_data(location, result)

                # Release the lock for this location.
                self.device.location_lock[location].release()

            # --- End of Timepoint Synchronization ---
            # All threads from all devices wait here, synchronizing the end of computation.
            self.device.device_barrier.wait()
            # Thread 0 of each device resets the timepoint event for the next cycle.
            if self.thread_id == 0:
                self.device.timepoint_done.clear()
