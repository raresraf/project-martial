"""
@file device.py
@brief This file defines a simulated device using a multi-level threading model with batched script execution.
@details The design features a main device thread that spawns and manages worker script threads in batches of 8.
         Synchronization is handled by a shared barrier and a set of shared, location-specific locks. The setup
         logic is fragile, relying on device 0 to initialize shared resources. A critical flaw is that the
         `get_data` and `set_data` methods are not thread-safe, creating a risk of race conditions.
"""


from threading import Event, Thread, Lock
import barrier

class Device(object):
    """
    @brief Represents a single device in the simulated network.
    @details This class is responsible for holding device data and state, and for coordinating the setup
             of shared synchronization objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        
        self.devices = None
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Initializes and distributes a shared barrier to all device threads.
        @warning The setup logic is fragile, as it assumes that this method is called by device 0
                 to correctly initialize the shared barrier for all devices.
        @param devices A list of all Device objects in the simulation.
        """
        
        shared_barrier = barrier.ReusableBarrier(len(devices))

        
        # Block Logic: Device 0 is responsible for distributing the single barrier instance
        # to all other device threads.
        if self.device_id == 0:
            for i in xrange(len(devices)):
                devices[i].thread.barrier = shared_barrier

        
        
        self.devices = devices

    def assign_script(self, script, location):
        """
        @brief Assigns a script and sets up a shared lock for its location if one doesn't exist.
        @details When a script for a new location is seen, a lock is created and distributed to the
                 `locations_lock` dictionary within every device's thread.
        @param script The script to be executed.
        @param location The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Block Logic: Create and distribute a shared lock for any new location.
            if location not in self.thread.locations_lock:
                loc_lock = Lock()
                for i in xrange(len(self.devices)):
                    self.devices[i].thread.locations_lock[location] = loc_lock

        else:
            # A `None` script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @warning This method is not thread-safe. It directly accesses the sensor_data dictionary
                 without acquiring a lock. It relies on an external caller to hold the lock.
        @return The sensor data or None if not found.
        """
        return self.sensor_data[location] \
        if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data at a given location.
        @warning This method is not thread-safe. It directly modifies the sensor_data dictionary
                 without acquiring a lock, which can lead to data corruption.
        @param location The location to update.
        @param data The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main control thread for a device, managing script execution in batches.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = None
        self.script_threads = []
        # A dictionary to hold locks, shared between all devices, indexed by location.
        self.locations_lock = {}

    def run(self):
        """
        @brief The main execution loop, which spawns and manages worker threads in batches of 8.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            
            # Invariant: Wait until all scripts for the current time step are assigned.
            self.device.timepoint_done.wait()

            count = 0
            
            # Block Logic: Process scripts by spawning worker threads in batches of 8.
            for (script, location) in self.device.scripts:
                # After starting 8 threads, wait for them to complete before starting more.
                # This serializes the work into batches.
                if count == 8:
                    count = 0
                    for i in xrange(len(self.script_threads)):
                        self.script_threads[i].join()
                    del self.script_threads[:]

                script_thread = ScriptThread(self.device, script, location,\ 
                    neighbours, count, self.locations_lock)

                self.script_threads.append(script_thread)
                script_thread.start()
                count = count + 1

            
            # Wait for the final batch of threads to complete.
            for i in xrange(len(self.script_threads)):
                self.script_threads[i].join()

            
            self.device.timepoint_done.clear()
            
            
            # Invariant: After all local work is done, wait at the global barrier.
            self.barrier.wait()

class ScriptThread(Thread):
    """
    @brief A short-lived worker thread that executes a single script.
    """

    def __init__(self, device, script, location, neighbours, i, locations_lock):
        Thread.__init__(self, name="Script Thread %d%d" % (device.device_id, i))
        self.device = device


        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.locations_lock = locations_lock


    def run(self):
        """
        @brief Executes the script logic within a location-specific lock.
        """
        script_data = []

        
        
        
        # Acquire the shared lock for this specific location to ensure that only one
        # script targeting this location runs at a time across the entire system.
        self.locations_lock[self.location].acquire()

        
        # Block Logic: Gathers data using non-thread-safe methods. This is only safe
        # because the external lock is held.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:

            
            result = self.script.run(script_data)



            for device in self.neighbours:
                device.set_data(self.location, result)

            self.device.set_data(self.location, result)

        
        self.locations_lock[self.location].release()