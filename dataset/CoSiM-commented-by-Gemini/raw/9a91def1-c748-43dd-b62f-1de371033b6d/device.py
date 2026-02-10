"""
@file device.py
@brief This file defines a simulated device for a distributed system that uses shared, location-specific locks
       for script execution, but lacks thread-safety in its data access methods.
@details Each device runs a single thread that processes assigned scripts. A key feature of this design is the
         dynamic creation and sharing of locks: the first time a script for a specific location is seen, a lock
         for that location is created and distributed to all other devices. However, the data access methods
         (`get_data`, `set_data`) do not use these locks internally, creating a significant risk of race conditions.
"""


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    """
    @brief Represents a single device in the simulated network.
    @details This class manages sensor data and script execution for a device. It participates in a
             lock-sharing scheme to serialize access to data locations across the network.
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
        # A fixed-size array to hold shared locks. The size 100 is a magic number.
        self.locks = [None] * 100
        self.devices = None
        self.barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs collective setup for the devices.
        @details Any device can initialize and distribute the global barrier if it hasn't been set yet.
        @param devices A list of all Device objects in the simulation.
        """
        
        self.devices = devices
        # This setup is racy if called concurrently, but works in a sequential setup.
        # The first device to check creates and propagates the barrier.
        if self.barrier is None:
            self.barrier = ReusableBarrierSem(len(devices))
            for i in self.devices:
                i.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script and sets up a shared lock for its location if one doesn't exist.
        @param script The script to be executed.
        @param location The location context for the script.
        """
        
        if script is not None:
            # Block Logic: This section implements a "first-one-in" lock initialization.
            # The first device to receive a script for a new location creates a lock and
            # broadcasts it to all other devices, ensuring they all share the same lock object.
            if self.locks[location] is None:
                self.locks[location] = Lock()
                for i in self.devices:
                    i.locks[location] = self.locks[location]

            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script signals that all scripts for the time step have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @warning This method is not thread-safe. It directly accesses the sensor_data dictionary
                 without acquiring the appropriate location-specific lock. It relies on an
                 external caller to hold the lock.
        @return The sensor data or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data at a given location.
        @warning This method is not thread-safe. It directly modifies the sensor_data dictionary
                 without acquiring the appropriate location-specific lock, which can lead to
                 data corruption if called concurrently.
        @param location The location to update.
        @param data The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main worker thread for a device, responsible for executing scripts.
    @details This thread acquires a location-specific lock before processing each script,
             but the underlying data access methods it calls are not internally synchronized.
    """
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device thread.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A `None` value for neighbors signals termination.
                break

            # Invariant: Wait until all scripts for the current time step are assigned.
            self.device.timepoint_done.wait()

            
            # Block Logic: Process all assigned scripts.
            for (script, location) in self.device.scripts:
                script_data = []

                # Acquire the shared lock for the specific location. This serializes the execution
                # of scripts operating on the same location across the entire system.
                self.device.locks[location].acquire()
                
                # The following data gathering and setting operations occur while the lock is held.
                # However, they call the non-thread-safe get_data and set_data methods.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    # Disseminate the result to neighbors and the local device.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                self.device.locks[location].release()

            self.device.timepoint_done.clear()
            
            # Invariant: After all work is done, wait at the global barrier to synchronize
            # with all other devices before starting the next time step.
            self.device.barrier.wait()