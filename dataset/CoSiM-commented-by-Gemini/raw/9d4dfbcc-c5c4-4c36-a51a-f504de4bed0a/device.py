"""
@file device.py
@brief This file defines a simulated device using a complex synchronization scheme with multiple
       barriers and a centralized list of location-based locks.
@details The design features a coordinator device (ID 0) that initializes two global barriers
         (`time_bar`, `script_bar`) and a shared list of locks, one for each data location.
         A key, and unusual, feature is that the main control thread participates in one of the
         barriers. The design's primary flaw is that the data access methods (`get_data`, `set_data`)
         are not thread-safe, creating a risk of race conditions.
"""


from threading import Event, Thread, Lock
from barrier import RBarrier


class Device(object):
    """
    @brief Represents a single device in the simulated network.
    @details The device's execution is controlled by a single thread that navigates a complex
             multi-barrier synchronization protocol.
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

        # A barrier to synchronize the start of the time step computation.
        self.time_bar = None
        # A barrier to synchronize the end of script assignment.
        self.script_bar = None
        # A shared list of locks, indexed by location, to serialize access to data points.
        self.devloc = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Initializes and distributes shared barriers and a list of location-based locks.
        @details The device with ID 0 is the coordinator. It finds the maximum location ID to
                 determine how many locks to create.
        @param devices A list of all Device objects.
        """
        if self.device_id == 0:
            
            self.time_bar = RBarrier(len(devices))
            self.script_bar = RBarrier(len(devices))

            
            for device in devices:
                device.time_bar = self.time_bar
                device.script_bar = self.script_bar

            
            # Block Logic: Find the highest location ID to size the lock list.
            maxim = 0
            for device in devices:
                loc_list = device.sensor_data.keys()
                loc_list.sort()
                if loc_list[-1] > maxim:
                    maxim = loc_list[-1]

            
            # Create a list of locks, one for each location.
            while maxim >= 0:
                self.devloc.append(Lock())
                maxim = maxim - 1

            
            # Distribute the shared list of locks to all devices.
            for device in devices:
                device.devloc = self.devloc


    def assign_script(self, script, location):
        """
        @brief Assigns a script and participates in barrier synchronization.
        @details When script assignment is finished (script is None), the calling thread
                 (the main simulation controller) blocks on the `script_bar` barrier, waiting
                 for all device threads to also acknowledge completion of this phase.
        @param script The script to execute.
        @param location The location context for the script.
        """

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            # Inline: The main thread blocks here, becoming a participant in the barrier.
            # This is a very unconventional design.
            self.script_bar.wait()



    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @warning This method is not thread-safe. It accesses the sensor_data dictionary
                 directly without using the appropriate location-specific lock from `devloc`.
        @return The sensor data or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data at a given location.
        @warning This method is not thread-safe. It modifies the sensor_data dictionary
                 directly without using the appropriate location-specific lock.
        @param location The location to update.
        @param data The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main worker thread for a device.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop, defined by a rigid multi-barrier synchronization sequence.
        """
        while True:

            
            neighbours = self.device.supervisor.get_neighbours()

            
            # Invariant 1: Wait for the main thread to signal that script assignment is complete.
            self.device.script_received.wait()

            
            # Invariant 2: Wait at the script barrier to ensure all devices (and the main thread)
            # have finished the script assignment phase.
            self.device.script_bar.wait()

            
            # Invariant 3: Wait at the time barrier. This acts as a starting gate, ensuring all
            # devices begin their computation at the same time.
            self.device.time_bar.wait()

            if neighbours is None:
                # Termination signal received.
                break

            
            # Block Logic: Process all assigned scripts sequentially.
            for (script, location) in self.device.scripts:

                
                # Acquire the specific lock for this location to prevent data races
                # between scripts operating on the same location.
                self.device.devloc[location].acquire()

                script_data = []

                
                # Data is gathered using non-thread-safe methods.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    # Data is set using non-thread-safe methods.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                
                self.device.devloc[location].release()