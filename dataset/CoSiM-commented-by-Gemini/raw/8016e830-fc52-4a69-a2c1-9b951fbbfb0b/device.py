"""
@file device.py
@brief Simulates a network of interconnected devices that process sensor data.
@details This script defines a `Device` class that operates in a multi-threaded
environment to simulate a distributed sensor network. Each device can execute
scripts on its sensor data and synchronize the results with its neighbors.
The simulation is coordinated by a supervisor and uses barriers and locks for
synchronization. This appears to be a model for a distributed data processing
or consensus algorithm.
"""


from threading import Event, Thread, Lock
from barrier import ReusableBarrier

class Device(object):
    """
    @class Device
    @brief Represents a single device in the simulated network.
    @details Each device has a unique ID, its own sensor data, and can communicate
    with a supervisor. It receives and executes scripts, and synchronizes with
    other devices using a reusable barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id The unique identifier for the device.
        @param sensor_data A dictionary representing the device's sensor data.
        @param supervisor A supervisor object that manages the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        
        self.locks = []
        
        self.barrier = ReusableBarrier(0)

    def __str__(self):
        """
        @brief String representation of the Device.
        @return A string identifying the device by its ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the network of devices, initializing locks and barriers.
        @details This method is intended to be called on the master device (device_id 0)
        to initialize synchronization primitives for the entire network. It creates
        a lock for each data location and a barrier for synchronizing all devices.
        @param devices A list of all Device objects in the network.
        """
        if self.device_id == 0:

            # Determine the total number of unique data locations across all devices.
            nr_locations = 0
            for i in xrange(len(devices)):
                nr_locations = max(nr_locations,
            		max(devices[i].sensor_data.keys()))

            # Create a lock for each location to ensure thread-safe access.
            for i in xrange(nr_locations + 1):
                self.locks.append(Lock())

            # A reusable barrier to synchronize all devices at the end of a timepoint.
            barrier = ReusableBarrier(len(devices))

            for i in xrange(len(devices)):
            	# Assign the shared barrier and locks to each device.
                devices[i].barrier = barrier
                
                for j in xrange(nr_locations + 1):
                    devices[i].locks.append(self.locks[j])

            # Start the main thread for each device.
            for i in xrange(len(devices)):
                devices[i].thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device.
        @param script The script to execute. If None, it signals the end of a timepoint.
        @param location The data location on which the script will operate.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data from a specific location.
        @param location The location from which to retrieve data.
        @return The sensor data at the given location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates sensor data at a specific location.
        @param location The location to update.
        @param data The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief The main execution thread for a Device.
    @details This thread runs a loop that processes scripts for each timepoint. It waits for
    scripts to be assigned, creates worker threads to execute them, and then waits at a
    barrier for all devices to complete the timepoint.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main loop of the device thread.
        """
        while True:
            # Get the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit loop if the simulation is over.

            
            # Wait until the supervisor signals that all scripts for the current
            # timepoint have been assigned.
            self.device.timepoint_done.wait()

          	
            worker_list = []

            # Create a worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                worker_list.append(Worker(self.device,
                	location, script, neighbours))

           	
            # Start all worker threads.
            for i in xrange(len(worker_list)):
                worker_list[i].start()

            # Wait for all worker threads to complete.
            for i in xrange(len(worker_list)):
                worker_list[i].join()

            # Wait at the barrier for all other devices to finish their timepoint.
            self.device.barrier.wait()

            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()


class Worker(Thread):
    """
    @class Worker
    @brief A worker thread that executes a single script on sensor data.
    @details The worker acquires a lock for a specific data location, gathers data from
    the device and its neighbors, runs the script, and then updates the data on all
    relevant devices with the script's result.
    """
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self, name="Worker")
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic of the worker thread.
        """
		
        # Acquire a lock on the data location to prevent race conditions.
        self.device.locks[self.location].acquire()

        script_data = []

		
        # Gather data from neighboring devices at the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

		
        # Gather data from the current device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
			
            # Execute the script with the gathered data.
            result = self.script.run(script_data)

			
            # Propagate the result to all neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)

			
            # Update the data on the current device.
            self.device.set_data(self.location, result)

		
        # Release the lock.
        self.device.locks[self.location].release()
