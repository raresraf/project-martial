from threading import Event, Thread, Lock
# Assumes a 'barrier.py' module with a ReusableBarrier class.
from barrier import ReusableBarrier


class Device(object):
    """
    Represents a device in a simulation that uses a dedicated worker thread
    for each assigned script.

    Architectural Role: This model uses a leader-follower pattern for setup.
    Device 0 acts as the "leader," determining the total number of data locations
    across the network, creating a corresponding list of locks, and distributing
    these locks along with a shared barrier to all other "follower" devices.
    Each device then processes its workload by spawning one new thread for every
    script it needs to execute in a time-step.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)

        # The barrier is initialized empty and populated by the leader during setup.
        self.barrier = ReusableBarrier(0)
        # The list of location-specific locks, populated during setup.
        self.locations_lock = []
        # A temporary list to hold worker threads for a single time-step.
        self.thread_list = []

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier and a list of location-specific locks.
        
        Functional Utility: This leader-driven method ensures that all devices share
        the exact same barrier and lock objects, which is critical for correct
        synchronization. The leader (device 0) is responsible for creating these
        objects and distributing them before starting the main thread of any device.
        """
        # A shared barrier for end-of-step synchronization.
        barrier = ReusableBarrier(len(devices))

        if self.device_id == 0: # Designates device 0 as the leader.
            # Block Logic: Determine the total number of unique data locations.
            locations = []
            for device in devices:
                if device is not None:
                    locations.append(max(device.sensor_data.keys()))
            no_locations = max(locations) + 1

            # Create a lock for each location.
            for i in xrange(no_locations):
                self.locations_lock.append(Lock())

            # Block Logic: Distribute shared objects to all devices.
            for device in devices:
                if device is not None:
                    device.barrier = barrier
                    # Assign the list of locks to each device.
                    for i in xrange(no_locations):
                        device.locations_lock.append(self.locations_lock[i])
                    # Start the main thread only after setup is complete.
                    device.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to the device or triggers the start of a time-step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from the device's local data store."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates data in the device's local data store."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class MyThread(Thread):
    """
    A worker thread responsible for executing a single script.
    
    Functional Utility: Each instance of this class handles one unit of work.
    It ensures thread safety by acquiring a location-specific lock before
    accessing any data and releasing it upon completion.
    """
    def __init__(self, device, neighbours, script, location):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location

    def run(self):
        # Acquire the specific lock for the target location to prevent data races.
        self.device.locations_lock[self.location].acquire()

        script_data = []
        # Data gathering phase (under lock).
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Script execution and data propagation phase (under lock).
        if script_data:
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

        # Release the lock for the location.
        self.device.locations_lock[self.location].release()

class DeviceThread(Thread):
    """
    The main control thread for a device, acting as a "thread-per-script" dispatcher.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Supervisor signals shutdown.

            # Wait for the supervisor to signal the start of a time-step.
            self.device.timepoint_done.wait()

            # Block Logic: Spawn one worker thread for each assigned script.
            # This provides a high degree of parallelism but may be inefficient
            # if many small scripts are assigned, due to thread creation overhead.
            for (script, location) in self.device.scripts:
                self.device.thread_list.append(MyThread(self.device, neighbours, script, location))

            # Start all worker threads for this time-step.
            for thread in self.device.thread_list:
                thread.start()

            # Wait for all worker threads to complete their execution.
            for thread in self.device.thread_list:
                thread.join()
            
            # Clear the list of completed threads.
            self.device.thread_list = []

            # Reset the event and wait at the barrier for all other devices to finish.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()