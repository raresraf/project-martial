import barrier
from threading import Event, Thread, Lock

class Device(object):
    """
    Represents a device in a simulation that uses a fixed number of worker
    threads to process scripts.

    Architectural Role: This device model uses the *last* device in the network
    as the "leader" for initializing shared synchronization objects (a barrier
    and location-based locks). The main device thread (`DeviceThread`) then
    distributes the total script workload across a fixed pool of 8 worker
    threads (`MyThread`) for parallel processing.
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

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.

        Functional Utility: The last device in the list acts as the leader. It
        is responsible for creating a shared barrier and a dictionary of
        location-based locks, which are then assigned to all other devices.
        """
        # Invariant: The last device in the list is the designated leader for setup.
        if self.device_id == len(devices) - 1:
            # Assumes `barrier.py` provides a `ReusableBarrierCond` class.
            my_barrier = barrier.ReusableBarrierCond(len(devices))
            my_dictionary = dict()
            # Create a lock for each unique data location across all devices.
            for dev in devices:
                for location, data in dev.sensor_data.iteritems():
                    if location not in my_dictionary:
                        my_dictionary[location] = Lock()
            # Assign the shared objects to all devices.
            for dev in devices:
                dev.barrier = my_barrier
                dev.dictionary = my_dictionary

    def assign_script(self, script, location):
        """Assigns a script to the device or triggers the time-step to start."""
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

class DeviceThread(Thread):
    """
    The main control thread for a device, which distributes scripts to worker threads.

    Functional Utility: This thread orchestrates the device's participation in
    the simulation. Its primary role is to partition the list of assigned scripts
    and delegate the work to a fixed pool of 8 `MyThread` instances, which
    then run in parallel.
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
            
            threads = []
            
            # Wait for the supervisor to signal that the time-step should begin.
            self.device.timepoint_done.wait()

            # Block Logic: Distribute the total script workload across 8 threads.
            # The following algorithm divides the scripts into 8 (mostly) equal chunks.
            # Note: This implementation is complex and can be fragile. A simpler
            # list-slicing approach might be more robust.
            div = len(self.device.scripts) // 8
            mod = len(self.device.scripts) % 8
            for division in range(8):
                if div > 0:
                    list_of_scripts = \
                        self.device.scripts[division * div: (division + 1) * div]
                else:
                    list_of_scripts = []
                # Distribute the remainder of the scripts one by one to the first `mod` threads.
                if mod > 0:
                    list_of_scripts.append(
                        self.device.scripts[len(self.device.scripts) - mod]
                    )
                    mod = mod - 1
                threads.append(MyThread(self.device, list_of_scripts, neighbours))

            # Start all worker threads and wait for them to complete.
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # Clear the event and wait at the barrier for all other devices.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

class MyThread(Thread):
    """
    A worker thread that processes a list of scripts sequentially.

    Functional Utility: Each instance of this class is given a sub-list of the
    total scripts for a time-step. It processes these scripts one by one,
    acquiring and releasing the appropriate location-based lock for each to
    ensure data consistency.
    """
    def __init__(self, device, scripts, neighbours):
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        # Process the assigned chunk of scripts sequentially.
        for (script, location) in self.scripts:
            # Acquire the lock for the specific location to ensure exclusive access.
            self.device.dictionary[location].acquire()
            script_data = []

            # Data gathering phase (under lock).
            for dev in self.neighbours:
                data = dev.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Script execution and data propagation phase (under lock).
            if script_data:
                result = script.run(script_data)
                
                for dev in self.neighbours:
                    dev.set_data(location, result)
                
                self.device.set_data(location, result)
            
            # Release the lock for the location.
            self.device.dictionary[location].release()