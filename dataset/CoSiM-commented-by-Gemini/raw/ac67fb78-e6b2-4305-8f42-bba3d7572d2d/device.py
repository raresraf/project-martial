from threading import Event, Thread, Lock
# Assumes the presence of a 'barrier.py' module with a ReusableBarrierCond class.
from barrier import ReusableBarrierCond

class Device(object):
    """
    Represents a device in a simulation using a leader-driven setup and
    batch-based script execution.

    Architectural Role: This class models a network device that processes data
    in a time-stepped simulation. It uses a leader-follower pattern where device 0
    is responsible for creating and distributing shared synchronization primitives
    (a barrier and location-based locks) to all other devices. Script execution
    is handled by its main thread, which batches tasks into groups of worker threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that all scripts for a time-step have been assigned.
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()
        # A dictionary of locks, keyed by data location, shared across all devices.
        self.dictLocks = {}

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects. (Leader-only method)

        Functional Utility: This method is executed by the designated leader device (device 0).
        It creates a reusable barrier and a comprehensive map of location-based locks,
        then actively pushes these shared objects to all other devices in the simulation
        by calling their `setup_mutualBarrier` method.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))

            # Block Logic: Create a lock for every unique data location across all devices.
            for device in devices:
                for location in device.sensor_data.keys():
                    if not self.dictLocks.has_key(location):
                        self.dictLocks[location] = Lock()
                # Push the shared barrier and locks to each follower device.
                device.setup_mutualBarrier(self.barrier, self.dictLocks)

    def setup_mutualBarrier(self, barrier, dictLocks):
        """
        Receives and stores shared synchronization objects. (Follower-only method)
        """
        if self.device_id != 0:
            self.barrier = barrier
            self.dictLocks = dictLocks

    def assign_script(self, script, location):
        """Assigns a script to be executed, or triggers the time-step to start."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A `None` script signals that all scripts for the time-step are ready.
            self.script_received.set()

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

def runScripts((script, location), neighbours, callingDevice):
    """
    The target function for a worker thread, executing a single script.

    Functional Utility: This function contains the core logic for a unit of work.
    It acquires a location-specific lock, gathers data from the local device and its
    neighbors, runs the script, and propagates the result back. It is executed by a
    generic `threading.Thread`.

    Args:
        (script, location) (tuple): The script to run and its target data location.
        neighbours (list[Device]): The list of neighboring devices.
        callingDevice (Device): The device instance that spawned this thread.
    """
    script_data = []
    
    # Acquire the shared lock for this specific location to ensure exclusive access.
    callingDevice.dictLocks[location].acquire()

    # Data gathering phase (under lock).
    for device in neighbours:
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)
            
    data = callingDevice.get_data(location)
    if data is not None:
        script_data.append(data)

    # Script execution and data propagation phase (under lock).
    if script_data:
        result = script.run(script_data)

        for device in neighbours:
            device.set_data(location, result)
            # Inefficiency: The local device's data is set on every iteration of
            # this loop, causing redundant writes.
            callingDevice.set_data(location, result)

    # Release the location-specific lock.
    callingDevice.dictLocks[location].release()

class DeviceThread(Thread):
    """
    The main control thread for a device, processing scripts in fixed-size batches.
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

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()

            threadsList = []
            index = 0
            nrScripts = len(self.device.scripts)
            
            # Block Logic: Process all assigned scripts in batches.
            # Invariant: This loop continues until all scripts (`nrScripts`) for the
            # current time-step have been processed. Scripts are handled in batches
            # of up to 8 to limit concurrency.
            while nrScripts:
                # Determine the size of the current batch.
                batch_size = 8 if nrScripts > 7 else nrScripts
                
                # Create threads for the current batch.
                for _ in range(batch_size):
                    threadsList.append(
                        Thread(target=runScripts, args=(self.device.scripts[index], neighbours, self.device))
                    )
                    index += 1
                
                nrScripts -= batch_size

                # Start and wait for all threads in the current batch to complete.
                for j in range(len(threadsList)):
                    threadsList[j].start()
                for j in range(len(threadsList)):
                    threadsList[j].join()

                # Reset the list for the next batch.
                threadsList = []

            # Clear the event and wait at the barrier for all other devices to finish.
            self.device.script_received.clear()
            self.device.barrier.wait()