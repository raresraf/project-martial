from threading import Event, Thread
from threading import Semaphore, Lock
# Assumes a 'Barrier.py' module with a ReusableBarrier class.
from Barrier import ReusableBarrier


class Device(object):
    """
    Represents a device in a simulation that uses a semaphore-bounded pool of
    worker threads to manage concurrency.

    Architectural Role: This model uses a leader (device 0) to initialize and
    distribute shared synchronization objects. The core of its design is a
    `Semaphore` that limits the number of concurrently executing script threads
    to a fixed size (8). This provides a robust and efficient way to handle a
    large number of scripts without overwhelming the system with too many threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        # Event to ensure the main thread doesn't run before setup is complete.
        self.initialization_event = Event()
        # A semaphore to limit the number of active worker threads to 8.
        self.free_threads = Semaphore(value=8)
        # A list of location-specific locks, populated during setup.
        self.locations = []
        self.barrier = None
        # A list to keep track of worker threads for the current time-step.
        self.device_threads = []

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier and location-based locks.
        (Leader-only method)
        """
        num_devices = len(devices)
        if self.device_id == 0: # Designates device 0 as the leader.
            locations = []
            # Note: The number of locations is hardcoded to 30, which is less
            # robust than dynamically calculating it from the sensor data.
            number_of_locations = 30
            while number_of_locations > 0:
                locations.append(Lock())
                number_of_locations -= 1

            barrier = ReusableBarrier(num_devices)

            # Distribute shared objects and signal all devices to start.
            for i in range(num_devices):
                devices[i].locations = locations
                devices[i].barrier = barrier
                devices[i].initialization_event.set()

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

    def clear_threads(self):
        """Waits for all worker threads from the current time-step to complete."""
        for thread in self.device_threads:
            thread.join()
        self.device_threads = []

    def shutdown(self):
        """Waits for all worker threads and the main device thread to terminate."""
        self.clear_threads()
        self.thread.join()

def execute(device, script, location, neighbours):
    """
    The target function for a worker thread, executing one script.
    
    Functional Utility: This function encapsulates the logic for a single unit
    of work. It uses a `with` statement for safe, exception-proof locking on
    the specific data location. Upon completion, it releases a semaphore to
    signal that a slot in the device's thread pool is now free.
    """
    # Use a `with` statement to ensure the location-specific lock is
    # acquired before and released after the critical section.
    with device.locations[location]:
        script_data = []
        # Data gathering phase.
        for dev in neighbours:
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        # Script execution and data propagation phase.
        if script_data:
            result = script.run(script_data)
            for dev in neighbours:
                dev.set_data(location, result)
            device.set_data(location, result)
            
    # Release the semaphore to signal that this thread has finished and a new
    # worker can be created.
    device.free_threads.release()

class DeviceThread(Thread):
    """
    The main control thread that dispatches scripts to a semaphore-bounded pool.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        # Wait until the leader has finished setting up shared resources.
        self.device.initialization_event.wait()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Supervisor signals shutdown.

            # Wait for the supervisor to signal the start of a time-step.
            self.device.timepoint_done.wait()
            
            # Block Logic: Dispatch all scripts using a semaphore to limit concurrency.
            for (script, location) in self.device.scripts:
                # This will block until a slot is free in the pool of 8.
                self.device.free_threads.acquire()
                
                device_thread = Thread(target=execute,
                           args=(self.device, script, location, neighbours))
                device_thread.start()
                self.device.device_threads.append(device_thread)
            
            self.device.timepoint_done.clear()

            # Wait for all started threads for this time-step to complete.
            self.device.clear_threads()
            
            # Synchronize with all other devices before starting the next time-step.
            self.device.barrier.wait()