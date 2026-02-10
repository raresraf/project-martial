from threading import *

class Device(object):
    """
    Represents a device in a simulation where script execution is serial.

    Architectural Role: This version of the Device model simplifies the execution
    by removing worker threads for individual scripts. All script processing is
    handled sequentially within the device's main `DeviceThread`. It uses a
    custom, locally-defined barrier for synchronization, which is owned by a
    single designated device.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): The local data store for the device.
            supervisor (object): The central supervisor managing the simulation.
        """
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
        Initializes a shared barrier owned by the first device in the list.
        
        Functional Utility: This method establishes a shared barrier for all devices.
        However, it relies on a fragile pattern where only device 0 creates the
        barrier, and all other devices must access it through the device 0 instance.
        """
        self.devices = devices
        # Invariant: The first device in the list is designated as the owner
        # of the synchronization barrier.
        if self == devices[0]:
            self.bar = MyReusableBarrier(len(devices))
        
        pass # This `pass` statement has no effect.

    def assign_script(self, script, location):
        """Assigns a script to the device or signals the start of a time-step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script acts as the trigger to start the time-step.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data from the device's local data store.

        Warning: This method is not thread-safe. Concurrent calls from different
        DeviceThreads (e.g., a neighbor reading data while this device's thread
        is modifying it) could lead to race conditions.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates data in the device's local data store.
        
        Warning: Like get_data, this method is not thread-safe.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a Device, executing scripts serially.

    Functional Utility: This thread manages the device's participation in the
    time-stepped simulation. Unlike other versions, it does not spawn separate
    worker threads for scripts. Instead, it iterates through its assigned scripts
    and executes them one by one within its own `run` loop.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to signal the start of a time-step.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Block Logic: Serial script execution.
            # Invariant: All assigned scripts are executed sequentially in a single thread.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Data gathering phase.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Script execution and data propagation phase.
                if script_data:
                    result = script.run(script_data)
                    
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

            # All devices wait at the barrier owned by device 0. This creates a
            # hard dependency on device 0.
            self.device.devices[0].bar.wait()


class MyReusableBarrier():
    """
    A custom, two-phase reusable barrier implementation.

    Warning: This implementation contains logical flaws in its counter reset
    mechanism that can lead to deadlocks or race conditions under heavy thread
    contention. The counters for each phase are reset prematurely.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait until all threads have reached the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            # Bug: This counter should be reset in phase2, not here. Resetting it
            # on every entry to phase1 can break the logic of the second phase.
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire()
         
    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            # Bug: This counter should be reset in phase1 (ideally only when the
            # count is 0), not on every entry to phase2. This makes the barrier
            # unreliable for reuse.
            self.count_threads1 = self.num_threads
         
        self.threads_sem2.acquire()