from threading import Lock, Semaphore, Event, Thread

class ReusableBarrier(object):
    """
    A correct, two-phase reusable barrier for thread synchronization.

    Functional Utility: This barrier ensures that a specified number of threads
    all arrive at a synchronization point before any are allowed to proceed.
    Its two-phase design (using two separate counters and semaphores) makes it
    safely reusable in iterative algorithms, such as time-stepped simulations,
    by preventing threads from one iteration from interfering with threads from
    the previous one.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters are wrapped in a list to be mutable across method calls.
        self.count_threads1 = [self.num_threads] # Counter for the first phase (arrival).
        self.count_threads2 = [self.num_threads] # Counter for the second phase (reset).
        self.count_lock = Lock() 
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase.

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            # Invariant: The last thread to arrive is responsible for releasing all others.
            if count_threads[0] == 0:
                # Release all waiting threads for this phase.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for this phase, making it ready for the next iteration.
                count_threads[0] = self.num_threads
        # All threads wait here until the last thread releases the semaphore.
        threads_sem.acquire()

class Device(object):
    """
    Represents a device in a highly serialized simulation model.

    Architectural Role: This device model uses a single, global lock for all
    script-processing activities across the entire simulation. While this ensures
    thread safety, it prevents any parallel computation between devices, making the
    simulation effectively sequential at the script execution level.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.barrier = None
        # A single, global lock to be shared by all devices.
        self.lock = None
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
        Initializes and distributes a shared barrier and a single global lock.

        Functional Utility: Device 0 acts as the leader, creating one barrier and
        one lock instance that are then shared with all other devices.
        """
        # Invariant: Setup is performed only once, coordinated by device 0.
        if devices[0].barrier is None:
            if self.device_id == devices[0].device_id:
                bariera = ReusableBarrier(len(devices))
                my_lock = Lock()
                for device in devices:
                    device.barrier = bariera
                    device.lock = my_lock

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

class DeviceThread(Thread):
    """
    The main control thread for a device, executing scripts serially under a global lock.

    Functional Utility: This thread manages the device's lifecycle. It processes
    its assigned scripts sequentially. Critically, it acquires a global lock before
    processing each script, which means only one device in the entire simulation can
    be executing a script at any given moment, effectively serializing the core
    computational work of the simulation.
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
            
            # Block Logic: Serial script processing under a global lock.
            for (script, location) in self.device.scripts:
                # Acquire the single global lock. This serializes all script
                # execution across all devices.
                self.device.lock.acquire()
                
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
                
                # Release the global lock, allowing another device to proceed.
                self.device.lock.release()
            
            # Clear the event and wait at the barrier for all devices to finish.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()