from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier():
    """
    A correct, two-phase reusable barrier for thread synchronization.

    Functional Utility: This barrier ensures that a specified number of threads
    all arrive at a synchronization point before any are allowed to proceed.
    Its two-phase design makes it safely reusable in iterative algorithms.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters are wrapped in a list to be mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
 
    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for _ in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()   

class Device(object):
    """
    Represents a device in a simulation featuring on-demand lock creation.

    Architectural Role: This device model attempts to initialize location-based
    locks in a "lazy" manner, creating them only when a script for that location
    is first assigned. However, this implementation contains a critical race
    condition. The synchronization pattern is also altered, with devices
    synchronizing at a barrier at the beginning of each time-step.
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
        Initializes and distributes a shared barrier and an empty lock dictionary.
        
        Functional Utility: Device 0 creates the shared objects. The lock dictionary
        is intended to be populated dynamically during the simulation.
        """
        if self.device_id == 0:
            nr_devices = len(devices)
            bar = ReusableBarrier(nr_devices)
            # The lock dictionary is created empty and shared among all devices.
            dictionar = dict()
            for D in devices:
                D.barrier = bar
                D.dictionar = dictionar
        pass

    def assign_script(self, script, location):
        """
        Assigns a script and attempts to create a lock for its location on-demand.
        
        Warning: This method contains a critical race condition. If multiple devices
        concurrently receive a script for the same *new* location, they can all
        check `if not (location in self.dictionar)`, find it true, and then attempt
        to create and assign a new lock. This check-then-set operation is not
        atomic, which can lead to lost updates and unpredictable locking behavior.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()
        
        # Block Logic: On-demand, non-thread-safe lock creation.
        if not (location in self.dictionar):
            L = Lock()
            self.dictionar[location] = L

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
    The main control thread for a device, featuring a start-of-step barrier.

    Functional Utility: This thread manages the device's lifecycle. It differs from
    other versions by waiting at a barrier at the *start* of the time-step, ensuring
    all devices begin processing simultaneously. Script execution is serial within
    this thread.
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

            # Wait for all devices to be ready to start the time-step.
            self.device.barrier.wait()

            # Block Logic: Serially process all assigned scripts for this time-step.
            for (script, location) in self.device.scripts:
                # Acquire lock for the specific location. Assumes lock was created
                # in assign_script (which has a race condition).
                self.device.dictionar[location].acquire()
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
                
                self.device.dictionar[location].release()

            # Wait for the supervisor to signal that this time-step is complete and
            # scripts for the next one have been assigned.
            self.device.timepoint_done.wait()