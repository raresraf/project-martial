from threading import Event, Thread, Semaphore, Lock

class ReusableBarrierCond():
    """
    A correct, two-phase reusable barrier based on Semaphores.
    
    Note: The class name `ReusableBarrierCond` is misleading, as the
    implementation uses Semaphores, not Condition variables.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
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
    Represents a device in a simulation that is critically flawed by a lack
    of data access synchronization.

    Architectural Role: This model uses a simple leader-follower pattern, where
    the device with the highest ID creates and distributes a shared barrier.
    However, it does not create or use any locks, making all shared data
    operations unsafe.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = list()

        self.bar = None
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id
    
    def assign_script(self, script, location):
        """Assigns a script to the device or triggers the start of a time-step."""
        if script is None:
            self.timepoint_done.set()
        else:
            self.scripts.append((script, location))
            self.script_received.set()
    
    def setup_devices(self, devices):
        """
        Initializes the shared barrier, with the highest-ID device acting as leader.
        """
        # Sort devices to easily find the one with the maximum ID.
        devices.sort(key=lambda x: x.device_id, reverse=True)
        id_maximum = devices[0].device_id
        
        if self.device_id == id_maximum: # Designates the max ID device as leader.
            barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                device.barrier = barrier
    
    def get_data(self, location):
        """
        Retrieves data. Warning: This operation is NOT thread-safe.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates data. Warning: This operation is NOT thread-safe.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device, executing scripts serially and without locks.
    
    Warning: The `run` method is critically flawed. It reads from and writes to
    neighboring devices' data stores without any locking, which will lead to
    unpredictable behavior and incorrect results due to race conditions.
    """
    def __init__(self, device):
        self.device = device
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        # This variable is initialized but never used.
        locki = 0

    def run(self):
        """The main simulation loop for the device."""
        neighbours = self.device.supervisor.get_neighbours()
        while neighbours is not None:
            # Wait for the supervisor to signal the start of a time-step.
            self.device.timepoint_done.wait()

            # Block Logic: Serial, non-thread-safe script execution.
            for (script, location) in self.device.scripts:
                script_data = list()

                # Unprotected data gathering phase. Race conditions will occur here.
                for neighbour in neighbours:
                    data = neighbour.get_data(location)
                    if data is None:
                        continue
                    script_data.append(data)
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Unprotected data propagation phase. Race conditions will occur here.
                if script_data:
                    result = script.run(script_data)
                    for neighbour in neighbours:
                        neighbour.set_data(location, result)
                    self.device.set_data(location, result)

            self.device.timepoint_done.clear()
            
            # Wait for all other devices to finish their (flawed) processing.
            self.device.barrier.wait()
            
            # Refresh neighbor list for the next iteration.
            neighbours = self.device.supervisor.get_neighbours()