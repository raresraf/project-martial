"""
@file device.py
@brief This file defines a simulated device for a distributed system. The system's execution is
       effectively serialized by a single global lock, and data access methods are not thread-safe.
@details The design uses a custom two-phase reusable barrier for synchronization between time steps.
         A single global lock is used around all script processing, which prevents any parallel
         execution between devices. A significant flaw is that the `get_data` and `set_data` methods
         do not use any locks, making them unsafe for concurrent access, although the global lock
         in the main loop mitigates this for script execution.
"""



from threading import Lock, Semaphore, Event, Thread

class ReusableBarrier(object):
    """
    @brief A custom, reusable two-phase barrier implemented with semaphores.
    @details This barrier ensures that a group of threads all wait at a synchronization point.
             The two-phase design prevents threads from one iteration from proceeding into the
             next before all threads have completed the current one.
    """
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock() 
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """
        @brief Blocks the calling thread until all participating threads have reached the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes one phase of the two-phase barrier.
        @param count_threads The counter for the current phase.
        @param threads_sem The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0: 
                # Last thread to arrive releases the semaphore for all waiting threads.
                i = 0
                while i < self.num_threads:
                    threads_sem.release() 
                    i += 1                
                count_threads[0] = self.num_threads  
        threads_sem.acquire() 
                              
                              

class Device(object):
    """
    @brief Represents a single device in the simulated network.
    @details Each device runs a single thread that processes scripts sequentially and uses
             shared objects for synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.barrier = None
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
        @brief Initializes and distributes a shared barrier and global lock.
        @details This method relies on device 0 to create the shared synchronization objects.
                 The logic is fragile and depends on the order of execution.
        @param devices A list of all Device objects.
        """
        if devices[0].barrier is None:
            if self.device_id == devices[0].device_id:
                bariera = ReusableBarrier(len(devices))
                my_lock = Lock()
                for device in devices:
                    device.barrier = bariera
                    device.lock = my_lock



    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @warning This method is not thread-safe. It accesses the shared sensor_data
                 dictionary without any locking mechanism.
        @return The sensor data or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data at a given location.
        @warning This method is not thread-safe. It modifies the shared sensor_data
                 dictionary without any locking mechanism.
        @param location The location to update.
        @param data The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main worker thread for a device, which processes all its scripts sequentially.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device thread.
        """
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A `None` value for neighbors signals termination.
                break



            # Invariant: Wait until all scripts for the current time step are assigned.
            self.device.timepoint_done.wait()

            
            # Block Logic: Process all assigned scripts sequentially.
            for (script, location) in self.device.scripts:
                # Inline: A single global lock is acquired here. This is a major performance bottleneck,
                # as it prevents any two devices from executing scripts concurrently. The entire
                # computation phase across all devices becomes sequential.
                self.device.lock.acquire()
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
                self.device.lock.release()
            
            self.device.timepoint_done.clear()
            # Invariant: Wait at the global barrier to synchronize with all other devices.
            self.device.barrier.wait()