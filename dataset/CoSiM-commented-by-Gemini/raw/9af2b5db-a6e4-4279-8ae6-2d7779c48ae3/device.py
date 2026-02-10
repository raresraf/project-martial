"""
@file device.py
@brief This file defines a simulated device for a distributed system. Each device processes scripts
       sequentially in a single thread, and a static, shared, two-phase barrier is used for
       synchronization between devices.
@details The design uses a custom two-phase barrier (named with Romanian variables like 'faza' for 'phase')
         that is stored as a static member of the `DeviceThread` class. A critical flaw is the lack
         of synchronization on data access (`get_data`, `set_data`), which can lead to race conditions
         when one device reads data from a neighbor that is simultaneously writing to that same data.
"""


from threading import *

class Barrier():
    """
    @brief A custom, reusable two-phase barrier implemented with semaphores.
    @details This barrier ensures that a group of threads all wait at a synchronization point. The
             two-phase implementation (`fazaI` and `fazaII`, meaning "phase I" and "phase II")
             prevents threads from one iteration from starting the next before all threads have
             completed the current one.
    """

    def __init__(self):
        self.threads_num = 0
        self.count1_threads = 0
        self.count2_threads = 0
        self.counter_lock = Lock()
        self.semafor1 = Semaphore(0)
        self.semafor2 = Semaphore(0)

    def init_devices (self, dev_nr):
        """
        @brief Initializes the barrier with the total number of participating threads (devices).
        @param dev_nr The number of devices that will use the barrier.
        """
        self.threads_num = dev_nr
        self.count1_threads = dev_nr
        self.count2_threads = dev_nr

    def fazaI (self):
        """ @brief The first phase of the two-phase barrier wait. """
        with self.counter_lock:
            self.count1_threads -= 1
            if self.count1_threads == 0:
                # Last thread to arrive releases the semaphore for all waiting threads.
                for i in range (self.threads_num):
                    self.semafor1.release()
                self.count1_threads = self.threads_num
        self.semafor1.acquire()

    def fazaII (self):
        """ @brief The second phase of the two-phase barrier wait. """
        with self.counter_lock:
            self.count2_threads -= 1
            if self.count2_threads == 0:
                # Last thread to arrive releases the semaphore for all waiting threads.
                for i in range (self.threads_num):
                    self.semafor2.release()
                self.count2_threads = self.threads_num
        self.semafor2.acquire()

    def wait(self):
        """ @brief Blocks the calling thread until all participating threads have reached the barrier. """
        self.fazaI()
        self.fazaII()

class Device(object):
    """
    @brief Represents a single device in the simulated network.
    @details Each device runs a main control thread to process scripts and synchronize.
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
        @brief Initializes the static, shared barrier with the number of devices.
        @param devices A list of all Device objects in the simulation.
        """
        DeviceThread.bariera.init_devices(len(devices))

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script signals that all scripts for the time step have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @warning This method is not thread-safe. It directly accesses sensor data without any locks,
                 which can cause race conditions if another device's thread is simultaneously
                 modifying this device's data via `set_data`.
        @return The sensor data or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data at a given location.
        @warning This method is not thread-safe. It directly modifies sensor data without any locks,
                 which can lead to data corruption if another device's thread is reading it at the same time.
        @param location The location to update.
        @param data The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief The main worker thread for a device, processing scripts sequentially.
    @details All instances of this class share a single static `Barrier` object for synchronization.
    """

    
    # A static barrier instance shared by all DeviceThreads.
    bariera = Barrier()

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
                break

            # Invariant: Wait until all scripts for the current time step are assigned.
            self.device.timepoint_done.wait()

            
            # Block Logic: Scripts for this device are processed sequentially in a single thread.
            # This prevents race conditions within the device but not between devices.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Block Logic: Gathers data from neighbors. This is a potential race condition site,
                # as a neighbor's thread could be modifying its data at the same time.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    # Disseminates the result to neighbors, also a potential race condition.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

            
            self.device.timepoint_done.clear()

            # Invariant: After all work is done, wait at the shared global barrier.
            DeviceThread.bariera.wait()