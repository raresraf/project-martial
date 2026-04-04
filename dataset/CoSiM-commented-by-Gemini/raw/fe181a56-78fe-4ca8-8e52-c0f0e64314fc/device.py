"""
This module defines an advanced simulation framework for a distributed device network.

It features a multi-threaded `Device` class that uses a pool of worker threads
(`DeviceThread`) to process scripts concurrently. A central device (ID 0)
coordinates the setup of shared synchronization primitives (barriers, locks)
for the entire network. This design allows for parallel script execution both
within a single device and across the network.
"""

from threading import Event, Thread, Semaphore, Lock
import Queue

class Device(object):
    """
    Represents a single, multi-threaded device in a distributed network simulation.

    Each device manages a pool of worker threads to process scripts and uses
    shared barriers and locks to synchronize with other devices in the network.
    Device 0 acts as a coordinator for setting up these shared resources.
    """
    
    
    num_threads = 8
    
    set_barrier = Event()
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data, keyed by location.
            supervisor (object): A supervisor object that manages the network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.aval_data = {}
        self.wait_threads = ReusableBarrier(Device.num_threads)
        self.neighbours = None
        self.thread = []
        # Each worker thread gets its own queue for script assignments.
        self.queues = [Queue.Queue() for i in range(Device.num_threads)]
        self.crt_que = 0
        for loc in sensor_data:
            self.aval_data[loc] = Event()
            self.aval_data[loc].set()
        self.lock = Lock()
        # Device 0 holds the global synchronization primitives.
        if device_id == 0:
            self.data_lock = {}
            self.wait_devices = None
            self.wholesomebarrier = None

    def __str__(self):
        """String representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared barriers and locks for all devices.

        This method relies on device 0 to act as the coordinator for creating
        and distributing the global synchronization objects.
        """
        
        # Find the coordinator device.
        for dev in devices:
            if dev.device_id == 0:
                info_dev = dev
                break
        # Coordinator device creates the shared resources.
        if self.device_id == 0:
            info_dev.wait_devices = ReusableBarrier(len(devices))
            info_dev.wholesomebarrier = ReusableBarrier(len(devices) * Device.num_threads)
            Device.set_barrier.set()
        # All devices create shared data locks via the coordinator.
        for dev in devices:
            for data in dev.sensor_data:
                if data not in info_dev.data_lock:
                    info_dev.data_lock[data] = Semaphore(1)
        Device.set_barrier.wait()
        # Start the worker thread pool for this device.
        for i in range(0, Device.num_threads):
            self.thread.append(DeviceThread(self, i, info_dev.data_lock,\
                 self.queues[i], info_dev.wait_devices, info_dev.wholesomebarrier))
            self.thread[i].start()

    def assign_script(self, script, location):
        """
        Assigns a script to a worker thread via its queue in a round-robin fashion.
        """
        
        if script is not None:
            self.scripts.append((script, location))
            self.queues[self.crt_que].put((script, location))
            self.crt_que += 1
            self.crt_que %= Device.num_threads
        else:
            # A None script signals the end of a timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's worker threads."""
        
        for i in self.thread:
            i.join()

class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads.
    
    Uses a two-phase implementation to allow for repeated use. The thread count
    is stored in a list to allow for mutable access within the phase method.
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
        """
        Executes one phase of the barrier synchronization.
        
        Args:
            count_threads (list): A list containing the current thread count for the phase.
            threads_sem (Semaphore): The semaphore used to block and release threads.
        """
        
        self.count_lock.acquire()
        count_threads[0] -= 1
        if count_threads[0] == 0:
            for i in range(self.num_threads):
                threads_sem.release()
            count_threads[0] = self.num_threads
        self.count_lock.release()
        threads_sem.acquire()

class DeviceThread(Thread):
    """
    A worker thread within a Device's internal thread pool.
    
    Each worker processes scripts assigned to it through a dedicated queue.
    """

    def __init__(self, device, id_thread, sem_loc, queue, wait_devices, wholesomebarrier):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.id_thread = id_thread
        self.device = device
        self.script_data = []
        self.aux = None
        self.script = None
        self.location = None
        self.sem_loc = sem_loc # Shared location locks from coordinator device
        self.data = 0
        self.result = 0
        self.queue = queue
        self.wait_devices = wait_devices # Shared barrier for all devices
        self.wholesomebarrier = wholesomebarrier # Shared barrier for all threads of all devices

    def run(self):
        """The main loop for the worker thread."""
        while True:
            
            # Thread 0 is responsible for fetching neighbor info for the device.
            if self.id_thread == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # This seems intended to re-queue scripts, but might be redundant
            # if scripts are only added once per timepoint.
            for i in range(self.id_thread, len(self.device.scripts), self.device.num_threads):
                self.queue.put(self.device.scripts[i])
            
            # Internal barrier for the device's own thread pool.
            self.device.wait_threads.wait()
            if self.device.neighbours is None:
                # Supervisor signals shutdown.
                break
            
            # Process scripts until the timepoint is done and the queue is empty.
            while (not self.queue.empty()) or (not self.device.timepoint_done.isSet()):
                self.script_data = []
                try:
                    
                    self.aux = self.queue.get(False)
                    self.script = self.aux[0]
                    self.location = self.aux[1]
                except Queue.Empty:
                    continue
                # Core logic: acquire lock, gather data, run script, update data, release lock.
                self.sem_loc[self.location].acquire()
                


                for device in self.device.neighbours:
                    self.data = device.get_data(self.location)
                    if self.data is not None:
                        self.script_data.append(self.data)
                
                self.data = self.device.get_data(self.location)
                if self.data is not None:
                    self.script_data.append(self.data)

                if self.script_data != []:
                    
                    self.result = self.script.run(self.script_data)
                    
                    for device in self.device.neighbours:
                        if device == self.device:
                            continue


                        if self.location in device.aval_data:
                            device.set_data(self.location, self.result)
                    self.device.set_data(self.location, self.result)
                self.sem_loc[self.location].release()

            
            # Global barrier for all threads across all devices.
            self.wholesomebarrier.wait()
            if self.id_thread == 0:
                
                # Thread 0 resets the timepoint event for the next cycle.
                self.device.timepoint_done.clear()
