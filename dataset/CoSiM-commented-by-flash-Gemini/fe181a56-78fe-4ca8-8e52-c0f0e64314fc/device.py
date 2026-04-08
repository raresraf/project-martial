"""
This module provides a simulation of a network of interconnected devices.

It employs a multi-queue producer-consumer model where each device has a pool
of worker threads, and each thread has its own dedicated work queue. The system
is synchronized using a hierarchical set of barriers.
"""

from threading import Event, Thread, Semaphore, Lock
import Queue

class Device(object):
    """
    Represents a single device in the simulation.

    This class acts as a dispatcher, receiving scripts and distributing them
    to its pool of worker threads in a round-robin manner.
    """
    
    num_threads = 8
    
    set_barrier = Event()
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): A dictionary of local sensor data.
            supervisor: The supervisor object providing network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.aval_data = {}
        # A barrier to synchronize worker threads within this device.
        self.wait_threads = ReusableBarrier(Device.num_threads)
        self.neighbours = None
        self.thread = []
        # Each worker thread gets its own dedicated queue.
        self.queues = [Queue.Queue() for i in range(Device.num_threads)]
        self.crt_que = 0
        for loc in sensor_data:
            self.aval_data[loc] = Event()
            self.aval_data[loc].set()
        self.lock = Lock()
        if device_id == 0:
            # Device 0 holds the globally shared synchronization objects.
            self.data_lock = {}
            self.wait_devices = None
            self.wholesomebarrier = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and shares global synchronization objects across all devices.

        Args:
            devices (list): All devices participating in the simulation.
        """
        for dev in devices:
            if dev.device_id == 0:
                info_dev = dev
                break
        if self.device_id == 0:
            # Device 0 creates the barriers.
            info_dev.wait_devices = ReusableBarrier(len(devices))
            info_dev.wholesomebarrier = ReusableBarrier(len(devices) * Device.num_threads)
            Device.set_barrier.set()
        
        # Device 0 creates a semaphore for each unique data location.
        for dev in devices:
            for data in dev.sensor_data:
                if data not in info_dev.data_lock:
                    info_dev.data_lock[data] = Semaphore(1)
        
        # All devices wait for setup to complete before starting threads.
        Device.set_barrier.wait()
        for i in range(0, Device.num_threads):
            self.thread.append(DeviceThread(self, i, info_dev.data_lock,\
                 self.queues[i], info_dev.wait_devices, info_dev.wholesomebarrier))
            self.thread[i].start()

    def assign_script(self, script, location):
        """
        Assigns a script to a worker thread's queue in a round-robin fashion.

        Args:
            script: The script to execute, or None to signal end of timepoint.
            location (int): The data location for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Distribute scripts to worker queues round-robin.
            self.queues[self.crt_que].put((script, location))
            self.crt_que += 1
            self.crt_que %= Device.num_threads
        else:
            # Signal to threads that no more scripts are coming for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data from the device's local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets data in the device's local sensor data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to complete."""
        for i in self.thread:
            i.join()

class ReusableBarrier(object):
    """
    A reusable, two-phase barrier for synchronizing a fixed number of threads.
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
        """Manages one phase of the barrier synchronization."""
        self.count_lock.acquire()
        count_threads[0] -= 1
        if count_threads[0] == 0:
            for i in range(self.num_threads):
                threads_sem.release()
            count_threads[0] = self.num_threads
        self.count_lock.release()
        threads_sem.acquire()

class DeviceThread(Thread):
    """A worker thread that executes scripts from its dedicated queue."""

    def __init__(self, device, id_thread, sem_loc, queue, wait_devices, wholesomebarrier):
        """
        Initializes the worker thread.

        Args:
            device (Device): The parent device.
            id_thread (int): The unique ID of this thread.
            sem_loc (dict): A shared dictionary mapping locations to semaphores.
            queue (Queue): The dedicated work queue for this thread.
            wait_devices (ReusableBarrier): Unused barrier (likely a remnant).
            wholesomebarrier (ReusableBarrier): The global barrier for all threads.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.id_thread = id_thread
        self.device = device
        self.script_data = []
        self.aux = None
        self.script = None
        self.location = None
        self.sem_loc = sem_loc
        self.data = 0
        self.result = 0
        self.queue = queue
        self.wait_devices = wait_devices
        self.wholesomebarrier = wholesomebarrier

    def run(self):
        """Main execution loop of the worker thread."""
        while True:
            # NOTE: The logic below for re-populating the queue seems redundant
            # as the `assign_script` method already populates it.
            if self.id_thread == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            for i in range(self.id_thread, len(self.device.scripts), self.device.num_threads):
                self.queue.put(self.device.scripts[i])
            
            # Synchronize threads within the same device.
            self.device.wait_threads.wait()
            if self.device.neighbours is None: # Shutdown signal.
                break
            
            # Process tasks until the queue is empty and the timepoint is marked as done.
            while (not self.queue.empty()) or (not self.device.timepoint_done.isSet()):
                self.script_data = []
                try:
                    self.aux = self.queue.get(False)
                    self.script = self.aux[0]
                    self.location = self.aux[1]
                except Queue.Empty:
                    continue
                
                # Acquire semaphore for the specific location.
                self.sem_loc[self.location].acquire()
                
                # Gather data from neighbors and local device.
                for device in self.device.neighbours:
                    self.data = device.get_data(self.location)
                    if self.data is not None:
                        self.script_data.append(self.data)
                
                self.data = self.device.get_data(self.location)
                if self.data is not None:
                    self.script_data.append(self.data)

                if self.script_data:
                    # Run script and update data.
                    self.result = self.script.run(self.script_data)
                    for device in self.device.neighbours:
                        if device == self.device:
                            continue
                        if self.location in device.aval_data:
                            device.set_data(self.location, self.result)
                    self.device.set_data(self.location, self.result)
                
                self.sem_loc[self.location].release()

            # Global sync point for all threads in the simulation.
            self.wholesomebarrier.wait()
            if self.id_thread == 0:
                # Thread 0 resets the timepoint event for the next cycle.
                self.device.timepoint_done.clear()
