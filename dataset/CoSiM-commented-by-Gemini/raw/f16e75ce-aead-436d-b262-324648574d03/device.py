"""
This module defines a simulated device and a reusable barrier for a
distributed sensor network.

The `Device` class represents a single node in the network, which processes
sensor data based on scripts assigned by a supervisor. The `ReusableBarrier`
class provides a synchronization mechanism for the devices.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A reusable barrier implemented using semaphores.

    This barrier synchronizes a fixed number of threads in two phases,
    allowing it to be reused multiple times.
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
        Causes a thread to wait at the barrier.

        When the required number of threads have called this method, all of them
        are released.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        A single phase of the barrier.

        Threads wait on a semaphore until all threads have arrived.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Pre-condition: If this is the last thread to arrive, release all
            # waiting threads for this phase.
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                    count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a single device in the simulated network.

    Each device runs a main thread to manage its lifecycle, including script
    execution and synchronization with other devices.

    Attributes:
        device_id (int): The unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data.
        supervisor: The supervisor object that manages the network.
        script_received (Event): An event to signal the arrival of new scripts.
        scripts (list): A list of scripts to be executed by the device.
        timepoint_done (Event): An event to signal the completion of a timepoint.
        thread (DeviceThread): The main thread for the device.
        locks (list): A list of locks for each location.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.locks = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device with shared resources for the network.

        The device with ID 0 is responsible for creating the barrier and
        location locks, which are then shared with all other devices.
        """
        
        # Pre-condition: This block is executed only by the device with ID 0,
        # which acts as the master for setting up shared resources.
        if self.device_id == 0:
            locks = []
            barrier = ReusableBarrier(len(devices))

            locations = -1
            for device in devices:
                for location, _ in device.sensor_data.iteritems():
                    if location > locations:
                        locations = location

            for _ in range(0, locations+1):
                locks.append(Lock())
            
            for device in devices:
                device.locks = locks
                device.barrier = barrier
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific location.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    """
    The main thread for a device.

    This thread manages the device's lifecycle, processing scripts for each
    timepoint and synchronizing with other devices.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread.

        It waits for scripts to be assigned, then distributes them among a
        fixed number of worker threads. After all scripts for a timepoint are
        executed, it waits at a barrier.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.script_received.wait()
            
            scripts_list = []
            for _ in range(0, 8):
                scripts_list.append([])
            
            index = 0
            # Block-level comment: Distributes the scripts among 8 lists to be
            # processed by worker threads.
            for (script, location) in self.device.scripts:
                scripts_list[index].append((script, location))
                index = (index + 1) % 8
            
            list_thread = []
            for lst in scripts_list:
                if len(lst) > 0:
                    list_thread.append(MyThread(self.device, neighbours, lst))
            
            for thr in list_thread:
                thr.start()
            
            for thr in list_thread:
                thr.join()

            
            self.device.barrier.wait()
            
            self.device.script_received.clear()

class MyThread(Thread):
    """
    A worker thread that executes a list of scripts.
    """
    
    def __init__(self, device, neighbours, lst):
        
        Thread.__init__(self)
        self.device = device
        self.lst = lst


        self.neighbours = neighbours

    def run(self):
        """
        Executes each script in the assigned list.

        For each script, it acquires a lock for the location, gathers data,
        runs the script, and updates the data.
        """
        for (script, location) in self.lst:
            script_data = []
            
            self.device.locks[location].acquire()
            # Invariant: Gathers data from all neighboring devices.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)
                
                self.device.set_data(location, result)

                
                # Invariant: The result is distributed to all neighbors.
                for device in self.neighbours:
                    device.set_data(location, result)
            self.device.locks[location].release()
        return