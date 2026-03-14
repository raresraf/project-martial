
"""
This module implements a multi-threaded simulation of a distributed system of devices.

This version features a manual implementation of a thread pool in the `DeviceThread`
class to manage concurrent script execution. It uses a `ReusableBarrier` for
synchronization between devices at the end of each timepoint.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue

class Device(object):
    """
    Represents a device in the distributed system.

    Each device runs in its own thread and communicates with a supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data for the device.
            supervisor: The supervisor object that manages the device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device with a list of other devices in the system.

        Initializes and shares a single barrier instance among all devices.

        Args:
            devices (list[Device]): A list of all devices in the system.
        """
        self.devices = devices
        self.barrier = ReusableBarrier(len(self.devices))
      
        for i in xrange(len(self.devices)):
            self.devices[i].barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script: The script to be executed.
            location: The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.

        Args:
            location: The location from which to retrieve data.

        Returns:
            The sensor data at the given location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location.

        Args:
            location: The location to update.
            data: The new data to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its thread."""
        self.thread.join()


class Worker(Thread):
    """
    A worker thread that executes a single script.
    """

    def __init__(self, device, neighbours, script, location):
        """
        Initializes the Worker.

        Args:
            device (Device): The device executing the script.
            neighbours (list[Device]): A list of neighboring devices.
            script: The script to execute.
            location: The location associated with the script.
        """
        Thread.__init__(self, name="Thread %d's Worker " % (device.device_id))
        
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location

    
    def run(self):
        """
        Executes the script.

        It gathers data from neighboring devices, runs the script, and then
        updates the data on all neighbors.
        """
        scriptData = []        
        data = self.device.get_data(self.location)
        
        if not data is None:
            scriptData.append(data)

        for device in self.neighbours:
            data = device.get_data(self.location)
            if not data is None:
                scriptData.append(data)


        if scriptData:
            
            newData = self.script.run(scriptData)

            for device in self.neighbours:
                device.set_data(self.location, newData)
            self.device.set_data(self.location, newData)

    def shutdown(self):
        """Shuts down the worker thread by joining it."""
        self.join()


class DeviceThread(Thread):
    """
    The main thread of execution for a Device.

    This thread manages a pool of worker threads to execute scripts concurrently.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)

        self.device = device
        

    def run(self):
        """
        The main loop for the device thread.

        It continuously waits for scripts, adds them to a queue, and manages a
        pool of worker threads to execute them.
        """
        
        q = Queue.Queue()

        
        listOfWorkers = []
        numberOfWorkers = 0

        while True:
            
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.script_received.wait()
            
            
            for (script, location) in self.device.scripts:
                q.put((script,location))
          
            while not q.empty():

                (script,location) = q.get()         

                
                if numberOfWorkers < 8:
                   
                    worker = Worker(self.device, neighbours, script, location)
                    listOfWorkers.append(worker)
                    worker.start()
                    numberOfWorkers += 1
                
                else:

                    index = -1
                    for i in range(len(listOfWorkers)):
                        if not listOfWorkers[i].is_alive():
                            listOfWorkers[i].shutdown()
                            index = i
                            break
                    listOfWorkers.remove(listOfWorkers[index])
                    
                    worker = Worker(self.device, neighbours, script, location)
                    listOfWorkers.insert(index,worker)
                    listOfWorkers[index].start()
                    numberOfWorkers+=1;


                q.task_done() 

            
            for i in range(len(listOfWorkers)):


                listOfWorkers[i].shutdown()

            self.device.timepoint_done.wait()       
            self.device.barrier.wait()              
            
            
            self.device.script_received.clear()     
            self.device.timepoint_done.clear()      



class ReusableBarrier():
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This barrier implementation uses a two-phase protocol to ensure that all
    threads wait at the barrier before any of them are allowed to proceed.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.counter_lock = Lock()                  
        self.threads_sem1 = Semaphore(0)            
        self.threads_sem2 = Semaphore(0)            
 
    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        Implements one phase of the barrier.

        Args:
            count_threads (list[int]): A list containing the count of remaining threads.
            threads_sem (Semaphore): The semaphore to signal when all threads have arrived.
        """
        with self.counter_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:               
                for i in range(self.num_threads):
                    threads_sem.release()           
                count_threads[0] = self.num_threads 
        threads_sem.acquire()     
