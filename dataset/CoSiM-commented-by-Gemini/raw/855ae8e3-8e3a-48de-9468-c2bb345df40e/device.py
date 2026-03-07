"""
This module implements a distributed device simulation framework in Python.

It features a unique synchronization model using two global locks and two reusable
barriers. The first device in the list acts as a coordinator to set up these
synchronization primitives. One lock appears to protect access to the shared
supervisor, while the other protects data update operations. The barriers are
used to synchronize the start and end of each simulation time step across all
devices. Worker threads for script execution are managed in a sequential,
batched manner.
"""

from threading import Event, Thread, Lock, Semaphore
from multiprocessing import cpu_count


class ReusableBarrier():
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a generic two-phase protocol with semaphores.
    The thread counter is stored in a single-element list, a technique to allow
    modification of the integer value across method calls (pass-by-reference-like).
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
 
    def wait(self):
        """Causes a thread to block until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the two-phase barrier protocol.

        Args:
            count_threads (list): A single-element list holding the current thread count.
            threads_sem (Semaphore): The semaphore to use for this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                # The last thread to arrive resets the counter and releases all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
                                                 


class Device(object):
    """
    Represents a device within the simulation.

    Each device is driven by a DeviceThread and coordinates with other devices
    using shared locks and barriers.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): The local sensor data for this device.
            supervisor (obj): The central supervisor managing the network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

    def __str__(self):
        """Returns a string representation of the device."""
        
        return "Device %d" % self.device_id

    def set_lock(self, lock1, lock2, barrier1, barrier2):
        """
        Assigns the shared synchronization primitives to this device.

        Args:
            lock1 (Lock): Lock for supervisor access.
            lock2 (Lock): Lock for data updates.
            barrier1 (ReusableBarrier): Barrier to signal script reception.
            barrier2 (ReusableBarrier): Barrier to signal end of a time step.
        """
        self.lock1=lock1
        self.lock2=lock2    
        self.script_received = barrier1
        self.timepoint_done = barrier2


    def setup_devices(self, devices):
        """
        Sets up the simulation environment.

        The first device in the list creates and distributes the shared locks and
        barriers to all other devices. It then starts the main thread for each device.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        

        
        if self.device_id==devices[0].device_id:
            lock1=Lock()
            lock2=Lock()
            barrier1=ReusableBarrier(len(devices))
            barrier2=ReusableBarrier(len(devices))
            for dev in devices:
                dev.set_lock(lock1, lock2, barrier1, barrier2)

        self.thread = DeviceThread(self)
        self.thread.start()        

    def assign_script(self, script, location):
        """Assigns a script to be run by the device."""
        
        if script is not None:
            self.scripts.append((script, location))

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread."""
        


        self.thread.join()


class MyThread(Thread):
    """A worker thread to execute one script."""
    def __init__(self, script, script_date, device, neighbours, location):
        """
        Initializes the worker thread.
        
        Args:
            script (obj): The script to execute.
            script_date (list): The input data for the script.
            device (Device): The parent device.
            neighbours (list): A list of neighboring devices.
            location (any): The data location being operated on.
        """
        Thread.__init__(self)
        self.script=script
        self.script_data=script_date
        self.result=None
        self.device=device
        self.neighbours=neighbours
        self.location=location

    def run(self):
        """
        Executes the script and updates data on the local device and its neighbors.
        
        A global lock (`lock2`) is used to protect all data update operations.
        """
        result = self.script.run(self.script_data)
        
        self.device.lock2.acquire()
        for device in self.neighbours:
            device.set_data(self.location, result)
        
        self.device.set_data(self.location, result)
        self.device.lock2.release()

class DeviceThread(Thread):
    """The main execution thread for a device."""
    

    def __init__(self, device):
        """Initializes the main thread for a given device."""
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop for the device.
        
        It coordinates the execution of scripts in batches and synchronizes with
        other devices using two barriers.
        """
        


        while True:
            
            # Use a global lock to get neighbor information from the supervisor.
            self.device.lock1.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.lock1.release()
            if neighbours is None:
                break # Supervisor signals shutdown.
    
            # Wait at a barrier until all devices are ready to process scripts.
            self.device.script_received.wait()
            
            threads=[]
            
            # Prepare worker threads for all assigned scripts.
            for (script, location) in self.device.scripts:
                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                        
                    threads.append(MyThread(script,script_data,self.device,neighbours,location))

            # Execute worker threads in sequential batches.
            # The size of a batch is twice the number of CPU cores.
            step=cpu_count()*2
            for i in range(0,len(threads),step):
                # Start a batch of threads.
                for j in range(step):
                    if i+j<len(threads):
                        threads[i+j].start()
                # Wait for the batch of threads to complete.
                for j in range(step):
                    if i+j<len(threads):
                        threads[i+j].join()

            
            # Wait at a barrier to signal that this device has finished its time step.
            self.device.timepoint_done.wait()
