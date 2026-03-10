


"""



Models a distributed network of devices that process sensor data concurrently.







This script provides an alternative implementation for a simulated system of



interconnected devices. Unlike a thread-pool based approach, it dynamically



spawns worker threads for each timepoint's tasks. It uses a custom,



reusable, semaphore-based barrier for synchronization.



"""







from threading import Event, Thread, Lock, Semaphore











class Device(object):



    """Represents a single device in the distributed sensor network.







    Each device manages its sensor data and executes assigned scripts. It elects one



    device as a master to initialize shared synchronization primitives.







    Attributes:



        device_id (int): A unique identifier for the device.



        sensor_data (dict): A dictionary holding the device's sensor readings.



        supervisor (Supervisor): An object for retrieving neighbor information.



        script_received (Event): Signals that a new script has been assigned.



        scripts (list): A list of (script, location) tuples to execute.



        timepoint_done (Event): Signals the completion of a simulation timepoint.



        thread (DeviceThread): The main thread of execution for this device.



        barrier (ReusableBarrierSem): A shared barrier for synchronization.



        map_locations (dict): A shared dictionary of locks for data locations.



    """



    







    def __init__(self, device_id, sensor_data, supervisor):



        """Initializes a Device instance."""



        self.device_id = device_id



        self.sensor_data = sensor_data



        self.supervisor = supervisor



        self.script_received = Event()



        self.scripts = []



        self.timepoint_done = Event()



        self.thread = DeviceThread(self)



        self.thread.start()







    def __str__(self):



        """Returns the string representation of the device."""



        return "Device %d" % self.device_id







    def setup_devices(self, devices):



        """Initializes shared synchronization objects for a group of devices.







        The device with the lowest `device_id` acts as the master, creating a



        single shared `ReusableBarrierSem` and a set of shared `Lock`s for all



        unique sensor data locations across all devices.







        Args:



            devices (list): A list of all Device objects in the simulation.



        """



        



        flag = True



        device_number = len(devices)







        



        for dev in devices:



            if self.device_id > dev.device_id:



                flag = False







        if flag == True:



            barrier = ReusableBarrierSem(device_number)



            map_locations = {}



            tmp = {}



            for dev in devices:



                dev.barrier = barrier



                tmp = list(set(dev.sensor_data) - set(map_locations))



                for i in tmp:



                    map_locations[i] = Lock()



                dev.map_locations = map_locations



                tmp = {}







    def assign_script(self, script, location):



        """Assigns a script to be executed by the device.







        A script value of None is a sentinel to signal the end of a timepoint.







        Args:



            script (Script): The script object to execute.



            location (str): The location context for the script execution.



        """



        if script is not None:



            self.scripts.append((script, location))



            self.script_received.set()



        else:



            self.timepoint_done.set()







    def get_data(self, location):



        """Retrieves sensor data for a specific location.







        Args:



            location (str): The location from which to retrieve data.







        Returns:



            The sensor data if the location exists, otherwise None.



        """



        return self.sensor_data[location] if location in self.sensor_data else None







    def set_data(self, location, data):



        """Updates sensor data for a specific location.







        Args:



            location (str): The location to update.



            data: The new sensor data.



        """



        if location in self.sensor_data:



            self.sensor_data[location] = data







    def shutdown(self):



        """Shuts down the device's execution thread."""



        self.thread.join()











class DeviceThread(Thread):



    """The main execution thread for a Device.







    This thread manages the device's lifecycle, orchestrating script execution



    and synchronization between timepoints by dynamically creating worker threads.







    Attributes:



        device (Device): The device instance this thread belongs to.



    """



    







    def __init__(self, device):



        """Initializes the DeviceThread."""



        Thread.__init__(self)



        self.device = device







    def run(self):



        """The main loop for the device thread.







        For each timepoint, it waits for scripts to be assigned, then spawns a



        fixed number of `SingleDeviceThread` workers to process them. After all



        workers complete, it synchronizes with other devices at a barrier.



        """



        while True:



            



            self.device.timepoint_done.clear()



            neighbours = self.device.supervisor.get_neighbours()



            if neighbours is None:



                break



            self.device.timepoint_done.wait()



            script_list = []



            thread_list = []



            index = 0



            for script in self.device.scripts:



                script_list.append(script)



            for i in xrange(8):



                thread = SingleDeviceThread(self.device, script_list, neighbours, index)











                thread.start()



                thread_list.append(thread)



            for i in xrange(len(thread_list)):



                thread_list[i].join()



            self.device.barrier.wait()







class SingleDeviceThread(Thread):



    """A worker thread to execute a single script task.







    Note: This implementation has a potential race condition. Multiple instances



    are created with the same `index` (0) and pop from the shared `script_list`,



    which is not a thread-safe operation and can lead to unpredictable behavior.







    Attributes:



        device (Device): The parent device.



        script_list (list): The shared list of scripts to execute.



        neighbours (list): A list of neighboring Device objects.



        index (int): The index used to pop a script from the script_list.



    """



    



    def __init__(self, device, script_list, neighbours, index):



        Thread.__init__(self)



        self.device = device



        self.script_list = script_list



        self.neighbours = neighbours



        self.index = index







    def run(self):



        """Pops one script from the shared list and executes it."""



        if self.script_list != []:



            (script, location) = self.script_list.pop(self.index)



            self.compute(script, location)







    def update(self, result, location):



        """Propagates the script result to the device and its neighbors."""



        for device in self.neighbours:



            device.set_data(location, result)



        self.device.set_data(location, result)







    def collect(self, location, neighbours, script_data):



        """Gathers data from the device and its neighbors for a location."""



        self.device.map_locations[location].acquire()



        for device in self.neighbours:



            



            data = device.get_data(location)



            if data is None:



                pass



            else:



                script_data.append(data)







        



        data = self.device.get_data(location)



        if data is not None:



            script_data.append(data)







    def compute(self, script, location):



        """Orchestrates data collection, script execution, and result update."""



        script_data = []



        self.collect(location, self.neighbours, script_data)







        if script_data == []:



            pass



        else:



            



            result = script.run(script_data)



            self.update(result, location)







        self.device.map_locations[location].release()







class ReusableBarrierSem():



    """A reusable barrier implemented using semaphores.







    This barrier ensures that a fixed number of threads all reach a



    synchronization point before any of them are allowed to proceed. It uses a



    two-phase mechanism to allow for reuse in iterative algorithms.







    Attributes:



        num_threads (int): The number of threads to synchronize.



        count_threads1 (int): Counter for threads entering the first phase.



        count_threads2 (int): Counter for threads entering the second phase.



        counter_lock (Lock): A lock to protect access to the counters.



        threads_sem1 (Semaphore): Semaphore for the first synchronization phase.



        threads_sem2 (Semaphore): Semaphore for the second synchronization phase.



    """



    







    def __init__(self, num_threads):



        """Initializes the ReusableBarrierSem."""



        self.num_threads = num_threads



        self.count_threads1 = self.num_threads











        self.count_threads2 = self.num_threads



        self.counter_lock = Lock()



        self.threads_sem1 = Semaphore(0)



        self.threads_sem2 = Semaphore(0)







    def wait(self):



        """Causes a thread to wait at the barrier until all threads arrive."""



        self.phase1()



        self.phase2()







    def phase1(self):



        """The first phase of the barrier synchronization."""



        with self.counter_lock:



            self.count_threads1 -= 1



            if self.count_threads1 == 0:



                for i in range(self.num_threads):



                    self.threads_sem1.release()



                self.count_threads1 = self.num_threads







        self.threads_sem1.acquire()







    def phase2(self):



        """The second phase of the barrier, allowing for reuse."""



        with self.counter_lock:



            self.count_threads2 -= 1



            if self.count_threads2 == 0:



                for i in range(self.num_threads):



                    self.threads_sem2.release()



                self.count_threads2 = self.num_threads







        self.threads_sem2.acquire()


