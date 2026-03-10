


"""



Models a distributed network of devices that process sensor data concurrently.







This script implements a device simulation where each device dynamically spawns a



new thread for every script execution within a timepoint. Synchronization is



managed by a custom reusable barrier and a complex lock-sharing mechanism.



"""







from threading import Thread, Event



from threading import Lock, Semaphore







class ReusableBarrier():



    """A reusable barrier implemented using semaphores for thread synchronization.







    This barrier ensures that a fixed number of threads all reach a



    synchronization point before any are allowed to proceed. It uses a two-phase



    (turnstile) mechanism to allow for reuse in iterative, multi-timepoint



    simulations.







    Attributes:



        num_threads (int): The number of threads to synchronize.



        count_threads1 (int): Counter for threads entering the first phase.



        count_threads2 (int): Counter for threads entering the second phase.



        counter_lock (Lock): A lock to protect access to the counters.



        threads_sem1 (Semaphore): Semaphore for the first synchronization phase.



        threads_sem2 (Semaphore): Semaphore for the second synchronization phase.



    """



    



    



    def __init__(self, num_threads):



        """Initializes the ReusableBarrier."""



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



        """The second phase of the barrier, allowing for safe reuse."""



        with self.counter_lock:



            self.count_threads2 -= 1



            if self.count_threads2 == 0:



                for i in range(self.num_threads):



                    self.threads_sem2.release()



                self.count_threads2 = self.num_threads   



        self.threads_sem2.acquire()







class RunScripts(Thread):



    """A dedicated thread to execute a single data processing script.







    An instance of this class is created for each script that needs to be run,



    encapsulating the logic for data gathering, execution, and result propagation.







    Attributes:



        device (Device): The parent device instance.



        location (str): The location context for the script.



        script (Script): The script object to be executed.



        neighbours (list): A list of neighboring Device objects.



    """



    



    def __init__(self, device, location, script, neighbours):



        """Initializes the script-running thread."""



        Thread.__init__(self)



        self.device = device



        self.location = location



        self.script = script



        self.neighbours = neighbours







    



    def run(self):



        """The main execution logic for the thread.







        It acquires a lock for the specific data location, collects data from the



        parent device and its neighbors, runs the script, and then updates the



        data on all relevant devices with the script's result.



        """



        self.device.location_lock[self.location].acquire()







        script_data = []



        



        for device in self.neighbours:



            data = device.get_data(self.location)



            if data is not None:



                script_data.append(data)







        



        data = self.device.get_data(self.location)



        if data is not None:



            script_data.append(data)







        if script_data != []:



            



            result = self.script.run(script_data)



            



            







            for device in self.neighbours:



                device.set_data(self.location, result)



                



            self.device.set_data(self.location, result)







        



        self.device.location_lock[self.location].release()







class Device(object):



    """Represents a single device in the distributed sensor network.







    This implementation creates a new thread for each script execution and uses a



    complex mechanism to share locks for data locations.







    Attributes:



        device_id (int): A unique identifier for the device.



        sensor_data (dict): A dictionary holding the device's sensor readings.



        supervisor (Supervisor): An object for retrieving neighbor information.



        scripts (list): A list of (script, location) tuples to execute.



        timepoint_done (Event): An event that signals the completion of a



                                simulation timepoint.



        thread (DeviceThread): The main thread of execution for this device.



        barrier (ReusableBarrier): A shared barrier for synchronization.



        location_lock (list): A list used to store location-based locks.



    """



    







    def __init__(self, device_id, sensor_data, supervisor):



        """Initializes a Device instance."""



        self.device_id = device_id



        self.sensor_data = sensor_data



        self.supervisor = supervisor



        self.script_received = Event()



        self.scripts = []



        self.devices = []



        self.timepoint_done = Event()











        self.thread = DeviceThread(self)



        self.barrier = None



        self.list_thread = []



        self.thread.start()



        self.location_lock = [None] * 200







    def __str__(self):



        """Returns the string representation of the device."""



        return "Device %d" % self.device_id







    def setup_devices(self, devices):



        """Initializes and shares the barrier among all devices."""



        nr_devices = len(devices)



        



        if self.barrier is None:



            barrier = ReusableBarrier(nr_devices)











            self.barrier = barrier







            for device in devices:



                if device.barrier is None:



                    device.barrier = barrier







        



        for device in devices:



            if device is not None:



                self.devices.append(device)











    def assign_script(self, script, location):



        """Assigns a script and manages the lock for its execution location.







        If a script is provided, it's added to the list. This method also



        implements a lazy, cooperative mechanism to initialize and share the lock



        for the specified location among all devices.







        Args:



            script (Script): The script to execute, or None to signal completion.



            location (int): The location index for the script.



        """



        lock_location = False







        if script is None:



            self.timepoint_done.set()















        else:



            



            self.scripts.append((script, location))



            if self.location_lock[location] is None:







                for device in self.devices:



                    if device.location_lock[location] is not None:



                        



                        self.location_lock[location] = device.location_lock[location]



                        lock_location = True



                        break







                if lock_location is False:



                    self.location_lock[location] = Lock()







            self.script_received.set()



            







    def get_data(self, location):



        """Retrieves sensor data for a specific location."""



        return self.sensor_data[location] if location in self.sensor_data else None







    def set_data(self, location, data):



        """Updates sensor data for a specific location."""



        if location in self.sensor_data:



            self.sensor_data[location] = data







    def shutdown(self):



        """Waits for the main device thread to terminate."""



        self.thread.join()







class DeviceThread(Thread):



    """The main execution thread for a Device.







    This thread orchestrates the device's lifecycle, spawning a new worker



    thread (`RunScripts`) for each assigned script in every timepoint.







    Attributes:



        device (Device): The device instance this thread belongs to.



    """



    







    def __init__(self, device):



        """Initializes the DeviceThread."""



        Thread.__init__(self, name="Device Thread %d" % device.device_id)



        self.device = device







    def run(self):



        """The main loop for the device thread.







        It waits for a timepoint to start, creates and runs a new thread for



        each script, waits for all of them to complete, and then synchronizes



        at the global barrier before starting the next timepoint.



        """



        



        while True:







            



            neighbours = self.device.supervisor.get_neighbours()



            if neighbours is None:



                break







            self.device.timepoint_done.wait()







            











            for (script, location) in self.device.scripts:



                thread = RunScripts(self.device, location, script, neighbours) 



                self.device.list_thread.append(thread)







            



            for thread_elem in self.device.list_thread:



                thread_elem.start()







            for thread_elem in self.device.list_thread:



                thread_elem.join()







            self.device.list_thread = []



            self.device.timepoint_done.clear()



            self.device.barrier.wait()


