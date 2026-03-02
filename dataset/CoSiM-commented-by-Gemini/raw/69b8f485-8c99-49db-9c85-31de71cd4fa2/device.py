


"""



Models a distributed system of devices for parallel data processing.







This module provides a more advanced simulation framework than other versions,



featuring a robust two-phase reusable barrier and a producer-consumer threading



model within each device. A single controller thread per device manages the



workflow for a time step, distributing tasks to a pool of worker threads via a queue.



"""







from Queue import Queue



from threading import Event, Thread, Lock, Semaphore











class ReusableBarrier(object):



    """



    A robust, reusable, two-phase barrier using semaphores.







    This implementation prevents race conditions where fast threads could loop



    around and re-enter the barrier before slow threads have left. It does this



    by requiring threads to pass through two distinct synchronization phases.



    """







    def __init__(self, num_threads):



        """



        Initializes the barrier for a given number of threads.







        Args:



            num_threads (int): The number of threads that must synchronize.



        """



        self.num_threads = num_threads



        # Counters are lists to be mutable across method calls.



        self.count_threads1 = [self.num_threads]



        self.count_threads2 = [self.num_threads]



        self.count_lock = Lock()



        self.threads_sem1 = Semaphore(0)



        self.threads_sem2 = Semaphore(0)







    def wait(self):



        """Blocks until all threads have called wait(), then resets for reuse."""



        self.phase(self.count_threads1, self.threads_sem1)



        self.phase(self.count_threads2, self.threads_sem2)







    def phase(self, count_threads, threads_sem):



        """



        Executes one phase of the two-phase barrier synchronization.







        Args:



            count_threads (list): The counter for the current phase.



            threads_sem (Semaphore): The semaphore for the current phase.



        """



        with self.count_lock:



            count_threads[0] -= 1



            if count_threads[0] == 0:



                # Last thread to arrive releases all other threads.



                for _ in range(self.num_threads):



                    threads_sem.release()



                count_threads[0] = self.num_threads # Reset for next use.



        threads_sem.acquire()











class Device(object):



    """



    Represents a device using a producer-consumer threading model.







    A single controller thread (`DeviceThread`) manages the device's state per



    timepoint, while a pool of `WorkerThread`s execute computation tasks from a queue.



    """







    def __init__(self, device_id, sensor_data, supervisor):



        """



        Initializes the device and its controller and worker threads.







        Args:



            device_id (int): The unique ID for this device.



            sensor_data (dict): The initial sensor data for this device.



            supervisor: The supervisor object managing the simulation.



        """



        self.device_id = device_id



        self.sensor_data = sensor_data



        self.supervisor = supervisor



        self.scripts = []



        self.timepoint_done = Event()







        # Shared synchronization objects to be provided by setup_devices.



        self.barrier = None



        self.locks = None # To be a dict of location-specific locks.







        # Internal producer-consumer components.



        self.queue = Queue()



        self.workers = [WorkerThread(self) for _ in range(8)]







        # The main controller thread for this device.



        self.thread = DeviceThread(self)



        self.thread.start()







        # Start all worker threads.



        for thread in self.workers:



            thread.start()







    def __str__(self):



        """Returns the string representation of the device."""



        return "Device %d" % self.device_id







    def setup_devices(self, devices):



        """



        Centrally configures shared synchronization objects for all devices.







        This method should be called on one "master" device (e.g., device 0).



        It creates and distributes a single system-wide barrier and a shared



        set of location-based locks to all devices in the simulation.







        Args:



            devices (list): A list of all Device objects in the simulation.



        """



        if self.device_id == 0:



            # Create a single barrier for all controller threads.



            barrier = ReusableBarrier(len(devices))







            # Create a shared lock for each unique data location.



            locks = {}



            for device in devices:



                for location in device.sensor_data:



                    if not location in locks:



                        locks[location] = Lock()







            # Distribute the shared objects to all devices.



            for device in devices:



                device.barrier = barrier



                device.locks = locks







    def assign_script(self, script, location):



        """



        Assigns a computation script, called by the supervisor.







        Args:



            script: The script object to be executed.



            location: The data location the script applies to.



        """



        if script is not None:



            self.scripts.append((script, location))



        else:



            # Sentinel None indicates all scripts are assigned for this timepoint.



            self.timepoint_done.set()







    def get_data(self, location):



        """Retrieves sensor data for a given location."""



        return self.sensor_data.get(location)







    def set_data(self, location, data):



        """Updates the sensor data for a given location."""



        if location in self.sensor_data:



            self.sensor_data[location] = data







    def shutdown(self):



        """Waits for the main controller thread to complete."""



        self.thread.join()











class WorkerThread(Thread):



    """



    A worker thread that processes computation tasks from a queue.



    """







    def __init__(self, device):



        """Initializes the worker thread."""



        Thread.__init__(self)



        self.device = device







    def run(self):



        """



        Main loop: waits for a task, executes it, and signals completion.



        """



        while True:



            item = self.device.queue.get()



            if item is None:



                # None is the shutdown signal.



                break







            (script, location) = item







            with self.device.locks[location]:



                script_data = []







                # Gather data from neighbors.



                for device in self.device.neighbours:



                    data = device.get_data(location)



                    if data is not None:



                        script_data.append(data)







                # Gather data from self.



                data = self.device.get_data(location)



                if data is not None:



                    script_data.append(data)







                if script_data:



                    # Run computation and broadcast result.



                    result = script.run(script_data)



                    for device in self.device.neighbours:



                        device.set_data(location, result)



                    self.device.set_data(location, result)







            self.device.queue.task_done()











class DeviceThread(Thread):



    """



    The main controller thread for a device.



    """







    def __init__(self, device):



        """Initializes the controller thread."""



        Thread.__init__(self, name="Device Thread %d" % device.device_id)



        self.device = device







    def run(self):



        """



        Orchestrates the device's activity for each time step.



        """



        while True:



            # Get neighbors for the current time step.



            self.device.neighbours = self.device.supervisor.get_neighbours()







            if self.device.neighbours is None:



                # Supervisor signals simulation shutdown.



                break







            # Wait for supervisor to assign all scripts for this timepoint.



            self.device.timepoint_done.wait()



            self.device.timepoint_done.clear()







            # Producer: Add all assigned scripts to the worker queue.



            for (script, location) in self.device.scripts:



                self.device.queue.put((script, location))







            # Wait for all workers to complete their tasks for this timepoint.



            self.device.queue.join()







            # Synchronize with all other devices before starting the next timepoint.



            self.device.barrier.wait()







        # --- Shutdown sequence ---



        # Signal all worker threads to exit.



        for _ in range(8):



            self.device.queue.put(None)







        # Wait for all worker threads to terminate.



        for thread in self.device.workers:



            thread.join()



			
