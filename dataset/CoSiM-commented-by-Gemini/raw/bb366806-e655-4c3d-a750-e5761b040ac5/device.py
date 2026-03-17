


"""



This module provides a simulation framework for a network of devices.







It appears to be a concatenation of several files, defining classes for the main `Device`



and its `DeviceThread`, as well as custom implementations of a `ThreadPool`, `Worker`,



and a `ReusableBarrier`. The simulation involves devices executing scripts, sharing data with



neighbors, and synchronizing at time steps.



"""







from threading import Event, Thread, Lock , Condition



# Note: The following imports are satisfied by class definitions within this same file.



from queue import Worker, ThreadPool



from reusable_barrier_semaphore import ReusableBarrier







class Device(object):



    """Represents a node in the simulated network.







    Each device has local sensor data, runs a set of scripts, and communicates



    with its neighbors under the coordination of a supervisor. It uses a thread



pool



    to execute scripts concurrently.



    """







    def __init__(self, device_id, sensor_data, supervisor):



        """Initializes the device and starts its main lifecycle thread.







        Args:



            device_id (int): A unique identifier for the device.



            sensor_data (dict): The local data held by this device.



            supervisor (object): The central supervisor for network information.



        """



        self.device_id = device_id



        self.sensor_data = sensor_data



        self.supervisor = supervisor



        self.script_received = Event()



        self.wait_neighbours = Event()



        self.scripts = []



        self.neighbours = []



        self.allDevices = []



        self.locks = []



        self.pool = ThreadPool(8)



        self.lock = Lock()



        self.thread = DeviceThread(self)



        self.thread.start()







    def __str__(self):



        """Returns the string representation of the device."""



        return "Device %d" % self.device_id







    def setup_devices(self, devices):



        """Initializes the network-wide barrier and data locks.







        Args:



            devices (list): A list of all devices in the simulation.



        """



        self.allDevices = devices



        self.barrier = ReusableBarrier(len(devices))







        # Pre-allocates a fixed number of locks for data locations.



        for i in range(0, 50):



            self.locks.append(Lock())







        pass







    def assign_script(self, script, location):



        """Assigns a script to be executed by the device's thread pool.







        Args:



            script (object): The script object to run.



            location (any): The data location the script operates on.



        """



        if script is not None:



            self.scripts.append((script, location))



            self.pool.add_task(self.executeScript,script,location)



        else:



            # A None script signals that all scripts for the timepoint have been assigned.



            self.script_received.set()







    def get_data(self, location):



        """Retrieves data from a specific location on this device.







        Args:



            location (any): The key for the desired data.







        Returns:



            The data at the given location, or None if not found.



        """



        return self.sensor_data[location] if location in self.sensor_data else None







    def set_data(self, location, data):



        """Updates data at a specific location on this device.







        Args:



            location (any): The key for the data.



            data (any): The new data value.



        """



        if location in self.sensor_data:



            self.sensor_data[location] = data







    def shutdown(self):



        """Waits for the device's main thread to terminate."""



        self.thread.join()







    def executeScript(self,script,location):



        """The core task executed by the thread pool for each assigned script.







        This method waits until the device's neighbors are known, then gathers data



        from itself and its neighbors, runs the script, and propagates the result back.



        



        Args:



            script (object): The script to execute.



            location (any): The data location to operate on.



        """



        # Invariant: Must wait for the main thread to set the neighbors for the current step.



        self.wait_neighbours.wait()



        script_data = []







        # Block Logic: Gathers data from neighboring devices.



        if not self.neighbours is None:



            for device in self.neighbours:



                device.locks[location].acquire()



                data = device.get_data(location)



                device.locks[location].release()







                if data is not None:



                    script_data.append(data)







        # Block Logic: Gathers data from the local device.



        self.locks[location].acquire()



        data = self.get_data(location)



        self.locks[location].release()







        if data is not None:



            script_data.append(data)



        



        # Executes the script only if some data was successfully gathered.



        if script_data != []:



            result = script.run(script_data)







            # Block Logic: Propagates the result back to all neighbors.



            if not self.neighbours is None:



                for device in self.neighbours:







                    device.locks[location].acquire()



                    device.set_data(location, result)



                    device.locks[location].release()







            # Block Logic: Updates the local device's data with the result.



            self.locks[location].acquire()



            self.set_data(location, result)



            self.locks[location].release()











class DeviceThread(Thread):



    """The main control loop thread for a single Device."""







    def __init__(self, device):



        """Initializes the main thread for a device.







        Args:



            device (Device): The parent device this thread controls.



        """



        Thread.__init__(self, name="Device Thread %d" % device.device_id)



        self.device = device







    def run(self):



        """Defines the main lifecycle of a device, processing data in time-steps.







        In each step, it retrieves neighbors from the supervisor, waits for all scripts



        for that step to be assigned, waits for the thread pool to complete execution,



        and finally synchronizes with all other devices at a barrier.



        """



        while True:



            self.device.script_received.clear()



            self.device.wait_neighbours.clear()







            self.device.neighbours = []



            self.device.neighbours = self.device.supervisor.get_neighbours()



            self.device.wait_neighbours.set()







            # A None from the supervisor signals the end of the simulation.



            if self.device.neighbours is None:



                self.device.pool.wait_completion()



                self.device.pool.terminateWorkers()



                self.device.pool.threadJoin()



                return



            



            # Repopulates the task queue with all scripts assigned so far for the new step.



            for (script, location) in self.device.scripts:



                self.device.pool.add_task(self.device.executeScript,script,location)







            # Waits for a signal indicating all scripts for this time-step have been queued.



            self.device.script_received.wait()



            # Waits for the thread pool to finish all tasks for this time-step.



            self.device.pool.wait_completion()







            # Invariant: All devices must synchronize at the barrier before the next time-step.



            for dev in self.device.allDevices:



                dev.barrier.wait()











# The following classes appear to be concatenated from other modules.







from Queue import Queue



from threading import Thread







class Worker(Thread):



    """A worker thread that consumes and executes tasks from a queue."""



    def __init__(self, tasks):



        """Initializes the worker.







        Args:



            tasks (Queue): The queue from which to fetch tasks.



        """



        Thread.__init__(self)



        self.tasks = tasks



        self.daemon = True



        self.terminate_worker = False



        self.start()







    def run(self):



        """The main loop for the worker, continuously processing tasks."""



        while True:



            func, args, kargs = self.tasks.get()



            # A None function is the signal to terminate.



            if func == None:



                self.tasks.task_done()



                break



            try: func(*args, **kargs)



            except Exception, e: print e



            self.tasks.task_done()











class ThreadPool:



    """A pool of worker threads that execute tasks from a shared queue."""



    def __init__(self, num_threads):



        """Initializes the thread pool and creates the workers.







        Args:



            num_threads (int): The number of worker threads to create.



        """



        self.tasks = Queue(99999)



        self.workers = []



        for _ in range(num_threads):



            self.workers.append(Worker(self.tasks))







    def add_task(self, func, *args, **kargs):



        """Adds a task to the queue for a worker to execute.







        Args:



            func (callable): The function to execute.



            *args: Positional arguments for the function.



            **kargs: Keyword arguments for the function.



        """



        self.tasks.put((func, args, kargs))







    def wait_completion(self):



        """Blocks until all tasks in the queue have been completed."""



        self.tasks.join()







    def terminateWorkers(self):



        """Sends a termination signal to all worker threads."""



        for worker in self.workers:



            worker.tasks.put([None,None,None])



            worker.terminate_worker = True







    def threadJoin(self):



        """Waits for all worker threads to complete their execution."""



        for worker in self.workers:



            worker.join()







from threading import *







class ReusableBarrier():



    """A reusable barrier for synchronizing a fixed number of threads.







    This implementation uses a two-phase protocol with two semaphores to ensure



    that threads from one "wave" do not overlap with threads from the next.



    """



    def __init__(self, num_threads):



        """



        Args:



            num_threads (int): The number of threads that will synchronize on this barrier.



        """



        self.num_threads = num_threads



        # Counters are stored in a list to be mutable across method calls.



        self.count_threads1 = [self.num_threads]



        self.count_threads2 = [self.num_threads]



        self.count_lock = Lock()                 



        self.threads_sem1 = Semaphore(0)         



        self.threads_sem2 = Semaphore(0)         



    



    def wait(self):



        """Causes a thread to wait at the barrier. Consists of two phases."""



        self.phase(self.count_threads1, self.threads_sem1)



        self.phase(self.count_threads2, self.threads_sem2)



    



    def phase(self, count_threads, threads_sem):



        """Executes one phase of the barrier synchronization.







        Args:



            count_threads (list): A list containing the counter for the current phase.



            threads_sem (Semaphore): The semaphore for the current phase.



        """



        with self.count_lock:



            count_threads[0] -= 1



            if count_threads[0] == 0:            



                # The last thread to arrive releases all other waiting threads.



                for i in range(self.num_threads):



                    threads_sem.release()        



                count_threads[0] = self.num_threads  



        threads_sem.acquire()                    



                                                 







class MyThread(Thread):



    def __init__(self, tid, barrier):



        Thread.__init__(self)



        self.tid = tid



        self.barrier = barrier



    



    def run(self):



        for i in xrange(10):



            self.barrier.wait()



            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",


