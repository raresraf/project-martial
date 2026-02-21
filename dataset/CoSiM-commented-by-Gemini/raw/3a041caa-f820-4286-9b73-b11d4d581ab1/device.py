


"""A distributed device simulation with decentralized synchronization setup.







This module implements a simulation of devices that operate in synchronized



time steps. It uses a custom `ReusableBarrier` for synchronization and a model



where each device's main thread spawns new worker threads for each script.







The initialization of shared resources (barriers and locks) is handled via a



decentralized, "gossip-like" protocol that is highly susceptible to race



conditions, making the setup non-deterministic.



"""







from threading import Event, Thread, Lock, Semaphore







class ReusableBarrier():



    """A custom, reusable barrier for synchronizing a fixed number of threads.







    This barrier uses a two-phase protocol with two semaphores to ensure it can



    be safely used multiple times in a loop without race conditions.



    """



    def __init__(self, num_threads):



        self.num_threads = num_threads



        self.count_threads1 = [self.num_threads]



        self.count_threads2 = [self.num_threads]



        self.count_lock = Lock()



        self.threads_sem1 = Semaphore(0)



        self.threads_sem2 = Semaphore(0)







    def wait(self):



        """Blocks the calling thread until all threads have reached the barrier."""



        self.phase(self.count_threads1, self.threads_sem1)



        self.phase(self.count_threads2, self.threads_sem2)







    def phase(self, count_threads, threads_sem):



        """Executes one phase of the barrier synchronization."""



        with self.count_lock:



            count_threads[0] -= 1



            if count_threads[0] == 0:  # The last thread to arrive



                # Release all waiting threads



                for _ in range(self.num_threads):



                    threads_sem.release()



                # Reset counter for the next use



                count_threads[0] = self.num_threads



        threads_sem.acquire()







class Device(object):



    """Represents a single device in the simulation.



    



    This class holds the device's state and is responsible for the complex,



    racy setup of shared synchronization objects.



    """







    def __init__(self, device_id, sensor_data, supervisor):



        """Initializes the device and starts its main control thread."""



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



        self.location_lock = [None] * 100







    def __str__(self):



        """Returns the string representation of the device."""



        return "Device %d" % self.device_id







    def setup_devices(self, devices):



        """Establishes a shared barrier among all devices.



        



        WARNING: This protocol is racy. If multiple devices call this



        concurrently, several barriers could be created, and the final



        state depends on execution order. There is no clear master device.



        """



        if self.barrier is None:



            barrier = ReusableBarrier(len(devices))



            self.barrier = barrier



            for device in devices:



                if device.barrier is None:



                    device.barrier = barrier







        for device in devices:



            if device is not None:



                self.devices.append(device)







    def assign_script(self, script, location):



        """Assigns a script and attempts to establish a shared lock for it.



        



        WARNING: The lock initialization protocol is decentralized and racy.



        When a script is assigned, this device checks if it has a lock for the



        script's location. If not, it polls other devices to find an existing



        lock. If none is found, it creates a new one. This can lead to multiple



        locks being created for the same location if devices run concurrently.



        """



        if script is not None:



            self.scripts.append((script, location))



            if self.location_lock[location] is None:



                found_lock = False



                for device in self.devices:



                    if device.location_lock[location] is not None:



                        self.location_lock[location] = device.location_lock[location]



                        found_lock = True



                        break



                if not found_lock:



                    self.location_lock[location] = Lock()



            self.script_received.set()



        else:



            self.timepoint_done.set()







    def get_data(self, location):



        """Gets data from a specific location."""



        return self.sensor_data.get(location)







    def set_data(self, location, data):



        """Sets data at a specific location."""



        if location in self.sensor_data:



            self.sensor_data[location] = data







    def shutdown(self):



        """Gracefully shuts down the device."""



        self.thread.join()







class NewThread(Thread):



    """A worker thread to execute one script."""



    def __init__(self, device, location, script, neighbours):



        Thread.__init__(self)



        self.device = device



        self.location = location



        self.script = script



        self.neighbours = neighbours







    def run(self):



        """The core logic for script execution.



        



        Acquires a location-specific lock, gathers data, runs the script,



        disseminates the result, and releases the lock.



        """



        script_data = []



        self.device.location_lock[self.location].acquire()



        



        # Gather data from neighbors and self.



        for device in self.neighbours:



            data = device.get_data(self.location)



            if data is not None:



                script_data.append(data)



        data = self.device.get_data(self.location)



        if data is not None:



            script_data.append(data)







        if script_data:



            # Run script and update data in the neighborhood.



            result = self.script.run(script_data)



            for device in self.neighbours:



                device.set_data(self.location, result)



            self.device.set_data(self.location, result)



            



        self.device.location_lock[self.location].release()







class DeviceThread(Thread):



    """The main control thread for a device's lifecycle."""







    def __init__(self, device):



        Thread.__init__(self, name="Device Thread %d" % device.device_id)



        self.device = device







    def run(self):



        """The main simulation loop.



        



        Waits for a signal, spawns worker threads for all assigned scripts,



        waits for them to complete, and then synchronizes at a global barrier.



        """



        while True:



            neighbours = self.device.supervisor.get_neighbours()



            if neighbours is None:



                break # Simulation exit condition.







            # Wait for the signal to start the current time step.



            self.device.timepoint_done.wait()







            # Create a new worker thread for each script.



            for (script, location) in self.device.scripts:



                thread = NewThread(self.device, location, script, neighbours)



                self.device.list_thread.append(thread)







            # Start and join all worker threads.



            for thread_elem in self.device.list_thread:



                thread_elem.start()



            for thread_elem in self.device.list_thread:



                thread_elem.join()



            self.device.list_thread = []







            # Prepare for the next time step and synchronize.



            self.device.timepoint_done.clear()



            self.device.barrier.wait()


