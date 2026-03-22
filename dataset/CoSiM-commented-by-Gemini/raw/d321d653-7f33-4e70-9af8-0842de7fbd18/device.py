


"""



This module implements a distributed device simulation.







Key architectural features:



- A leader-election mechanism (based on the lowest device ID) is used to have



  one device create and distribute shared resources.



- Shared resources include a global `ReusableBarrier` for all devices and a



  global dictionary of `locks` for each data location.



- Each device's main thread (`DeviceThread`) spawns a new worker thread



  (`ScriptThread`) for each script assigned in a time step, which is a



  performance anti-pattern.



- A critical race condition exists in the lazy initialization of locks within



  the `ScriptThread`, making the current implementation not thread-safe.







Note: The file imports `ReusableBarrier` from a local module but also defines



a class with the same name. This analysis assumes the local definition is used.



The script uses Python 2 syntax.



"""











from threading import Lock, Thread, Event



# This import is shadowed by a local class definition of the same name below.



from reusable_barrier_semaphore import ReusableBarrier











class Device(object):



    """



    Represents a device node in the simulation.



    



    It participates in a leader-election to initialize shared synchronization



    primitives and manages its own control thread (`DeviceThread`).



    """







    def __init__(self, device_id, sensor_data, supervisor):



        self.device_id = device_id



        self.sensor_data = sensor_data



        self.supervisor = supervisor



        self.script_received = Event()



        self.scripts = []



        self.timepoint_done = Event()



        self.thread = DeviceThread(self)



        self.thread.start()



        # --- Globally Shared Objects ---



        self.barrier = None



        self.locks = None







    def __str__(self):



        return "Device %d" % self.device_id







    def setup_devices(self, devices):



        """



        Performs collective setup of shared resources via leader-election.







        The device with the lowest ID is elected as the leader to create the



        shared barrier and lock dictionary, which are then distributed to all



        other devices.



        



        :param devices: A list of all Device objects in the simulation.



        """



        # Elect the device with the minimum ID as the leader.



        leader = devices[0]



        for device in devices:



            if device.device_id < leader.device_id:



                leader = device







        if self.device_id == leader.device_id:



            # The leader creates and distributes the shared objects.



            self.barrier = ReusableBarrier(len(devices))



            self.locks = {}



            for device in devices:



                device.barrier = self.barrier



                device.locks = self.locks







    def assign_script(self, script, location):



        """Assigns a script for the current time step."""



        if script is not None:



            self.scripts.append((script, location))



            self.script_received.set()



        else:



            # A None script signals the end of assignments for the timepoint.



            self.timepoint_done.set()







    def get_data(self, location):



        """Non-thread-safe method to get data."""



        return self.sensor_data.get(location)







    def set_data(self, location, data):



        """Non-thread-safe method to set data."""



        if location in self.sensor_data:



            self.sensor_data[location] = data







    def shutdown(self):



        self.thread.join()















class DeviceThread(Thread):



    """



    The main control thread for a device.



    



    For each time step, it spawns a new thread for each assigned script,



    waits for them to complete, and then synchronizes with all other devices.



    """







    def __init__(self, device):



        Thread.__init__(self, name="Device Thread %d" % device.device_id)



        self.device = device







    def run(self):



        """Main execution loop driven by synchronized time steps."""



        while True:



            neighbours = self.device.supervisor.get_neighbours()



            if neighbours is None:



                break







            # Wait until supervisor signals all scripts are assigned for this step.



            self.device.timepoint_done.wait()







            # Create and start a new worker thread for every script.



            threads = []



            for (script, location) in self.device.scripts:



                script_thread = ScriptThread(self.device, script, location, neighbours)



                threads.append(script_thread)



                script_thread.start()







            # Wait for all worker threads for this device to finish.



            for thread in threads:



                thread.join()







            self.device.timepoint_done.clear()







            # Wait at the global barrier for all other devices to finish their step.



            self.device.barrier.wait()











class ScriptThread(Thread):



    """



    A short-lived worker thread that executes one script.



    """







    def __init__(self, device, script, location, neighbours):



        Thread.__init__(self, name="Script Processing Thread for Device %d" % device.device_id)



        self.device = device



        self.script = script



        self.location = location



        self.neighbours = neighbours







    def run(self):



        """



        Executes a single script, including data gathering and result distribution.



        """



        # CRITICAL FLAW: This lazy initialization of locks is not thread-safe.



        # Two threads trying to create a lock for the same new location at the



        # same time can lead to a race condition.



        if self.location not in self.device.locks:



            self.device.locks[self.location] = Lock()







        # Acquire the global lock for this location before processing.



        with self.device.locks[self.location]:



            script_data = []



            



            # Gather data from neighbors and the local device.



            for device in self.neighbours:



                data = device.get_data(self.location)



                if data is not None:



                    script_data.append(data)



            data = self.device.get_data(self.location)



            if data is not None:



                script_data.append(data)







            if script_data:



                # Execute the script and broadcast the results.



                result = self.script.run(script_data)



                for device in self.neighbours:



                    device.set_data(self.location, result)



                self.device.set_data(self.location, result)











from threading import *







class ReusableBarrier():



    """



    A reusable barrier implemented with Semaphores.







    This allows a group of threads to all wait for each other to reach a



    certain point before any of them are allowed to proceed.



    """



    def __init__(self, num_threads):



        self.num_threads = num_threads



        self.count_threads1 = [self.num_threads]



        self.count_threads2 = [self.num_threads]



        self.count_lock = Lock()



        self.threads_sem1 = Semaphore(0)



        self.threads_sem2 = Semaphore(0)







    def wait(self):



        """Blocks until all threads have called this method."""



        self.phase(self.count_threads1, self.threads_sem1)



        self.phase(self.count_threads2, self.threads_sem2)



 



    def phase(self, count_threads, threads_sem):



        """Executes a single phase of the barrier synchronization."""



        with self.count_lock:



            count_threads[0] -= 1



            if count_threads[0] == 0:



                # Last thread to arrive releases all others for this phase.



                for i in range(self.num_threads):



                    threads_sem.release()



                # Reset counter for the next use of the barrier.



                count_threads[0] = self.num_threads



        threads_sem.acquire()


