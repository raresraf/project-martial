






"""



@brief A non-functional device simulation with a flawed thread pool.



@file device.py







This module attempts to implement a device simulation using a fixed-size thread



pool. However, the implementation is riddled with severe architectural flaws,



race conditions, and bugs that render it non-functional.







WARNING: SEVERE ARCHITECTURAL FLAWS - THIS CODE IS NON-FUNCTIONAL.



1.  **Catastrophic Race Condition in `setup_devices`**: The setup logic is



    completely broken. Each device attempts to overwrite the `barrier` and `lock`



    attributes of ALL other devices. The final state depends entirely on which



    thread finishes this process last, making synchronization unpredictable and



    incorrect.



2.  **Broken Thread Pool Management**: The `DeviceThread` implements a complex and



    unsafe algorithm to manage its "pool" of 8 threads. It relies on checking



    `is_alive()` and manually replacing threads in a list, a pattern that is



    unreliable and fails to handle the case where all threads are busy.



3.  **Flawed `ReusableBarrier`**: This file includes a `ReusableBarrierSem` that,



    like in other versions, contains a deadlock-prone anti-pattern of holding a



    lock while releasing semaphores.



"""







from threading import Event, Thread, Lock, Semaphore







class Device(object):



    """Represents a device node in the broken simulation."""







    def __init__(self, device_id, sensor_data, supervisor):



        self.device_id = device_id



        self.sensor_data = sensor_data



        self.supervisor = supervisor



        self.script_received = Event()



        self.scripts = []



        self.timepoint_done = Event()



        self.lock = {}



        self.barrier = None



        self.devices = []



        self.thread = DeviceThread(self)



        self.thread.start()







    def __str__(self):



        return "Device %d" % self.device_id







    def setup_devices(self, devices):



        """



        FATALLY FLAWED setup method. It creates a race condition where the last



        device to execute this function dictates the shared resources for ALL devices.



        """



        self.devices = devices



        # Each device creates its own barrier and lock dictionary...



        self.barrier = ReusableBarrierSem(len(self.devices))



        for location in self.sensor_data:



            self.lock[location] = Lock()



        for device in devices:



            for location in device.sensor_data:



                self.lock[location] = Lock()







        # ...and then tries to force them upon all other devices. This will fail.



        for i in xrange(len(self.devices)):



            self.devices[i].barrier = self.barrier



            self.devices[i].lock = self.lock







    def assign_script(self, script, location):



        """Adds a script to the device's workload for the current time step."""



        if script is not None:



            self.scripts.append((script, location))



        else:



            self.script_received.set()



            self.timepoint_done.set()







    def get_data(self, location):



        return self.sensor_data.get(location)







    def set_data(self, location, data):



        if location in self.sensor_data:



            self.sensor_data[location] = data







    def shutdown(self):



        self.thread.join()











class MyThread(Thread):



    """A worker thread for executing a single script."""



    def __init__(self, my_id, device, neighbours, lock, script, location):



        Thread.__init__(self, name="Thread %d from device %d" % (my_id, device.device_id))



        self.device = device



        self.my_id = my_id



        self.neighbours = neighbours



        self.lock = lock # This is the flawed, overwritten dictionary of locks.



        self.script = script



        self.location = location







    def run(self):



        """



        Executes the script. While it uses a `with` statement correctly, the



        lock object itself is likely not properly shared due to setup flaws.



        """



        with self.lock[self.location]:



            script_data = []



            # Gather data from neighbors and self.



            for device in self.neighbours:



                data = device.get_data(self.location)



                if data is not None:



                    script_data.append(data)



            data = self.device.get_data(self.location)



            if data is not None:



                script_data.append(data)







            # If data exists, execute the script and update values.



            if script_data:



                result = self.script.run(script_data)



                for device in self.neighbours:



                    device.set_data(self.location, result)



                self.device.set_data(self.location, result)







    def shutdown(self):



        self.join()











class DeviceThread(Thread):



    """



    The main control thread, which contains broken logic for managing a



    fixed-size pool of worker threads.



    """



    def __init__(self, device):



        Thread.__init__(self, name="Device Thread %d" % device.device_id)



        self.device = device



        self.numThreads = 0



        self.listThreads = []







    def run(self):



        """Main simulation loop."""



        while True:



            neighbours = self.device.supervisor.get_neighbours()



            if neighbours is None:



                break







            self.device.script_received.wait()







            # --- WARNING: Flawed and unsafe thread pool management ---



            for (script, location) in self.device.scripts:



                if len(self.listThreads) < 8:



                    # Add new threads until the pool size is 8.



                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)



                    self.listThreads.append(thread)



                    thread.start()



                    self.numThreads += 1



                else:



                    # Attempt to find a finished thread to replace. This is not a



                    # reliable or safe way to manage a thread pool. It fails to



                    # handle the case where all threads are still busy.



                    index = -1



                    for i in xrange(len(self.listThreads)):



                        if not self.listThreads[i].is_alive():



                            self.listThreads[i].join()



                            index = i



                    



                    if index != -1:



                        self.listThreads.pop(index)



                        thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)



                        self.listThreads.insert(index, thread)



                        thread.start()



                        self.numThreads += 1







            # Wait for all threads in the current list to complete.



            for i in xrange(len(self.listThreads)):



                self.listThreads[i].join()







            self.device.timepoint_done.wait()



            



            self.device.script_received.clear()



            self.device.timepoint_done.clear()



            



            # This will fail because self.device.barrier is not a shared object.



            self.device.barrier.wait()











class ReusableBarrierSem():



    """



    A flawed implementation of a reusable two-phase barrier using semaphores.



    This is a copy of other flawed barriers in this dataset.



    """



    def __init__(self, num_threads):



        self.num_threads = num_threads



        self.count_threads1 = self.num_threads



        self.count_threads2 = self.num_threads



        self.counter_lock = Lock()               



        self.threads_sem1 = Semaphore(0)         



        self.threads_sem2 = Semaphore(0)         







    def wait(self):



        self.phase1()



        self.phase2()







    def phase1(self):



        with self.counter_lock:



            self.count_threads1 -= 1



            if self.count_threads1 == 0:



                for i in range(self.num_threads):



                    self.threads_sem1.release()



                self.count_threads1 = self.num_threads



        self.threads_sem1.acquire()







    def phase2(self):



        with self.counter_lock:



            self.count_threads2 -= 1



            if self.count_threads2 == 0:



                for i in range(self.num_threads):



                    self.threads_sem2.release()



                self.count_threads2 = self.num_threads



        self.threads_sem2.acquire()




