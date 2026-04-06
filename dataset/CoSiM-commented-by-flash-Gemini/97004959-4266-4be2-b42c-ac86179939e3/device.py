




"""
@97004959-4266-4be2-4b2c-ac86179939e3/device.py
@brief Defines classes for device simulation in a distributed system, including reusable barrier synchronization and script execution management.
Functional Utility: This module orchestrates device behavior in a simulated distributed environment, enabling synchronized data processing and script execution across multiple interconnected devices.
Domain: Distributed Systems, Concurrency, Simulation.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue

class ReusableBarrier(object):
    """
    Implements a reusable barrier synchronization primitive for a fixed number of threads.

    This barrier allows a group of `num_threads` to halt execution at a specific
    point (`wait` method) until all threads in the group have arrived. Once all
    threads have reached the barrier, they are all released simultaneously.
    The barrier can then be reset and reused for subsequent synchronization points.

    The implementation uses a two-phase approach to ensure reusability without
    deadlock. Each phase involves:
    - A `count_lock` to atomically decrement a thread counter.
    - A `Semaphore` to block arriving threads and release them when the counter reaches zero.

    Attributes:
        num_threads (int): The total number of threads expected to participate in the barrier.
        count_threads1 (list): Internal counter for the first phase of the barrier.
        count_threads2 (list): Internal counter for the second phase of the barrier.
        count_lock (Lock): A lock to protect access to the thread counters.
        threads_sem1 (Semaphore): Semaphore for the first phase of thread synchronization.
        threads_sem2 (Semaphore): Semaphore for the second phase of thread synchronization.
    """
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        
        with self.count_lock:
            count_threads[0] -= 1
            
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.result_queue = Queue.Queue()
        self.set_lock = Lock()
        self.neighbours_lock = None
        self.neighbours_barrier = None

        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == devices[0].device_id:
            self.neighbours_lock = Lock()
            self.neighbours_barrier = ReusableBarrier(len(devices))
        
        else:
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier

        self.thread.start()

    def assign_script(self, script, location):
        


        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        
        self.set_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_lock.release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []

    def run(self):
        while True:

            
            self.device.neighbours_lock.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.neighbours_lock.release()

            if neighbours is None:
                break

            
            self.device.script_received.wait()

            
            self.workers = []
            for i in range(8):
                self.workers.append(DeviceWorker(self.device, i, neighbours))

            
            for (script, location) in self.device.scripts:

                
                added = False
                for worker in self.workers:
                    if location in worker.locations:
                        worker.add_script(script, location)
                        added = True

                
                if added == False:
                    minimum = len(self.workers[0].locations)
                    chosen_worker = self.workers[0]
                    for worker in self.workers:
                        if minimum > len(worker.locations):
                            minimum = len(worker.locations)
                            chosen_worker = worker

                    chosen_worker.add_script(script, location)

            
            for worker in self.workers:
                worker.start()

            


            for worker in self.workers:
                worker.join()

            
            self.device.neighbours_barrier.wait()
            self.device.script_received.clear()


class DeviceWorker(Thread):
    

    def __init__(self, device, worker_id, neighbours):
        

        Thread.__init__(self)
        self.device = device
        self.worker_id = worker_id
        self.scripts = []
        self.locations = []
        self.neighbours = neighbours

    def add_script(self, script, location):
        
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        

        for (script, location) in zip(self.scripts, self.locations):

            script_data = []
            
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            
            if script_data != []:
                res = script.run(script_data)

                for device in self.neighbours:
                    device.set_data(location, res)
                self.device.set_data(location, res)

    def run(self):
        self.run_scripts()
