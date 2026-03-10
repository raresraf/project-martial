"""
This module implements a simulation framework for distributed devices,
focusing on concurrent execution of scripts and synchronized data processing.
It utilizes a `Device` class to represent each simulated entity,
a `DeviceThread` for main control flow, and `MyThread` workers for
executing individual scripts. A `ReusableBarrier` facilitates global
synchronization across devices, and a shared `lock_hash` manages
location-specific data access.
"""


from threading import Semaphore, Event, Lock, Thread, Condition # Condition imported for ReusableBarrier

class ReusableBarrier(object):
    """
    A reusable double-barrier synchronization primitive implemented using semaphores.

    This barrier allows a fixed number of threads (`num_threads`) to wait for
    each other to reach a common point before any can proceed. It is designed
    to be reusable across multiple synchronization points within a larger simulation loop.
    """
    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must arrive
                               at the barrier before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase of the barrier.


        self.count_threads2 = self.num_threads # Counter for the second phase of the barrier.
        self.counter_lock = Lock() # Lock to protect the counters during updates.
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase, initialized to block all threads.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase, initialized to block all threads.

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        `num_threads` threads have also called `wait()`.
        """
        
        self.phase1() # Executes the first synchronization phase.


        self.phase2() # Executes the second synchronization phase, enabling reusability.

    def phase1(self):
        """
        Manages the first phase of the double-barrier synchronization.

        Block Logic: Decrements a shared counter. When it reaches zero, all threads
                     have arrived at the barrier for this phase. It then releases
                     all waiting threads via the first semaphore and resets the
                     counter for the next phase.
        """
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in xrange(self.num_threads): # Note: `xrange` is Python 2.x specific.
                    self.threads_sem1.release() # Releases all threads blocked on `threads_sem1`.
                self.count_threads1 = self.num_threads # Resets the counter for barrier reusability.

        self.threads_sem1.acquire() # Blocks the current thread until it's released by the semaphore.

    def phase2(self):
        """
        Manages the second phase of the double-barrier synchronization.

        Block Logic: Decrements a shared counter. When it reaches zero, all threads
                     have passed through the first phase. It then releases all
                     waiting threads via the second semaphore and resets the
                     counter for the next cycle, making the barrier fully reusable.
        """
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in xrange(self.num_threads): # Note: `xrange` is Python 2.x specific.
                    self.threads_sem2.release() # Releases all threads blocked on `threads_sem2`.
                self.count_threads2 = self.num_threads # Resets the counter for barrier reusability.

        self.threads_sem2.acquire() # Blocks the current thread until it's released by the semaphore.

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.none_script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.timepoint_end = 0
        self.barrier = None
        self.lock_hash = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        
        self.barrier = barrier

    def set_locks(self, lock_hash):
        
        self.lock_hash = lock_hash

    def setup_devices(self, devices):
        
        
        ids_list = []
        for dev in devices:
            ids_list.append(dev.device_id)


        if self.device_id == min(ids_list):
            
            self.barrier = ReusableBarrier(len(devices))
            self.lock_hash = {}

            for dev in devices:
                for location in dev.sensor_data:
                    if location not in self.lock_hash:
                        self.lock_hash[location] = Lock()

            
            
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.set_barrier(self.barrier)
                    dev.set_locks(self.lock_hash)


    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.none_script_received.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.semaphore = Semaphore(value=8)

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.none_script_received.wait()
            self.device.none_script_received.clear()

            thread_list = []

            
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, neighbours, script, location,
                    self.semaphore)


                thread.start()
                thread_list.append(thread)

            for i in xrange(len(thread_list)):
                thread_list[i].join()

            
            
            
            self.device.barrier.wait()

class MyThread(Thread):
    

    def __init__(self, device, neighbours, script, location, semaphore):
        

        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.semaphore = semaphore

    def run(self):
        
        self.semaphore.acquire()

        self.device.lock_hash[self.location].acquire()

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

        
        self.device.lock_hash[self.location].release()

        
        self.semaphore.release()
