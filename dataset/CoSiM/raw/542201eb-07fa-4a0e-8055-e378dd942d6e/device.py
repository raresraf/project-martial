


from threading import Event, Thread, Lock
from reusable_barrier_semaphore import ReusableBarrier
import Queue
NUMBER_OF_THREADS = 8

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.data_lock = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        barrier = ReusableBarrier(len(devices))

        if self.barrier is None:
            self.barrier = barrier

        for device in devices:

            if device.barrier is None:
                device.barrier = barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    
    location_locks = {}

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = []
        self.scripts_queue = Queue.Queue()

    def run(self):
        
        for _ in range(NUMBER_OF_THREADS):
            self.threads.append(ScriptThread(self.scripts_queue))

        for script_thread in self.threads:
            script_thread.start()

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            
            if neighbours is None:
                for script_thread in self.threads:
                    self.scripts_queue.put(MyObjects(None, None, None, None,
                                                     False, None))
                break

            
            self.device.timepoint_done.wait()
            for (script, location) in self.device.scripts:

                
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()

                self.scripts_queue.put(MyObjects(self.device, location, script,
                                                 neighbours, True,
                                                 self.location_locks),
                                       block=True, timeout=None)
            self.device.timepoint_done.clear()

            
            self.device.barrier.wait()

        for script_thread in self.threads:
            script_thread.join()

class ScriptThread(Thread):
    

    def __init__(self, queue):
        
        Thread.__init__(self, name="Script Thread")
        self.queue = queue

    def run(self):
        
        while True:

            
            my_objects = self.queue.get(block=True, timeout=None)

            
            if my_objects.stop == False:
                break

            
            my_objects.location_locks[my_objects.location].acquire()

            script_data = []
            
            for device in my_objects.neighbours:
                data = device.get_data(my_objects.location)
                if data is not None:
                    script_data.append(data)

            
            my_objects.device.data_lock.acquire()
            data = my_objects.device.get_data(my_objects.location)
            my_objects.device.data_lock.release()

            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = my_objects.script.run(script_data)

                
                for device in my_objects.neighbours:
                    device.data_lock.acquire()
                    device.set_data(my_objects.location, result)
                    device.data_lock.release()

                
                my_objects.device.data_lock.acquire()
                my_objects.device.set_data(my_objects.location, result)
                my_objects.device.data_lock.release()

            my_objects.location_locks[my_objects.location].release()

class MyObjects():
    

    def __init__(self, device, location, script, neighbours, stop, location_locks):
        
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.stop = stop


        self.location_locks = location_locks
from threading import *

class ReusableBarrier():
    def __init__(self, num_threads):
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
                                                 

class MyThread(Thread):
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",