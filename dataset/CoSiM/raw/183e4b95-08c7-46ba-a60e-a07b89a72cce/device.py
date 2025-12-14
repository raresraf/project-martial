


from threading import Event, Thread, Lock , Condition
from queue import Worker, ThreadPool
from reusable_barrier_semaphore import ReusableBarrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
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
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.allDevices = devices
        self.barrier = ReusableBarrier(len(devices))

        for i in range(0, 50):
            self.locks.append(Lock())

        pass

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.pool.add_task(self.executeScript,script,location)
        else:
            self.script_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

    def executeScript(self,script,location):

        self.wait_neighbours.wait()
        script_data = []

        if not self.neighbours is None:
            for device in self.neighbours:
                device.locks[location].acquire()
                data = device.get_data(location)
                device.locks[location].release()

                if data is not None:
                    script_data.append(data)

        self.locks[location].acquire()
        data = self.get_data(location)
        self.locks[location].release()

        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = script.run(script_data)

            if not self.neighbours is None:
                for device in self.neighbours:

                    device.locks[location].acquire()
                    device.set_data(location, result)
                    device.locks[location].release()

            self.locks[location].acquire()
            self.set_data(location, result)
            self.locks[location].release()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

        while True:
            self.device.script_received.clear()
            self.device.wait_neighbours.clear()

            self.device.neighbours = []


            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.wait_neighbours.set()

            if self.device.neighbours is None:
                self.device.pool.wait_completion()
                self.device.pool.terminateWorkers()
                self.device.pool.threadJoin()
                return

            for (script, location) in self.device.scripts:


                self.device.pool.add_task(self.device.executeScript,script,location)

            self.device.script_received.wait()
            self.device.pool.wait_completion()

            for dev in self.device.allDevices:
                dev.barrier.wait()



from Queue import Queue
from threading import Thread

class Worker(Thread):
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.terminate_worker = False
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            if func == None:
                self.tasks.task_done()
                break
            try: func(*args, **kargs)
            except Exception, e: print e
            self.tasks.task_done()


class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = Queue(99999)
        self.workers = []
        for _ in range(num_threads):
            self.workers.append(Worker(self.tasks))

    def add_task(self, func, *args, **kargs):
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        self.tasks.join()

    def terminateWorkers(self):
        for worker in self.workers:
            worker.tasks.put([None,None,None])
            worker.terminate_worker = True

    def threadJoin(self):
        for worker in self.workers:
            worker.join()
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