

from rr import ReusableBarrier
from threading import Event, Thread, Lock


L_LOCKS = {}
LOCK = Lock()
BARRIER = None

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.event = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id > 0:
            self.event.wait()
            self.thread.start()
        else:
            global BARRIER
            BARRIER = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id > 0:
                    device.event.set()

            self.thread.start()

    def assign_script(self, script, location):
        
        if script is None:
            self.timepoint_done.set()
        else:
            self.scripts.append((script, location))


    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data \
	else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
	
        while True:
            global BARRIER
            BARRIER.wait()
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()
            cs = self.device.scripts

            
            for (script, location) in self.device.scripts:
                global LOCK
                LOCK.acquire()

                global L_LOCKS
                if not location in L_LOCKS.keys():
                    L_LOCKS[location] = Lock()
                L_LOCKS[location].acquire()
                LOCK.release()

                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
		    
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                L_LOCKS[location].release()

            
            self.device.timepoint_done.clear()
from threading import Thread, Lock, Semaphore

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
            print "I'm Thread " + str(self.tid) + \
            " after barrier, in step " + str(i) + "\n",

