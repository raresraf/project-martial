


from threading import Event, Thread, Lock, Semaphore
from reusable_barrier_condition import ReusableBarrier
import multiprocessing
import Queue

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.baariera = None
        self.dicti = {}
        self.device_master = None
    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.device_master = self
            self.baariera = ReusableBarrier(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    self.device_master = device

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            if not self.device_master.dicti.has_key(location):
                self.device_master.dicti[location] = Lock()

        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]

        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class ThreadExecutor(Thread):
    


    def __init__(self, device_thd):
        Thread.__init__(self)
        self.device_thd = device_thd

    def run(self):

        while True:
            
            self.device_thd.sem_produce.acquire()
            
            item = self.device_thd.coada.get()
            
            if item is None:
                break
            neighbours = item[0]
            script = item[1]
            location = item[2]

            script_data = []
            
            
            self.device_thd.device.device_master.dicti[location].acquire()
            
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device_thd.device.get_data(location)
            if data is not None:
                script_data.append(data)



            if script_data != []:
                
                result = script.run(script_data)
                
                for device in neighbours:
                    device.set_data(location, result)
                
                self.device_thd.device.set_data(location, result)
            
            self.device_thd.device.device_master.dicti[location].release()



class DeviceThread(Thread):
    
    def create_workers(self, device_thd):
        
        lista_workers = []
        for _ in xrange(self.numar_proc):
            aux_t = ThreadExecutor(device_thd)
            lista_workers.append(aux_t)

        for thd in lista_workers:


            thd.start()

        return lista_workers

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.numar_proc = multiprocessing.cpu_count()*6
        self.sem_produce = Semaphore(0)
        self.coada = Queue.Queue(maxsize=0)
        self.lista_workers = self.create_workers(self)



    def run(self):

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                
                
                for _ in xrange(self.numar_proc):
                    self.coada.put(None)
                    self.sem_produce.release()

                for item in self.lista_workers:
                    item.join()
                break
            
            self.device.timepoint_done.wait()

            for (script, location) in self.device.scripts:
                
                item = (neighbours, script, location)
                self.coada.put(item)
                self.sem_produce.release()
            
            self.device.device_master.baariera.wait()
            
            self.device.timepoint_done.clear()
from threading import *
 
class ReusableBarrier():
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
 
    def wait(self):
        self.cond.acquire()                      
        self.count_threads -= 1;
        if self.count_threads == 0:
            self.cond.notify_all()               
            self.count_threads = self.num_threads    
        else:
            self.cond.wait();                    
        self.cond.release();                     


class MyThread(Thread):
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
 
    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",