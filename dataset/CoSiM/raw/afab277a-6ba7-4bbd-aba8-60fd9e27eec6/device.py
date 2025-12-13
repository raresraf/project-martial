


from threading import Event, Thread, Lock, Semaphore
import Queue

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        

        self.devices = devices
        self.barrier = ReusableBarrier(len(self.devices))
      
        for i in xrange(len(self.devices)):
            self.devices[i].barrier = self.barrier

    def assign_script(self, script, location):
        
        

        if script is not None:
            self.scripts.append((script, location))
            
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class Worker(Thread):

    def __init__(self, device, neighbours, script, location):
        


        Thread.__init__(self, name="Thread %d's Worker " % (device.device_id))
        
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location

    
    def run(self):
        
        scriptData = []        
        data = self.device.get_data(self.location)
        
        if not data is None:
            scriptData.append(data)

        for device in self.neighbours:
            data = device.get_data(self.location)
            if not data is None:
                scriptData.append(data)


        if scriptData:
            
            newData = self.script.run(scriptData)

            


            for device in self.neighbours:
                device.set_data(self.location, newData)
            self.device.set_data(self.location, newData)

    def shutdown(self):
        self.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)

        self.device = device
        

    def run(self):
        
        
        q = Queue.Queue()

        
        listOfWorkers = []
        numberOfWorkers = 0

        while True:
            
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.script_received.wait()
            
            
            for (script, location) in self.device.scripts:
                q.put((script,location))
          
            while not q.empty():

                (script,location) = q.get()         

                
                if numberOfWorkers < 8:
                   
                    worker = Worker(self.device, neighbours, script, location)
                    listOfWorkers.append(worker)
                    worker.start()
                    numberOfWorkers += 1
                
                else:

                    index = -1
                    for i in range(len(listOfWorkers)):
                        if not listOfWorkers[i].is_alive():
                            listOfWorkers[i].shutdown()
                            index = i
                            break
                    listOfWorkers.remove(listOfWorkers[index])
                    
                    worker = Worker(self.device, neighbours, script, location)
                    listOfWorkers.insert(index,worker)
                    listOfWorkers[index].start()
                    numberOfWorkers+=1;


                q.task_done() 

            
            for i in range(len(listOfWorkers)):


                listOfWorkers[i].shutdown()

            self.device.timepoint_done.wait()       
            self.device.barrier.wait()              
            
            
            self.device.script_received.clear()     
            self.device.timepoint_done.clear()      



class ReusableBarrier():
    

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.counter_lock = Lock()                  
        self.threads_sem1 = Semaphore(0)            
        self.threads_sem2 = Semaphore(0)            
 
    def wait(self):
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        with self.counter_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:               
                for i in range(self.num_threads):
                    threads_sem.release()           
                count_threads[0] = self.num_threads 
        threads_sem.acquire()     