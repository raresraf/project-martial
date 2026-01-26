


from threading import enumerate, Event, Thread, Lock, Semaphore

class ReusableBarrierSem():

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
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            self.count_threads1 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    self.barrier = device.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
    def shutdown(self):
        
        self.thread.join()

class Node(Thread):

    def __init__(self, script, script_data):

        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        self.result = None
         
    def run(self):
        self.result = self.script.run(self.script_data)

    def join(self):
        Thread.join(self)
        return (self.script, self.result)


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            thread_list=[]
            scripts_result = {}
            scripts_data = {}
            if neighbours is None:
                break

            string = ""
            for neighbour in neighbours:
                string = string + " " + str(neighbour)
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            for (script, location) in self.device.scripts:
                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                scripts_data[script] = script_data
                if script_data != []:
                    nod = Node(script,script_data)
                    thread_list.append(nod)
            for nod in thread_list:
                
                nod.start()
            for nod in thread_list:
                key ,value = nod.join()
                scripts_result[key] = value
            for (script, location) in self.device.scripts:
                
                if scripts_data[script] != []:
                    for device in neighbours:
                        device.set_data(location, scripts_result[script])
                        
                    self.device.set_data(location, scripts_result[script])
            
            self.device.barrier.wait()
