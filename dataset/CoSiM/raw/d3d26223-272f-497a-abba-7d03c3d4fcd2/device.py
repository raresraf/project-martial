


from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem


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
        
        self.time_bar = None
        
        
        self.script_bar = None
        self.devloc = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:
            
            self.time_bar = ReusableBarrierSem(len(devices))
            self.script_bar = ReusableBarrierSem(len(devices))

            
            for device in devices:
                device.time_bar = self.time_bar
                device.script_bar = self.script_bar

            
            maxim = 0
            for device in devices:
                loc_list = device.sensor_data.keys()
                loc_list.sort()
                if loc_list[-1] > maxim:
                    maxim = loc_list[-1]

            
            while maxim >= 0:
                self.devloc.append(Lock())
                maxim -= 1

            
            for device in devices:
                device.devloc = self.devloc


    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.script_bar.wait()



    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()



class ParallelScript(Thread):
    
    def __init__(self, device_thread):
        
        Thread.__init__(self)
        self.device_thread = device_thread
    def run(self):
        while True:
            self.device_thread.sem_scripts.acquire()
            
            nod = self.device_thread.to_procces[0]
            
            del self.device_thread.to_procces[0]
            if nod is None:
                break
            
            neighbours, script, location = nod[0], nod[1], nod[2]


            
            
            self.device_thread.device.devloc[location].acquire()

            script_data = []

            
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                for device in neighbours:
                    device.set_data(location, result)
                
                self.device_thread.device.set_data(location, result)

            
            self.device_thread.device.devloc[location].release()



class DeviceThread(Thread):
    
    def create_pool(self, device_thread):
        
        pool = []
        for _ in xrange(self.numar_procesoare):
            aux_t = ParallelScript(device_thread)
            pool.append(aux_t)
            
            
            aux_t.start()
        return pool




    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.sem_scripts = Semaphore(0)
        self.numar_procesoare = 8 
        self.pool = self.create_pool(self)
        
        self.to_procces = []

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                
                
                for _ in range(self.numar_procesoare):
                    self.to_procces.append(None)
                    self.sem_scripts.release()
                for item in self.pool:
                    item.join()
                break
            
            self.device.script_received.wait()
            
            for (script, location) in self.device.scripts:
                
                
                nod = (neighbours, script, location)
                
                self.to_procces.append(nod)
                
                
                self.sem_scripts.release()

            
            self.device.script_bar.wait()

            
            self.device.time_bar.wait()
            
            self.device.script_received.clear()
