


from threading import Event, Thread, Condition, Lock

class Barrier(object):
    def __init__(self, num_threads=0):
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

        
class Device(object):
    
    
    
    

    DeviceBarrier = Barrier()
    DeviceLocks = []


    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        
        self.script_received = Event()
        self.scripts = []


        self.locations = []
        
        self.DeviceLocks = []


        self.currentScript = 0
        self.scriptNumber = 0
        
        
        self.timepoint_done = Event()

        
        self.neighbours = []
        self.neighbours_event = Event()
        self.lockScript = Lock()
        self.barrier = Barrier(8)
        
        
        
        self.thread = DeviceThread(self, True)
        self.thread.start()
        self.threads = []
        
        for _ in range(7):
            newThread = DeviceThread(self, False)
            self.threads.append(newThread)
            newThread.start()
        
        

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
           
    
        
        size = len(devices)
        Device.DeviceBarrier = Barrier(size)
        if Device.DeviceLocks==[]:
            self.updateLocks()


    def getNeighbours(self):
        return self.supervisor.supervisor.testcase.num_locations

    def updateLocks(self):
        for _ in range(self.getNeighbours()):
            Device.DeviceLocks.append(Lock())

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append(script)
            self.locations.append(location)
            self.scriptNumber += 1
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()
        for myThread in self.threads:
            myThread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, isInitiator):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.isInitiator = isInitiator
      
    def neighboursOperation(self):
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_event.set()
                self.device.currentScript = 0

    def reserve(self):
            self.device.lockScript.acquire()
            index = self.device.currentScript
            self.device.currentScript += 1
            self.device.lockScript.release()    
            
            return index    

    def acquireLocation(self, location):
        Device.DeviceLocks[location].acquire()

    def releaseLocation(self, location):
        Device.DeviceLocks[location].release()

    def ThreadWait(self):
        self.device.barrier.wait()

    def CheckForInitiator(self):
        return self.isInitiator

    def finishUp(self):
        self.ThreadWait()
        if self.CheckForInitiator():
            self.device.neighbours_event.clear()
            self.device.timepoint_done.clear()
        self.ThreadWait()
        if self.CheckForInitiator():
            Device.DeviceBarrier.wait()



    def run(self):
        
        while True:
            


            if self.isInitiator == True:
                
                self.neighboursOperation()
            self.device.neighbours_event.wait()
            if self.device.neighbours is None:
                break
            self.device.timepoint_done.wait()
            
            while True:
                index = self.reserve()


                
                
                if index >= self.device.scriptNumber:
                    break
                location = self.device.locations[index]
                script = self.device.scripts[index]
                self.acquireLocation(location)
                script_data = []
                
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                self.releaseLocation(location)

            self.finishUp()