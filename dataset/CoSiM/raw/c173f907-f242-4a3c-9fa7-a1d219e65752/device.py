


from threading import Event, Thread, Lock
from barrier import *

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []  
        self.scriptType = ""
        self.timepoint_done = Event()

        self.allDevices = []
        self.devices_setup = Event()

        
        self.barrierLoop = []        
        
        self.canRequestResourcesLock = Lock()
        


        self.myResourceLock = { loc : Lock() for loc in self.sensor_data.keys() }
        
        self.neighbours = []
        
        self.numWorkers = 8

        
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        self.allDevices = devices
        self.barrierLoop = ReusableBarrierCond(len(devices))
        self.devices_setup.set()


    def assign_script(self, script, location):
        

        if script is not None:
            
            self.scripts.append((script, location))
            self.scriptType = "SCRPIT"
            self.script_received.set()
        else:
            
            self.scriptType = "TIMEPOINT"
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        
        ret = self.sensor_data[location] if location in self.sensor_data else None
        return ret

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()




class WorkerThread(Thread):
    

    def __init__(self, device, listOfIndexes):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.listOfIndexes = listOfIndexes
    
    def run(self):
        
        
        for i in self.listOfIndexes:
            (script, location) = self.device.scripts[i]

            
            self.device.allDevices[0].canRequestResourcesLock.acquire()
            
            if location in self.device.myResourceLock:
                self.device.myResourceLock[location].acquire()
            for device in self.device.neighbours:
                if self.device.device_id != device.device_id:
                      if location in device.myResourceLock:
                            device.myResourceLock[location].acquire()
            self.device.allDevices[0].canRequestResourcesLock.release()


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
                
                
                self.device.set_data(location, result)
                
                for device in self.device.neighbours:
                    device.set_data(location, result)

            
            if location in self.device.myResourceLock:
                self.device.myResourceLock[location].release()
            for device in self.device.neighbours:
                if self.device.device_id != device.device_id:
                      if location in device.myResourceLock:
                            device.myResourceLock[location].release()



class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

        
        self.device.devices_setup.wait()

        while True:

            
            self.device.allDevices[0].barrierLoop.wait()

            
            self.device.neighbours = self.device.supervisor.get_neighbours()
                
            
            if self.device.neighbours is None:
                break
            
            
            while True:
                self.device.script_received.wait()
                self.device.script_received.clear()
                if self.device.scriptType == "SCRIPT":
                    continue
                self.device.timepoint_done.wait()
                self.device.timepoint_done.clear()
                break

            
            
            workerThreadList = []
            indexesList = []
        
            
            for i in range(self.device.numWorkers):
                indexesList.append([])
            for i in range(len(self.device.scripts)):
                indexesList[i%self.device.numWorkers].append(i)
                
            
            for i in range(self.device.numWorkers):
                if indexesList[i] != []:
                    workerThread = WorkerThread(self.device,indexesList[i])
                    workerThreadList.append(workerThread)
                    workerThread.start()

            
            for i in range(self.device.numWorkers):
                if indexesList[i] != []:
                    workerThreadList[i].join()

