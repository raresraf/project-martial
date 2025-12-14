


from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()
        self.dictLocks = {}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))

            for device in devices:
                for location in device.sensor_data.keys():
                    if self.dictLocks.has_key(location) == False:
                        self.dictLocks[location] = Lock()
                device.setup_mutualBarrier(self.barrier, self.dictLocks)


    def setup_mutualBarrier(self, barrier, dictLocks):
        
        if self.device_id != 0:
            self.barrier = barrier
            self.dictLocks = dictLocks


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


def runScripts((script, location), neighbours, callingDevice):
    

    script_data = []
    
    callingDevice.dictLocks[location].acquire()
    for device in neighbours:
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)
            
    data = callingDevice.get_data(location)
    if data is not None:
        script_data.append(data)

    if script_data != []:
        
        result = script.run(script_data)

        


        for device in neighbours:
            device.set_data(location, result)
            
            callingDevice.set_data(location, result)
    callingDevice.dictLocks[location].release()



class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()

            threadsList = []
            index = 0
            nrScripts = len(self.device.scripts)
            
            while nrScripts:
                
                if nrScripts > 7:
                    for j in range(8):
                        threadsList.append(
                        Thread(target=runScripts, args=
                        (self.device.scripts[index], neighbours, self.device)))
                        index += 1
                    nrScripts = nrScripts - 8
                else:
                    for j in range(nrScripts):
                        threadsList.append(
                        Thread(target=runScripts, args=
                        (self.device.scripts[index], neighbours, self.device)))
                        index += 1
                    nrScripts = 0

                
                for j in range(len(threadsList)):
                    threadsList[j].start()

                
                for j in range(len(threadsList)):
                    threadsList[j].join()

                threadsList = []

            
            self.device.script_received.clear()
            
            self.device.barrier.wait()
