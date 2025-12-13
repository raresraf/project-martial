


from threading import Event, Thread, Lock
from barrier import RBarrier


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
            
            self.time_bar = RBarrier(len(devices))
            self.script_bar = RBarrier(len(devices))

            
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
                maxim = maxim - 1

            
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


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

        while True:

            
            neighbours = self.device.supervisor.get_neighbours()

            
            self.device.script_received.wait()

            
            self.device.script_bar.wait()

            
            self.device.time_bar.wait()

            if neighbours is None:
                break

            
            for (script, location) in self.device.scripts:

                
                self.device.devloc[location].acquire()

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

                
                self.device.devloc[location].release()
