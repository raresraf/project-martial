


from threading import Event, Thread, Lock
import Barrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()


        self.scripts = []
        self.setup_done = Event()
        self.devices = []
        self.barrier = None
        self.locks = None
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        for device in devices:
            if self.device_id != device.device_id:
                self.devices.append(device)

        if self.device_id == 0:
        
            self.barrier = Barrier.Barrier(len(devices))
            
            self.locks = {}
            
            for device in devices:
                device.barrier = self.barrier
                device.locks = self.locks
                
                

        self.setup_done.set()

    def assign_script(self, script, location):
        

        if script is not None:
            if ~((self.locks).has_key(location)):
            
                self.locks[location] = Lock()
                
            self.scripts.append((script, location))

        else:
            self.script_received.set()
            

    def get_data(self, location):
        
        res = None
        if location in self.sensor_data:
            res = self.sensor_data[location]

        return res

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    @staticmethod
    def split(script_list, number):
        
       
        res = [[] for i in range(number)]
        i = 0
        while i < len(script_list):
            part = script_list[i]
            res[i%number].append(part)
            i = i + 1
       
        return res

    def run_scripts(self, scripts, neighbours):
        
        for (script, location) in scripts:
            with self.device.locks[location]:
            

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


    def run(self):

        self.device.setup_done.wait()
        
        for device in self.device.devices:
            device.setup_done.wait()
            

        while True:

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()
            

            if len(self.device.scripts) != 0:
            
                scripts_list = self.split(self.device.scripts, 8)
                
                

                thread_list = []
                
                for scripts in scripts_list:
                    new_thread = Thread(target=self.run_scripts,
                                                     args=(scripts, neighbours))
                    
                    thread_list.append(new_thread)
                    

                for thread in thread_list:
                    thread.start()
                    

                for thread in thread_list:
                    thread.join()
                    


            self.device.script_received.clear()
            

            self.device.barrier.wait()
            
            
