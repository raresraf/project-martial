


from threading import Event, Thread, Lock
import barrier

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
        self.locks = None
        self.barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        devices[0].barrier = barrier.ReusableBarrierSem(len(devices))
        devices[0].locks = {}
        list_index = list(range(len(devices)))
        for i in list_index[1:len(devices)]:
            devices[i].barrier = devices[0].barrier
            devices[i].locks = devices[0].locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def thread_script(self, neighbours, script, location):
        

        
        script_data = []
        if location not in self.device.locks:
            self.device.locks[location] = Lock()

        


        self.device.locks[location].acquire()

        
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

        
        self.device.locks[location].release()

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            threads_script = []
            for (script, location) in self.device.scripts:
                
                thread = Thread(target=self.thread_script,
                    args=(neighbours, script, location))
                thread.start()
                threads_script.append(thread)

            
            for j in xrange(len(threads_script)):
                threads_script[j].join()

            
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
