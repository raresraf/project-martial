

import barrier
from threading import Event, Thread, Lock


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

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == len(devices) - 1:
            my_barrier = barrier.ReusableBarrierCond(len(devices))
            my_dictionary = dict()
            for dev in devices:
                for location, data in dev.sensor_data.iteritems():
                    if location not in my_dictionary:
                        my_dictionary[location] = Lock()
            for dev in devices:
                dev.barrier = my_barrier
                dev.dictionary = my_dictionary

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data \
        else None

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
            if neighbours is None:
                break
            threads = []
            div = len(self.device.scripts) // 8
            mod = len(self.device.scripts) % 8
            self.device.timepoint_done.wait()

            for division in range(8):
                if div > 0:
                    list_of_scripts = \
                    self.device.scripts[division * div: (division+1) * div]
                else:
                    list_of_scripts = []
                if mod > 0:
                    list_of_scripts.append\
                    (self.device.scripts[len(self.device.scripts) - mod])
                    mod = mod - 1
                threads.append(MyThread\


                (self.device, list_of_scripts, neighbours))

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            self.device.timepoint_done.clear()
            self.device.barrier.wait()

class MyThread(Thread):
    

    def __init__(self, device, scripts, neighbours):
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):

	
        for (script, location) in self.scripts:
            self.device.dictionary[location].acquire()
            script_data = []

	    
            for dev in self.neighbours:
                data = dev.get_data(location)
                if data is not None:
                    script_data.append(data)
	    
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
	        
                result = script.run(script_data)

		
                
                for dev in self.neighbours:
                    dev.set_data(location, result)
		
                self.device.set_data(location, result)
            self.device.dictionary[location].release()
