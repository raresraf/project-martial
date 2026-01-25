


from threading import Event, Thread, Lock
from barrier import ReusableBarrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        
        self.locks = []
        
        self.barrier = ReusableBarrier(0)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:

            
            nr_locations = 0
            for i in xrange(len(devices)):
                nr_locations = max(nr_locations,
            		max(devices[i].sensor_data.keys()))

            
            for i in xrange(nr_locations + 1):
                self.locks.append(Lock())

            
            barrier = ReusableBarrier(len(devices))

            for i in xrange(len(devices)):
            	
                devices[i].barrier = barrier
                
                for j in xrange(nr_locations + 1):
                    devices[i].locks.append(self.locks[j])

            
            for i in xrange(len(devices)):
                devices[i].thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

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
            if neighbours is None:
                break

            


            self.device.timepoint_done.wait()

          	
            worker_list = []

            
            for (script, location) in self.device.scripts:
                worker_list.append(Worker(self.device,
                	location, script, neighbours))

           	
            for i in xrange(len(worker_list)):
                worker_list[i].start()

            
            for i in xrange(len(worker_list)):
                worker_list[i].join()

            
            self.device.barrier.wait()

            
            self.device.timepoint_done.clear()


class Worker(Thread):
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self, name="Worker")
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):

		
        self.device.locks[self.location].acquire()

        script_data = []

		
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

		
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
			
            result = self.script.run(script_data)

			
            for device in self.neighbours:
                device.set_data(self.location, result)

			
            self.device.set_data(self.location, result)

		
        self.device.locks[self.location].release()
