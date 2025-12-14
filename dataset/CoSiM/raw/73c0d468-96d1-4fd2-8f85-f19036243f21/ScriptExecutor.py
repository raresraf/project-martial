
from threading import Event, Thread, Lock, Semaphore
class ScriptExecutor(Thread):
	
	def __init__(self, device, neighbours, current_location, script):
		
		Thread.__init__(self)
		self.device = device


		self.script = script
		self.neighbours = neighbours
		self.current_location = current_location

	def run(self):
		
		self.device.location_locks[self.current_location].acquire()
		script_data = []

		
		for device in self.neighbours:
			data = device.get_data(self.current_location)
			if data is not None:
				script_data.append(data)
		
		data = self.device.get_data(self.current_location)
		if data is not None:
			script_data.append(data)

		if script_data != []:
			
			result = self.script.run(script_data)

			
			for device in self.neighbours:
				device.set_data(self.current_location, result)
			
			self.device.set_data(self.current_location, result)
		self.device.location_locks[self.current_location].release()
		self.device.threads_limit.release()


from threading import Event, Thread, Lock, Semaphore
from ReusableBarrier import ReusableBarrier
from ScriptExecutor import ScriptExecutor

class Device(object):
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        
        self.location_locks = []

        
        self.barrier = None

        
        self.threads = []
        self.threads_limit = Semaphore(8)
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            
            my_locations = []

            for device in devices:
                locations = device.sensor_data.keys()
                for location in locations:
                    
                    if location not in my_locations:
                        my_locations.append(location)
                        self.location_locks.append(Lock())

            
            barrier = ReusableBarrier(len(devices))

            
            for device in devices:
                device.location_locks = self.location_locks
                device.barrier = barrier

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
            self.device.threads = []

            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                    
                self.device.threads_limit.acquire()
                self.device.threads.append(ScriptExecutor(
                    self.device, neighbours, location, script
                    ))
            for thread in self.device.threads:
                thread.start()
            for thread in self.device.threads:
                thread.join()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
