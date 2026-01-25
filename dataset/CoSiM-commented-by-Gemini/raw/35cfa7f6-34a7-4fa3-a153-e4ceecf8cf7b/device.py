


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
import time


class Device(object):
	

	def __init__(self, device_id, sensor_data, supervisor):
		
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor
		self.script_received = Event()
		self.cb = Event()
		self.scripts = []
		self.timepoint_done = Event()
		self.thread = DeviceThread(self)
		self.thread.start()
		self.data_lock = Lock()
		self.dev_l = []
		self.no_dev = -1
		self.max = 0
		self.location_lock = []


	def __str__(self):
		
		return "Device %d" % self.device_id


	def setup_devices(self, devices):
		

		
		
		
		for x in devices:
			for val in x.sensor_data:
				
				if val > self.max:
					self.max = val
		
		self.dev_l = devices
		self.no_dev = len(devices)
	
	def get_max_dev(self):
		return self.max
	
	def get_no_dev(self):
		return self.no_dev

	def assign_script(self, script, location):
		
		
		
		if script is not None:
			self.scripts.append((script, location))
			self.script_received.set()
		else:
			self.script_received.set()

	def get_data(self, location):
		
		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		
		if location in self.sensor_data:
			self.data_lock.acquire()
			self.sensor_data[location] = data
			self.data_lock.release()

	def set_b(self, barr,locat):
		self.bar = barr
		self.cb.set()
		self.location_lock = locat

	def shutdown(self):
		
		self.thread.join()


class DeviceThread(Thread):
	

	def __init__(self, device):
		
		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device




	def run(self):

		i = 0
		
	


		
		while True:
		
		
			
			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break
		




		
		
			if self.device.device_id == 0 and i == 0:
				max = 0
				location_lock = []
				for dev in self.device.dev_l:
					if dev.get_max_dev() > max:
						max = dev.get_max_dev()
				max = max + 1
				for i in range(max):
					location_lock.append(Lock())

				self.bar = ReusableBarrierSem(self.device.get_no_dev())
				i=2
				for dev in self.device.dev_l:
					dev.set_b(self.bar,location_lock)
		
			else:


				self.device.cb.wait()
				

			self.device.bar.wait()
		
		
			self.device.script_received.wait()
			
		
		
		
		
		

			
			for (script, location) in self.device.scripts:
				self.device.location_lock[location].acquire()
			
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
					
					self.device.location_lock[location].release()

			
		
			self.device.bar.wait()
			self.device.timepoint_done.set()
		
			
			self.device.timepoint_done.wait()

