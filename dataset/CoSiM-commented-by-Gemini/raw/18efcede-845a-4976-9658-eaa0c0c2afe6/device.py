

from barrier import ReusableBarrierCond
from threading import Lock, Event, Thread

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
		self.list_of_devices = []
		self.lock_for_data = Lock()
		self.lock_for_location = Lock()
		self.number_of_devices = 0
		self.reusable_bar = Event()

	def __str__(self):
		
		return "Device %d" % self.device_id

	def setup_devices(self, devices):
		

		
		self.list_of_devices = devices
		self.number_of_devices = len(devices)

	def assign_script(self, script, location):
		
		if script is not None:
			self.scripts.append((script, location))
		self.script_received.set()

	def get_data(self, location):
		
		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		
		
		if location in self.sensor_data:
			self.lock_for_data.acquire()
			self.sensor_data[location] = data
			self.lock_for_data.release()

	def shutdown(self):
		
		self.thread.join()

	def set_barrier(self, barrier):
		self.reusable_bar.set()
		self.bar = barrier


class DeviceThread(Thread):
	

	def __init__(self, device):
		
		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device

	def run(self):
		ok = 0

		
		while True:
			
			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break

			
			if ok == 0 and self.device.device_id == 0: 
				self.bar = ReusableBarrierCond(self.device.number_of_devices)
				for dev in self.device.list_of_devices:
					dev.set_barrier(self.bar)
				ok += 1
			
			if ok == 0:
				


				self.device.reusable_bar.wait() 

			
			self.device.bar.wait()

			
			self.device.script_received.wait()

			
			for (script, location) in self.device.scripts:
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

					
					self.device.lock_for_location.acquire()
					for device in neighbours:
						device.set_data(location, result)
					
					self.device.set_data(location, result)
					self.device.lock_for_location.release()

			
			self.device.bar.wait()
