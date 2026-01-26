


from threading import Event, Thread, Lock
from Barrier import *


class Device(object):
	

	def __init__(self, device_id, sensor_data, supervisor):
		
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor


		self.script_received = Event()
		self.scripts = []
		self.num_devices = 0
		self.thread = DeviceThread(self)
		self.thread.start()

	def __str__(self):
		
		return "Device %d" % self.device_id

	def setup_devices(self, devices):
		
		
		
		self.num_devices = len(devices)
		if devices[0].thread.barrier is None:
			devices[0].thread.barrier = Barrier(self.num_devices, str(self.device_id))
			for i in range (1, len(devices)):
				devices[i].thread.barrier = devices[0].thread.barrier

	def assign_script(self, script, location):
		
		if script is not None:
			with self.thread.lock:
				self.scripts.append((script, location))
		else:
			self.script_received.set()

	def get_data(self, location):
		
		with self.thread.lock:
			return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		
		with self.thread.lock:
			if location in self.sensor_data:
				if self.sensor_data[location] < data:
					self.sensor_data[location] = data

	def shutdown(self):
		
		self.thread.join()


class DeviceThread(Thread):
	

	def __init__(self, device):
		
		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device
		self.barrier = None
		self.lock = Lock()

	def run(self):
		
		while True:
			
			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break

			self.device.script_received.wait()
			self.device.script_received.clear()

			
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

					
					for device in neighbours:
						device.set_data(location, result)
					
					self.device.set_data(location, result)

			
			
			
			self.barrier.wait()
