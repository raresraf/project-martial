


from threading import *
from barrier import ReusableBarrierCond
import Queue


class Device(object):
	
	def __init__(self, device_id, sensor_data, supervisor):
		
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor
		self.script_received = Event()
		self.scripts = []
		self.timepoint_done = Event()
		self.devices = []
		self.location_lock=[]
		self.queue=Queue.Queue()
		self.event_start = Event()
		self.barrier = None
		self.thread = DeviceThread(self)
		self.thread.start()
		self.script_semaphore = Semaphore(value=0)
		self.stop_workers = False

		



	def __str__(self):
		
		return "Device %d" % self.device_id

	def setup_devices(self, devices):
		

		if self.device_id == 0:
			self.barrier = ReusableBarrierCond(len(devices))
			for i in range(100):
				self.location_lock.append(Lock())
			for dev in devices:
				dev.location_lock = self.location_lock
				dev.barrier = self.barrier
				dev.event_start.set()
		
		self.devices=devices

		

		

	def assign_script(self, script, location):
		
		if script is not None:
			self.scripts.append((script, location))
		else:
			self.timepoint_done.set()

	def get_data(self, location):
		


		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		
		if location in self.sensor_data:
			self.sensor_data[location] = data

	def shutdown(self):
		
		self.thread.join()

class Worker(Thread):

	def __init__(self,device,worker_id):


		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device
		self.worker_id = worker_id



	def run(self):
		while True:
			self.device.script_semaphore.acquire()
			if self.device.stop_workers is True:
				break

			tuplu=self.device.queue.get()
			script = tuplu[0]
			location = tuplu[1]
			neighbours = tuplu[2]
			lock = tuplu[3]

			with lock:
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






class DeviceThread(Thread):
	

	def __init__(self, device):
		


		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device

	def run(self):
		

		self.device.event_start.wait()
		t=[]
		for i in range(8):
			thread=Worker(self.device,i)
			t.append(thread)

		for i in range(8):
			t[i].start()

		while True:
			

			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break

			self.device.timepoint_done.wait()

			

			for (script, location) in self.device.scripts:
				self.device.queue.put((script , location , neighbours, self.device.location_lock[location]))
				self.device.script_semaphore.release()

			
			self.device.timepoint_done.clear()


			self.device.barrier.wait()

		self.device.stop_workers = True 

		for i in range(8):
			self.device.script_semaphore.release()

		for i in range(8):
			t[i].join()

		
