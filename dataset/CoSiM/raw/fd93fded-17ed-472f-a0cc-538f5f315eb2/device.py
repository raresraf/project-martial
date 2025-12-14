


from threading import Event, Lock
from barrier import ReusableBarrierCond
from device_thread import DeviceThread

import multiprocessing

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        

        
        self.neighbours = []

        self.device_id = device_id


        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.scripts_aux = []
        self.counter = 0
        self.timepoint_done = Event()

        
        self.pop_script_lock = Lock()

        
        
        self.devices_barrier = None

        
        
        
        
        
        self.location_locks = {}
        for location in self.sensor_data:
            self.location_locks[location] = Lock()

        
        
        
        
        self.global_location_locks = {}

        self.threads = []
        self.number_of_threads = 4 * multiprocessing.cpu_count()

        
        self.threads_barrier = ReusableBarrierCond(self.number_of_threads)

        for i in range(self.number_of_threads):
            self.threads.append(DeviceThread(self, i))

        for i in range(self.number_of_threads):
            self.threads[i].start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices_barrier(self, barrier):
        
        self.devices_barrier = barrier

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:
            self.devices_barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.devices_barrier = self.devices_barrier
                    device.global_location_locks = self.global_location_locks

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
            self.scripts_aux.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] \
            if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for i in range(self.number_of_threads):
            self.threads[i].join()


from threading import Thread, Lock

class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        while True:
            if self.thread_id == 0:
                self.device.timepoint_done.clear()
                for script in self.device.scripts:
                    self.device.scripts_aux.append(script)
                self.device.neighbours = self.device.supervisor.get_neighbours()

            self.device.threads_barrier.wait()

            if self.device.neighbours is None:
                break

            
            while True:
                if self.device.timepoint_done.is_set() \
                        and len(self.device.scripts_aux) == 0:
                    break

                self.device.pop_script_lock.acquire()
                if len(self.device.scripts_aux) > 0:
                    (script, location) = self.device.scripts_aux.pop(0)
                    self.device.pop_script_lock.release()
                else:
                    self.device.pop_script_lock.release()
                    continue

                script_data = []


                
                if location not in self.device.global_location_locks:
                    self.device.global_location_locks[location] = Lock()

                
                
                
                
                with self.device.global_location_locks[location]:
                    for device in self.device.neighbours:
                        if location in device.sensor_data:

                            
                            device.location_locks[location].acquire()
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)

                    if self.device not in self.device.neighbours:
                        if location in self.device.sensor_data:
                            self.device.location_locks[location].acquire()
                            data = self.device.get_data(location)
                            if data is not None:
                                script_data.append(data)

                if script_data != []:
                    result = script.run(script_data)

                    for device in self.device.neighbours:
                        if location in device.sensor_data:
                            device.set_data(location, result)
                            device.location_locks[location].release()

                    if self.device not in self.device.neighbours:
                        if location in self.device.sensor_data:
                            self.device.set_data(location, result)


                            self.device.location_locks[location].release()

            
            self.device.threads_barrier.wait()

            
            if self.thread_id == 0:
                self.device.devices_barrier.wait()
