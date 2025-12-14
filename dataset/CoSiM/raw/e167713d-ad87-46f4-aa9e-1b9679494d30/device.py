


from threading import Event, Thread, Lock
from barrier import Barrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbours = []
        self.scripts = []
        self.temp_scripts = []

        
        self._thread_list = []
        self.timepoint_done = Event()
        self.device_lock = Lock()
        self.script_list_lock = Lock()
        self.locations_locks = {}
        self.device_thread_barrier = None
        self.thread_number = 8

        
        for thread_id in xrange(self.thread_number):
            thread = DeviceThread(self, thread_id)
            self._thread_list.append(thread)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
 
        
        if self.device_thread_barrier == None:
            self.device_thread_barrier = Barrier(len(devices) * self.thread_number)
            for dev in devices:
                dev.device_thread_barrier = self.device_thread_barrier

        
        max_location = -1
        if not self.locations_locks:
            for dev in devices:
                for key in dev.sensor_data:
                    if key > max_location:
                        max_location = key
            for i in xrange(max_location + 1):
                self.locations_locks[i] = Lock()
            for dev in devices:
                dev.locations_locks = self.locations_locks

        
        for thread_id in xrange(self.thread_number):
            self._thread_list[thread_id].start()

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
        

        for i in xrange(len(self._thread_list)):
            self._thread_list[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.thread_id = thread_id

    def run(self):

        while True:

            self.device.device_thread_barrier.wait()
            if self.thread_id == 0:
                
                self.device.neighbours = self.device.supervisor.get_neighbours()

            self.device.device_thread_barrier.wait()
            
            if self.device.neighbours is None:
                break

            if self.thread_id == 0:
                
                self.device.timepoint_done.wait()
                self.device.timepoint_done.clear()
                self.device.temp_scripts = list(self.device.scripts)

            self.device.device_thread_barrier.wait()

            done_iter = False
            while True:
                item = ()
                
                self.device.script_list_lock.acquire()
                if len(self.device.temp_scripts) > 0:
                    item = self.device.temp_scripts.pop(0)
                else:
                    done_iter = True
                self.device.script_list_lock.release()

                
                if done_iter:
                    break

                script = item[0]
                location = item[1]

                
                self.device.locations_locks[location].acquire()

                script_data = []
                
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in self.device.neighbours:
                        device.device_lock.acquire()
                        device.set_data(location, result)
                        device.device_lock.release()

                    
                    self.device.device_lock.acquire()
                    self.device.set_data(location, result)
                    self.device.device_lock.release()

                self.device.locations_locks[location].release()
