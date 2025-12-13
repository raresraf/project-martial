


from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

import Queue

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.is_available = Lock()
        self.neighbours = []
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.script_queue = Queue.Queue()
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            shared_barrier = ReusableBarrierCond(len(devices))
            location_lock = {}
            for device in devices:
                device.shared_barrier = shared_barrier
                device.location_lock = location_lock

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

        self.script_received.set()

    def get_data(self, location):
        
        self.is_available.acquire()
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else:
            data = None
        self.is_available.release()
        return data

    def set_data(self, location, data):
        
        self.is_available.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.is_available.release()

    def shutdown(self):
        
        self.thread.join()


class ScriptObject(object):
    

    def __init__(self, script, location, stop_execution):
        self.script = script
        self.location = location
        self.stop_execution = stop_execution

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None

        self.threads = []
        for _ in range(8):
            worker_thread = Thread(target=self.script_compute)
            self.threads.append(worker_thread)
            worker_thread.start()

    def script_compute(self):
        
        while True:
            script_object = self.device.script_queue.get()
            if script_object.stop_execution is True:
                self.device.script_queue.task_done()
                break

            script = script_object.script
            location = script_object.location

            if location not in self.device.location_lock:
                self.device.location_lock[location] = Lock()
            self.device.location_lock[location].acquire()

            
            script_data = []
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            
            data = self.device.get_data(location)


            if data is not None:
                script_data.append(data)

            
            if script_data != []:
                result = script.run(script_data)
                
                self.device.is_available.acquire()
                if location in self.device.sensor_data:
                    self.device.sensor_data[location] = result
                self.device.is_available.release()

                
                
                for device in self.neighbours:
                    device.set_data(location, result)

            if self.device.location_lock[location].locked():
                self.device.location_lock[location].release()
            self.device.script_queue.task_done()



    def run(self):
        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                for _ in range(8):
                    self.device.script_queue.put(ScriptObject(None, None, True))
                self.stop_all_threads()
                break

            self.device.timepoint_done.wait()

            for (script, location) in self.device.scripts:
                new_scriptobject = ScriptObject(script, location, False)
                self.device.script_queue.put(new_scriptobject)

            self.device.script_queue.join()
            
            self.device.shared_barrier.wait()
            self.device.timepoint_done.clear()

    def stop_all_threads(self):
        
        for thread in self.threads:
            thread.join()

        self.threads = []
