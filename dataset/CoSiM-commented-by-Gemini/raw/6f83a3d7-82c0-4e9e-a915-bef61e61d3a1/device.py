


from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

import Queue

class Device(object):
    

    def set_shared_barrier(self, shared_barrier):
        
        self.shared_barrier = shared_barrier

    def set_shared_location_locks(self, shared_location_locks):
        
        self.shared_location_locks = shared_location_locks

    def lock_location(self, location):
        
        if location not in self.shared_location_locks:
            self.shared_location_locks[location] = Lock()
        self.shared_location_locks[location].acquire()

    def release_location(self, location):
        
        self.shared_location_locks[location].release()

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.is_available = Lock()
        self.neighbours = []
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            shared_barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                device.set_shared_barrier(shared_barrier)

        if self.device_id == 0:
            shared_location_locks = {}
            for device in devices:
                device.set_shared_location_locks(shared_location_locks)
        

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

        self.script_received.set()

    def get_data(self, location):
        
        self.is_available.acquire()
        data = self.sensor_data[location] \
                if location in self.sensor_data else None
        self.is_available.release()
        return data

    def set_data(self, location, data):
        
        self.is_available.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.is_available.release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def do_in_parallel(self):
        
        while True:
            args = self.queue.get()
            script = args["script"]
            location = args["location"]

            if script is None:
                self.queue.task_done()
                break

            self.device.lock_location(location)

            
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

            self.device.release_location(location)

            self.queue.task_done()

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue.Queue()
        self.threads = []
        self.neighbours = None
        for _ in range(8):
            generated_thread = Thread(target=self.do_in_parallel)
            generated_thread.daemon = True
            self.threads.append(generated_thread)
            generated_thread.start()



    def run(self):
        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                for _ in range(8):
                    self.queue.put({"script":None, "location":None})
                for thread in self.threads:
                    thread.join()
                self.threads = []
                break

            self.device.timepoint_done.wait()

            for (script, location) in self.device.scripts:
                self.queue.put({"script":script, "location": location})

            self.queue.join()

            self.device.shared_barrier.wait()

            self.device.timepoint_done.clear()
