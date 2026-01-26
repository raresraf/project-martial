


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem as Barrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        

        
        
        self.timepoint_done = None

        
        
        self.lock = None

        
        
        self.todo_scripts = []

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        
        
        
        

        if self.timepoint_done is None:
            self.timepoint_done = Barrier(len(devices))
            for device in devices:
                if device.timepoint_done is None:
                    device.timepoint_done = self.timepoint_done

        if self.lock is None:
            self.lock = {}
            for device in devices:
                if device.lock is None:
                    device.lock = self.lock

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
            
            
            
            if location not in self.lock:
                self.lock[location] = Lock()
        else:
            self.script_received.set()

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

            self.device.script_received.wait()

            
            for script in self.device.scripts:
                self.device.todo_scripts.append(script)

            
            nr_subthreads = min(8, len(self.device.scripts))
            
            subthreads = []
            
            
            scripts_lock = Lock()

            


            while len(subthreads) < nr_subthreads:
                
                subthread = ScriptThread(scripts_lock, self, neighbours)
                subthreads.append(subthread)

            
            for subthread in subthreads:
                subthread.start()

            
            for subthread in subthreads:
                subthread.join()

            
            self.device.script_received.clear()

            
            self.device.timepoint_done.wait()

class ScriptThread(Thread):
    

    def __init__(self, scripts_lock, parent, neighbours):
        
        Thread.__init__(self)
        self.scripts_lock = scripts_lock
        self.parent = parent
        self.neighbours = neighbours

    def run(self):
        

        
        
        
        self.scripts_lock.acquire()
        length = len(self.parent.device.todo_scripts)
        if length > 0:
            current_script = self.parent.device.todo_scripts.pop()
        else:
            current_script = None
        self.scripts_lock.release()

        while current_script is not None and self.neighbours is not None:
            
            (script, location) = current_script
            
            script_data = []

            
            
            
            self.parent.device.lock[location].acquire()

            
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.parent.device.get_data(location)
            if data is not None:
                script_data.append(data)



            if script_data != []:
                
                result = script.run(script_data)

                
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.parent.device.set_data(location, result)

            self.parent.device.lock[location].release()

            
            
            
            self.scripts_lock.acquire()
            length = len(self.parent.device.todo_scripts)
            if length > 0:
                current_script = self.parent.device.todo_scripts.pop()
            else:
                current_script = None
            self.scripts_lock.release()
