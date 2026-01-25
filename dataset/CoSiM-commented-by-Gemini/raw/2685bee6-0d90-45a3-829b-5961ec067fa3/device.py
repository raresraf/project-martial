
from threading import Event, Thread, Lock, Condition
from Queue import Queue


class ReusableBarrierCond(object):
    

    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        
        
        self.cond = Condition()

    def wait(self):
        
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            
            self.cond.wait()
        
        self.cond.release()



class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor


        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.data_locks = {}
        for location in sensor_data:
            self.data_locks[location] = Lock()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            self.data_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_locks[location].release()

    def shutdown(self):
        
        self.thread.join()

class ThreadPool(object):
    def __init__(self, helper, no_threads):
        self.helper = helper
        self.queue = Queue(no_threads)
        for i in range(len(helper.scripts)):
            self.queue.put((helper.scripts[i][0], helper.scripts[i][1], helper.neighbours))
        self.threads = [Thread(target=self.run) for _ in range(no_threads)]
        for thread in self.threads:
            thread.start()

    def run(self):
        while True:
            script, location, neighbours = self.queue.get()

            if not neighbours or not script:
                self.queue.task_done()
                break

            script_data = []
            for device in neighbours:
                if device.device_id != self.helper.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            data = self.helper.device.get_data(location)
            if data is not None:
                script_data.append(data)
            if script_data != []:
                result = script.run(script_data)
                for device in neighbours:
                    if device.device_id != self.helper.device.device_id:
                        device.set_data(location, result)
                self.helper.device.set_data(location, result)

            self.queue.task_done()

    def join(self):
        self.queue.join()

    def close(self):
        self.join()

        for _ in self.threads:
            self.queue.put((None, None, None))

        for thread in self.threads:
            thread.join()


class Helper(object):
 
    def __init__(self, device):

        self.device = device
        self.neighbours = None
        self.scripts = None
        self.pool = None

    def set_neighbours_and_scripts(self, neighbours, scripts):

        self.scripts = scripts
        self.neighbours = neighbours
        if not self.pool:
            self.pool = ThreadPool(self, 8)
        else:
            for i in range(len(scripts)):
                self.pool.queue.put((scripts[i][0], scripts[i][1], neighbours))

    def close_pool(self):
        if self.pool:
            self.pool.close()


class DeviceThread(Thread):
    

    def __init__(self, device):
      
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.helper = None

    def run(self):



        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.helper = Helper(self.device)
            
            while True:
                if (self.device.script_received.is_set() or
                self.device.timepoint_done.is_set()):
                    
                    
                    
                    if self.device.script_received.is_set():
                        self.device.script_received.clear()
                        self.helper.set_neighbours_and_scripts(neighbours,
							self.device.scripts)
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break
            self.helper.close_pool()
            self.device.barrier.wait()
