


import Queue
from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

class ScriptThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device

    def run(self):

        while True:

            
            (script, location) = self.device.scripts_queue.get()

            
            
            if (script, location) == (None, None):
                self.device.scripts_queue.put((None, None))
                break


            script_data = []

            
            with self.device.lcks[location]:
                
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
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

            
            self.device.scripts_queue.task_done()


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.start_scripts = Event()
        
        self.timepoint_done = ReusableBarrierCond(0)
        
        self.used_barrier = False
        
        self.neighbours = []

        
        self.scripts_queue = Queue.Queue()
        
        self.lcks = {}

        
        self.thread = DeviceThread(self)
        
        self.thread.start()

        
        self.thread_pool = []
        
        self.init_thread_pool(self.thread_pool)


    def init_thread_pool(self, pool):
        

        
        for i in xrange(8):
            thread = ScriptThread(self)
            pool.append(thread)

        
        for i in xrange(len(pool)):
            pool[i].start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        for i in self.sensor_data.keys():
            
            if self.lcks.has_key(i) is False:
                
                self.lcks[i] = Lock()
                
                for j in xrange(len(devices)):
                    if devices[j].device_id != self.device_id:
                        devices[j].lcks[i] = self.lcks[i]

        
        if self.used_barrier is False:
            
            self.timepoint_done.count_threads = len(devices)
            self.timepoint_done.num_threads = len(devices)
            
            for i in xrange(len(devices)):
                devices[i].used_barrier = True
                if devices[i].device_id != self.device_id:
                    devices[i].timepoint_done = self.timepoint_done


    def assign_script(self, script, location):
        

        if script is not None:
            
            self.scripts.append((script, location))
        else:
            
            self.script_received.set()

    def get_data(self, location):
        

        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        

        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        

        
        for i in xrange(len(self.thread_pool)):
            self.thread_pool[i].join()

        
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device

    def run(self):

        while True:

            
            self.device.neighbours = self.device.supervisor.get_neighbours()

            
            
            if self.device.neighbours is None:
                self.device.scripts_queue.put((None, None))
                break

            
            self.device.script_received.wait()


            self.device.script_received.clear()

            
            for (script, location) in self.device.scripts:
                self.device.scripts_queue.put((script, location))

            
            self.device.scripts_queue.join()
            
            if self.device.used_barrier is True:
                self.device.timepoint_done.wait()
