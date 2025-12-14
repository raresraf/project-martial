


from threading import Thread, Lock, Event

class MyThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self)
        self.device = device
        self.scripts_list = []
        self.neighbours_list = []
        self.permission = Event()
        self.finish = Event()
        self.thread_killed = 0
        self.lists = Lock()

    def run(self):
        while True:
            
            self.permission.wait()

            
            if self.thread_killed == 1:
                break

            
            self.permission.clear()
            self.finish.clear()

            
            while len(self.scripts_list) > 0 and len(self.neighbours_list) > 0:
                
                self.lists.acquire()
                script, place = self.scripts_list[0]
                neighbours = self.neighbours_list[0]

                del self.scripts_list[0]
                del self.neighbours_list[0]
                self.lists.release()

                
                self.device.scripts_locks[place].acquire()

                
                data_list = []
                data = self.device.get_data(place)
                if data is not None:
                    data_list.append(data)

                
                for neighbour in neighbours:
                    data = neighbour.get_data(place)
                    if data is not None:
                        data_list.append(data)

                if len(data_list) > 0:
                    
                    result = script.run(data_list)

                    
                    self.device.set_data(place, result)

                    
                    for neighbour in neighbours:
                        neighbour.set_data(place, result)

                
                self.device.scripts_locks[place].release()

            
            self.finish.set()


from threading import Event, Thread, Lock
from MyBarrier import MyBarrier
from MyThread import MyThread

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []

        
        self.ready = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

        
        self.threads = []
        self.nr_threads = 8 

        for _ in range(0, self.nr_threads):
            thread = MyThread(self)
            self.threads.append(thread)
            thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == 0:
            
			
            nr_devices = len(devices)
            barrier = MyBarrier(nr_devices)

            places = []
            locks = []
            for device in devices:
                places.extend(device.sensor_data.keys())
                data = len(device.sensor_data.keys())
                for _ in range(data):
                    locks.append(Lock())
                device.barrier = barrier
                device.scripts_locks = locks
                device.ready.set()

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
        

		
        for thread in self.threads:
            thread.join()
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        

        Thread.__init__(self)
        self.device = device

    def run(self):

        
        self.device.ready.wait()

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            
            if neighbours is None:
                for thread in self.device.threads:
                    thread.thread_killed = 1
                    thread.permission.set()
                break

            
            self.device.script_received.wait()

            
			
            scr = len(self.device.scripts)
            for i in range(0, scr):
                crt = i % self.device.nr_threads
                self.device.threads[crt].scripts_list.append(self.device.scripts[i])
                self.device.threads[crt].neighbours_list.append(neighbours)

            
            for thread in self.device.threads:
                thread.permission.set()

            
            self.device.script_received.clear()

            
            for thread in self.device.threads:
                thread.finish.wait()

            
            self.device.barrier.wait()
