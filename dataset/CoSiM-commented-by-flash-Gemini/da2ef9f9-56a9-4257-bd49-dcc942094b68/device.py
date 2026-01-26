


from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierCond
from Queue import Queue

THREAD_NR = 8

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.setup_finished = Event()
        self.dataLock = Lock()
        self.shared_lock = Lock()
        self.thread_queue = Queue(0)
        self.wait_get_neighbours = ReusableBarrierCond(THREAD_NR)
        self.thread_pool = []
        self.neighbours = []

        for i in range(0, THREAD_NR):
            thread = DeviceThread(self, i)
            self.thread_pool.append(thread)
            self.thread_queue.put(thread)
            thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.reusable_barrier = ReusableBarrierCond(len(devices) * THREAD_NR)
            self.location_locks = {}
            for device in devices:
                if device.device_id != self.device_id:
                    device.set_location_locks(self.location_locks)
                    device.set_barrier(self.reusable_barrier)

            self.setup_finished.set()

    def set_barrier(self, reusable_barrier):
        self.reusable_barrier = reusable_barrier
        self.setup_finished.set()

    def set_location_locks(self, location_locks):
        self.location_locks = location_locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            if location not in self.location_locks:
                self.location_locks[location] = Lock()

            thread = self.thread_queue.get()
            thread.give_script(script, location)

            
        else:
            for (s, l) in self.scripts:
                thread = self.thread_queue.get()
                thread.give_script(s, l)

            for thread in self.thread_pool:
                thread.give_script(None, None)


    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for i in range(THREAD_NR):
            self.thread_pool[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, ID):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id = ID
        self.script_queue = Queue(0)

    def give_script(self, script, location):
        self.script_queue.put((script, location))

    def run(self):

        while True:
            
            self.device.setup_finished.wait()

            
            if self.id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            
            self.device.wait_get_neighbours.wait()

            if self.device.neighbours is None:
                break

            while True:
                (script, location) = self.script_queue.get()

                if script is None:
                    break

                self.device.location_locks[location].acquire()

                script_data = []
                


                for device in self.device.neighbours:
                    device.dataLock.acquire()
                    data = device.get_data(location)
                    device.dataLock.release()

                    if data is not None:
                        script_data.append(data)

                
                self.device.dataLock.acquire()
                data = self.device.get_data(location)
                self.device.dataLock.release()
                
                if data is not None:
                   script_data.append(data)

                self.device.location_locks[location].release()

                if script_data != []:
                    
                    result = script.run(script_data)
                    
                    self.device.location_locks[location].acquire()

                    
                    for device in self.device.neighbours:
                        device.dataLock.acquire()
                        device.set_data(location, result)
                        device.dataLock.release()

                    


                    self.device.dataLock.acquire()
                    self.device.set_data(location, result)
                    self.device.dataLock.release()
                    self.device.location_locks[location].release()

               
                self.device.thread_queue.put(self)

            
            self.device.reusable_barrier.wait()