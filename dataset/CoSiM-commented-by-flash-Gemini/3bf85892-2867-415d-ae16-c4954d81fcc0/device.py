


from threading import Event, Thread, Lock
from Queue import Queue
from reusable_barrier_condition import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.location_locks = {}
        self.barrier = None
        self.num_threads = 8
        self.queue = Queue(self.num_threads)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        if self.barrier is None:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
            for device in devices:
                device.location_locks = self.location_locks



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

class WorkerThread(Thread):
    

    def __init__(self, queue, device):
        Thread.__init__(self)
        self.queue = queue
        self.device = device

    def run(self):

        
        
        
        
        while True:
            data_tuple = self.queue.get()

            if data_tuple == (None, None, None):
                break

        
            self.device.location_locks[data_tuple[1]].acquire()
            script_data = []
            


            for device in data_tuple[2]:
                data = device.get_data(data_tuple[1])
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(data_tuple[1])
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = data_tuple[0].run(script_data)

                
                for device in data_tuple[2]:
                    device.set_data(data_tuple[1], result)
                
                self.device.set_data(data_tuple[1], result)
            self.device.location_locks[data_tuple[1]].release()



class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        threads = []

        for i in range(self.device.num_threads):
            thread = WorkerThread(self.device.queue, self.device)
            threads.append(thread)
            threads[i].start()

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()



            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location, neighbours))



            
            self.device.barrier.wait()
            self.device.timepoint_done.clear()

        

        for i in range(self.device.num_threads):
            self.device.queue.put((None, None, None))

        for i in range(self.device.num_threads):
            threads[i].join()


from threading import Condition

class ReusableBarrier(object):
    
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
