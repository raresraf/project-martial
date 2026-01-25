


from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.queue = Queue()
        self.worker_threads = []
        self.worker_threads_no = 8
        self.timepoint_done = Event()

        for _ in range(0, self.worker_threads_no):
            worker = WorkerThread(self, self.queue)
            worker.start()
            self.worker_threads.append(worker)

        if device_id == 0:
            devices_no = len(supervisor.supervisor.testcase.devices)
            self.barrier = ReusableBarrierCond(devices_no)
            self.dict_location_lock = {}
        else:
            
            self.barrier = None
            self.dict_location_lock = None

        self.all_devs = None

        self.master_thread = DeviceThread(self)
        self.master_thread.start()


    def __str__(self):
        
        return "Device %d" % self.device_id


    def setup_devices(self, devices):
        
        if self.device_id != 0:
            for dev in devices:
                if dev.device_id == 0:
                    self.barrier = dev.barrier
                    self.dict_location_lock = dev.dict_location_lock
                    break

        self.all_devs = devices


    def assign_script(self, script, location):
        
        if script is not None:
            if location not in self.dict_location_lock.keys():


                self.dict_location_lock[location] = Lock()
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()


    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]

        return None


    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data


    def shutdown(self):
        
        for i in range(0, self.worker_threads_no):
            self.worker_threads[i].join()

        self.master_thread.join()


class WorkerThread(Thread):
    

    def __init__(self, device, queue):
        

        Thread.__init__(self)
        self.device = device
        self.queue = queue


    def run(self):

        while True:

            (scr_loc, neighbours) = self.queue.get()
            if neighbours is None:
                return

            (script, location) = scr_loc
            script_data = []

            with self.device.dict_location_lock[location]:
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

            self.queue.task_done()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):



        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                
                for _ in range(0, self.device.worker_threads_no):
                    self.device.queue.put((None, None))
                break

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            for src_loc in self.device.scripts:
                self.device.queue.put((src_loc, neighbours))

            
            self.device.queue.join()
            
            self.device.barrier.wait()

