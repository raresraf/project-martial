

import Queue
from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        
        self.devices = []

        
        self.event_setup = Event()

        
        self.barrier_device = None

        
        self.locations_lock = []

        
        self.data_set_lock = Lock()

        self.thread = DeviceThread(self)
        self.thread.start()

        
        self.device_shutdown_order = False

        
        self.work_queue = Queue.Queue()

        
        self.worker_barrier = ReusableBarrierCond(8)

        
        self.data_semaphore = Semaphore(value=0)

        
        self.worker_semaphore = Semaphore(value=0)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier_device = ReusableBarrierCond(len(devices))

            for _ in range(25):
                self.locations_lock.append(Lock())

            for dev in devices:
                dev.devices = devices
                dev.barrier_device = self.barrier_device
                dev.locations_lock = self.locations_lock
                dev.event_setup.set()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        with self.data_set_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

    	
        self.device.event_setup.wait()

        
        list_threads = []
        for i in range(8):
            thrd = WorkerThread(self.device, self.device.locations_lock, self.device.work_queue, i)
            list_threads.append(thrd)

        for thrd in list_threads:
            thrd.start()

        
        script_number = 0

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                tup = (script, location, neighbours)

                
                self.device.work_queue.put(tup)

                
                self.device.data_semaphore.release()

                script_number += 1


            self.device.timepoint_done.clear()

            
            self.device.barrier_device.wait()

        
        for _ in xrange(script_number):


            self.device.worker_semaphore.acquire()

        
        self.device.device_shutdown_order = True

        
        for _ in xrange(8):
            self.device.data_semaphore.release()

        
        for thrd in list_threads:
            thrd.join()


class WorkerThread(Thread):
    

    def __init__(self, device, locations_lock, work_queue, worker_id):
        

        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.device = device
        self.locations_lock = locations_lock
        self.work_queue = work_queue
        self.worker_id = worker_id




    def run(self):

        while  True:
        	
            self.device.data_semaphore.acquire()

            
            if self.device.device_shutdown_order is True:
                break

            
            tup = self.work_queue.get()
            script = tup[0]
            location = tup[1]
            neighbours = tup[2]

            
            with self.locations_lock[location]:
                script_data = []
                
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
            
            self.device.worker_semaphore.release()

        
        self.device.worker_barrier.wait()
