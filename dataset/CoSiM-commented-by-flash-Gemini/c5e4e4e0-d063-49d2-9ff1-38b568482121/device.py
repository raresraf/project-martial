

import Queue
from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    

    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        
        self.phase1()
        self.phase2()

    def phase1(self):
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):


                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class MyThread(Thread):
    

    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier

    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.barrier = None
        self.timepoint_done = Event()
        self.lock_data = Lock()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.all_devices = None
        self.locations = []
        self.lock_locations = []


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        for location in range(len(self.sensor_data)):
            if self.sensor_data.get(location) is not None:
                if location not in self.locations:
                    self.locations.append(location)


        self.all_devices = devices

        for device in self.all_devices:
            for location in device.locations:
                if location not in self.locations:
                    self.locations.append(location)



        self.locations.sort()

        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))

            
            for _ in self.locations:
                lock = Lock()
                self.lock_locations.append(lock)

            
            for device in self.all_devices:
                device.set_barrier(self.barrier)
                device.set_lock_locations(self.lock_locations)


    def assign_script(self, script, location):
        
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_barrier(self, barrier):
        
        self.barrier = barrier

    def set_lock_locations(self, lock_locations):
        
        self.lock_locations = lock_locations

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
            self.device.lockForLocations = []
            if neighbours is None:
                break
            self.device.barrier.wait()

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            queue_list = []
            index_list = []

            for item in self.device.scripts:

                (_, location) = item

                if location not in index_list:
                    index_list.append(location)
                    temp_queue = Queue.Queue()
                    temp_queue.put(item)
                    queue_list.append(temp_queue)
                else:
                    index = index_list.index(location)
                    queue_list[index].put(item)


            
            th_list = []

            
            for queue in queue_list:
                worker = Thread(target=split_work, args=(self.device, neighbours, queue, ))
                worker.setDaemon(True)
                th_list.append(worker)
                worker.start()

            for thr in th_list:
                thr.join()


def split_work(device, neighbours, queue_param):

    
    while True:
        try:
            
            (script, location) = queue_param.get(False)
        except Queue.Empty:

            break
        else:
            if location in device.locations:
                device.lock_locations[location].acquire()
                script_data = []
                

                for device_temp in neighbours:
                    data = device_temp.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = device.get_data(location)

                if data is not None:
                    script_data.append(data)
                if script_data != []:
                    

                    result = script.run(script_data)
                    
                    for device_temp in neighbours:
                        device_temp.set_data(location, result)
                    
                    device.set_data(location, result)
                queue_param.task_done()
                device.lock_locations[location].release()
