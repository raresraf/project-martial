/**
 * @file device.py
 * @brief Semantic documentation for device.py. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */


from Queue import Queue
from threading import Thread, Condition, Event

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

class MyThread(object):
    


    def __init__(self, threads_count):
        

        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None
        self.create(threads_count)
        self.start()

    def create(self, threads_count):
        

        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

    def start(self):
        

        for thread in self.threads:
            thread.start()

    def set_device(self, device):
        
        self.device = device

    def execute(self):
        

        while True:

            neighbours, script, location = self.queue.get()

            if neighbours is None and script is None:
                self.queue.task_done()
                return

            self.run_script(neighbours, script, location)
            self.queue.task_done()

    def run_script(self, neighbours, script, location):
        

        
        for (script, location) in self.device.scripts:
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

    def put(self, neighbours, script, location):
        

        self.queue.put((neighbours, script, location))

    def wait_threads(self):
        

        self.queue.join()

    def end_threads(self):
        

        self.wait_threads()

        for _ in xrange(len(self.threads)):
            self.put(None, None, None)

        for thread in self.threads:
            thread.join()

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = False
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.location_cond = {location: Condition() for location in sensor_data}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            self.send_barrier(devices, self.barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        

        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        
        self.barrier = barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            self.location_cond[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_cond[location].release()

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = MyThread(8)

    def run(self):

        self.threads.set_device(self.device)

        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            while True:

                
                if self.device.script_received or self.device.timepoint_done.wait():
                    if self.device.script_received:
                        self.device.script_received = False
                        
                        for (script, location) in self.device.scripts:
                            self.threads.put(neighbours, script, location)
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_received = True
                        break

            
            self.threads.wait_threads()

            
            self.device.barrier.wait()

        
        self.threads.end_threads()
