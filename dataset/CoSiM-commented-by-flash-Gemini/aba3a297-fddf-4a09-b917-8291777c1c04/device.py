


from threading import Event, Thread, Lock

from barrier import ReusableBarrierCond
from threadpool import ThreadPool

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        
        
        self.locations_locks = []

        for location in sensor_data:
            self.locations_locks.append((location, Lock()))

        self.locations_locks = dict(self.locations_locks)

        self.barrier = None

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
            self.locations_locks[location].acquire()
            return self.sensor_data[location]

        return None


    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations_locks[location].release()


    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(8, device)



    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            while True:

                
                if self.device.timepoint_done.wait() and not self.device.script_received.is_set():
                    self.device.timepoint_done.clear()
                    self.device.script_received.set()
                    break

                


                if self.device.script_received.is_set():
                    self.device.script_received.clear()

                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task(script, location, neighbours)


            
            self.thread_pool.tasks_queue.join()

            
            self.device.barrier.wait()

        
        self.thread_pool.join_threads()


from threading import Thread
from Queue import Queue

class ThreadPool(object):
    

    def __init__(self, number_threads, device):
        
        self.number_threads = number_threads
        self.device_threads = []
        self.device = device
        self.tasks_queue = Queue(number_threads)

        for _ in xrange(0, number_threads):


            thread = Thread(target=self.apply_scripts)
            self.device_threads.append(thread)

        for thread in self.device_threads:
            thread.start()

    def apply_scripts(self):
        
        while True:

            script, location, neighbours = self.tasks_queue.get()

            
            if neighbours is None and script is None:
                self.tasks_queue.task_done()
                return

            script_data = []
            
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                for device in neighbours:


                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                self.device.set_data(location, result)


            self.tasks_queue.task_done()


    def submit_task(self, script, location, neighbours):
        
        self.tasks_queue.put((script, location, neighbours))


    def join_threads(self):
        
        self.tasks_queue.join()

        for _ in xrange(0, len(self.device_threads)):
            self.submit_task(None, None, None)

        for thread in self.device_threads:
            thread.join()
