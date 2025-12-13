


from Queue import Queue
from threading import Thread

class MyThread(object):
    

    def __init__(self, threads_count):
        

        self.queue = Queue(threads_count)

        self.threads = []
        self.device = None

        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

        for thread in self.threads:
            thread.start()

    def execute(self):
        

        while True:

            neighbours, script, location = self.queue.get()

            if neighbours is None:
                if script is None:
                    self.queue.task_done()
                    return



            self.run_script(neighbours, script, location)
            self.queue.task_done()

    def run_script(self, neighbours, script, location):
        

        script_data = []

        
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is None:
                    continue

                script_data.append(data)

        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            
            result = script.run(script_data)

            
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    continue

                device.set_data(location, result)

            


            self.device.set_data(location, result)

    def end_threads(self):
        

        self.queue.join()

        for _ in xrange(len(self.threads)):
            self.queue.put((None, None, None))

        for thread in self.threads:
            thread.join()




from threading import Event, Thread, Lock

from barrier import Barrier
from MyThread import MyThread


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

        self.new_adds()

        self.thread.start()

    def new_adds(self):
        

        self.barrier = None
        self.locations = {}
        for location in self.sensor_data:
            self.locations[location] = Lock()
        self.script_arrived = False

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            for dev in devices:
                if dev.device_id == 0:
                    continue

                dev.barrier = self.barrier

    def assign_script(self, script, location):
        

        self.set_boolean(script)
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def set_boolean(self, script):
        

        if script is not None:
            self.script_arrived = True


    def acquire_location(self, location):
        
        if location in self.sensor_data:
            self.locations[location].acquire()

    def get_data(self, location):
        
        self.acquire_location(location)
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        self.thread_pool = MyThread(8)

    def run(self):

        self.thread_pool.device = self.device

        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            while True:

                
                if self.device.script_arrived or self.device.timepoint_done.wait():
                    if self.device.script_arrived:
                        self.device.script_arrived = False

                        
                        for (script, location) in self.device.scripts:
                            self.thread_pool.queue.put((neighbours, script, location))
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_arrived = True
                        break

            
            self.thread_pool.queue.join()

            
            self.device.barrier.wait()

        
        self.thread_pool.end_threads()
