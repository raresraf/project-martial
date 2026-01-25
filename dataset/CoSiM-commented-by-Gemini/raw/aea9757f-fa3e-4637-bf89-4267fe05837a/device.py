


from threading import Lock, Semaphore, Event, Thread
from sets import Set

class Barrier(object):
    
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    



    barrier = None
    lock_list = []
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()
        self.devices = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.devices = devices
        if Device.barrier is None:
            Device.barrier = Barrier(len(devices))
        if Device.lock_list == []:
            zones = []
            for dev in devices:
                zones.extend(dev.sensor_data.keys())

            Device.lock_list = [Lock() for i in range(len(Set(zones)))]

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

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
            if neighbours is None:
                break

            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            tasks = [[] for i in xrange(8)]
            i = 0
            for (script, location) in self.device.scripts:
                tasks[i%8].append((script, location))
                i += 1

            script_threads = []
            for i in xrange(8):
                if tasks[i%8] != []:
                    thr = ScriptThread(self.device, neighbours, tasks[i%8])
                    script_threads.append(thr)
                    thr.start()

            for i in xrange(len(script_threads)):
                script_threads[i].join()

            Device.barrier.wait()

class ScriptThread(Thread):
    
    def __init__(self, device, neighbours, scripts):
        
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.scripts = scripts

    def run(self):
        
        for (script, location) in self.scripts:
            Device.lock_list[location].acquire()
            script_data = []
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)


                for device in self.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            Device.lock_list[location].release()
