


from threading import *

dictionary = {}


class ReusableBarrier(object):
    

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

def pseudo_singleton(count):
    
    global dictionary
    if not dictionary.has_key(count):
        dictionary[count] = ReusableBarrier(count)
    return dictionary[count]

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.lock = Lock()
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self, 0)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.barrier = pseudo_singleton(len(devices))

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def listoflists(self, list, number):
        size = int(len(list) / number)
        chunks = []
        for i in xrange(number):
            chunks.append(list[0 + size * i: size * (i + 1)])
        for i in xrange(len(list) - size * number):
            chunks[i % number].append(list[(size * number) + i])
        return chunks

    def __init__(self, device, id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = id

    class Instance(Thread):
        
        def __init__(self, device, listfromlist, neighbours):
            Thread.__init__(self, name="Instance")
            self.device = device
            self.listfromlist = listfromlist
            self.neighbours = neighbours

        def set_data_for_all_devices(self, location, result):
            for device in self.neighbours:
                self.device.lock.acquire()
                device.set_data(location, max(result, device.get_data(location)))
                self.device.lock.release()
            self.device.lock.acquire()
            self.device.set_data(location, max(result, self.device.get_data(location)))


            self.device.lock.release()

        def run(self):
            script_data = []
            for (script, location) in self.listfromlist:
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                if script_data != []:
                    result = script.run(script_data)
                    self.set_data_for_all_devices(location, result)


    def run(self):
        while True:
            neighbours = self.device.supervisor.get_neighbours()


            if neighbours is None:
                break
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            list_of_scripts = self.listoflists(self.device.scripts, 8)
            instances = []
            for i in range(8):
                if len(list_of_scripts):
                    instances.append(self.Instance(self.device, list_of_scripts[i], neighbours))
            for index in range(len(instances)):
                instances[index].start()
            for index in range(len(instances)):
                instances[index].join()
            self.device.barrier.wait()
