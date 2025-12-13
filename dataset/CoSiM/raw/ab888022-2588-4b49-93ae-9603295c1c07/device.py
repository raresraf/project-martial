
from threading import *


MAX_THREADS = 8

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

dictionary = dict()

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
        self.locks = dict()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        if not dictionary.has_key(len(devices)):
            dictionary[len(devices)] = ReusableBarrier(len(devices))
        self.barrier = dictionary[len(devices)]

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
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def build_lists(self, lst, number):
        result_list = []        
        size = int(len(lst) / number)
        for i in range(number):
            row = size * i
            column = size * (i + 1)
            result_list.append(lst[row: column])
        for i in range(len(lst) - size * number):
            row = i % number
            column = i + (size * number)
            result_list[row].append(lst[column])
        return result_list

    class Instance(Thread):
        def __init__(self, device, big_list, neighbours):
            Thread.__init__(self, name="Instance")
            self.neighbours = neighbours            
            self.big_list = big_list
            self.device = device

        def run(self):
            script_data = []
            for (script, location) in self.big_list:
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                if script_data != []:
                    result = script.run(script_data)
                    self.all_devices_set_data(location, result)

        def my_max(self, a, b):
            if a >= b:
                return a
            return b


        def all_devices_set_data(self, location, result):
            for device in self.neighbours:
                with self.device.lock:
                    device.set_data(location, self.my_max(result, device.get_data(location)))
            with self.device.lock:
                result_data = self.my_max(result, self.device.get_data(location))
                self.device.set_data(location, result_data)


    def run(self):
        while True:
            threads_scripts = list()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            list_of_scripts = self.build_lists(self.device.scripts, MAX_THREADS)
            for i in range(MAX_THREADS):
                if list_of_scripts != []:
                    threads_scripts.append(self.Instance(self.device, list_of_scripts[i], neighbours))
            for script_thread in threads_scripts:
                script_thread.start()
            for script_thread in threads_scripts:
                script_thread.join()
            self.device.barrier.wait()