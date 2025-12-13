
from threading import Thread

class CoreThread(Thread):
    
    def __init__(self, device, script_id, neighbours):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script_id = script_id
        self.neighbours = neighbours

    def run(self):
        
        (script, location) = self.device.scripts[self.script_id]

        self.device.semaphores_list[location].acquire()

        script_data = []

        self.device.lock1.acquire()

        for device in self.neighbours:
            data = device.get_data(location)

            if data is not None:
                script_data.append(data)

        data = self.device.get_data(location)

        self.device.lock1.release()

        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = script.run(script_data)

            for device in self.neighbours:

                self.device.lock2.acquire()

                device.set_data(location, result)
                self.device.set_data(location, result)

                self.device.lock2.release()

        self.device.semaphores_list[location].release()


from threading import Thread, Event, Semaphore, Lock
from core_thread import CoreThread
from barrier import ReusableBarrierCond


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.semaphores_list = []
        self.semaphore_setup_devices = Semaphore(1)
        self.lock1 = Lock()
        self.lock2 = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        barrier = ReusableBarrierCond(len(devices))

        for _ in range(50):
            semaphore = Semaphore(1)
            self.semaphores_list.append(semaphore)

        if self.device_id == 0:
            for device in devices:
                device.semaphores_list = self.semaphores_list
                device.barrier = barrier
                device.semaphore_setup_devices.acquire()
        self.thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

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

    def run(self):
        

        self.device.semaphore_setup_devices.release()

        while True:

            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break

            self.device.script_received.wait()
            self.device.script_received.clear()

            cores = []
            script_id = 0
            scripts_number = len(self.device.scripts)

            for i in xrange(scripts_number):

                core_thread = CoreThread(self.device, script_id, neighbours)
                cores.append(core_thread)
                script_id = script_id + 1

            begin = 0

            if scripts_number > 8:

                while scripts_number >= 0:

                    if scripts_number >= 8:

                        index = begin
                        i = 0

                        while i < 8:
                            cores[index + i].start()

                        index = begin
                        i = 0

                        while i < 8:
                            cores[index + i].join()

                        scripts_number = scripts_number - 8

                        begin = begin + 8

                    else:

                        index = begin
                        i = 0

                        while i < scripts_number:
                            cores[index + i].start()

                        index = begin
                        i = 0

                        while i < scripts_number:
                            cores[index + i].join()

                        break
            else:

                for index in xrange(scripts_number):
                    cores[index].start()

                for index in xrange(scripts_number):
                    cores[index].join()

            self.device.barrier.wait()
