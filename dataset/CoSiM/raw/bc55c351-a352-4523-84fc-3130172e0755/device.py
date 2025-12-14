



from threading import Event, Thread, Semaphore
from barrier import ReusableBarrierSem
from Queue import Queue
from copy import deepcopy
from time import sleep
from random import random

class Device(object):
    

    
    NR_THREADS = 8

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor


        self.devices = []
        self.script_received = Event()
        self.scripts = []
        self.script_queue = Queue()
        self.loc_dev_semaphore = Semaphore(value=1)
        self.loc_info = {}
        self.script_reset = False
        self.semaphore_devices = Semaphore()
        self.update_semaphore = Semaphore(value=1)
        self.current_script_state = {}
        self.has_neighbours = False
        self.neighbours_semaphore = Semaphore(value=1)
        self.neighbours = []
        self.barrier_devices = ReusableBarrierSem(0)
        self.barrier_threads = ReusableBarrierSem(Device.NR_THREADS)
        self.queue_semaphore = Semaphore(value=1)
        self.queue_init_semaphore = Semaphore(value=1)
        self.threads = []

        
        for count in range(0, Device.NR_THREADS):
            self.threads.append(DeviceThread(self))
            self.threads[count].start()


    def __str__(self):
        
        return "Device %d" % self.device_id


    def setup_devices(self, devices):
        
        
        same_barrier = ReusableBarrierSem(len(devices) * Device.NR_THREADS)
        
        semaphore_devices = Semaphore(value=1)

        self.devices = devices

        
        for device in devices:
            device.barrier_devices = same_barrier
            device.semaphore_devices = semaphore_devices


    def assign_script(self, script, location):
        
        
        if script is not None:
            self.scripts.append((script, location))

            
            for device in self.devices:
                device.add_location(location)
        else:
            
            self.script_received.set()


    def add_location(self, location):
        
        self.update_semaphore.acquire()

        
        if location in self.loc_info:
            self.loc_info[location] = False
        else:
            self.loc_info.update({location : False})

        self.update_semaphore.release()


    def check_location(self, location):
        
        self.semaphore_devices.acquire()

        res = False

        
        if self.current_script_state[location] == False:
            res = True
            
            for device in self.devices:
                device.current_script_state[location] = True

        self.semaphore_devices.release()

        return res


    def free_location(self, location):
        
        self.semaphore_devices.acquire()
        for device in self.devices:
            device.current_script_state[location] = False

        self.semaphore_devices.release()


    def get_data(self, location):
        
        result = None

        if location in self.sensor_data:
            result = self.sensor_data[location]

        return result


    def set_data(self, location, data):
        

        if location in self.sensor_data:
            self.sensor_data[location] = data


    def get_current_neighbours(self):
        
        self.neighbours_semaphore.acquire()
        if self.has_neighbours == False:
            self.neighbours = self.supervisor.get_neighbours()
            self.has_neighbours = True

        self.neighbours_semaphore.release()

        return self.neighbours


    def reset_neigbours(self):
        
        self.has_neighbours = False


    def init_queue(self):
        
        self.queue_init_semaphore.acquire()
        if self.script_queue.empty() == True:
            for (script, location) in self.scripts:
                self.script_queue.put((script, location))

        self.queue_init_semaphore.release()

    
    def again(self):
        
        self.script_reset = False

    
    def reset_script_state(self):
        
        self.loc_dev_semaphore.acquire()

        if self.script_reset == False:
            self.current_script_state = deepcopy(self.loc_info)
            self.script_reset = True

        self.loc_dev_semaphore.release()


    def shutdown(self):
        
        for count in range(0, Device.NR_THREADS):
            self.threads[count].join()






class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):

        while True:
            
            neighbours = self.device.get_current_neighbours()
            if neighbours is None:
                break

            
            self.device.script_received.wait()

            
            self.device.init_queue()

            
            self.device.barrier_devices.wait()

            
            self.device.reset_script_state()

            
            self.device.barrier_devices.wait()

            
            self.device.script_received.clear()

            while True:
                
                self.device.queue_semaphore.acquire()

                
                if self.device.script_queue.empty():
                    self.device.queue_semaphore.release()
                    break

                
                (script, location) = self.device.script_queue.get()

                
                if self.device.check_location(location) == False:
                    
                    last_script = self.device.script_queue.empty()

                    
                    self.device.script_queue.put((script, location))

                    self.device.queue_semaphore.release()

                    
                    if last_script:
                        
                        sleep(random() * 0.3)
                    continue

                self.device.queue_semaphore.release()

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

                
                self.device.free_location(location)

            
            self.device.barrier_devices.wait()

            
            self.device.reset_neigbours()

            
            self.device.again()

            
            self.device.barrier_threads.wait()

            
            self.device.barrier_devices.wait()


        