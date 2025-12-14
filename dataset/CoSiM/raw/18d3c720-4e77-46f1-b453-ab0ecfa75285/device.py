


from threading import Event, Thread, Lock
from reference import CommonReference
import multiprocessing

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.number_of_processors = multiprocessing.cpu_count()

        
        self.wait_for_reference = Event()

        
        
        self.synch_reference = None

        
        self.thread_list = []

        
        
        self.location_locks = {}
        for entry in self.sensor_data:
            self.location_locks[entry] = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        self.devices = devices

        
        
        if self.device_id == 0:
            self.synch_reference = CommonReference(len(self.devices))
            for dev in self.devices:
                if dev.device_id != 0:
                    dev.synch_reference = self.synch_reference

            
            

            for dev in self.devices:
                dev.wait_for_reference.set()

        else:
            
            
            self.wait_for_reference.wait()

    def assign_script(self, script, location):
        

        if script is not None:
            
            self.scripts.append((script, location))

        else:
            
            self.script_received.set()

    def get_data(self, location):
        

        
        if location in self.sensor_data:
            self.location_locks[location].acquire()
        return self.sensor_data[location] if location in \
               self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

            
            self.location_locks[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def simple_task(self, neighbours, script, location):
        
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


    def run_tasks(self, neighbours, list_of_tuples):
        

        for (script, location) in list_of_tuples:
            self.simple_task(neighbours, script, location)


    def run(self):
        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            
            self.device.script_received.wait()
            self.device.script_received.clear()

            
            
            self.device.synch_reference.first_barrier.wait()

            
            if self.device in neighbours:
                neighbours.remove(self.device)

            
            self.list_of_thread_lists = []
            for i in range(self.device.number_of_processors):
                self.list_of_thread_lists.append([])

            
            
            load_factor = 2
            if len(self.device.scripts) <= (load_factor * self.device.number_of_processors):
                for (script, location) in self.device.scripts:
                    self.device.thread_list.append(Thread(target=self.simple_task, args=(neighbours, script, location)))
                    self.device.thread_list[-1].start()

                for i in range(len(self.device.thread_list)):
                    self.device.thread_list[i].join()

                del self.list_of_thread_lists[:]
            else:
                i = 0
                for (script, location) in self.device.scripts:
                    self.list_of_thread_lists[i % self.device.number_of_processors].append((script, location))
                    i += 1

                for i in range(self.device.number_of_processors):
                    if len(self.list_of_thread_lists[i]) is not 0:
                        self.device.thread_list.append(Thread(target=self.run_tasks, args=(neighbours, self.list_of_thread_lists[i])))
                        self.device.thread_list[-1].start()

                for i in range(len(self.device.thread_list)):
                    self.device.thread_list[i].join()

                del self.list_of_thread_lists[:]

            
            self.device.synch_reference.second_barrier.wait()



from barrier import SimpleBarrier
from threading import Lock

class CommonReference(object):
    
    def __init__(self, number_of_devices):
        

        self.lock = Lock()
        self.first_barrier = SimpleBarrier(number_of_devices)
        self.second_barrier = SimpleBarrier(number_of_devices)
