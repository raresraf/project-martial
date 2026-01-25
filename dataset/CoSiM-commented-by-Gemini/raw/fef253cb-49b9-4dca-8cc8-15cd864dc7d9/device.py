

import barrier
from threading import Event, Thread, Lock


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.script_taken = Event()
        self.assign_script_none = Event()
        self.script_de_orice_fel = Event()
        self.assign_script_not_none = Event()
        self.bariera = None
        self.bariera_join = None
        self.barrier_time = None
        self.flag_terminate = False
        self.script_sent = Lock()
        self.script_sent_thread = Lock()
        self.barrier_lock = Lock()
        self.counter = 0
        self.flag_received = Event()
        self.got_neighbours = Event()
        self.barrier_clear_events = None
        self.flag_script_received = False
        self.flag_script_taken = False
        self.flag_assign_script = 2
        self.flag_get_neigbours = False
        self.get_neighbours_lock = Lock()
        self.index_lock = Lock()
        self.i = 0
        self.scripts = []
        self.neighbours = None
        self.devices = []
        self.count_threads = []
        self.locations_locks = []
        self.timepoint_done = Event()
        self.initialize = Event()
        self.put_take_data = Lock()
        self.thread = DeviceThread(self)
        self.threads = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        self.devices = devices
        self.count_threads = [len(self.devices)]

        if self.device_id == 0:
            
            locations = []
            for device in self.devices:
                l = []
                for key in device.sensor_data.keys():
                    l.append(key)
                for locatie in l:
                    locations.append(locatie)
            maxim = max(locations)
            self.locations_locks = [None] * (maxim + 1)
            for locatie in locations:
                if self.locations_locks[locatie] is None:
                    lock = Lock()
                    self.locations_locks[locatie] = lock

            self.bariera = barrier.ReusableBarrierCond(len(self.devices))
            num_threads = len(self.devices) * 8
            self.bariera_join = barrier.ReusableBarrierCond(num_threads)
            self.barrier_time = barrier.ReusableBarrierCond(num_threads)
            self.barrier_clear_events = barrier.ReusableBarrierCond(num_threads)

            for device in self.devices:
                device.i = 0
                device.bariera = self.bariera
                device.counter = len(self.devices)
                device.barrier_time = self.barrier_time
                device.barrier_clear_events = self.barrier_clear_events
                device.locations_locks = self.locations_locks

        self.thread.start()
        i = 0
        while i < 7:
            dev = WorkerThread(self)
            dev.start()
            self.threads.append(dev)
            i = i + 1
        
        self.initialize.set()

    def assign_script(self, script, location):
        
        with self.script_sent:
            if script is not None:
                self.scripts.append((script, location))
            else:
                self.scripts.append((script, location))
                self.script_received.set()
                self.timepoint_done.set()

    def get_data(self, location):
        
        with self.put_take_data:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        with self.put_take_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        for thread in self.threads:
            thread.join()
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                self.device.flag_terminate = True
                self.device.got_neighbours.set()
                break
            
            self.device.got_neighbours.set()
            
            self.device.script_received.wait()
            while True:
                with self.device.index_lock:
                    if self.device.i >= len(self.device.scripts):
                        break
                    (script, location) = self.device.scripts[self.device.i % 8]
                    self.device.i = self.device.i + 1
                if script is not None:
                    lock = self.device.locations_locks[location]
                    with lock:
                        
                        script_data = []
                        
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data != []:
                            
                            result = script.run(script_data)

                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            self.device.set_data(location, result)
                if script is None:
                    break


            self.device.barrier_clear_events.wait()
            self.device.script_received.clear()
            self.device.got_neighbours.clear()
            self.device.barrier_time.wait()

class WorkerThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            self.device.got_neighbours.wait()
            if self.device.flag_terminate == True:
                break
            
            self.device.script_received.wait()
            while True:
                with self.device.index_lock:
                    if self.device.i >= len(self.device.scripts):
                        break
                    (script, location) = self.device.scripts[self.device.i % 8]
                    self.device.i = self.device.i + 1

                if script is not None:
                    lock = self.device.locations_locks[location]
                    with lock:
                        
                        script_data = []
                        
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data != []:
                            
                            result = script.run(script_data)

                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            self.device.set_data(location, result)
                if script is None:
                    break
            self.device.barrier_clear_events.wait()
            self.device.barrier_time.wait()
