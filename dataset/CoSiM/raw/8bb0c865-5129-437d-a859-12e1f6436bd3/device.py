


from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        
        self.timepoint_done = Event()
        
        self.barrier_set = Event()
        
        self.script_dict = {}
        
        self.location_lock_dict = {}
        self.barrier = None

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def set_synchronization(self, barrier, location_lock_dict):
        
        self.barrier = barrier
        self.location_lock_dict = location_lock_dict
        self.barrier_set.set()


    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            location_lock_dict = {}
            
            for device in devices:
                for location in device.sensor_data.keys():
                    if location_lock_dict.has_key(location) == False:
                        location_lock_dict[location] = Lock()


            for device in devices:
                device.set_synchronization(barrier, location_lock_dict)


    def assign_script(self, script, location):
        
        if script is not None:
            
            if self.script_dict.has_key(location) == False:
                self.script_dict[location] = []
            self.script_dict[location].append(script)
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

    def run(self):
        
        self.device.barrier_set.wait()


        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            nr_locations = len(self.device.script_dict)
            nr_threads = min(nr_locations, 8) 

            
            if nr_locations != 0:
                
                threads = []
                for i in xrange(nr_threads - 1):
                    threads.append(DeviceThreadHelper(self.device, i + 1,
                    	nr_locations, nr_threads, neighbours))
                for thread in threads:
                    thread.start()

                
                locations_list = self.device.script_dict.items()
                my_list = locations_list[0: nr_locations : nr_threads]


                for (location, script_list) in my_list:



                    for script in script_list:
                        script_data = []

                        
                        self.device.location_lock_dict[location].acquire()
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

                        self.device.location_lock_dict[location].release()


                for thread in threads:
                    thread.join()

            
            self.device.barrier.wait()
            self.device.timepoint_done.clear()



class DeviceThreadHelper(Thread):
    

    def __init__(self, device, helper_id, num_locations, pace, neighbours):
        Thread.__init__(self)
        self.device = device
        self.my_id = helper_id
        self.num_locations = num_locations
        self.pace = pace
        self.neighbours = neighbours

    def run(self):
    	
        locations_list = self.device.script_dict.items()
        my_list = locations_list[self.my_id: self.num_locations : self.pace]

        
        for (location, script_list) in my_list:

            for script in script_list:
                script_data = []

                self.device.location_lock_dict[location].acquire()
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

                self.device.location_lock_dict[location].release()
