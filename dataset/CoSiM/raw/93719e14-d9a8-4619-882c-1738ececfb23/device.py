


from threading import Event, Thread, Semaphore
from reusable_barrier import TimePointsBarrier, ClassicBarrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.threads = []
        self.neighbours = []
        self.num_threads = 8

        self.locations_semaphore = None
        self.devices_barrier = None
        self.neighbours_barrier = None
        self.all_devices = None

    def set_neighbours(self, new_neighbours):
        
        self.neighbours = new_neighbours


    def set_devices_barrier(self, barrier):
        
        self.devices_barrier = barrier


    def set_locations_semaphore(self, locations_semaphore):
        
        self.locations_semaphore = locations_semaphore

    def get_locations(self, location_list):
        
        for location in self.sensor_data:
            location_list.append(location)


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        

        if self.device_id == 0:
            barrier = ClassicBarrier(len(devices))
            locations = []
            locations_semaphore = []
            for device in devices:
                device.get_locations(locations)

            locations = sorted(list(set(locations)))

            for i in range(0, len(locations)):
                locations_semaphore.append(Semaphore(value=1))

            for device in devices:
                device.set_devices_barrier(barrier)
                device.set_locations_semaphore(locations_semaphore)

        self.all_devices = devices
        self.neighbours_barrier = TimePointsBarrier(self.num_threads, self)


        for i in range(self.num_threads):
            current_thread = DeviceThread(self, i)
            current_thread.start()
            self.threads.append(current_thread)

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def has_data(self, location):
        
        if location in self.sensor_data:
            return True
        return False
    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for i in range(self.num_threads):


            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        
        
        while True:
            

            self.device.neighbours_barrier.wait()

            if self.device.neighbours is None:
                break

            self.device.timepoint_done.wait()

            if len(self.device.neighbours) != 0:
                devices_with_date = []
                
                for index in range(
                        self.thread_id,
                        len(self.device.scripts),
                        self.device.num_threads):
                    (script, location) = self.device.scripts[index]
                    
                    
                    script_data = []
                    self.device.locations_semaphore[location].acquire()

                    for device in self.device.neighbours:
                        if device.has_data(location):
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                                devices_with_date.append(device)

                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                        devices_with_date.append(self.device)

                    if script_data != []:
                        
                        result = script.run(script_data)
                        for device in devices_with_date:
                            device.set_data(location, result)
                        devices_with_date = []

                    self.device.locations_semaphore[location].release()


