


from threading import Event, Thread, Lock
from Queue import Queue, Empty
from barrier import Barrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        
        self.num_threads = 8
        
        self.scripts = []
        
        self.jobs_queue = Queue()
        
        self.neighbours = []

        
        self.scripts_received = Event()
        self.scripts_received_barrier = Barrier(self.num_threads)
        self.scripts_processed_barrier = Barrier(self.num_threads)
        self.neighbours_received_barrier = Barrier(self.num_threads)

        
        self.location_locks = {}
        self.timepoint_barrier = None

        
        self.threads = [DeviceThread(self, i) for i in xrange(self.num_threads)]


        for i in xrange(self.num_threads):
            self.threads[i].start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        leader = min([device.device_id for device in devices])

        
        
        if self.device_id == leader:
            
            locations_set = set()
            for device in devices:
                locations_set.update(device.sensor_data.keys())
            locations = list(locations_set)
            self.location_locks = {location : Lock() for location in locations}

            
            
            self.timepoint_barrier = Barrier(len(devices))

            
            for device in devices:
                device.location_locks = self.location_locks
                device.timepoint_barrier = self.timepoint_barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.jobs_queue.put((script, location))
        else:
            self.scripts_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for i in xrange(self.num_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, id_thread):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id_thread = id_thread

    def run(self):
        

        leader = 0
        while True:
            
            if self.id_thread == leader:


                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            self.device.neighbours_received_barrier.wait()

            if self.device.neighbours is None:
                break

            
            self.device.scripts_received.wait()
            self.device.scripts_received_barrier.wait()
            self.device.scripts_received.clear()

            
            while True:
                try:
                    (script, location) = self.device.jobs_queue.get_nowait()
                except Empty:
                    break

                self.run_script(script, location)
            
            self.device.scripts_processed_barrier.wait()

            
            
            if self.id_thread == leader:
                for script in self.device.scripts:
                    self.device.jobs_queue.put(script)
                
                self.device.timepoint_barrier.wait()

    def run_script(self, script, location):
        

        
        with self.device.location_locks[location]:
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
