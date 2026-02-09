/**
 * @file device.py
 * @brief Semantic documentation for device.py. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */



from threading import Thread, Event, Lock
from barrier import ReusableBarrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.locations = []
        self.locks = []


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            for device in devices:
                for data in device.sensor_data:
                    if data not in self.locations:
                        self.locations.append(data)
            self.locations.sort()

            for i in xrange(len(self.locations)):
                self.locks.append(Lock())

            self.barrier = ReusableBarrier(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    self.locations = device.locations
                    self.locks = device.locks
                    self.barrier = device.barrier


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] \
        if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = []
        self.neighbours = []

    def run(self):
        
        while True:
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                break

            self.device.timepoint_done.wait()
            num_scripts = len(self.device.scripts)
            self.scripts.extend(self.device.scripts)

            if num_scripts < 8:
                num_threads = num_scripts
            else:
                num_threads = 8

            threads = []
            for i in xrange(num_threads):


                thread = Thread(target=self.run_script)
                thread.start()
                threads.append(thread)

            for i in xrange(len(threads)):
                threads[i].join()

            self.device.timepoint_done.clear()
            self.device.barrier.wait()

    def run_script(self):
        
        while not self.scripts == []:
            (script, location) = self.scripts.pop()


            self.device.locks[location].acquire()
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
            self.device.locks[location].release()

