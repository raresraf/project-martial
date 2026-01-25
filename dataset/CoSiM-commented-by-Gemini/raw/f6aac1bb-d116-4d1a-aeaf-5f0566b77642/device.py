


from threading import Event, Thread, Lock
from reusable_barrier_semaphore import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data


        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.location_locks = []
        self.next_timepoint_barrier = ReusableBarrier(0)
        self.thread = DeviceThread(self)

    def __str__(self):
        
        return "Device %d" % self.device_id

    @staticmethod
    def count_locations(devices, devices_nr):
        
        locations_number = 0

        for i in range(devices_nr):
            for location in devices[i].sensor_data.keys():
                if location > locations_number:
                    locations_number = location

        locations_number = locations_number + 1

        return locations_number

    def setup_devices(self, devices):
        
        devices_nr = len(devices)
        
        next_timepoint_barrier = ReusableBarrier(devices_nr)

        
        if self.device_id == 0:
            locations_number = self.count_locations(devices, devices_nr)

            
            for i in range(locations_number):
                lock = Lock()
                self.location_locks.append(lock)

            
            for i in range(devices_nr):


                for j in range(locations_number):
                    devices[i].location_locks.append(self.location_locks[j])

                devices[i].next_timepoint_barrier = next_timepoint_barrier
                devices[i].thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

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
        
        scriptsolvers = []

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break

            self.device.script_received.wait()
            self.device.script_received.clear()

            
            for (script, location) in self.device.scripts:
                scriptsolvers.append(
                    ScriptSolver(self.device, script, neighbours, location))

            workers_nr = len(scriptsolvers)

            for index in range(workers_nr):
                scriptsolvers[index].start()

            for index in range(workers_nr):


                scriptsolvers[index].join()

            
            scriptsolvers = []

            
            self.device.next_timepoint_barrier.wait()


class ScriptSolver(Thread):
    
    def __init__(self, device, script, neighbours, location):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location

    def collect_data(self, neighbours, location):
        
        data_script = []
        own_data = self.device.get_data(location)

        
        if own_data is not None:
            data_script.append(own_data)

        
        for device in neighbours:

            data = device.get_data(location)

            if data is not None:
                data_script.append(device.get_data(location))

        return data_script

    def update_data(self, neighbours, location, run_result):
        
     	
        self.device.set_data(location, run_result)

        
        for device in neighbours:
            device.set_data(location, run_result)

    def solve(self, script, neighbours, location):
        
        
        self.device.location_locks[location].acquire()

        data_script = self.collect_data(neighbours, location)

        
        if data_script != []:
            
            run_result = script.run(data_script)



            self.update_data(neighbours, location, run_result)

        
        self.device.location_locks[location].release()

    def run(self):
        self.solve(self.script, self.neighbours, self.location)
from threading import *

class ReusableBarrier():
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
                                                 
