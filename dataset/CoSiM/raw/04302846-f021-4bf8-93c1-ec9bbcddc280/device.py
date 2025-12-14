




from threading import Thread, Semaphore, Lock, Event


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
                nr_threads = self.num_threads
                while nr_threads > 0:
                    threads_sem.release()
                    nr_threads -= 1
                    
                count_threads[0] = self.num_threads  
        threads_sem.acquire()   
                                
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

        self.locks = None
        self.barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        
        if all(element is None for element in [self.barrier, self.locks]):
            barrier = ReusableBarrier(len(devices))
            locks = []
            
            max_locations = 0
            for device in devices:
                for location in device.sensor_data.keys():
                    if location > max_locations:
                        max_locations = location

            for location in range(max_locations + 1):
                locks.append(Lock())

            
            
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))

        else:
            self.script_received.set()
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
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            number_of_scripts = len(self.device.scripts)

            
            if number_of_scripts != 0:

            	
                if number_of_scripts < 8:
                    number_of_threads = number_of_scripts
                else:
                    number_of_threads = 8

                workers_list = []

                
                
                for i in range(number_of_threads):
                    worker = Worker(self.device, neighbours)
                    workers_list.append(worker)

                current_thread = 0
                average_scripts = 0

                
                
                

                if number_of_threads > 0:
                    average_scripts = len(self.device.scripts) / number_of_threads
                aux_average = average_scripts

                
                
                
                for(script, location) in self.device.scripts:
                    if aux_average > 0:
                        workers_list[current_thread].scripts.append((script, location))
                        aux_average -= 1
                    if aux_average == 0:
                        aux_average = average_scripts

                        
                        
                        if current_thread < number_of_threads - 1:
                            current_thread += 1
                        else:
                            current_thread = 0
                
                for i in range(number_of_threads):
                    workers_list[i].start()

                
                for i in range(number_of_threads):
                    workers_list[i].join()

            self.device.timepoint_done.clear()
            
            self.device.barrier.wait()

class Worker(Thread):
    
    def __init__(self, device, neighbours):
        Thread.__init__(self)
        self.device = device
        self.scripts = []


        self.neighbours = neighbours

    def run(self):
    
        for (script, location) in self.scripts:
            script_data = []

            
            self.device.locks[location].acquire()

            
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
