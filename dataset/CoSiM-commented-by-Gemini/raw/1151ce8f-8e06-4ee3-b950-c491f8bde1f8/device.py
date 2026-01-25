


from threading import Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        


        self.counter_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase1()
        self.phase2()

    def phase1(self):
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()


                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()


def start_threads(threads):
    
    for thread in threads:
        thread.start()

def join_threads(threads):
    
    for thread in threads:
        thread.join()



def create_semaphores(devices):

    
    max_locations = 0

    for dev in devices:
        
        temp_max = max(dev.sensor_data, key=int)
        if max_locations < temp_max:
            max_locations = temp_max

    for dev in devices:
        for i in range(0, max_locations + 1):
            dev.sems_locations[i] = Semaphore(1)



class Device(Thread):
    
    def __init__(self, device_id, sensor_data, supervisor):

        

        Thread.__init__(self)
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        self.workers = []
        self.lock = None
        self.bariera = None
        self.bar_workers = None
        self.neighbours = None

        
        self.script_received = Semaphore(0)
        
        self.sems_locations = {}

        self.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        


        
        if self.device_id == 0:

            bariera = ReusableBarrierSem(len(devices))
            lock = Lock()

            for dev in devices:
                dev.bariera = bariera
                dev.lock = lock

            create_semaphores(devices)



    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            
            
            self.script_received.release() 

    def get_data(self, location):
        

        

        
        self.sems_locations[location].acquire()
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None


    def set_data(self, location, data):
        

        self.sensor_data[location] = data
        self.sems_locations[location].release()

    def shutdown(self):
        

        join_threads(self.workers)
        self.join()


    def run(self):

        while True:


            
            self.neighbours = self.supervisor.get_neighbours()
            if self.neighbours is None:
                break

            
            
            self.script_received.acquire()


            
            self.workers = []

            
            if self.scripts is not None:

                
                for (script, location) in self.scripts:
                    worker = Worker(self, self.neighbours, location, script)
                    self.workers.append(worker)

                
                start_threads(self.workers)
                
                join_threads(self.workers)

            
            self.bariera.wait()





class Worker(Thread):
    

    def __init__(self, parent, vecini, location, script):
        
        Thread.__init__(self)
        self.parent = parent
        self.worker_id = parent.device_id
        
        self.parent_neighbours = vecini

        
        self.location_for_script = location

        
        self.script = script

    def __str__(self):
        return "Worker " + str(self.worker_id)

    def run(self):

        script_data = []


        
        for neighbour in self.parent_neighbours:

            
            if neighbour.device_id != self.worker_id:
                data = neighbour.get_data(self.location_for_script)
                if data is not None:
                    script_data.append(data)

        
        data = self.parent.get_data(self.location_for_script)


        if data is not None:
            script_data.append(data)


        
        if script_data != []:
            

            result = self.script.run(script_data)


            
            for neighbour in self.parent_neighbours:
                
                if neighbour.device_id != self.worker_id:
                    neighbour.set_data(self.location_for_script, result)

            
            self.parent.set_data(self.location_for_script, result)

