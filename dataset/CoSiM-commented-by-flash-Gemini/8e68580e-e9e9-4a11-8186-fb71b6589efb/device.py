


from threading import Event, Thread, Lock
import reusable_barrier_semaphore


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        
        self.number_threads_per_device = 8
        
        
        self.number_locations = 100

        
        

        self.barrier_timepoint = None
        
        
        self.barrier_get_neighbours = \
            reusable_barrier_semaphore.\
            ReusableBarrier(self.number_threads_per_device)

        
        
    
        self.barrier_reset_counters = \
            reusable_barrier_semaphore.\
            ReusableBarrier(self.number_threads_per_device)


        
        self.script_access = Lock()

        
        self.script_access_index = -1

        
        self.finished_scripts = 0

        
        self.exit_simulation = 0

        
        self.neighbours = []

        
        self.all_devices = []

        
        
        self.locks_location_update_data = []
        for _ in xrange(self.number_locations):
            self.locks_location_update_data.append(Lock())

        
        self.event_access_data = Event()

        self.thread_id = 0
        self.thread_list = []

        
        for self.thread_id in xrange(self.number_threads_per_device):
            self.thread_list.append(DeviceThread(self, self.thread_id))

        
        for thread in self.thread_list:
            thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        self.all_devices = devices
        
        
        

        if (self.device_id is 0):
            self.barrier_timepoint = \
                reusable_barrier_semaphore. \
                ReusableBarrier(len(self.all_devices) * \
                    self.number_threads_per_device)
            i = 0
            for i in xrange(len(self.all_devices)):
                devices[i].barrier_timepoint = self.barrier_timepoint


                devices[i].locks_location_update_data = \
                self.locks_location_update_data
                

    def assign_script(self, script, location):
        

        self.scripts.append((script, location))

        
        self.event_access_data.set()


    def get_data(self, location):
        

        return self.sensor_data[location] if  \
            location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            
            self.sensor_data[location] = data
            

    def shutdown(self):
        
        thread = None
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        
        self.thread_id = thread_id

        
        self.script_to_access = 0
        


    def run(self):
        
        
        while True:

            

            if (self.thread_id == 0):
                self.device.neighbours = \
                self.device.supervisor.get_neighbours()
                if self.device.neighbours is None:
                    self.device.exit_simulation = 1
 
            
            
            self.device.barrier_get_neighbours.wait()

            
            if (self.device.exit_simulation == 1):
                break
            
            
            
            while (self.device.finished_scripts == 0):

                
                self.device.script_access.acquire()

                self.device.script_access_index += 1
                self.script_to_access = self.device.script_access_index

                
                

                if (self.device.script_access_index >= \
                    len(self.device.scripts)):
                    
                    self.device.event_access_data.wait()                        
                
                
                if (self.device.finished_scripts == 0):
                    self.device.event_access_data.clear()       
                
                self.device.script_access.release()

                
                if (self.device.finished_scripts == 1):
                    break
      
                (script, location) = self.device.scripts[self.script_to_access]
       
                if (script is None):
              
                    self.device.finished_scripts = 1

                    
                    
                    self.device.event_access_data.set()
                    break
                    
                
                

                self.device.locks_location_update_data[location].acquire()
                script_data = []

                

                if (self.device.neighbours != []):
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                if (script_data != []):

                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)



                    result = script.run(script_data)
             
                    for device in self.device.neighbours:

                        device.set_data(location, result)
            
                    self.device.set_data(location, result)

                
                self.device.locks_location_update_data[location].release()

            
            self.script_to_access = 0


            self.device.barrier_reset_counters.wait()
            if (self.thread_id == 0):
                
  
                
                

                self.device.scripts.pop()
                self.device.event_access_data.clear()
                self.device.script_access_index = -1
                self.device.finished_scripts = 0

            
            self.device.barrier_timepoint.wait()
