


from __future__ import division


from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier
from math import ceil



class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        
        self.timepoint_done = Event()

        
        
        self.neighbours = None

        
        self.scripts = []

        
        self.threads = []

        
        
        
        self.l_loc_dev = {}

        
        
        
        self.l_all_threads = None

        
        
        self.b_all_threads = None

        
        
        
        
        
        
        
        b_local = ReusableBarrier(8)

        
        
        
        e_local = Event()

        
        for i in xrange(8):
            thread = DeviceThread(self, i, b_local, e_local)
            self.threads.append(thread)
            thread.start()


    def __str__(self):
        
        return "Device %d" % self.device_id


    def setup_devices(self, devices):
        
        if devices[0] == self:
            nr_of_threads = sum([len(device.threads) for device in devices])
            barrier = ReusableBarrier(nr_of_threads)
            loc_dev_lock = {
                (device.device_id, location_id): Lock()
                for device in devices
                for location_id in device.sensor_data
                }

            set_data_lock = Lock()

            for device in devices:
                device.b_all_threads = barrier
                device.l_loc_dev = loc_dev_lock
                device.l_all_threads = set_data_lock


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))

        else:
            self.timepoint_done.set()


    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]

        return None


    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data


    def access_data(self, location):
        

        self.l_all_threads.acquire()

        if location in self.sensor_data:
            self.l_loc_dev[(self.device_id, location)].acquire()

        for device in self.neighbours:
            if device != self and location in device.sensor_data:
                device.l_loc_dev[(device.device_id, location)].acquire()
        self.l_all_threads.release()


    def release_data(self, location):
        
        if location in self.sensor_data:
            self.l_loc_dev[(self.device_id, location)].release()

        for device in self.neighbours:
            if device != self and location in device.sensor_data:
                device.l_loc_dev[(device.device_id, location)].release()


    def shutdown(self):
        
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, id_thread, barrier, event):
        
        Thread.__init__(
            self,
            name="Device Thread {0}-{1}".format(device.device_id, id_thread)
            )

        self.device = device
        self.id_thread = id_thread
        self.barrier = barrier
        self.event = event


    def run(self):

        while True:

            
            if self.device.threads[0] == self:
                
                
                self.device.neighbours = self.device.supervisor.get_neighbours()

                
                self.event.set()

            else:
                
                
                self.event.wait()


            
            if self.device.neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            
            self.barrier.wait()

            
            
            
            
            if self.device.threads[0] == self:
                self.device.timepoint_done.clear()
                self.event.clear()

            
            partition_size = int(ceil(
                len(self.device.scripts) /
                len(self.device.threads)
                ))

            
            down_lim = self.id_thread * partition_size
            up_lim = min(down_lim + partition_size, self.device.scripts)

            for (script, location) in self.device.scripts[down_lim : up_lim]:

                
                


                self.device.access_data(location)

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

                
                
                self.device.release_data(location)


            
            
            self.device.b_all_threads.wait()
