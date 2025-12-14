


from threading import Event, Thread, Lock
from my_barrier import ReusableBarrierCond

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id


        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = []
        self.scripts = []
        self.new_scripts = []
        self.timepoint_done = []
        self.threads = []
        self.nxt_thr_to_rcv_scr = 0
        self.data_access = Lock()
        self.scripts_access = []
        self.new_scripts_access = []
        self.barrier1 = None
        self.barrier2 = None
        self.locs_acc = [] 
        self.neighbours = None


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            bar1 = ReusableBarrierCond(len(devices) * 8)
            
            for dev in devices:
                dev.barrier1 = bar1
                

        for i in range(8):
            self.threads.append(DeviceThread(self, i))
            self.threads[i].start()
            self.scripts.append([])
            self.new_scripts.append([])
            self.script_received.append(Event())
            self.timepoint_done.append(Event())
            self.scripts_access.append(Lock())
            self.new_scripts_access.append(Lock())
        
        if self.device_id == 0:
            max_loc = -1
            for dev in devices:
                for loc in dev.sensor_data.keys():
                    if loc > max_loc:
                        max_loc = loc
            locs_locks = [] 
            for i in range(max_loc+1):
                locs_locks.append(Lock())
            for dev in devices:
                dev.locs_acc = locs_locks



    def assign_script(self, script, location):
        


        if script is not None:
            
            i = self.nxt_thr_to_rcv_scr
            self.nxt_thr_to_rcv_scr = (self.nxt_thr_to_rcv_scr + 1) % 8
            self.new_scripts[i].append((script, location))
            self.script_received[i].set()
        else:
            for j in range(8):
                self.timepoint_done[j].set()
                self.script_received[j].set()



    def get_data(self, location):
        

        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for i in range(8):


            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, id_thread):
        
        Thread.__init__(self, name="Device %d Thread %d" % (device.device_id, id_thread))
        self.device = device
        self.crt_tp = 0
        self.id_thread = id_thread

    def run(self):

        while True:
            
            
            if self.id_thread is 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.crt_tp += 1
            self.device.barrier1.wait()
            
            if self.device.neighbours is None:
                break

            
            for (script, location) in self.device.scripts[self.id_thread]:
                self.device.locs_acc[location].acquire()
                self.procces_script(script, location, self.device, self.device.neighbours)
                self.device.locs_acc[location].release()

            self.device.timepoint_done[self.id_thread].wait()
            self.device.timepoint_done[self.id_thread].clear()

            
            for (script, location) in self.device.new_scripts[self.id_thread]:
                self.device.locs_acc[location].acquire()
                self.procces_script(script, location, self.device, self.device.neighbours)
                self.device.locs_acc[location].release()
                self.device.scripts[self.id_thread].append((script, location))

            self.device.new_scripts[self.id_thread] = []
            

            self.device.barrier1.wait()

    def procces_script(self, script_func, location, crt_device, neighbours):
        
        script_data = []
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = crt_device.get_data(location)
        if data is not None:
            script_data.append(data)
        if script_data != []:
            
            result = script_func.run(script_data)

            
            for device in neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)
