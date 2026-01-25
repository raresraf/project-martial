


from threading import Event, Thread, Lock
import supervisor
from barrier import ReusableBarrierSem


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.locations = []
        self.get_data_lock = Lock()
        self.ready = Event()
        self.devices = None
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        self.devices = devices
        barrier = ReusableBarrierSem(len(devices))
        if self.device_id == 0:
            i = 0
            while i < 150:  
                self.locations.append(Lock())
                i = i + 1


            for dev in devices:
                dev.barrier = barrier
                dev.locations = self.locations
                dev.ready.set()

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        
        with self.get_data_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="%d" % device.device_id)
        self.device = device

    def run(self):
        
        self.device.ready.wait()

        while True:
            neigh = self.device.supervisor.get_neighbours()
            if neigh is None:
                break

            
            self.device.script_received.wait()
            self.device.script_received.clear()

            rem_scripts = len(self.device.scripts)

            threads = []
            i = 0
            while i < rem_scripts:
                threads.append(MyThread(self.device, neigh, self.device.scripts, i))
                i = i + 1

            if rem_scripts < 8:
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            else:
                pos = 0
                while rem_scripts != 0:
                    if rem_scripts > 8:


                        for i in range(pos, pos + 8):
                            threads[i].start()
                        for i in range(pos, pos + 8):
                            threads[i].join()
                        pos = pos + 8
                        rem_scripts = rem_scripts - 8
                    else:
                        for i in range(pos, pos + rem_scripts):
                            threads[i].start()
                        for i in range(pos, pos + rem_scripts):
                            threads[i].join()
                        rem_scripts = 0

            self.device.barrier.wait()


class MyThread(Thread):
    

    def __init__(self, device, neigh, scripts, index):
        
        Thread.__init__(self, name="%d" % device.device_id)
        self.device = device
        self.neigh = neigh
        self.scripts = scripts
        self.index = index

    def run(self):
        
        (script, loc) = self.scripts[self.index]
        self.device.locations[loc].acquire()
        info = []
        for neigh_iter in self.neigh:
            aux_data = neigh_iter.get_data(loc)
            if aux_data is not None:
                info.append(aux_data)
        aux_data = self.device.get_data(loc)
        if aux_data is not None:
            info.append(aux_data)
        if info != []:
            result = script.run(info)
            for neigh_iter in self.neigh:
                neigh_iter.set_data(loc, result)
                self.device.set_data(loc, result)
        self.device.locations[loc].release()
