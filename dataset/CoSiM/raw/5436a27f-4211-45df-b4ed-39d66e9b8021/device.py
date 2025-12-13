


from threading import Event, Thread, Lock, Condition, Semaphore


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = None


        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.leader = -1
        
        self.location_locks = []
        if device_id == 0:
            
            self.finishedthread = 0

            
            self.condition = Condition()

            
            self.can_start = Event()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        self.devices = devices

        
        for i in xrange(len(self.devices)):
            if devices[i].device_id == 0:
                self.leader = i
                break

        if self.device_id == 0:
            self.can_start.clear()

            
            maximum = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > maximum:
                        maximum = location

            for _ in range(0, maximum + 1):
                self.location_locks.append(Lock())

            for device in devices:
                device.location_locks = self.location_locks

            self.can_start.set()
        else:
            devices[self.leader].can_start.wait()

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

    def finished(self):
        


        self.devices[self.leader].condition.acquire()
        self.devices[self.leader].finishedthread += 1

        if self.devices[self.leader].finishedthread != len(self.devices):


            self.devices[self.leader].condition.wait()
        else:
            self.devices[self.leader].condition.notifyAll()
            self.devices[self.leader].finishedthread = 0

        self.devices[self.leader].condition.release()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.slavelist = SlaveList(device)

    def run(self):
        
        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                self.slavelist.shutdown()
                break

            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            for (script, location) in self.device.scripts:
                self.slavelist.do_work(script, location, neighbours)

            self.slavelist.event_wait()

            self.device.finished()

class SlaveList(object):
    
    def __init__(self, device):


        self.device = device
        self.event = Event()
        self.event.set()
        self.semaphore = Semaphore(8)
        self.lock = Lock()
        self.slavelist = []
        self.readythreads = []
        


        for _ in xrange(8):
            thread = Slave(self, self.device)
            self.slavelist.append(thread)
            self.readythreads.append(thread)
            thread.start()

    def do_work(self, script, location, neighbours):
        
        if self.event.isSet():
            self.event.clear()
        self.semaphore.acquire()
        self.lock.acquire()
        slave = self.readythreads.pop(0)
        self.lock.release()
        slave.do_work(script, location, neighbours)

    def shutdown(self):
        
        for slave in self.slavelist:
            slave.imdone = True
            slave.semaphore.release()
            slave.join()

    def slave_done(self, slave):
        
        self.lock.acquire()
        self.readythreads.append(slave)

        if self.event.isSet() == False:
            if len(self.readythreads) == 8:
                self.event.set()

        self.lock.release()
        self.semaphore.release()

    def event_wait(self):
        
        self.event.wait()



class Slave(Thread):
    
    def __init__(self, slavelist, device):
        
        Thread.__init__(self)


        self.slavelist = slavelist
        
        self.semaphore = Semaphore(0)
        self.device = device
        self.script = None
        self.location = None
        self.neighbours = None
        self.imdone = False

    def do_work(self, script, location, neighbours):
        
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.semaphore.release()


    def run(self):
        while True:
            self.semaphore.acquire()
            values = []
            if self.imdone is True:
                break
            self.device.location_locks[self.location].acquire()
            
            
            data = self.device.get_data(self.location)
            if data is not None:
                values.append(data)
            
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    values.append(data)
            

            if values != []:
                
                result = self.script.run(values)

                
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)
            self.device.location_locks[self.location].release()
            self.slavelist.slave_done(self)
