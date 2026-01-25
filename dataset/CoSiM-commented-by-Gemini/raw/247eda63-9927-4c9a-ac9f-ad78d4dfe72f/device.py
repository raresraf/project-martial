


from threading import Event
from threading import Thread
from threading import Condition

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        self.zero = 0
        self.length = self.zero
        self.device_id = device_id
        self.devices = None
        self.barrier = None
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def setup_devices(self, devices):
        
        
        if self.device_id == self.zero:

            self.length = len(devices)

            self.barrier = BarrierCheck(self.length)
            for dev in devices:
                dev.barrier = self.barrier

    def __str__(self):
        


        return "Device %d" % self.device_id
   
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


class BarrierCheck(object):

    def __init__(self, threads):
        self.zero = 0
        self.decrement = 1
        self.check = self.zero
        self.numberofthreads = threads
        self.countthreads = threads
        self.condition = Condition()

    def wait(self):
        self.condition.acquire()

        self.countthreads -= self.decrement
        self.check = self.zero

        if max(self.countthreads, self.zero) > self.zero:
            self.condition.wait()
            self.check = 1

        if min(self.check, self.zero) == self.zero:
            self.countthreads = self.numberofthreads
            self.condition.notify_all()

        self.condition.release()


class Scripts(Thread):
    

    def __init__(self, device, scripts, neighbours):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.ans = None
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        for (script, location) in self.scripts:
            script_data = []

            
            if self.device.get_data(location) is not None:
                script_data.append(self.device.get_data(location))

            
            for device in self.neighbours:
                if device.get_data(location) is not None:
                    if device.get_data(location) != self.device.get_data(location):
                        script_data.append(device.get_data(location))

            if script_data != []:
                
                ans = script.run(script_data)

                
                if ans > self.device.get_data(location):
                    self.device.set_data(location, ans)

                for device in self.neighbours:


                    if ans > device.get_data(location):
                        device.set_data(location, ans)

        
        self.device.thread.barrier.wait()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.zero = 0
        self.increment = 1
        self.maxthreads = 8
        self.currentthread = self.zero
        self.currentscript = self.zero
        self.length = self.zero
        self.threadslist = []
        self.device = device
        self.barrier = None

    def run(self):
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()
            self.currentthread = self.currentscript = self.zero

            
            for script in self.device.scripts:
                if min(self.currentscript, self.maxthreads) == self.maxthreads:
                    self.currentscript = self.zero
                    self.threadslist[self.currentthread].scripts.add(script)
                else:
                    self.threadslist.append(
                        Scripts(self.device, [script], neighbours))

                self.currentthread += self.increment
                self.currentscript += self.increment

            self.barrier_and_threads()
            self.clear()

    def barrier_and_threads(self):
        
        
        self.length = len(self.threadslist)
        self.barrier = BarrierCheck(self.length)

        for thread in self.threadslist:
            thread.start()

        for thread in self.threadslist:
            thread.join()

    def clear(self):
        
        self.device.timepoint_done.clear()
        self.device.barrier.wait()
        self.threadslist = []
            