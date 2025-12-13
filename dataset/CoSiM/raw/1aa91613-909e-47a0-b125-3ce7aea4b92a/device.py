


from threading import Event, Thread, Lock
from Queue import Queue
import barrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.everyone = []
        self.barrier = None
        self.locations_lock = [None] * 100
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == 0:

            self.barrier = barrier.ReusableBarrierCond(len(devices))

            for device in devices:
                device.barrier = self.barrier

            self.everyone = devices

            for device in devices:
                device.everyone = self.everyone

            for i in range(100):
                self.locations_lock[i] = Lock()

            for device in devices:
                device.locations_lock = self.locations_lock

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
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
        self.workers = []
        self.jobs = Queue(maxsize=0)
        self.exit = False
        self.time_done = False
        self.neighbours = []

    def run(self):

        while True:

            self.time_done = False

            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                self.time_done = True
                self.exit = True

            if self.exit is False:
                self.device.timepoint_done.wait()

            if self.time_done is False:
                if self.neighbours == []:
                    self.time_done = True

            if self.time_done is False:

                for i in range(8):
                    thread = Workerr(self, self.neighbours)
                    self.workers.append(thread)

                for i in range(8):
                    self.workers[i].start()

                for script in self.device.scripts:
                    self.jobs.put(script)

                self.jobs.join()

                for i in range(8):
                    self.jobs.put(None)

                self.jobs.join()

                for i in range(8):
                    self.workers[i].join()

                self.workers = []

                self.time_done = True

            if self.time_done is True:

                self.device.timepoint_done.clear()
                self.device.everyone[0].barrier.wait()

                if self.exit is True:
                    break

class Workerr(Thread):
    

    def __init__(self, device, neighbours):
        
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours

    def run(self):

        while True:

            job_to_do = self.device.jobs.get()

            if job_to_do is None:
                self.device.jobs.task_done()
                break

            self.device.device.locations_lock[job_to_do[1]].acquire()
            script_data = []

            for device in self.neighbours:



                data = device.get_data(job_to_do[1])

                if data is not None:
                    script_data.append(data)

            data = self.device.device.get_data(job_to_do[1])

            if data is not None:
                script_data.append(data)

            if script_data != []:

                result = (job_to_do[0]).run(script_data)

                for device in self.neighbours:

                    device.set_data(job_to_do[1], result)



                    self.device.device.set_data(job_to_do[1], result)

            self.device.device.locations_lock[job_to_do[1]].release()

            self.device.jobs.task_done()
