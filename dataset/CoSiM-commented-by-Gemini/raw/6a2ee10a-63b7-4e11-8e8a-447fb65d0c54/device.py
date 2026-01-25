


from threading import *




class ReusableBarrier(object):
    

    def __init__(self, numOfTh):
        self.numOfTh = numOfTh
        self.threads = [{}, {}]
        self.threads[0]['count'] = numOfTh
        self.threads[1]['count'] = numOfTh
        self.threads[0]['sem'] = Semaphore(0)
        self.threads[1]['sem'] = Semaphore(0)
        self.lock = Lock()

    def wait(self):
        for i in range(0, 2):
            with self.lock:
                self.threads[i]['count'] -= 1
                if self.threads[i]['count'] == 0:
                    for _ in range(self.numOfTh):
                        self.threads[i]['sem'].release()
                    self.threads[i]['count'] = self.numOfTh
            self.threads[i]['sem'].acquire()


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        


        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor


        self.scripts = []
        self.timepoint_done = Event()

        self.threads = []
        self.no_threads = 8
        self.tBariera = None
        self.locks = []
        self.sLock = Lock()
        self.iBariera = ReusableBarrier(8)


        self.etLock = Lock()
        self.lastScripts = []

        if device_id == 0:
            self.init_event = Event()

        for tid in range(self.no_threads):
            thread = DeviceThread(self, tid)
            self.threads.append(thread)


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id != 0:
            i = 0
            while (i < len(devices) and devices[i].device_id != 0):
                i += 1
            if i < len(devices):
                devices[i].init_event.wait()
                self.tBariera = devices[i].tBariera
                self.locks = devices[i].locks
        else:
            aux = 0
            self.tBariera = ReusableBarrier(len(devices))


            for d in devices:
                aux += len(d.sensor_data)
            self.locks = [RLock() for _ in range(aux)]
            self.init_event.set()

        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        
        if script is None:
            self.etLock.acquire()
            self.timepoint_done.set()
        else:
            self.sLock.acquire()
            self.scripts.append((script, location))
            self.sLock.release()

    def get_data(self, location):
        



        if location in self.sensor_data:
            self.locks[location].acquire()

        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        

        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        if self.thread_id == 0:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device in self.device.neighbours:
                self.device.neighbours.remove(self.device)

        while True:
            if self.thread_id == 0:
                self.device.sLock.acquire()
                self.device.scripts += self.device.lastScripts
                self.device.lastScripts = []
                self.device.sLock.release()
            self.device.iBariera.wait()
            neighbours = self.device.neighbours
            if neighbours is None:
                break

            while len(self.device.scripts) != 0:
                script = None

                self.device.sLock.acquire()
                if len(self.device.scripts) != 0:

                    script, location = self.device.scripts.pop(0)
                    self.device.lastScripts.append((script, location))
                self.device.sLock.release()

                if script:
                    script_data = []

                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        result = script.run(script_data)



                        for device in neighbours:
                            device.set_data(location, result)

                        self.device.set_data(location, result)
            self.device.timepoint_done.wait()

            while len(self.device.scripts) != 0:
                script = None

                self.device.sLock.acquire()
                if len(self.device.scripts) != 0:

                    script, location = self.device.scripts.pop(0)
                    self.device.lastScripts.append((script, location))
                self.device.sLock.release()

                if script:
                    script_data = []

                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        result = script.run(script_data)



                        for device in neighbours:
                            device.set_data(location, result)

                        self.device.set_data(location, result)

            self.device.iBariera.wait()

            if self.thread_id == 0:
                self.device.tBariera.wait()

                self.device.neighbours = self.device.supervisor.get_neighbours()
                if self.device.neighbours and self.device in self.device.neighbours:
                    self.device.neighbours.remove(self.device)

                self.device.timepoint_done.clear()
                self.device.etLock.release()
