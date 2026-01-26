


from threading import Event, Thread, Lock, Condition


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

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        lock_set = {}
        barrier = ReusableBarrier(len(devices))
        idx = len(devices) - 1

        while idx >= 0:
            current_device = devices[idx]
            current_device.barrier = barrier
            for current_location in current_device.sensor_data:
                lock_set[current_location] = Lock()
            current_device.lock_set = lock_set
            idx = idx - 1

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
        else:
            pass

    def shutdown(self):
        
        self.thread.join()

class ReusableBarrier(object):
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

    def print_barrier(self):
        print self.num_threads, self.count_threads

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device

    def run(self):
        nr_threads = 8
        while True:
            self.device.timepoint_done.clear()
            neigh = self.device.supervisor.get_neighbours()
            self.device.barrier.wait()
            if neigh is None:
                break
            self.device.timepoint_done.wait()
            execute_script = []
            threads = []
            for script in self.device.scripts:
                execute_script.append(script)
            for i in xrange(nr_threads):
                threads.append(MakeUpdate(self.device, neigh, execute_script))
                threads[i].start()

            for t in threads:
                t.join()
            self.device.barrier.wait()

class MakeUpdate(Thread):


    def __init__(self, device, neighbours, execute_script):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.execute_script = execute_script

    def run(self):
        if len(self.execute_script) != 0:
            collected = []
            (script, location) = self.execute_script.pop()
            self.device.lock_set[location].acquire()
            for neigh_c in self.neighbours:
                collected.append(neigh_c.get_data(location))
            collected.append(self.device.get_data(location))

            if collected != []:
                result = script.run(collected)
                self.device.set_data(location, result)
                for neigh_c in self.neighbours:
                    neigh_c.set_data(location, result)
            self.device.lock_set[location].release()
