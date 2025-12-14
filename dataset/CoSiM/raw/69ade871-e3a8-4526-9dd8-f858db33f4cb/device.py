


from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

class Device(object):
    

    
    _CORE_COUNT = 8

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id

        self.sensor_data = sensor_data
        self.busy_data = {}

        for key in sensor_data:
            self.busy_data[key] = {"busy": False, "queue": []}

        self.device_barrier = None 
        self.supervisor = supervisor
        self.scripts = []
        self.listeners = 0
        self.script_queue = Queue()
        self.processing_finished = Event()
        self.devices_done = Event()
        self.thread_barrier = Barrier(Device._CORE_COUNT)
        self.working_threads = 0
        self.received_stop = False


        self.neighbours = []
        self.thread_lock = Lock()
        self.data_lock = Lock()
        self.script_lock = Lock()

        
        self.threads = []

        for i in xrange(Device._CORE_COUNT):


            self.threads.append(DeviceThread(self))
            self.threads[i].start()



    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        device = self

        for dev in devices:
            if dev.device_id < device.device_id:
                device = dev

        
        if device == self:
            self.device_barrier = Barrier(len(devices))

            for dev in devices:
                dev.device_barrier = self.device_barrier

    def assign_script(self, script, location):
        

        self.script_lock.acquire()


        if script is not None:
            self.scripts.append((script, location))
            self.script_queue.put((script, location, False))
        else:
            self.received_stop = True

            
            
            self.script_queue.put(None)

        self.script_lock.release()

    def increase_listeners(self):
        

        self.script_lock.acquire()

        self.listeners = self.listeners + 1

        self.script_lock.release()

    def decrease_listeners(self):
        
        self.script_lock.acquire()

        self.listeners = self.listeners - 1

        self.script_lock.release()

    def should_stop_thread(self):
        

        self.script_lock.acquire()

        val = self.script_queue.empty() and \
              self.listeners == 0 and \
              (self.received_stop is True)

        if val is True:
            
            
            self.script_queue.put(None)


        self.script_lock.release()

        return val

    def thread_start_timestep(self):
        

        self.thread_lock.acquire()

        
        
        if self.working_threads == 0:
            self.neighbours = self.supervisor.get_neighbours()

            
            if self.neighbours is not None:
                if self in self.neighbours:
                    self.neighbours.remove(self)

        self.working_threads = self.working_threads + 1

        self.thread_lock.release()

    def decr_working_threads(self):
        

        self.thread_lock.acquire()

        self.working_threads = self.working_threads - 1
        val = self.working_threads

        self.thread_lock.release()

        return val

    def get_working_threads(self):
        

        self.thread_lock.acquire()

        val = self.working_threads

        self.thread_lock.release()

        return val

    def finish(self):
        
        self.device_barrier.wait()

        self.start_timestep()



    
    def start_timestep(self):
        

        self.thread_lock.acquire()

        self.script_lock.acquire()

        
        while self.script_queue.empty() is False:
            self.script_queue.get()

        
        for script in self.scripts:
            self.script_queue.put((script[0], script[1], False))

        self.received_stop = False

        self.script_lock.release()

        self.devices_done.set()
        self.devices_done.clear()

        self.thread_lock.release()

    def get_data(self, location):
        
        result = None

        self.data_lock.acquire()

        if location in self.sensor_data:
            result = self.sensor_data[location]

        self.data_lock.release()

        return result

    def get_data_with_listener(self, location, listener):
        

        result = None

        self.data_lock.acquire()

        if location in self.sensor_data:

            if self.busy_data[location]["busy"] is True:
                
                self.busy_data[location]["queue"].append(listener)
                result = False
            else:
                
                self.busy_data[location]["busy"] = True
                result = self.sensor_data[location]

        self.data_lock.release()

        return result

    def set_data(self, location, data):
        

        self.data_lock.acquire()

        self.sensor_data[location] = data

        self.data_lock.release()

    def set_data_with_listener(self, location, data):
        

        self.data_lock.acquire()

        self.sensor_data[location] = data

        if location not in self.busy_data:
            
            self.busy_data[location] = {"busy": False, "queue": []}
        else:
            
            self.busy_data[location]["busy"] = False

            if len(self.busy_data[location]["queue"]) > 0:
                listener = self.busy_data[location]["queue"].pop(0)

                listener[1].put((listener[0], location, True))

        self.data_lock.release()

    def shutdown(self):
        


        for i in xrange(len(self.threads)):
            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    
    
    def _get_script_data(self, script, location, neighbours):
        data = []
        locked_devices = []

        res = self.device.get_data_with_listener(
            location,
            (script, self.device.script_queue)
        )

        if res is False:
            
            return False
        elif res is not None:
            data.append(res)
            locked_devices.append((self.device, res))

        for neighbour in neighbours:
            res = neighbour.get_data_with_listener(
                location,
                (script, self.device.script_queue)
            )

            if res is False:
                
                
                for dev in locked_devices:
                    dev[0].set_data_with_listener(location, dev[1])

                return False
            elif res is not None:
                data.append(res)
                locked_devices.append((neighbour, res))

        if len(data) == 0:
            return None

        return data

    def _update_devices(self, location, value, neighbours):
        self.device.set_data_with_listener(location, value)

        for neighbour in neighbours:
            neighbour.set_data_with_listener(location, value)

    def _loop(self, neighbours):
        

        while self.device.should_stop_thread() is False:

            res = self.device.script_queue.get()

            if res is None:
                continue

            
            (script, location, is_listener) = res

            if is_listener is True:

                self.device.decrease_listeners()

            data = self._get_script_data(script, location, neighbours)

            if data is False:
                
                self.device.increase_listeners()
            elif data is not None:
                new_value = script.run(data)


                self._update_devices(location, new_value, neighbours)

        self.device.thread_barrier.wait()

        val = self.device.decr_working_threads()

        if val == 0:
            self.device.finish()
        else:
            self.device.devices_done.wait()

        self.device.thread_barrier.wait()


    def run(self):
        while True:
            self.device.thread_start_timestep()

            neighbours = self.device.neighbours

            if neighbours is None:
                break

            self._loop(neighbours)

        self.device.thread_barrier.wait()

        val = self.device.decr_working_threads()

        if val == 0:
            self.device.finish()

class Barrier(object):
    

    def __init__(self, thread_count):
        self.lock = Lock()
        self.sem1 = Semaphore(0)
        self.sem2 = Semaphore(0)
        self.thread_count = thread_count
        self.thread_count1 = [thread_count]
        self.thread_count2 = [thread_count]


    def wait(self):
        
        self._phase(self.thread_count1, self.sem1)
        self._phase(self.thread_count2, self.sem2)

    def _phase(self, thread_count, sem):
        self.lock.acquire()

        thread_count[0] = thread_count[0] - 1
        value = thread_count[0]

        self.lock.release()

        if value == 0:
            thread_count[0] = self.thread_count
            for _ in xrange(self.thread_count):
                sem.release()

        sem.acquire()
