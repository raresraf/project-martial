




from threading import Event, Thread, Lock
from barrier import Barrier
from device_thread import DeviceOwnThread

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        self.new_sensor_data = {}
        self.new_sensor_data_lock = Lock()

        self.other_devices = []

        
        self.ready_to_get_scripts = Event()

        
        self.got_all_scripts = Event()

        
        self.assign_script_lock = Lock()

        
        self.set_data_lock = Lock()

        
        self.start_loop_barrier = Barrier()

        
        self.got_scripts_barrier = Barrier()

        
        self.everyone_done = Barrier()

        
        self.location_mutex = []

        self.get_neighbours_lock = Lock()

        self.data_ready = Event()
        self.data_ready.set()

        
        self.own_threads = []
        self.power = 20
        for _ in range(0, self.power): 
            new_thread = DeviceOwnThread(self)
            self.own_threads.append(new_thread)
            new_thread.start()

        
        
        self.own_threads_rr = 0

        
        self.initialized = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

    def get_main_device(self):
        
        min_device = self
        min_id = self.device_id
        for device in self.other_devices:
            if device.device_id < min_id:
                min_device = device
                min_id = device.device_id
        return min_device

    
    def get_start_loop_barrier(self):
        
        return self.get_main_device().start_loop_barrier

    
    def get_got_scripts_barrier(self):
        
        return self.get_main_device().got_scripts_barrier

    def get_get_neighbours_lock(self):
        
        return self.get_main_device().get_neighbours_lock

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        
        for device in devices:
            self.other_devices.append(device)

        if self.get_main_device() == self:
            self.start_loop_barrier.set_n(len(devices))
            self.got_scripts_barrier.set_n(len(devices))
            self.everyone_done.set_n(len(devices))

            
            number_of_locations = 0
            for device in devices:
                current_number = max(device.sensor_data)
                if current_number > number_of_locations:
                    number_of_locations = current_number

            for _ in range(0, number_of_locations + 1):
                self.location_mutex.append(Lock())

        
        self.ready_to_get_scripts.set()

        
        self.initialized.set()

    def assign_script(self, script, location):
        
        self.ready_to_get_scripts.wait()
        self.assign_script_lock.acquire()
        if script is not None:
            self.own_threads[self.own_threads_rr].assign_script(script, location)
            self.own_threads_rr = (self.own_threads_rr + 1) % len(self.own_threads)
        else:
            self.ready_to_get_scripts.clear()
            self.data_ready.clear()
            self.got_all_scripts.set()
        self.assign_script_lock.release()

    def get_data(self, location):
        
        self.data_ready.wait()
        self.set_data_lock.acquire()
        result = self.sensor_data[location] if location in self.sensor_data else None
        self.set_data_lock.release()
        return result

    def get_temp_data(self, location):
        
        result = self.sensor_data[location] if location in self.sensor_data else None
        return result

    def set_data(self, location, data):
        
        self.set_data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_data_lock.release()

    def shutdown(self):
        
        for dot in self.own_threads:
            dot.join()
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        self.device.initialized.wait()
        self.device.ready_to_get_scripts.set()
        self.device.data_ready.clear()
        while True:
            
            self.device.get_start_loop_barrier().wait()

            self.device.get_get_neighbours_lock().acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.get_get_neighbours_lock().release()

            for dot in self.device.own_threads:
                dot.waiting_for_permission.wait()
            for dot in self.device.own_threads:
                dot.neighbours = neighbours
                dot.waiting_for_permission.clear()
                dot.start_loop_condition.acquire()
                dot.start_loop_condition.notify_all()
                dot.start_loop_condition.release()


            if neighbours is None:
                break

            
            self.device.got_all_scripts.wait()
            self.device.got_all_scripts.clear()
            self.device.data_ready.clear()

            
            self.device.get_got_scripts_barrier().wait()

            
            for dot in self.device.own_threads:
                dot.execute_scripts_event.set()

            
            for dot in self.device.own_threads:
                dot.done.wait()

            self.device.get_main_device().everyone_done.wait()

            self.device.data_ready.set()
            
            for dot in self.device.own_threads:
                dot.done.clear()
                dot.execute_scripts_event.clear()

            
            self.device.ready_to_get_scripts.set()



from threading import Event, Thread, Condition

class DeviceOwnThread(Thread):
    


    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

        
        self.done = Event()

        
        self.scripts = []

        
        self.execute_scripts_event = Event()

        
        self.start_loop_condition = Condition()

        
        self.waiting_for_permission = Event()

        self.neighbours = []

    def assign_script(self, script, location):
        
        self.scripts.append((script, location))

    def execute_scripts(self):
        
        for (script, location) in self.scripts:
            self.device.get_main_device().location_mutex[location].acquire()
            script_data = []
            
            for device in self.neighbours:
                data = device.get_temp_data(location)
                if data is not None:
                    script_data.append(data)

            
            data = self.device.get_temp_data(location)
            if data is not None:
                script_data.append(data)
            if script_data != []:
                
                result = script.run(script_data)



                for device in self.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            self.device.get_main_device().location_mutex[location].release()

    def run(self):
        while True:
            
            self.start_loop_condition.acquire()
            self.waiting_for_permission.set()
            self.start_loop_condition.wait()
            self.start_loop_condition.release()

            
            if self.neighbours is None:
                break


            
            
            self.execute_scripts_event.wait()

            self.execute_scripts()

            
            self.done.set()
