

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None
        


        self.script_running = Lock()
        self.timepoint_done = Event()
        
        self.data_locks = dict()
        
        self.queue = Queue()
        
        self.available_threads = 14

        for loc in sensor_data:
            self.data_locks.__setitem__(loc, Lock())

        self.can_get_data = Lock()

        self.master = None
        self.script_over = False
        self.alive = True
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        self.barrier = ReusableBarrier(len(devices))

        
        self.master = devices[0]

    def assign_script(self, script, location):
        
        if script is not None:
            
            self.script_running.acquire()
            self.scripts.append((script, location))
            self.queue.put_nowait((script, location))
            self.script_received.set()
        else:
            self.script_running.acquire()
            self.timepoint_done.set()

    def get_data(self, location):
        

        

        self.can_get_data.acquire()
        return_value = self.sensor_data[location] if location in self.sensor_data else None
        self.can_get_data.release()
        return return_value

    def get_device_data(self, location):
        

        if location not in self.sensor_data:
            return None

        

        self.data_locks.get(location).acquire()

        new_data = self.sensor_data[location]

        self.data_locks.get(location).release()

        return new_data

    def set_data(self, location, data):
        
        if location in self.sensor_data:

            

            self.data_locks.get(location).acquire()
            self.sensor_data[location] = data
            self.data_locks.get(location).release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):



        while True:
            self.device.can_get_data.acquire()
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:

                

                self.device.master.barrier.wait()

                self.device.can_get_data.release()
                return

            

            script_instance = Scripter(self.device, neighbours)

            script_instance.start()

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            self.device.script_over = True
            self.device.script_received.set()

            

            script_instance.join()

            
            for (script, location) in self.device.scripts:
                self.device.queue.put_nowait((script, location))

            self.device.script_over = False

            

            self.device.master.barrier.wait()

            self.device.can_get_data.release()
            self.device.script_running.release()


class Scripter(Thread):
    

    def __init__(self, device, neighbours):
        


        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours

    def run(self):
        

        list_executors = []

        for iterator in range(1, self.device.available_threads):
            executor = ScriptExecutor(self.device, self.device.queue, self.neighbours, iterator)
            list_executors.append(executor)
            executor.start()

        while True:
            
            self.device.script_received.wait()
            self.device.script_received.clear()

            if self.device.script_over:

                
                
                

                for iterator in range(1, self.device.available_threads):
                    self.device.queue.put((None, None))

                
                for executor in list_executors:
                    executor.join()

                
                self.device.queue = Queue(-1)
                return

            self.device.script_running.release()


class ScriptExecutor(Thread):
    

    def __init__(self, device, queue, neighbours, identifier):
        
        Thread.__init__(self, name="Script Executor %d" % identifier)
        self.device = device
        self.queue = queue
        self.neighbours = neighbours

    def run(self):
        
        while True:
            

            (script, location) = self.queue.get()
            if script is None:
                return

            script_data = []
            
            for device in self.neighbours:
                data = device.get_device_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_device_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                


                result = script.run(script_data)
                
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)


class ReusableBarrier:
    

    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for iterator in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
