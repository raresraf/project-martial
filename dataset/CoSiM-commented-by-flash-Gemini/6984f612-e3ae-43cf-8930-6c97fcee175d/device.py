"""
@6984f612-e3ae-43cf-8930-6c97fcee175d/device.py
@brief This module implements a distributed simulation or data processing system.

It defines three core classes:
- `ReusableBarrier`: A custom barrier synchronization mechanism for threads.
- `Device`: Represents a computational node that manages its sensor data,
  worker threads, and coordinates with a supervisor and other devices.
- `DeviceThread`: Worker threads spawned by a `Device` to execute assigned scripts
  and manage data access.

The system relies on `threading` primitives (Lock, Event, Thread, Condition)
for concurrency and synchronization, allowing parallel processing of scripts
across multiple devices and within a single device.

Algorithm:
- Decentralized processing: Each `Device` operates semi-autonomously.
- Timepoint synchronization: Devices (via their threads) synchronize at discrete timepoints using a custom barrier.
- Concurrent script execution: `DeviceThread`s execute scripts in parallel.
- Distributed locking: Location-specific locks ensure data consistency across devices.
- Load balancing: Scripts are spread among worker threads based on their `id_thread`.

Time Complexity:
- `Device.__init__`: O(N_threads) where N_threads is number of threads.
- `Device.setup_devices`: O(D * L + D * N_threads) where D is number of devices, L is number of locations, N_threads is number of threads per device.
- `DeviceThread.run`: O(T * S * N_neighbors * L_locations) where T is timepoints, S is scripts per thread, N_neighbors is number of neighbors, L_locations is number of locations.
Space Complexity:
- `Device`: O(L) for locks per location, O(N_threads) for threads.
- `ReusableBarrier`: O(1).
- `DeviceThread`: O(1) beyond script and data storage.
"""

from threading import Lock, Event, Thread, Condition

class ReusableBarrier():
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


class Device(object):
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.gotneighbours = Event()
        self.zavor = Lock()
        self.threads = []
        self.neighbours = []
        self.nthreads = 8
        self.barrier = ReusableBarrier(1)
        self.lockforlocation = {}
        self.num_locations = supervisor.supervisor.testcase.num_locations
        for i in xrange(self.nthreads):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(self.nthreads):
            self.threads[i].start()


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        barrier = ReusableBarrier(devices[0].nthreads*len(devices))
        lockforlocation = {}
        for i in xrange(0, devices[0].num_locations):
            lock = Lock()
            lockforlocation[i] = lock
        for i in xrange(0, len(devices)):
            devices[i].barrier = barrier


            devices[i].lockforlocation = lockforlocation


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
        
        for i in xrange(self.nthreads):
            self.threads[i].join()


class DeviceThread(Thread):
    
    def __init__(self, device, id_thread):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id_thread = id_thread

    def run(self):
        
        while True:
            

            self.device.zavor.acquire()
            
            if self.device.gotneighbours.is_set() == False:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.gotneighbours.set()
            self.device.zavor.release()
            

            if self.device.neighbours is None:
                break

            
            self.device.timepoint_done.wait()
            
            myscripts = []
            for i in xrange(self.id_thread, len(self.device.scripts), self.device.nthreads + 1):
                myscripts.append(self.device.scripts[i])

            

            for (script, location) in myscripts:
                self.device.lockforlocation[location].acquire()
                script_data = []
                
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    


                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
                self.device.lockforlocation[location].release()

            
            self.device.barrier.wait()


            
            if self.id_thread == 0:
                self.device.timepoint_done.clear()
                self.device.gotneighbours.clear()
            
            self.device.barrier.wait()
