"""
This module implements a device simulation framework using a master-worker
thread model within each device, designed for a Python 2 environment.

Each device consists of a single master thread that coordinates tasks for a
timepoint and a pool of worker threads that execute those tasks.
Synchronization between devices is achieved with a shared barrier, while
intra-device communication uses a task queue and events.

Classes:
    ReusableBarrier: A simple condition-based reusable barrier.
    Device: Represents a device, managing its master and worker threads.
    DeviceThreadMaster: The master thread for a device, dispatching tasks.
    DeviceThreadWorker: A worker thread that executes computational scripts.
"""
from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty


class ReusableBarrier(object):
    """
    A reusable barrier for thread synchronization using a Condition variable.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a set number of threads."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks the calling thread until all threads reach the barrier."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Represents a device in the simulation, using a master-worker thread architecture.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): The local data for the device.
            supervisor: The supervisor object managing the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.neighbours = []

        self.barrier = None
        self.locks = []
        self.timepoint_done = Event()
        self.tasks_ready = Event()
        self.tasks = Queue()
        self.simulation_ended = False

        
        self.master = DeviceThreadMaster(self)
        self.master.start()

        
        self.workers = []
        for i in xrange(8):
            worker = DeviceThreadWorker(self, i)
            self.workers.append(worker)
            worker.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device [%d]" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for all devices in the simulation.

        The device with ID 0 creates a global barrier and a shared list of locks.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            locks = [Lock() for _ in xrange(24)]
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """Assigns a script to be executed for a given location."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its master and worker threads."""
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceThreadMaster(Thread):
    """
    The master thread for a device, which orchestrates work for a timepoint.
    """
    def __init__(self, device):
        """Initializes the master thread."""
        Thread.__init__(self, name="Device [%d] Thread Master" % device.device_id)
        self.device = device

    def run(self):
        """The main loop for the master thread."""
        while True:
            
            self.device.neighbours = self.device.supervisor.get_neighbours()

            
            if self.device.neighbours is None:
                # A None neighbor list is the shutdown signal.
                self.device.simulation_ended = True
                self.device.tasks_ready.set()
                
                break

            
            self.device.timepoint_done.wait()

            
            for task in self.device.scripts:
                self.device.tasks.put(task)

            
            self.device.tasks_ready.set()

            


            self.device.tasks.join()

            
            self.device.tasks_ready.clear()
            self.device.timepoint_done.clear()

            
            self.device.barrier.wait()


class DeviceThreadWorker(Thread):
    """A worker thread that executes tasks from a shared queue."""
    def __init__(self, device, thread_id):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device [%d] Thread Worker [%d]" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main loop for the worker thread."""
        while not self.device.simulation_ended:
            
            self.device.tasks_ready.wait()

            try:
                
                script, location = self.device.tasks.get(block=False)

                
                self.device.locks[location].acquire()

                script_data = []

                
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                
                if len(script_data) > 0:
                    
                    result = script.run(script_data)

                    
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                
                self.device.locks[location].release()

                
                self.device.tasks.task_done()
            except Empty:
                pass
