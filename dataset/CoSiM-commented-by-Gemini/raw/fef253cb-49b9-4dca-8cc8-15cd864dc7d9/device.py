"""
This module defines a highly complex, multi-threaded device simulation.

It relies on a custom 'barrier' module and employs an unusual threading model
with a main 'DeviceThread' and a pool of 'WorkerThread's that perform nearly
identical tasks. The synchronization logic is intricate, involving numerous
Event and Lock objects. Device 0 acts as a coordinator for setting up shared
state, including multiple global barriers and location-based locks.
"""
import barrier
from threading import Event, Thread, Lock


class Device(object):
    """
    Represents a device in the simulation, managing a complex state and a
    heterogeneous pool of worker threads.
    
    Attributes:
        A large number of synchronization primitives (Events, Locks) are used
        to manage the device's state machine.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.script_taken = Event()
        self.assign_script_none = Event()
        self.script_de_orice_fel = Event()
        self.assign_script_not_none = Event()
        self.bariera = None
        self.bariera_join = None
        self.barrier_time = None
        self.flag_terminate = False
        self.script_sent = Lock()
        self.script_sent_thread = Lock()
        self.barrier_lock = Lock()
        self.counter = 0
        self.flag_received = Event()
        self.got_neighbours = Event()
        self.barrier_clear_events = None
        self.flag_script_received = False
        self.flag_script_taken = False
        self.flag_assign_script = 2
        self.flag_get_neigbours = False
        self.get_neighbours_lock = Lock()
        self.index_lock = Lock()
        self.i = 0
        self.scripts = []
        self.neighbours = None
        self.devices = []
        self.count_threads = []
        self.locations_locks = []
        self.timepoint_done = Event()
        self.initialize = Event()
        self.put_take_data = Lock()
        # The device has a main 'DeviceThread' and a pool of 'WorkerThread's.
        self.thread = DeviceThread(self)
        self.threads = []

    def __str__(self):
        """String representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources via the coordinator device (ID 0).
        """
        
        
        self.devices = devices
        self.count_threads = [len(self.devices)]

        if self.device_id == 0:
            
            # Coordinator device calculates required lock array size.
            locations = []
            for device in self.devices:
                l = []
                for key in device.sensor_data.keys():
                    l.append(key)
                for locatie in l:
                    locations.append(locatie)
            maxim = max(locations) if locations else -1
            self.locations_locks = [None] * (maxim + 1)
            for locatie in locations:
                if self.locations_locks[locatie] is None:
                    lock = Lock()
                    self.locations_locks[locatie] = lock

            # Coordinator creates multiple global barriers for synchronization.
            self.bariera = barrier.ReusableBarrierCond(len(self.devices))
            num_threads = len(self.devices) * 8
            self.bariera_join = barrier.ReusableBarrierCond(num_threads)
            self.barrier_time = barrier.ReusableBarrierCond(num_threads)
            self.barrier_clear_events = barrier.ReusableBarrierCond(num_threads)

            # Distribute shared resources to all other devices.
            for device in self.devices:
                device.i = 0
                device.bariera = self.bariera
                device.counter = len(self.devices)
                device.barrier_time = self.barrier_time
                device.barrier_clear_events = self.barrier_clear_events
                device.locations_locks = self.locations_locks

        # Start the main device thread and the worker pool.
        self.thread.start()
        i = 0
        while i < 7:
            dev = WorkerThread(self)
            dev.start()
            self.threads.append(dev)
            i = i + 1
        
        self.initialize.set()

    def assign_script(self, script, location):
        """
        Appends a script to the shared script list for the device.
        """
        
        with self.script_sent:
            if script is not None:
                self.scripts.append((script, location))
            else:
                # A None script signals the end of the timepoint.
                self.scripts.append((script, location))
                self.script_received.set()
                self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data, protected by a lock."""
        
        with self.put_take_data:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data, protected by a lock."""
        
        with self.put_take_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Joins all worker threads and the main device thread."""
        
        for thread in self.threads:
            thread.join()
        self.thread.join()

class DeviceThread(Thread):
    """
    The main thread for a device, responsible for fetching neighbor data.
    It also participates in processing scripts from the shared list.
    """

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the thread."""
        while True:
            # This thread is solely responsible for getting neighbor data from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                self.device.flag_terminate = True
                self.device.got_neighbours.set()
                break
            
            self.device.got_neighbours.set()
            
            # Wait for the signal that all scripts for the timepoint have been assigned.
            self.device.script_received.wait()
            # This loop pulls scripts from a shared list and processes them.
            # It competes with the WorkerThreads.
            while True:
                with self.device.index_lock:
                    if self.device.i >= len(self.device.scripts):
                        break
                    # The modulo logic here seems incorrect, as it doesn't guarantee
                    # unique script processing if i grows beyond the number of threads.
                    (script, location) = self.device.scripts[self.device.i % 8]
                    self.device.i = self.device.i + 1
                if script is not None:
                    # Core script processing logic.
                    lock = self.device.locations_locks[location]
                    with lock:
                        
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
                if script is None:
                    break


            # Synchronize with other threads at multiple barrier points.
            self.device.barrier_clear_events.wait()
            self.device.script_received.clear()
            self.device.got_neighbours.clear()
            self.device.barrier_time.wait()

class WorkerThread(Thread):
    """
    A worker thread that processes scripts from a shared list. Its logic is
    largely identical to DeviceThread, creating a race condition for scripts.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            
            # Wait for neighbor data to be available.
            self.device.got_neighbours.wait()
            if self.device.flag_terminate == True:
                break
            
            # Wait for all scripts to be assigned for the timepoint.
            self.device.script_received.wait()
            # This loop is a near-duplicate of the one in DeviceThread.
            while True:
                with self.device.index_lock:
                    if self.device.i >= len(self.device.scripts):
                        break
                    # The modulo logic here is shared and likely buggy.
                    (script, location) = self.device.scripts[self.device.i % 8]
                    self.device.i = self.device.i + 1

                if script is not None:
                    # Core script processing logic, duplicated from DeviceThread.
                    lock = self.device.locations_locks[location]
                    with lock:
                        
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
                if script is None:
                    break
            # Synchronize with other threads.
            self.device.barrier_clear_events.wait()
            self.device.barrier_time.wait()
