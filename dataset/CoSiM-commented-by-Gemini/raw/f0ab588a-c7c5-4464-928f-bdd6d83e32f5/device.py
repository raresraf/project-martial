"""
This module implements a multi-threaded device simulation where each device
has its own pool of worker threads. It uses a two-level barrier system for
synchronization.
"""

from threading import Event, Thread, Lock, Condition

class ReentrantBarrier(object):
    """
    A reusable barrier that allows a set of threads to wait for each other.
    """
    def __init__(self, num_threads):
        """Initializes the barrier."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Causes a thread to wait at the barrier. When all threads have arrived,
        they are all released.
        """
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
    Represents a device in the simulation, managing a pool of worker threads.
    """
    # Class-level variables shared across all devices.
    barrier = None
    devices_lock = Lock()
    locations = []
    nrloc = 0

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its worker threads."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.sensor_data_lock = Lock()
        self.supervisor = supervisor
        self.gen_lock = Lock()
        self.script_lock = Lock()
        self.script_event = Event()
        self.scripts = []
        self.working_scripts = []
        self.neighbour_request = False
        self.neighbour = None
        self.timepoint_done = False
        self.reinit_barrier = None
        
        self.threads_num = 8
        self.threads = []
        for i in xrange(self.threads_num):
            self.threads.append(DeviceThread(self, i))

    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for all devices, including the global barrier
        and location locks.
        """
        with self.gen_lock:
            self.reinit_barrier = ReentrantBarrier(self.threads_num)

        with Device.devices_lock:
            # Initialize a lock for each sensor location.
            Device.nrloc = max(Device.nrloc, (max(self.sensor_data.keys())+1))
            while Device.nrloc != len(Device.locations):
                Device.locations.append(Lock())

            # Initialize the global barrier for all threads of all devices.
            if Device.barrier is None:
                Device.barrier = ReentrantBarrier((len(devices) \
                        *self.threads_num))
        for i in xrange(self.threads_num):
            self.threads[i].start()

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        with self.script_lock:
            if script is not None:
                self.scripts.append((script, location))
                self.working_scripts.append((script, location))
            else:
                self.timepoint_done = True
            self.script_event.set()

    def get_data(self, location):
        """Gets sensor data in a thread-safe manner."""
        with self.sensor_data_lock:
            return self.sensor_data[location] \
                    if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data in a thread-safe manner."""
        with self.sensor_data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down all worker threads for this device."""
        for i in xrange(self.threads_num):
            self.threads[i].join()

class DeviceThread(Thread):
    """
    A worker thread belonging to a device. Each device has a pool of these.
    """
    def __init__(self, device, thread_nr):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.t_num = thread_nr

    def run_script(self, script, location):
        """Executes a script, ensuring exclusive access to the location."""
        with Device.locations[location]:
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

    def run(self):
        """Main loop for the worker thread."""
        while True:
            # Wait for all threads from all devices to finish the previous timepoint.
            Device.barrier.wait()
            
            # Re-initialize for the new timepoint.
            with self.device.script_lock:
                if len(self.device.working_scripts) == 0:
                    self.device.working_scripts = list(self.device.scripts)
                    self.device.timepoint_done = False
                    self.device.neighbour_request = False

            # Wait for all threads of this device to re-initialize.
            self.device.reinit_barrier.wait()
            
            # One thread per device gets the neighbors for the current timepoint.
            Device.devices_lock.acquire()
            if self.device.neighbour_request == False:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbour_request = True
            Device.devices_lock.release()

            if self.device.neighbours is None:
                break # End of simulation.

            # This loop processes the scripts for the current timepoint.
            while True:
                self.device.script_lock.acquire()
                if len(self.device.working_scripts) != 0:
                    (script, location) = self.device.working_scripts.pop()
                    self.device.script_lock.release()
                elif self.device.timepoint_done == True:
                    self.device.script_lock.release()
                    break
                else:
                    # Wait for more scripts to be assigned.
                    self.device.script_event.clear()
                    self.device.script_lock.release()
                    self.device.script_event.wait()
                    continue
                
                if script is not None:
                    self.run_script(script, location)
