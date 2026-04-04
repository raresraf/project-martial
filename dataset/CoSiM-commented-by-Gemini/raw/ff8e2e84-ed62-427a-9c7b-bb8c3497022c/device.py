"""
This module defines a device simulation framework with a complex, phased execution model.

A main `DeviceThread` orchestrates the workflow by gathering data, spawning
short-lived `ScriptThread` workers for execution, and then propagating results.
Synchronization is managed through a combination of a custom `ReusableBarrier`,
Events, and multiple `RLock` instances. Device setup appears to be coordinated
but the locking logic for initialization is intricate.
"""

from threading import Condition, Event, RLock, Thread

class ReusableBarrier(object):
    """
    A reusable barrier implemented using a `threading.Condition`.

    Includes a non-standard `reinit` method that appears intended for shutdown
    but may be unsafe as it modifies the thread count of an active barrier.
    """
    
    def __init__(self, num_threads):
        
        self.num_threads = num_threads


        self.count_threads = self.num_threads
        self.cond = Condition()

    def reinit(self):
        """Decrements the expected thread count and re-waits, likely for shutdown."""
        
        self.cond.acquire()
        self.num_threads -= 1
        self.cond.release()
        self.wait()

    def wait(self):
        """Blocks the calling thread until all threads have called `wait()`."""
        
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
    Represents a device that uses a main thread to manage batches of script-executing
    worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.start = Event()
        self.scripts = []


        self.scripts_to_process = []
        self.timepoint_done = Event()
        self.nr_script_threats = 0
        self.thread = DeviceThread(self)
        self.thread.start()
        self.script_threats = []
        self.barrier_devices = None
        self.neighbours = None
        self.cors = 8 # Number of script threads to spawn per batch
        self.lock = None # Shared lock for the scripts_to_process list
        self.lock_self = None # Lock to protect initialization of other shared locks
        self.results = {}
        self.results_lock = None

    def __str__(self):
        """String representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared locks and a global barrier for all devices.
        The initialization logic is complex and relies on careful locking.
        """
        
        for script in self.scripts:
            self.lock.acquire()
            self.scripts_to_process.append(script)
            self.lock.release()

        if not self.lock_self:
            lock = RLock()
            for device in devices:
                device.lock_self = lock

        self.lock_self.acquire()
        if not self.lock:
            rlock = RLock()
            for device in devices:
                device.lock = rlock
        self.lock_self.release()

        self.lock_self.acquire()
        if not self.results_lock:
            results_lock = RLock()
            for device in devices:
                device.results_lock = results_lock
        self.lock_self.release()

        self.lock_self.acquire()
        if not self.barrier_devices:
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier_devices = barrier
                device.start.set()
        self.lock_self.release()



    def assign_script(self, script, location):
        """
        Adds a script to the central script list for the device.
        """
        

        if script is not None:
            self.lock.acquire()
            self.scripts.append((script, location))
            self.scripts_to_process.append((script, location))
            self.script_received.set()
            self.lock.release()
        else:
            # A None script signals the end of a timepoint.
            self.lock.acquire()
            self.timepoint_done.set()
            self.script_received.set()
            self.lock.release()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else:
            data = None
        return data

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
        
    def shutdown(self):
        """Joins the device's main thread."""
        
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. Orchestrates data gathering,
    spawning of worker threads, and result propagation.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.device.neighbours = None

    def run(self):
        """Main execution loop."""
        
        self.device.start.wait()
        while True:
            # Refresh the processing queue from the main script list for the timepoint.
            self.device.scripts_to_process = []
            for script in self.device.scripts:
                self.device.scripts_to_process.append(script)

            


            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                self.device.barrier_devices.reinit() # Unsafe barrier usage for shutdown
                break

            self.device.results = {}


            # Loop to process scripts in batches.
            while True:
                if not self.device.timepoint_done.is_set():
                    self.device.script_received.wait()
                    self.device.script_received.clear()

                
                if len(self.device.scripts_to_process) == 0:
                    if self.device.timepoint_done.is_set():
                        break

                
                # Process scripts in batches of size `self.device.cors`.
                while len(self.device.scripts_to_process):
                    list_threats = []
                    self.device.script_threats = []
                    self.device.nr_script_threats = 0
                    
                    # Pop a batch of scripts from the shared queue.
                    while len(self.device.scripts_to_process) and self.device.nr_script_threats < self.device.cors:
                        script, location = self.device.scripts_to_process.pop(0)
                        list_threats.append((script, location))
                        self.device.nr_script_threats += 1

                    # Data gathering phase (serialized in this main thread).
                    for script, location in list_threats:
                        script_data = []
                        
                        neighbours = self.device.neighbours
                        for device in neighbours:
                            device.lock_self.acquire()
                            data = device.get_data(location)
                            device.lock_self.release()
                            if data is not None:
                                script_data.append(data)
                        
                        self.device.lock_self.acquire()
                        data = self.device.get_data(location)


                        self.device.lock_self.release()
                        if data is not None:
                            script_data.append(data)

                        # Spawn a lightweight worker thread for execution only.
                        thread_script_d = ScriptThread(self.device, script, location, script_data)

                        self.device.script_threats.append(thread_script_d)
                        thread_script_d.start()

                    # Wait for the current batch of worker threads to finish.
                    for thread in self.device.script_threats:
                        thread.join()

            
            # Data propagation phase (serialized in this main thread).
            for location, result in self.device.results.iteritems():
                
                for device in self.device.neighbours:
                    device.lock_self.acquire()
                    device.set_data(location, result)
                    device.lock_self.release()
                
                self.device.lock_self.acquire()
                self.device.set_data(location, result)
                self.device.lock_self.release()

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            self.device.barrier_devices.wait()

class ScriptThread(Thread):
    """
    A lightweight worker thread that only executes a script with pre-gathered data.
    """

    def __init__(self, device, script, location, script_data):


        
        Thread.__init__(self, name="Device Script Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.script_data = script_data

    def run(self):
        """Runs the script and stores the result in a shared dictionary."""
        
        if self.script_data != []:
            
            result = self.script.run(self.script_data)
            
            self.device.results_lock.acquire()
            self.device.results[self.location] = result
            self.device.results_lock.release()
        self.device.nr_script_threats -= 1
