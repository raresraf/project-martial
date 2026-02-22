


"""
This file models a distributed sensor network simulation.

The architecture uses a "thread-per-task" model, where the main `DeviceThread`
spawns a new `Worker` thread for each computational script in a time step.
Shared resources like a barrier and location-based locks are intended to be
managed by a master device (device 0).

@warning This script contains several critical concurrency flaws, including an
         unsafe barrier implementation, race conditions in setup, and a
         deadlock-prone locking strategy in the worker thread.
"""

from threading import Event, Thread, Condition, Lock


class Device(object):
    """
    Represents a single device node in the network.

    @warning The `setup_devices` method contains a race condition. Non-master
             devices modify the shared `locations` dictionary without a lock,
             which can lead to data corruption if multiple devices are set up
             concurrently.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.scripts_done = Event()
        self.my_lock = Lock() # A per-device lock.
        
        # Shared resources, intended to be set up by device 0.
        self.locations = None
        self.barrier = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources. Master (device 0) creates the resources,
        while other devices get a reference to them.
        """
        # Master device setup.
        if self.device_id is 0:
            self.locations = {}
            self.barrier = ReusableBarrier(len(devices));
            # Discovers locations only from its own data.
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass
                else:
                    self.locations[loc] = Lock()
        
        # Other devices setup.
        else:
            self.locations = devices[0].locations
            self.barrier = devices[0].get_barrier()
            # FLAW: Concurrent modification of the shared `locations` dict.
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass
                else:
                    self.locations[loc] = Lock()

        # Start the main orchestrator thread for this device.
        self.thread = DeviceThread(self, self.barrier, self.locations)
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script for the upcoming time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_done.set()

    def get_data(self, location):
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

    def get_barrier(self):
        return self.barrier

class DeviceThread(Thread):
    """
    Orchestrator thread for a device, using a "thread-per-task" model.
    """

    def __init__(self, device, barrier, locations):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.locations = locations

    def run(self):
        """Main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal.

            # 1. Wait for scripts to be assigned for the time step.
            self.device.scripts_done.wait()
            self.device.scripts_done.clear()
            
            # 2. Create and start a new worker thread for each script.
            workers = []
            for (script, location) in self.device.scripts:
                w = Worker(self.device, neighbours, script, location, self.locations)
                workers.append(w)
                w.start()

            # 3. Wait for all worker threads for this time step to complete.
            for w in workers:
                w.join()

            # 4. Synchronize with all other devices before the next time step.
            self.barrier.wait()

class Worker(Thread):
    """
    A short-lived worker thread that executes a single script.

    @warning The locking strategy in `run` is prone to deadlock. It acquires a
             location-specific lock and then, while holding it, tries to acquire
             device-specific locks. If two workers acquire different location
             locks but then need to acquire the same device lock that the other
             worker's device holds, they can deadlock.
    """
    def __init__(self, device, neighbours, script, location, locations):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.locations = locations

    def run(self):
        # 1. Acquire the lock for the specific data location.
        self.locations[self.location].acquire()
        
        script_data = [] 
        # 2. Aggregate data. This involves acquiring more locks.
        for device in self.neighbours:
            # Acquiring a second lock while holding the first can cause deadlock.
            device.my_lock.acquire()
            data = device.get_data(self.location)
            device.my_lock.release()
            if data is not None:
                script_data.append(data)
        
        self.device.my_lock.acquire()
        data = self.device.get_data(self.location)
        self.device.my_lock.release()
        if data is not None:
            script_data.append(data)

        # 3. Compute and disseminate results.
        if script_data != []:
            result = self.script.run(script_data)
            
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()
            
        # 4. Release the initial location lock.
        self.locations[self.location].release()



class ReusableBarrier():
    """
    A synchronization barrier implemented with a Condition variable.

    @warning This is a non-reusable, single-phase barrier. It is NOT safe for use
             inside a loop. It is vulnerable to a "stray wakeup" race condition
             where a fast thread can loop and re-enter `wait()` before slow
             threads have exited, corrupting the barrier's state.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
 
    def wait(self):
        self.cond.acquire()                      
        self.count_threads -= 1;
        if self.count_threads == 0:
            self.cond.notify_all()               
            self.count_threads = self.num_threads    
        else:
            self.cond.wait();                    
        self.cond.release();                     
                     