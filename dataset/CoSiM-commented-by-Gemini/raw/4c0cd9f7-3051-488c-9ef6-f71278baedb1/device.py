"""
This module implements a highly complex, multi-layered simulation framework
for a network of devices.

The architecture is notable for its use of both class-level (static) and
instance-level synchronization primitives. It employs a two-phase semaphore-based
reusable barrier for global synchronization among devices. Each device, in turn,
manages a pool of its own worker threads, coordinated by a dedicated controller
thread and another layer of barriers and events.
"""

from threading import Event, Thread, Lock, Semaphore, Lock

class ReusableBarrier(object):
    """
    A reusable barrier implemented with two semaphores and a lock.

    This is a classic two-phase barrier implementation. The first phase ensures
    all threads have arrived at the barrier, and the second phase ensures all
    threads have been released before the barrier can be used again.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0) # Controls entry to the first phase
        self.threads_sem2 = Semaphore(0) # Controls entry to the second phase

    def wait(self):
        """Causes a thread to wait at the barrier, proceeding in two phases."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread arrived, release all threads for the first phase.
                for _ in range(self.num_threads):


                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use.

        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread arrived, release all for the second phase.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use.

        self.threads_sem2.acquire()



class Device(object):
    """
    Represents a device node using a complex internal threading model.

    This class uses shared, class-level static resources for global
    synchronization (`bar1`, `event1`, `locck`) and instance-level resources
    for coordinating its own internal worker threads.
    """
    
    # Class-level (static) variables shared by all device instances.
    bar1 = ReusableBarrier(1) # Global barrier for all devices.
    event1 = Event()          # Global event to signal initial setup completion.
    locck = []                # Global list of locks for data locations.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and its internal controller and worker threads.
        """
        
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = []

        
        # A list of events for signaling between this device's controller
        # thread and its worker threads.
        self.event = []
        for _ in xrange(11):
            self.event.append(Event())

        
        self.nr_threads_device = 8
        
        self.nr_thread_atribuire = 0
        
        
        # A barrier for this device's own controller and worker threads.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1)

        
        # The controller thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

        
        # The pool of worker threads for this device.
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources.
        
        Intended to be run by device 0 to set up the class-level static
        barrier and locks.
        """
        
        self.devices = devices
        
        if self.device_id == 0:
            for _ in xrange(30):
                Device.locck.append(Lock())
            Device.bar1 = ReusableBarrier(len(devices))
            
            # Signal all devices that setup is complete.
            Device.event1.set()

    def assign_script(self, script, location):
        """
        Assigns a script to one of the device's worker threads in round-robin.
        """
        
        if script is not None:
            # The script is assigned to a specific worker thread.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%
            self.nr_threads_device
        else:
            # A None script signals the end of the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. Not internally thread-safe."""
        
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data. Not internally thread-safe."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the controller and all worker threads."""
        
        self.thread.join()
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    """
    The controller thread for a single Device.

    Its primary role is to manage synchronization between simulation steps,
    signaling its worker threads to start processing and then synchronizing
    with them and all other devices.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.contor = 0 # Counter for the event list.

    def run(self):
        # Wait for the global setup signal from device 0.
        Device.event1.wait()

        while True:
            
            # Get the current list of neighbors for this simulation step.
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                # Shutdown signal received.
                self.device.event[self.contor].set()
                break

            
            # Wait for the supervisor to signal the end of the current timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            # Signal this device's worker threads to start processing.
            self.device.event[self.contor].set()
            self.contor += 1

            
            
            # Wait for this device's own workers to finish their tasks.
            self.device.bar_threads_device.wait()

            
            
            # Wait for all other devices to finish their steps.
            Device.bar1.wait()

class ThreadAux(Thread):
    """
    A worker thread that executes assigned scripts.
    """
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device
        self.script_loc = {} # Holds scripts specifically assigned to this worker.
        self.contor = 0

    def run(self):
        while True:
            
            
            # Wait for the signal from the controller thread to begin work.
            self.device.event[self.contor].wait()
            self.contor += 1

            
            neigh = self.device.thread.neighbours
            if neigh is None:
                # Shutdown signal.
                break

            # Process all scripts assigned to this worker.
            for script in self.script_loc:
                location = self.script_loc[script]
                
                # Acquire a global lock for the specific data location.
                Device.locck[location].acquire()
                script_data = []

                # Flawed data gathering: only data from the last neighbor is used.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Run the script and distribute the result.
                    result = script.run(script_data)
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                
                Device.locck[location].release()

            
            # Synchronize with the controller and other local workers.
            self.device.bar_threads_device.wait()