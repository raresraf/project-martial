```
Models a device in a distributed simulation using a hierarchical, two-level
synchronization model and round-robin work distribution.

This module defines a simulation where each device has a master thread and a
fixed pool of worker threads. A global barrier synchronizes the master threads
of all devices. Within each device, a local barrier and event system
synchronize the master with its workers. Work is assigned to workers in a
round-robin fashion.

Classes:
    ReusableBarrier: A custom two-phase thread barrier.
    Device: Manages device state and orchestrates its master/worker threads.
    DeviceThread: The master/coordinator thread for a single device.
    ThreadAux: A worker thread that executes a subset of scripts.
```

from threading import Lock, Event, Thread, Condition, Semaphore

class ReusableBarrier(object):
    """
    A custom, two-phase implementation of a reusable barrier.
    
    Threads calling wait() must pass through two distinct synchronization phases,
    ensuring that no thread can start a new 'wait' cycle until all threads
    have exited the previous one.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads reach the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """Executes the first synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """Executes the second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()



class Device(object):
    """
    Represents a device, its state, and its internal master-worker architecture.
    """
    # Class attributes are used for resources shared across all device instances.
    bar1 = ReusableBarrier(1) # Global barrier for master threads.
    event1 = Event() # Global event to signal end of setup.
    locck = [] # Global list of locks for data locations.

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = []
        
        # A list of events for local master-worker synchronization.
        self.event = []
        for _ in xrange(11):
            self.event.append(Event())
        
        self.nr_threads_device = 8
        self.nr_thread_atribuire = 0 # Index for round-robin assignment.
        
        # A local barrier for synchronizing the master and its workers.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1)

        self.thread = DeviceThread(self) # The master thread.
        self.thread.start()

        self.threads = [] # The pool of worker threads.
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs a one-time, centralized setup of globally shared resources.
        """
        self.devices = devices
        if self.device_id == 0:
            # Initialize shared locks for all possible data locations.
            for _ in xrange(30):
                Device.locck.append(Lock())
            # Initialize the global barrier for all device master threads.
            Device.bar1 = ReusableBarrier(len(devices))
            
            # Signal that setup is complete.
            Device.event1.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the next worker thread in a round-robin fashion.
        """
        if script is not None:
            # Add the script to the personal workload of a specific worker thread.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%
            self.nr_threads_device
        else:
            # A None script signals the end of script assignments for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its threads."""
        self.thread.join()
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    """The master/coordinator thread for a single device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.contor = 0

    def run(self):
        """The main control loop for the master thread."""
        Device.event1.wait() # Wait for global setup to finish.

        while True:
            # This thread is solely responsible for fetching the neighbor list.
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                # Signal shutdown to worker threads.
                self.device.event[self.contor].set()
                break

            # Wait for the supervisor to assign all scripts for the timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Signal this device's worker threads to start processing.
            self.device.event[self.contor].set()
            self.contor += 1
            
            # Wait at the local barrier until all its workers have finished.
            self.device.bar_threads_device.wait()
            
            # Wait at the global barrier for all other devices' masters to finish.
            Device.bar1.wait()

class ThreadAux(Thread):
    """A worker thread that executes a statically assigned set of scripts."""
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device
        self.script_loc = {} # Holds this thread's scripts for the current timepoint.
        self.contor = 0

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            
            # Wait for the signal from the master thread to begin processing.
            self.device.event[self.contor].wait()
            self.contor += 1

            neigh = self.device.thread.neighbours
            if neigh is None:
                break # Shutdown signal received.

            # Process all scripts assigned to this specific thread.
            for script in self.script_loc:
                location = self.script_loc[script]
                
                Device.locck[location].acquire()
                script_data = []

                # Gather data from neighbors.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from the parent device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute script and propagate results.
                if script_data != []:
                    result = script.run(script_data)
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                Device.locck[location].release()

            # Signal completion to the master by waiting at the local barrier.
            self.device.bar_threads_device.wait()

