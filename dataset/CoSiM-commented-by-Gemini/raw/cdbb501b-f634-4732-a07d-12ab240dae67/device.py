"""
This module simulates a distributed network of devices that process sensor data.

This is a complex, multi-threaded implementation featuring a two-level
synchronization hierarchy.

Architecture:
1. Each `Device` maintains its own persistent pool of worker threads (`ThreadAux`).
2. A master thread (`DeviceThread`) on each device acts as a synchronizer. It
   does not perform computations itself but orchestrates the worker threads.
3. Worker threads wait for a signal from their master thread to begin processing
   scripts for a given time step.
4. A local barrier (`bar_threads_device`) synchronizes the worker threads with
   their master thread, ensuring all local computations for a time step are done.
5. A global, static barrier (`Device.bar1`) synchronizes all `Device` instances
   together, ensuring all devices complete a time step before any can move on.
6. Data consistency for specific locations is maintained using a global, static
   list of `Lock` objects (`Device.locck`).
"""



from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A custom, two-phase, reusable barrier for thread synchronization.

    This implementation uses two semaphores to prevent threads from one barrier
    "generation" from proceeding before all threads from the previous generation
    have been released.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock()
        # Semaphores to control the two phases of the barrier.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all threads have reached the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread arrives, releases all waiting threads for this phase.
                for _ in range(self.num_threads):


                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase to prevent race conditions for reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread arrives, releases all waiting threads for this phase.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()



class Device(object):
    """
    Represents a device node in the network.

    Manages a pool of persistent worker threads (`ThreadAux`) and a master
    synchronization thread (`DeviceThread`). Shared concurrency primitives are
    stored as static class variables.
    """
    
    # --- Static variables shared across all Device instances ---
    bar1 = ReusableBarrier(1) # Global barrier for synchronizing all devices.
    event1 = Event()          # Global event to start the simulation.
    locck = []                # Global list of location-based locks.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device, its master thread, and its worker thread pool.
        """
        
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = []

        
        # A list of events for master-worker signaling within this device.
        self.event = []
        for _ in xrange(11):
            self.event.append(Event())

        
        self.nr_threads_device = 8
        
        self.nr_thread_atribuire = 0 # For round-robin script assignment.
        
        
        # Local barrier for this device's master and worker threads.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1)

        
        # The master synchronization thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

        
        # The persistent pool of worker threads for this device.
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared static concurrency objects."""
        self.devices = devices
        
        if self.device_id == 0:
            # Device 0 is the master, setting up static shared resources.
            for _ in xrange(30):
                Device.locck.append(Lock())
            Device.bar1 = ReusableBarrier(len(devices))
            
            Device.event1.set() # Signal all devices to start their main loops.

    def assign_script(self, script, location):
        """Assigns a script to one of the worker threads in a round-robin fashion."""
        if script is not None:
            # Assign script to the next available worker thread's dictionary.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%
            self.nr_threads_device
        else:
            # A `None` script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location. Not thread-safe on its own;
        relies on external locking (e.g., in the worker thread).
        """
        return self.sensor_data[location] if location in 
        self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location. Not thread-safe on its own;
        relies on external locking.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully shuts down the master and worker threads for this device."""
        self.thread.join()
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    """
    Master synchronization thread for a single device.
    Orchestrates its worker threads (`ThreadAux`) and synchronizes with other devices.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.contor = 0

    def run(self):
        """Main simulation loop for synchronization."""
        Device.event1.wait() # Wait for the global start signal.

        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                # End of simulation: signal workers to terminate and then exit.
                self.device.event[self.contor].set()
                break

            
            # 1. Wait for scripts for the current timepoint to be assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            # 2. Signal this device's worker threads to start processing.
            self.device.event[self.contor].set()
            self.contor += 1

            
            
            # 3. Wait on the local barrier until all its workers have finished.
            self.device.bar_threads_device.wait()

            
            
            # 4. Wait on the global barrier until all other devices are finished.
            Device.bar1.wait()

class ThreadAux(Thread):
    """
    A persistent worker thread that executes assigned scripts.
    """
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device
        self.script_loc = {} # Scripts assigned to this specific thread.
        self.contor = 0

    def run(self):
        """Main loop for the worker thread."""
        while True:
            
            
            # 1. Wait for the signal from the master DeviceThread to start work.
            self.device.event[self.contor].wait()
            self.contor += 1

            
            neigh = self.device.thread.neighbours
            if neigh is None:
                # End of simulation signal received.
                break

            for script in self.script_loc:
                location = self.script_loc[script]
                
                
                # Acquire the global lock for this location.
                Device.locck[location].acquire()
                script_data = []

                # Gather data from self and neighbours.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Run script and update data.
                    result = script.run(script_data)
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                
                # Release the global lock for this location.
                Device.locck[location].release()

            
            # 2. Wait on the local barrier to signal completion to the master thread.
            self.device.bar_threads_device.wait()
