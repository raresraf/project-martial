from threading import Lock, Event, Thread, Semaphore

class ReusableBarrierSem():
    """
    A reusable barrier for synchronizing a fixed number of threads, implemented
    using semaphores and a two-phase protocol.
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
        """The first synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """The second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a device node. This implementation uses class-level static variables
    for shared state and a pool of identical DeviceThreads that perform all work.
    """
    # Class-level variables shared by all Device instances.
    location_locks = []
    barrier = None
    nr_t = 8 # Number of threads per device.

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.neighbours_event = Event()
        self.threads = []
        
        # Create and start a pool of worker threads.
        for i in xrange(Device.nr_t): # Note: uses Python 2 `xrange`.
            self.threads.append(DeviceThread(self, i))
        for i in xrange(Device.nr_t):
            self.threads[i].start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes a single barrier shared by all threads of all devices.
        """
        Device.barrier = ReusableBarrierSem(Device.nr_t * len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script and lazily creates a shared lock for its location.
        
        WARNING: This method has a race condition. Multiple threads could check
        for the existence of a location lock and try to create it simultaneously.
        """
        if location not in [elem[0] for elem in Device.location_locks]:
            Device.location_locks.append((location, Lock()))

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all of this device's threads to complete."""
        for i in xrange(Device.nr_t):
            self.threads[i].join()

class DeviceThread(Thread):
    """
    A worker thread for a device. Each device runs a pool of these threads.
    The threads handle coordination, work distribution, and execution.
    """
    def __init__(self, device, index):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.index = index # Unique index (0-7) for this thread within its device.
        self.neighbours = None

    def run(self):
        """Main execution loop for the worker thread."""
        while True:
            # Thread 0 of each device is responsible for getting the neighbor list.
            if self.index == 0:
                self.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_event.set() # Signal other threads on this device.
            else:
                self.device.neighbours_event.wait() # Wait for thread 0.
                self.neighbours = self.device.threads[0].neighbours
            
            if self.neighbours is None:
                break # Shutdown signal.

            # All threads wait for the supervisor to assign all scripts.
            self.device.timepoint_done.wait()

            # Statically distribute scripts among threads based on index.
            # e.g., thread 0 takes scripts 0, 8, 16...
            for j in range(self.index, len(self.device.scripts), Device.nr_t):
                location = self.device.scripts[j][1]
                script = self.device.scripts[j][0]
                
                # Inefficiently find the lock for the current location.
                lock_to_use = None
                for loc, lock in Device.location_locks:
                    if loc == location:
                        lock_to_use = lock
                        break
                
                with lock_to_use:
                    script_data = []
                    # Gather data from neighbors.
                    for device in self.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    # Gather data from own device.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    # Execute script and broadcast results.
                    if script_data:
                        result = script.run(script_data)
                        for device in self.neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)

            # All threads from all devices synchronize here after finishing work.
            Device.barrier.wait()
            
            # Thread 0 on each device resets the events for the next timepoint.
            if self.index == 0:
                self.device.timepoint_done.clear()
                self.device.neighbours_event.clear()
                
            # A second barrier to ensure all threads see the cleared events
            # before the next iteration begins.
            Device.barrier.wait()
