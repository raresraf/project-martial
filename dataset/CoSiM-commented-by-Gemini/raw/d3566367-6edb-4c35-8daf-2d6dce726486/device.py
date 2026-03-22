"""
This module implements a highly complex and likely flawed distributed device
simulation.

Key architectural features:
- Each device has its own pool of worker threads (`DeviceThread`).
- A "master" device (device 0) creates and distributes global resources: a
  `timepoint_barrier` for inter-device synchronization and a list of `locks`.
- Only a "master" worker (`thread_id == 0`) from each device's pool
  participates in the global `timepoint_barrier`.
- Workers within a single device use their own `internal_barrier`.
- The `DeviceThread.run` method contains duplicated code blocks, which appears
  to be a significant logical bug.
- Locking for data access is encapsulated within the `get_data` and `set_data`
  methods.

Note: This script uses Python 2 syntax.
"""



from threading import Event, Thread, Lock, Semaphore, RLock




class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads.
    It uses a two-phase protocol based on semaphores.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all `num_threads` have called this method."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a device node, managing its own pool of worker threads and
    participating in a global synchronization scheme.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.scripts = []
        self.timepoint_done = Event()

        self.threads = []
        self.no_threads = 8
        
        # --- Synchronization Primitives ---
        self.timepoint_barrier = None # Global barrier for master workers.
        self.locks = []               # Global list of location locks.
        self.scripts_lock = Lock()
        self.internal_barrier = ReusableBarrier(self.no_threads) # For own workers.
        self.end_timepoint = Lock()
        
        self.last_scripts = []

        if device_id == 0:
            self.init_event = Event() # Used by master to signal setup completion.

        # Create this device's internal worker thread pool.
        for thread_id in range(self.no_threads):
            thread = DeviceThread(self, thread_id)
            self.threads.append(thread)


    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources using a master-worker pattern and starts threads.
        Device 0 creates and distributes the global barrier and lock list.
        """
        if self.device_id == 0:
            # Master device creates the global barrier for one thread per device.
            self.timepoint_barrier = ReusableBarrier(len(devices))

            no_location = 0
            for device in devices:
                no_location += len(device.sensor_data)

            # Create a global list of re-entrant locks for all possible locations.
            self.locks = [RLock() for _ in range(no_location)]
            self.init_event.set() # Signal that global resources are ready.
        else:
            # Other devices wait for the master and copy references.
            for device in devices:
                if device.device_id == 0:
                    device.init_event.wait()
                    self.timepoint_barrier = device.timepoint_barrier
                    self.locks = device.locks
        
        # All devices start their worker threads after setup.
        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        """Assigns a script for the current time step."""
        if script is not None:
            with self.scripts_lock:
                self.scripts.append((script, location))
        else:
            self.end_timepoint.acquire()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safely gets data for a location by acquiring a global lock.
        """
        ret = None
        if location in self.sensor_data:
            self.locks[location].acquire()
            ret = self.sensor_data[location]
        return ret

    def set_data(self, location, data):
        """
        Thread-safely sets data for a location and releases the global lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread in a device's pool with complex, confusing logic.
    Only the master worker (thread_id==0) participates in global sync.
    """
    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device %d, Thread %d" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main execution loop, containing confusing and duplicated logic."""
        if self.thread_id == 0:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device in self.device.neighbours:
                self.device.neighbours.remove(self.device)

        while True:
            # All workers in a device sync with each other.
            self.device.internal_barrier.wait()
            
            neighbours = self.device.neighbours
            if neighbours is None:
                break

            # --- Work-Stealing Loop 1 ---
            # All workers in the pool compete to process scripts.
            while len(self.device.scripts) != 0:
                script = None
                with self.device.scripts_lock:
                    if len(self.device.scripts) != 0:
                        script, location = self.device.scripts.pop(0)
                        self.device.last_scripts.append((script, location))
                
                if script:
                    # ... (script execution logic) ...
                    script_data = []
                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None: script_data.append(data)
                    data = self.device.get_data(location)
                    if data is not None: script_data.append(data)
                    if script_data:
                        result = script.run(script_data)
                        for device in neighbours: device.set_data(location, result)
                        self.device.set_data(location, result)

            # All workers wait for the supervisor to signal the end of the script
            # assignment phase for the current timepoint.
            self.device.timepoint_done.wait()

            # BUG: This is a duplicated block of code. The `scripts` list would have
            # been emptied by the first loop, so this second loop will do nothing.
            # This appears to be a copy-paste error.
            while len(self.device.scripts) != 0:
                script = None
                with self.device.scripts_lock:
                    if len(self.device.scripts) != 0:
                        script, location = self.device.scripts.pop(0)
                        self.device.last_scripts.append((script, location))

                if script:
                    script_data = []
                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None: script_data.append(data)
                    data = self.device.get_data(location)
                    if data is not None: script_data.append(data)
                    if script_data:
                        result = script.run(script_data)
                        for device in neighbours: device.set_data(location, result)
                        self.device.set_data(location, result)


            # All workers in a device sync again before the global barrier.
            self.device.internal_barrier.wait()

            # Only the master worker participates in the global timepoint barrier.
            if self.thread_id == 0:
                self.device.timepoint_barrier.wait()
                
                # The master worker prepares for the next time step.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                if self.device.neighbours:
                    if self.device in self.device.neighbours:
                        self.device.neighbours.remove(self.device)

                self.device.timepoint_done.clear()
                self.device.end_timepoint.release()
