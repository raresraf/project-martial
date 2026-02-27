from threading import Event, Thread, Lock, Semaphore

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
        """Blocks the calling thread until all threads have reached the barrier."""
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
    Represents a device node which creates new worker threads for each script
    in a timepoint, up to a fixed limit.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

        self.neighbours = []
        self.alldevices = []
        self.barrier = None
        self.threads = []
        self.threads_number = 8 # Max number of worker threads to spawn per timepoint.
        self.locks = [None] * 100 # Fixed-size list for location-based locks.
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Propagates a shared barrier instance to all devices and stores a
        list of all devices for later use.
        """
        if self.barrier is None:
            barrier = ReusableBarrierSem(len(devices))
            self.barrier = barrier
            for d in devices:
                if d.barrier is None:
                    d.barrier = barrier
        for device in devices:
            if device is not None:
                self.alldevices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script and sets up a shared lock for its location.
        The lock management is complex: it searches all other devices for an
        existing lock for the location before creating a new one.
        """
        if script is not None:
            self.scripts.append((script, location))
            no_lock_for_location = False
            # Inefficiently search for an existing lock on other devices.
            for device in self.alldevices:
                if device.locks[location] is not None:
                    self.locks[location] = device.locks[location]
                    no_lock_for_location = True
                    break
            if not no_lock_for_location:
                self.locks[location] = Lock()
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
        """Waits for the device's master thread to complete."""
        self.thread.join()

class MyThread(Thread):
    """
    A worker thread that is created to execute a single script.
    """
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script: locks the location, gathers data, runs the script,
        broadcasts the result, and releases the lock.
        """
        with self.device.locks[self.location]:
            script_data = []
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            # Gather data from the parent device.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data:
                result = self.script.run(script_data)
                # Broadcast result to all devices.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)

class DeviceThread(Thread):
    """
    The master thread for a device. It spawns, starts, and joins new worker
    threads for each timepoint's workload.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main loop: waits for scripts, spawns workers, waits for them to complete,
        and then synchronizes with other devices.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal from supervisor.

            self.device.timepoint_done.wait() # Wait for all scripts to be assigned.
            self.device.neighbours = neighbours

            count = 0
            # Create a new worker thread for each script, up to a limit of 8.
            # Note: This is a bug. If there are more than 8 scripts, they are ignored.
            for (script, location) in self.device.scripts:
                if count >= self.device.threads_number:
                    break
                count = count + 1
                thread = MyThread(self.device, location, script, neighbours)
                self.device.threads.append(thread)

            # Start and then immediately wait for all workers to complete.
            for thread in self.device.threads:
                thread.start()
            for thread in self.device.threads:
                thread.join()
            self.device.threads = [] # Clear the list of finished threads.

            self.device.timepoint_done.clear()
            self.device.barrier.wait() # Synchronize with other devices.
