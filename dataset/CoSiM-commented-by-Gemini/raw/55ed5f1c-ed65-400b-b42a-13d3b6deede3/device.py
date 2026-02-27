from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads, implemented
    using semaphores and a two-phase protocol.
    """
    def __init__(self, num_threads):
        """Initializes the barrier for a specific number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads reach the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a device node which uses a master-worker threading model to process scripts.
    It coordinates with other devices using a complex, multi-layered synchronization scheme.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device, its master thread, and its pool of worker threads.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        # A dictionary of locks, one per device, shared across all devices.
        self.lock = {}
        self.supervisor = supervisor
        self.scripts = []
        # Events for signaling between supervisor, master, and workers.
        self.script_received = Event()
        self.timepoint_done = Event()
        self.setup_done = Event()
        self.terminate = Event()
        self.neighbours = []

        self.barrier = None # Inter-device barrier for all master threads.
        # Intra-device barrier for the master and its 8 workers.
        self.threads_barrier = ReusableBarrier(9)
        
        # The master thread that orchestrates work for this device.
        self.master = DeviceThread(self, self.terminate, self.barrier, self.threads_barrier, self.setup_done)
        self.master.start()

        # The pool of worker threads that execute scripts.
        self.threads = []
        for _ in range(8):
            thread = Worker(self.master, self.terminate, self.threads_barrier)
            self.threads.append(thread)
            thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Centralized setup run by device 0 to create and distribute shared
        synchronization objects (barrier and locks) to all other devices.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for dev in devices:
                self.lock[dev] = Lock() # Create a lock for each device.
            for dev in devices:
                if dev.device_id != 0:
                    # Share the barrier and lock dictionary with other devices.
                    dev.barrier = self.barrier
                    dev.lock = self.lock
                    dev.setup_done.set()
            self.setup_done.set() # Signal that device 0 is done.

    def assign_script(self, script, location):
        """Assigns a script to the device or signals the end of a timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by terminating and joining all threads."""
        self.terminate.set()
        for i in range(8):
            # Unblock any waiting worker threads so they can terminate.
            self.threads[i].script_received.set()
            self.threads[i].join()
        self.master.join()


class DeviceThread(Thread):
    """
    The master thread for a single Device. It coordinates with other devices
    and distributes work to its own pool of worker threads.
    """
    def __init__(self, device, terminate, barrier, threads_barrier, setup_done):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = []
        self.terminate = terminate
        self.barrier = barrier # Inter-device barrier.
        self.threads_barrier = threads_barrier # Intra-device barrier.
        self.setup_done = setup_done

    def run(self):
        """Main execution loop with complex multi-stage synchronization."""
        self.setup_done.wait() # 1. Wait for global setup to complete.
        self.device.barrier.wait() # 2. Initial sync with all other devices.

        while True:
            self.device.barrier.wait() # 3. Sync at the start of a new timepoint.
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break # Supervisor signals shutdown.

            self.device.timepoint_done.wait() # 4. Wait for all scripts for this timepoint.
            self.device.timepoint_done.clear()
            self.device.barrier.wait() # 5. Sync to ensure all devices have received scripts.

            # Distribute scripts among the 8 worker threads.
            scripts = [[] for _ in range(8)]
            for i, script in enumerate(self.device.scripts):
                scripts[i % 8].append(script)

            # Signal workers to start processing.
            for i in range(8):
                self.device.threads[i].scripts = scripts[i]
                self.device.threads[i].script_received.set()

            # 6. Wait for all local workers to finish their tasks for this timepoint.
            if not self.terminate.is_set():
                self.threads_barrier.wait()


class Worker(Thread):
    """A worker thread that executes a subset of a device's scripts."""
    def __init__(self, master, terminate, barrier):
        Thread.__init__(self)
        self.master = master # Reference to the parent DeviceThread.
        self.script_received = Event()
        self.terminate = terminate
        self.scripts = []
        self.barrier = barrier # The intra-device barrier.

    @staticmethod
    def append_data(device, location, script_data):
        """Safely reads data from a device using its specific lock."""
        with device.lock[device]:
            data = device.get_data(location)
        if data is not None:
            script_data.append(data)

    @staticmethod
    def set_data(device, location, result):
        """Safely writes data to a device using its specific lock."""
        with device.lock[device]:
            device.set_data(location, result)

    def run(self):
        """Main loop: waits for work, executes scripts, and synchronizes."""
        while True:
            self.script_received.wait() # 1. Wait for the master to assign scripts.
            self.script_received.clear()

            if self.terminate.is_set():
                break
            
            if self.scripts is not None:
                for (script, location) in self.scripts:
                    # Gather data from neighbors and the parent device.
                    script_data = []
                    if self.master.neighbours is not None:
                        for device in self.master.neighbours:
                            self.append_data(device, location, script_data)
                    self.append_data(self.master.device, location, script_data)

                    # Execute script and broadcast results.
                    if script_data:
                        result = script.run(script_data)
                        if self.master.neighbours is not None:
                            for device in self.master.neighbours:
                                self.set_data(device, location, result)
                        self.set_data(self.master.device, location, result)
            
            # 2. Signal completion to the master by waiting at the intra-device barrier.
            self.barrier.wait()
