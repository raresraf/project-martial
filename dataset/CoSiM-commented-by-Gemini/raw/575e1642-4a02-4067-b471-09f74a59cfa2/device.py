from threading import Event, Lock, Thread, RLock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier for synchronizing a fixed number of threads, implemented
    using semaphores and a two-phase protocol.
    """
    def __init__(self, num_threads):
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
    Represents a device node that processes scripts in parallel using a
    spawn-and-join threading model for each computational step.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # Re-entrant locks allow a thread to acquire the same lock multiple times.
        self.lock = RLock()
        self.script_lock = RLock()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier. Device 0 is responsible
        for creating the barrier, which is then assigned to all other devices.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """Atomically assigns a script to the device."""
        with self.script_lock:
            if script is not None:
                self.scripts.append((script, location))
                self.script_received.set()
            else:
                self.timepoint_done.set()

    def get_data(self, location):
        """Thread-safely retrieves data from a specific sensor location."""
        with self.lock:
            return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Thread-safely updates data at a specific sensor location."""
        with self.lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's master thread to complete."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The master thread for a device. For each timepoint, it spawns a new pool
    of worker threads to execute scripts.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main loop: waits for scripts, spawns workers, waits for them, and
        then synchronizes with other devices.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal.
            
            # Add self to neighbors list to process own data.
            neighbours.append(self.device)

            self.device.timepoint_done.wait() # Wait for all scripts for this timepoint.

            # Create a new set of threads for the current timepoint's workload.
            num_threads = 8
            threads = [Thread(target=self.concurrent_work,
                              args=(neighbours, i, num_threads)) for i in range(num_threads)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join() # Wait for all workers to complete.

            # After local work is done, synchronize with all other devices.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()

    def concurrent_work(self, neighbours, thread_id, num_threads):
        """
        The logic executed by each worker thread. It processes a subset of
        the device's scripts.
        """
        # Get the subset of scripts assigned to this thread.
        for (script, location) in self.keep_assigned(self.device.scripts, thread_id, num_threads):
            script_data = []
            
            # Gather data from all neighbors (including self).
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            if script_data:
                result = script.run(script_data)

                # Aggregate results by taking the max and broadcasting it.
                for device in neighbours:
                    # get_data acquires a lock, so this is a critical section.
                    res = max(result, device.get_data(location))
                    # set_data releases the lock.
                    device.set_data(location, res)

    def keep_assigned(self, scripts, thread_id, num_threads):
        """
        Implements static work distribution, assigning a strided subset of
        scripts to each worker thread based on its ID.
        """
        assigned_scripts = []
        for i, script in enumerate(scripts):
            if i % num_threads == thread_id:
                assigned_scripts.append(script)
        return assigned_scripts
