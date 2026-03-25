from threading import enumerate, Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    An attempted implementation of a reusable barrier for a fixed number of
    threads. It uses two semaphores to create two distinct synchronization phases.

    @note This implementation contains a bug in the phase reset logic. In
    `phase1`, it resets the counter for `phase2`, and vice-versa. This can
    lead to deadlocks or race conditions if the barrier is used multiple times,
    as a thread entering a new `phase1` might do so before a slow thread has
    left the previous `phase2`, causing the counter to be reset prematurely.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()       
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """Blocks the thread until all participating threads have called wait."""
        self.phase1()
        self.phase2()

    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            # Buggy: Resets the counter for the *other* phase.
            self.count_threads2 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            # Buggy: Resets the counter for the *other* phase.
            self.count_threads1 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a device in a simulated distributed system. Each device runs
    a main thread which in turn spawns worker threads for script execution.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that all scripts for a timepoint have been assigned.
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for synchronization among all devices.
        Device 0 creates the barrier, and other devices get a reference to it.
        @note The lookup for device 0 results in O(N^2) complexity for setup.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    self.barrier = device.barrier
                    break

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script triggers processing.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal the main device thread to start processing the collected scripts.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data
            
    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class Node(Thread):
    """
    A worker thread to execute a single script.
    """
    def __init__(self, script, script_data):
        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        self.result = None
         
    def run(self):
        """Executes the script and stores the result."""
        self.result = self.script.run(self.script_data)

    def join(self):
        """
        Waits for the thread to complete and returns the script and its result.
        @note Overriding join() to return a value is unconventional.
        """
        Thread.join(self)
        return (self.script, self.result)


class DeviceThread(Thread):
    """
    The main thread for a device, orchestrating the work for each timepoint.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main loop: waits for scripts, executes them in parallel, updates data,
        and synchronizes with other devices.
        """
        while True:
            # Pre-condition: Get neighbors for the new timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Termination signal.

            thread_list=[]
            scripts_result = {}
            scripts_data = {}
            
            # Block until all scripts for the timepoint are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            # Block Logic: Data Gathering and Concurrent Execution
            # 1. Gather data for each script.
            for (script, location) in self.device.scripts:
                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                scripts_data[script] = script_data
                
                # 2. If data is available, create a worker thread for the script.
                if script_data:
                    nod = Node(script, script_data)
                    thread_list.append(nod)
            
            # 3. Start all worker threads.
            for nod in thread_list:
                nod.start()

            # 4. Collect results from all worker threads.
            for nod in thread_list:
                key, value = nod.join()
                scripts_result[key] = value

            # Block Logic: Data Dissemination
            # Update data on self and all neighbors with the script results.
            # @note If multiple scripts modify the same location, the last one
            # in the list determines the final value.
            for (script, location) in self.device.scripts:
                if script in scripts_result:
                    result = scripts_result[script]
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
            
            # Invariant: Synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()