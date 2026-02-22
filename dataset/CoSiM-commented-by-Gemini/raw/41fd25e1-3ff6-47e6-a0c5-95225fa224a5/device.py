"""
This file models a distributed sensor network simulation.

This implementation uses a "thread-per-task" model, where the main `DeviceThread`
spawns a new `MyThread` for each computational script. It features a unique
and complex architecture where shared resources (like the main barrier and data
locks) are centralized on a single device instance (`devices[0]`) which all
other devices must reference.

Classes:
    ReusableBarrier: A correct two-phase reusable barrier for synchronization.
    Device: Represents a node, holding state and a list of all other devices.
    DeviceThread: Orchestrator that creates and manages worker threads in batches.
    MyThread: A short-lived worker thread that executes a single task.
"""


from threading import Event, Thread, Lock, Semaphore, RLock


class ReusableBarrier(object):
    """A reusable barrier for synchronizing a fixed number of threads.
    
    This is a correct, two-phase implementation that is safe for use in loops.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        # Counters are stored in a list to be mutable across method calls.
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
        """Executes one phase of the two-phase barrier protocol."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                i = 0
                while i < self.num_threads:
                    threads_sem.release()
                    i += 1
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """Represents a single device node in the network.
    
    @note This design centralizes shared resources. The main barrier and the
          dictionary of location locks are stored only on the `devices[0]`
          instance, creating a single point of reference and a fragile
          architecture.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.devices = [] # Each device will hold a list of all other devices.
        self.script_received = Event()
        self.timepoint_done = Event() # Not used in this implementation's logic.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the centralized shared resources on device 0."""
        for device in devices:
            self.devices.append(device)
        # The barrier and locks are only stored on the first device in the list.
        self.devices[0].barrier = ReusableBarrier(len(self.devices))
        self.devices[0].locations_lock = {}

    def assign_script(self, script, location):
        """Assigns a script for the upcoming time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
            # Signal that all scripts have been received for this time step.
            self.script_received.set()

    def get_data(self, location):
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    Orchestrates the execution of tasks for a single device per time step.

    @warning The thread management logic in `run` is highly complex and
             inefficient. It manually starts, joins, and removes threads from a
             list in fixed-size batches. A simpler design would be to start
             all threads, then join all threads in two separate loops.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.num_threads = 8 # Size of the execution batches.

    def run(self):
        while True:
            threads = []
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal.
            
            # 1. Wait until all scripts for the time step have been assigned.
            self.device.script_received.wait()

            # 2. Create a new worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                thread = MyThread(self, script, location, neighbours)
                threads.append(thread)

            # 3. Execute the created threads in batches of `num_threads`.
            rounds = len(self.device.scripts) / self.num_threads
            leftovers = len(self.device.scripts) % self.num_threads
            
            # Process full batches.
            while rounds > 0:
                for j in xrange(self.num_threads):
                    threads[j].start()
                for j in xrange(self.num_threads):
                    threads[j].join()
                for j in xrange(self.num_threads):
                    threads.pop(0)
                rounds -= 1
            
            # Process the remaining leftover threads.
            for j in xrange(leftovers):
                threads[j].start()
            for j in xrange(leftovers):
                threads[j].join()
            for j in xrange(leftovers):
                threads.pop(0)

            del threads[:]
            
            # 4. Wait at the global barrier to synchronize with all other devices.
            self.device.devices[0].barrier.wait()
            
            # Reset for the next time step.
            self.device.script_received.clear()


class MyThread(Thread):
    """
    A short-lived worker thread that executes a single script task.

    @warning The lazy initialization of locks in the `run` method is not
             thread-safe. If two threads for the same new location execute the
             `if not in` check concurrently, they can both attempt to create
             a lock, leading to a race condition where one lock is overwritten.
             This check-then-set action needs its own lock to be safe.
    """

    def __init__(self, device_thread, script, location, neighbours):
        Thread.__init__(self)
        self.device_thread = device_thread
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        # Unsafe lazy initialization of location locks.
        if self.location not in\
                self.device_thread.device.devices[0].locations_lock:
            self.device_thread.device.devices[0].locations_lock[self.location]\
                = RLock()

        # Acquire lock and perform the task.
        with self.device_thread.device.devices[0].locations_lock[self.location]:
            # Aggregate data.
            script_data = []
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device_thread.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            # Compute and disseminate result.
            if script_data != []:
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device_thread.device.set_data(self.location, result)
