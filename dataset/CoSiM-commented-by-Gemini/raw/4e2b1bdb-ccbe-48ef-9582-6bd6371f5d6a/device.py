"""
This module implements a device simulation framework where each device's master
thread spawns a new pool of temporary worker threads for each simulation step.

The key architectural features are:
- A two-phase semaphore-based reusable barrier for global synchronization.
- A centralized setup mechanism that creates a shared set of locks for all
  unique data locations across all devices.
- A master `DeviceThread` for each device that, upon receiving scripts,
  divides the work and creates new `ScriptThread` workers to handle the load
  for that single step.

While the locking mechanism is robust, the pattern of creating and destroying
threads in every step is generally inefficient compared to using a persistent
thread pool.
"""


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A reusable barrier implemented with two semaphores and a lock.

    This implementation uses single-element lists for its counters to allow them
    to be passed by reference and modified within a helper method, a common
    Python technique for simulating mutable integers.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters are wrapped in lists to be mutable.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """A generic method for one phase of the two-phase barrier."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread arrived, release all waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire()
        

class Device(object):
    """
    Represents a device node, holding state and configuration.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.locks = []
        self.barrier = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared simulation resources.
        
        Run by a master device (id 0), this identifies all unique data locations,
        creates a lock for each, and shares the locks and a global barrier
        with all other devices.
        """
        global_barrier = ReusableBarrier(len(devices))
        self.barrier = global_barrier
        locations = []

        # Find all unique locations across all devices.
        for device in devices:
            device.barrier = global_barrier
            for data in device.sensor_data:
                if data not in locations:
                    locations.append(data)

        # Create one lock for each unique location.
        for location in locations:
            lock = Lock()
            self.locks.append((location, lock))

        # Share the list of locks with all devices.
        for device in devices:
            device.locks = self.locks

    def assign_script(self, script, location):
        """Assigns a script or signals the end of script assignment."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's master thread to terminate."""
        self.thread.join()

class ScriptThread(Thread):
    """
    A temporary worker thread that executes a subset of scripts for one step.
    """
    def __init__(self, device, neighbours, scripts):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.scripts = scripts

    def run(self):
        """Executes the assigned scripts."""
        locks = dict(self.device.locks)
        for (script, location) in self.scripts:
            # Use a 'with' statement for safe, automatic lock acquisition/release.
            with locks[location]:
                script_data = []
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    result = script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)


class DeviceThread(Thread):
    """
    The persistent master thread for a device, orchestrating work for each step.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal

            # Wait for supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()

            # --- Inefficient Threading Model ---
            # This section creates a new set of threads for every simulation
            # step, which is generally inefficient due to the overhead of
            # thread creation and destruction.
            
            # Divide the scripts into 8 chunks.
            divided_scripts = [[] for _ in range(8)]
            index = 0
            for (script, location) in self.device.scripts:
                divided_scripts[index].append((script, location))
                index = (index + 1) % 8

            # Create, start, and join a new thread for each chunk of scripts.
            threads = []
            for s_list in divided_scripts:
                if s_list:
                    thread = ScriptThread(self.device, neighbours, s_list)
                    threads.append(thread)
                    thread.start()
            for thread in threads:
                thread.join()

            # This wait is redundant as timepoint_done is set with script_received.
            self.device.timepoint_done.wait()
            
            # Synchronize with all other devices before the next step.
            self.device.barrier.wait()
            
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            self.device.scripts = []