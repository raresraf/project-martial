from threading import Event, Thread, Lock, Condition, Semaphore

class Device(object):
    """
    Represents a device node that coordinates with a "leader" device (device 0)
    for synchronization and uses a pool of slave threads for script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its master control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.devices = None
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.leader = -1
        self.location_locks = [] # A list of locks for fine-grained data access.
        
        # Only the leader device (id 0) creates the master synchronization objects.
        if device_id == 0:
            self.finishedthread = 0 # Counter for the manual barrier.
            self.condition = Condition() # A Condition variable for a custom global barrier.
            self.can_start = Event() # Event to signal that setup is complete.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources. Device 0 acts as the leader, creating location
        locks and signaling other devices when setup is complete.
        """
        self.devices = devices

        # Find the leader device (id 0). Note: uses Python 2 `xrange`.
        for i in xrange(len(self.devices)):
            if devices[i].device_id == 0:
                self.leader = i
                break

        if self.device_id == 0:
            self.can_start.clear()
            # Find the max location ID to size the lock list.
            maximum = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > maximum:
                        maximum = location
            
            # Create a lock for each location.
            for _ in range(0, maximum + 1):
                self.location_locks.append(Lock())

            # Share the lock list with all other devices.
            for device in devices:
                device.location_locks = self.location_locks
            
            self.can_start.set() # Signal that setup is done.
        else:
            # Non-leader devices wait for the leader to finish setup.
            devices[self.leader].can_start.wait()

    def assign_script(self, script, location):
        """Assigns a script to be run or signals the end of a timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set() # All scripts for this timepoint are assigned.

    def get_data(self, location):
        """Gets data from a sensor location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the master thread to finish."""
        self.thread.join()

    def finished(self):
        """
        A custom, centralized barrier implementation using a Condition variable.
        All devices report to the leader (device 0), which coordinates the barrier.
        """
        leader_device = self.devices[self.leader]
        with leader_device.condition:
            leader_device.finishedthread += 1
            if leader_device.finishedthread != len(self.devices):
                leader_device.condition.wait() # Wait for all other devices.
            else:
                # Last device has arrived, notify all waiting devices.
                leader_device.condition.notifyAll()
                leader_device.finishedthread = 0 # Reset for the next barrier.

class DeviceThread(Thread):
    """The master thread for a device, managing a pool of slave threads."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.slavelist = SlaveList(device) # Manages the worker thread pool.

    def run(self):
        """Main loop: waits for scripts, dispatches them to slaves, and synchronizes."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                self.slavelist.shutdown()
                break

            self.device.timepoint_done.wait() # Wait for supervisor to assign all scripts.
            self.device.timepoint_done.clear()

            # Dispatch all scripts for this timepoint to the slave pool.
            for (script, location) in self.device.scripts:
                self.slavelist.do_work(script, location, neighbours)

            self.slavelist.event_wait() # Wait for all local slaves to finish.
            self.device.finished() # Participate in the global inter-device barrier.

class SlaveList(object):
    """A thread pool manager for a device's 'Slave' worker threads."""
    def __init__(self, device):
        self.device = device
        self.event = Event()
        self.event.set()
        # Semaphore limits active tasks to the number of slave threads (8).
        self.semaphore = Semaphore(8)
        self.lock = Lock()
        self.slavelist = [] # List of all slave threads.
        self.readythreads = [] # A queue of available slave threads.
        
        for _ in xrange(8): # Note: uses Python 2 `xrange`.
            thread = Slave(self, self.device)
            self.slavelist.append(thread)
            self.readythreads.append(thread)
            thread.start()

    def do_work(self, script, location, neighbours):
        """Pops a ready slave from the pool and assigns it a task."""
        if self.event.isSet():
            self.event.clear()
        self.semaphore.acquire() # Block if all slaves are busy.
        with self.lock:
            slave = self.readythreads.pop(0)
        slave.do_work(script, location, neighbours)

    def shutdown(self):
        """Shuts down all slave threads in the pool."""
        for slave in self.slavelist:
            slave.imdone = True
            slave.semaphore.release() # Unblock the slave so it can terminate.
            slave.join()

    def slave_done(self, slave):
        """
        Callback for when a slave finishes its task. It returns the slave to
        the ready pool and signals the master if all slaves are now ready.
        """
        with self.lock:
            self.readythreads.append(slave)
            if not self.event.isSet() and len(self.readythreads) == 8:
                self.event.set() # Signal to the master that all work is done.
        self.semaphore.release() # Allow another task to be dispatched.

    def event_wait(self):
        """Allows the master thread to wait for all slaves to complete."""
        self.event.wait()

class Slave(Thread):
    """A worker thread that executes a single script task."""
    def __init__(self, slavelist, device):
        Thread.__init__(self)
        self.slavelist = slavelist
        # This semaphore acts as a personal "go" signal for this slave.
        self.semaphore = Semaphore(0)
        self.device = device
        self.script = None
        self.location = None
        self.neighbours = None
        self.imdone = False # Shutdown flag.

    def do_work(self, script, location, neighbours):
        """Loads task data into the slave and releases its semaphore to run."""
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.semaphore.release() # Wakes up the run() method.

    def run(self):
        """Main loop: waits for a task, executes it, and reports back."""
        while True:
            self.semaphore.acquire() # Wait for a task from do_work().
            if self.imdone:
                break
            
            with self.device.location_locks[self.location]:
                values = []
                # Gather data from parent device and neighbors.
                data = self.device.get_data(self.location)
                if data is not None:
                    values.append(data)
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        values.append(data)
                
                # Execute script and broadcast results.
                if values:
                    result = self.script.run(values)
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    self.device.set_data(self.location, result)
            
            # Notify the pool manager that this slave is now done and ready.
            self.slavelist.slave_done(self)
