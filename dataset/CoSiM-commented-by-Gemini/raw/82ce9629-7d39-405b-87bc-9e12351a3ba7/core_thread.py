"""
This module contains a multi-threaded device simulation.

Its architecture is based on a "thread-per-task" model, where the main `DeviceThread`
for each device creates a new `CoreThread` for every script it needs to execute.
It then attempts to manage the execution of these threads in batches.

WARNING: The implementation is highly complex and contains several inefficiencies and
design flaws:
- A new thread is created for every task, which is inefficient.
- The `DeviceThread` uses a manual, convoluted loop to run workers in batches,
  a task better suited for a standard thread pool and queue.
- The `CoreThread` uses a confusing and inefficient locking strategy involving two
  separate global locks in addition to a location-specific semaphore, with one
  lock being acquired and released inside a loop during the write-back phase.
"""

from threading import Thread, Event, Semaphore, Lock

# Forward-declaration of ReusableBarrierCond which is defined later.
# In a real scenario, this would be in a separate file.
class ReusableBarrierCond(object):
    pass

class CoreThread(Thread):
    """
    A short-lived worker thread that executes exactly one script.
    """
    def __init__(self, device, script_id, neighbours):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script_id = script_id
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script logic using a complex and inefficient locking scheme.
        """
        (script, location) = self.device.scripts[self.script_id]

        # Acquire a semaphore for this location, acting as a location-specific lock.
        self.device.semaphores_list[location].acquire()

        script_data = []

        # Acquire a global lock for the data gathering phase.
        self.device.lock1.acquire()
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(location)
        self.device.lock1.release()

        # The original code was missing appending the device's own data.
        if data is not None:
            script_data.append(data)

        if script_data:
            result = script.run(script_data)

            # Inefficiently acquire and release a second global lock for every
            # single device write-back operation.
            for device in self.neighbours:
                self.device.lock2.acquire()
                device.set_data(location, result)
                # Redundantly sets its own data on every neighbor iteration.
                self.device.set_data(location, result)
                self.device.lock2.release()

        self.device.semaphores_list[location].release()

class Device(object):
    """Represents a device and its resources in the simulation."""
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        
        # --- Synchronization Primitives ---
        self.semaphores_list = []
        # This semaphore is used to block the DeviceThread until setup is complete.
        self.semaphore_setup_devices = Semaphore(0) # Locked initially
        self.lock1 = Lock() # A global lock for data gathering.
        self.lock2 = Lock() # A global lock for data writing.
        self.barrier = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources."""
        barrier = ReusableBarrierCond(len(devices))
        
        # Create a shared list of semaphores to act as location-specific locks.
        if self.device_id == 0:
            semaphores_list = [Semaphore(1) for _ in range(50)]
            for device in devices:
                device.semaphores_list = semaphores_list
                device.barrier = barrier
                # The semaphore starts locked; this call will block until the
                # corresponding DeviceThread's run() method releases it.
                device.semaphore_setup_devices.acquire()
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be executed."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data. Not thread-safe on its own."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data. Not thread-safe on its own."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class DeviceThread(Thread):
    """
    The main controller thread for a device. Manages the creation and execution
    of worker threads in manually-coded batches.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        # Release the setup semaphore, allowing the main thread to continue
        # from the setup_devices call.
        self.device.semaphore_setup_devices.release()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # --- Worker Creation and Batching ---
            # Create one worker thread for every script.
            cores = [CoreThread(self.device, i, neighbours) for i in range(len(self.device.scripts))]
            
            # This complex block manually runs the created threads in batches of 8.
            # This is an inefficient way to manage a thread pool.
            scripts_number = len(self.device.scripts)
            begin = 0
            if scripts_number > 8:
                while scripts_number > 0:
                    batch_size = min(8, scripts_number)
                    # Start a batch
                    for i in range(batch_size):
                        cores[begin + i].start()
                    # Join the batch
                    for i in range(batch_size):
                        cores[begin + i].join()
                    
                    scripts_number -= batch_size
                    begin += batch_size
            else:
                for core in cores:
                    core.start()
                for core in cores:
                    core.join()
            
            self.device.scripts = [] # Reset for the next timepoint.
            # Synchronize with all other devices before the next cycle.
            self.device.barrier.wait()


class ReusableBarrierCond(object):
    """A barrier implementation using `threading.Condition`."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks until all `num_threads` have called this method."""
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                self.cond.notify_all()
                self.count_threads = self.num_threads
            else:
                self.cond.wait()
