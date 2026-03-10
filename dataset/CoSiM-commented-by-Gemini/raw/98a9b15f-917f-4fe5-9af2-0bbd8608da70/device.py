
"""
Models a distributed network of devices that process sensor data concurrently.

This script presents a complex device simulation architecture. It utilizes a
combination of global (class-level) and instance-level synchronization
primitives. Each device maintains a persistent pool of worker threads, which are
orchestrated by a main device thread using a series of events and barriers.
"""
from threading import Event, Thread, Lock, Semaphore, Lock

class ReusableBarrier(object):
    """A reusable barrier implemented using semaphores for thread synchronization.

    This barrier ensures that a fixed number of threads all reach a
    synchronization point before any of them are allowed to proceed. It uses a
    two-phase (turnstile) mechanism to allow for reuse in iterative algorithms.
    """
    
    def __init__(self, num_threads):
        """Initializes the ReusableBarrier."""
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads arrive."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):


                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase of the barrier, allowing for safe reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()



class Device(object):
    """Represents a single device in the distributed sensor network.

    This implementation uses global (class-level) primitives for inter-device
    synchronization and maintains its own pool of persistent worker threads.

    Class Attributes:
        bar1 (ReusableBarrier): A global barrier for all devices.
        event1 (Event): A global event to start all device threads.
        locck (list): A global list of locks for data locations.

    Instance Attributes:
        device_id (int): A unique identifier for the device.
        thread (DeviceThread): The main orchestration thread for this device.
        threads (list): A pool of persistent worker threads (`ThreadAux`).
    """
    
    # --- Class-level (global) synchronization primitives ---
    bar1 = ReusableBarrier(1)
    event1 = Event()
    locck = []

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance and its persistent worker threads."""
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = []

        # A series of events for signaling between the main thread and workers.
        self.event = []
        for _ in xrange(11):
            self.event.append(Event())

        self.nr_threads_device = 8
        self.nr_thread_atribuire = 0
        
        # A barrier to synchronize the main thread and its own workers.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1)

        self.thread = DeviceThread(self)
        self.thread.start()

        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes the global synchronization primitives.

        This method is called on all devices, but the device with ID 0 is
        responsible for creating the shared locks and the global barrier.
        """
        self.devices = devices
        
        if self.device_id == 0:
            for _ in xrange(30):
                Device.locck.append(Lock())
            Device.bar1 = ReusableBarrier(len(devices))
            
            Device.event1.set()

    def assign_script(self, script, location):
        """Assigns a script to one of the worker threads in a round-robin fashion."""
        if script is not None:
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%\
            self.nr_threads_device
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main and worker threads to shut down the device."""
        self.thread.join()
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    """The main orchestration thread for a single Device.

    This thread does not perform computations itself. Its role is to manage the
    simulation timepoints, signal its worker threads to start computation, and
    handle the two-level barrier synchronization.
    """
    def __init__(self, device):
        """Initializes the main DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.contor = 0

    def run(self):
        """The main control loop for the device.

        It waits for a global start signal, then enters a loop for each
        timepoint. In each loop, it signals its workers, waits for them to
        finish (via a local barrier), and then waits on a global barrier.
        """
        Device.event1.wait()

        while True:
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                self.device.event[self.contor].set()
                break

            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Signal this device's worker threads to start processing.
            self.device.event[self.contor].set()
            self.contor += 1

            # Wait for this device's workers to finish.
            self.device.bar_threads_device.wait()

            # Wait for all other devices to finish the timepoint.
            Device.bar1.wait()

class ThreadAux(Thread):
    """A persistent worker thread that performs script computations.

    Each worker has a dictionary of assigned scripts and locations. It waits for
    a signal from its parent `DeviceThread` before starting computations.
    """
    def __init__(self, device):
        """Initializes the worker thread."""
        Thread.__init__(self)
        self.device = device
        self.script_loc = {}
        self.contor = 0

    def run(self):
        """The main loop for the worker thread.

        It waits for a signal, then processes all its assigned scripts. After
        computation, it synchronizes with its parent `DeviceThread`.
        """
        while True:
            # Wait for the signal from the parent DeviceThread.
            self.device.event[self.contor].wait()
            self.contor += 1

            neigh = self.device.thread.neighbours
            if neigh is None:
                break

            for script in self.script_loc:
                location = self.script_loc[script]
                
                # Acquire a global lock for the specific location.
                Device.locck[location].acquire()
                script_data = []

                # Collect data from neighbors and self.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Run script and propagate results.
                if script_data != []:
                    result = script.run(script_data)
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                Device.locck[location].release()

            # Synchronize with the parent DeviceThread and other workers.
            self.device.bar_threads_device.wait()
