"""
A simulation framework for a network of communicating devices using a
hierarchical threading model.

This script defines a complex system where each `Device` not only runs its own
main control thread (`DeviceThread`) but also manages a pool of worker threads
(`ThreadAux`). The simulation proceeds in synchronized time-steps, orchestrated
by a combination of global and local barriers, events, and a global lock pool.

The main components are:
- ReusableBarrier: A synchronization primitive to ensure a group of threads
  waits for each other.
- Device: Represents a node in the network. It initializes a `DeviceThread`
  and a pool of `ThreadAux` worker threads. It assigns incoming scripts to its
  workers in a round-robin fashion.
- DeviceThread: The main control loop for a Device, responsible for
  synchronizing with other devices on a global scale and coordinating its
  local pool of worker threads.
- ThreadAux: A worker thread that executes a subset of scripts for a device.
  It handles data gathering from neighbors, script execution, and result
  dissemination, using a global lock to ensure data consistency for a given
  'location'.

Synchronization is managed at three levels:
1.  **Global Device Synchronization**: A static barrier (`Device.bar1`) ensures
    all `DeviceThread` instances are synchronized at the end of a time-step.
2.  **Intra-Device Synchronization**: A local barrier (`bar_threads_device`)
    and a series of events (`event`) synchronize the main `DeviceThread` with
    its own worker `ThreadAux` threads.
3.  **Data Location Synchronization**: A static list of locks (`Device.locck`)
    is used to protect access to data based on its 'location', preventing race
    conditions between worker threads from different devices.
"""
from threading import Event, Thread, Lock, Semaphore, Lock


class ReusableBarrier(object):
    """
    Implements a reusable barrier using two semaphores for two-phase signaling.

    This forces a specified number of threads to wait until all have arrived.
    It is reusable for multiple synchronization points.

    Attributes:
        num_threads (int): The number of threads to wait for.
        counter_lock (Lock): A mutex to protect internal counters.
        threads_sem1 (Semaphore): Semaphore for the first synchronization phase.
        threads_sem2 (Semaphore): Semaphore for the second synchronization phase.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase for reusability."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a device node with a pool of worker threads.

    Manages a main `DeviceThread` and multiple `ThreadAux` workers. Scripts
    are distributed among the workers.

    Class Attributes:
        bar1 (ReusableBarrier): A global barrier for all devices.
        event1 (Event): A global event to kick-start the simulation.
        locck (list): A global pool of locks indexed by script location.
    """
    # Static members shared across all Device instances.
    bar1 = ReusableBarrier(1)
    event1 = Event()
    locck = []

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its associated threads."""
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.devices = []

        # A list of events to signal worker threads for each time-step.
        self.event = []
        for _ in xrange(11):
            self.event.append(Event())

        self.nr_threads_device = 8
        # Index for round-robin assignment of scripts to worker threads.
        self.nr_thread_atribuire = 0
        
        # Local barrier for the main thread and all its worker threads.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device + 1)

        # Main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Pool of worker threads for executing scripts.
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes static, global resources shared by all devices.

        This should be called on one device only (e.g., device_id 0).
        """
        self.devices = devices
        
        if self.device_id == 0:
            # Initialize the global pool of location-based locks.
            for _ in xrange(30):
                Device.locck.append(Lock())
            # Initialize the global barrier for all devices.
            Device.bar1 = ReusableBarrier(len(devices))
            
            # Signal all devices to start their main loops.
            Device.event1.set()

    def assign_script(self, script, location):
        """
        Assigns a script to one of the device's worker threads.

        Args:
            script: The script to be executed.
            location: The location context for the script.
        """
        if script is not None:
            # Assign script to the next worker thread in a round-robin fashion.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            self.nr_thread_atribuire = (self.nr_thread_atribuire + 1) %
                self.nr_threads_device
        else:
            # A None script indicates all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in 
            self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main thread and all worker threads to terminate."""
        self.thread.join()
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    """The main control thread for a single Device."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.contor = 0  # Counter for time-steps/events.

    def run(self):
        """
        Main orchestration loop.

        Waits for global start, then repeatedly gets neighbors, signals local
        workers to run, and synchronizes with local workers and global devices.
        """
        Device.event1.wait()  # Wait for the global start signal.

        while True:
            # Get neighbors for the current time-step.
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                # Supervisor signals termination.
                self.device.event[self.contor].set() # Unblock workers to exit.
                break

            # Wait for supervisor to finish assigning all scripts for this step.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Unblock worker threads to start processing their assigned scripts.
            self.device.event[self.contor].set()
            self.contor += 1

            # --- Intra-Device Sync Point ---
            # Wait for all local worker threads to finish their work.
            self.device.bar_threads_device.wait()

            # --- Global Sync Point ---
            # Wait for all other devices to finish their time-step.
            Device.bar1.wait()


class ThreadAux(Thread):
    """A worker thread that executes a batch of scripts for a Device."""
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device
        self.script_loc = {} # Holds {script: location} assignments.
        self.contor = 0

    def run(self):
        """
        Main worker loop.

        Waits for a signal from its parent DeviceThread, executes all its
        assigned scripts, and then synchronizes on a local barrier.
        """
        while True:
            # Wait for the signal from the parent DeviceThread to start.
            self.device.event[self.contor].wait()
            self.contor += 1

            neigh = self.device.thread.neighbours
            if neigh is None:
                # Termination signal received.
                break

            # Process all assigned scripts.
            for script in self.script_loc:
                location = self.script_loc[script]
                
                # --- Data Location Sync ---
                # Acquire the global lock for this specific location.
                Device.locck[location].acquire()
                
                # Block Logic: Gather data from neighbors and self.
                script_data = []
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute script and disseminate results.
                if script_data:
                    result = script.run(script_data)
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the global lock for this location.
                Device.locck[location].release()

            # --- Intra-Device Sync Point ---
            # Signal to the parent DeviceThread that this worker is done.
            self.device.bar_threads_device.wait()
