"""
A simulation framework for a network of communicating devices using a
master/worker threading model for script execution.

This script defines a system where each `Device` has a single master thread
(`DeviceThread`) that orchestrates a pool of `Worker` threads. This design
allows for concurrent execution of scripts within a device, coordinated by a
set of shared synchronization primitives.

The main components are:
- ReusableBarrierSem: A semaphore-based reusable barrier for synchronization.
- Device: Represents a node. It creates and manages one `DeviceThread` and
  a pool of `Worker` threads. It also participates in setting up a global,
  shared dictionary of per-device locks.
- DeviceThread: The master thread for a device. It synchronizes with other
  devices, receives script assignments, distributes these scripts to its
  worker pool, and waits for them to complete.
- Worker: A worker thread that receives a batch of scripts from its master,
  executes them, and synchronizes upon completion.

Key Architectural Points:
- **Master/Worker Pattern**: Each device uses a single master to dispatch
  work to multiple workers.
- **Global Per-Device Locks**: A central dictionary of locks is created and
  shared among all devices. Access to a specific device's data is
  controlled by acquiring the lock associated with that device object,
  preventing data races.
- **Two-Level Synchronization**:
  1. A global barrier (`barrier`) synchronizes all master `DeviceThread`s
     across the network at key points in each time-step.
  2. A local barrier (`threads_barrier`) synchronizes the master thread with
     its own pool of worker threads within a single device.
"""
from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    Implements a reusable barrier using two semaphores for two-phase signaling.
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
    Represents a device node, managing a master thread and a worker pool.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        # This global lock dictionary is shared among all devices.
        self.lock = {}
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_done = Event()
        self.terminate = Event() # Global termination signal.
        self.neighbours = []

        self.barrier = None # Global barrier for inter-device sync.
        # Local barrier for this device's master and worker threads.
        self.threads_barrier = ReusableBarrierSem(9) # 8 workers + 1 master
        
        self.master = DeviceThread(self, self.terminate, self.barrier, self.threads_barrier,
                                   self.setup_done)
        self.master.start()

        self.threads = []
        for _ in range(8):
            thread = Worker(self.master, self.terminate, self.threads_barrier)
            self.threads.append(thread)
            thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes the shared barrier and lock dictionary.

        This is called by the primary device (id 0) to set up resources for
        the entire simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            # Create a global lock for each device, stored in a shared dict.
            for dev in devices:
                self.lock[dev] = Lock()
            # Distribute the barrier and the lock dictionary to other devices.
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier
                    dev.lock = self.lock
                    dev.setup_done.set() # Signal that setup is complete.
            self.setup_done.set()

    def assign_script(self, script, location):
        """Assigns a script to be executed in the current time-step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the master and worker threads for this device."""
        self.terminate.set()
        # Unblock any waiting worker threads so they can check the terminate flag.
        for i in range(8):
            self.threads[i].script_received.set()
            self.threads[i].join()
        self.master.join()


class DeviceThread(Thread):
    """
    The master thread for a device, dispatching scripts to workers.
    """
    def __init__(self, device, terminate, barrier, threads_barrier, setup_done):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = []
        self.terminate = terminate
        self.barrier = barrier
        self.threads_barrier = threads_barrier
        self.setup_done = setup_done

    def run(self):
        """Main orchestration loop for the device's time-step."""
        self.setup_done.wait() # Wait for global setup to finish.
        self.device.barrier.wait() # Initial sync with all other devices.

        while True:
            # --- Global Sync Point 1 ---
            self.device.barrier.wait()
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break

            # Wait for supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # --- Global Sync Point 2 ---
            self.device.barrier.wait()

            # Manually distribute scripts to the worker threads in round-robin.
            scripts = [[] for _ in range(8)]
            for i in range(len(self.device.scripts)):
                scripts[i % 8].append(self.device.scripts[i])

            # Signal each worker to start processing its batch of scripts.
            for i in range(8):
                self.device.threads[i].scripts = scripts[i]
                self.device.threads[i].script_received.set()

            # --- Local Sync Point ---
            # Wait for all 8 workers + self to reach the local barrier.
            if not self.terminate.is_set():
                self.threads_barrier.wait()


class Worker(Thread):
    """A worker thread that executes a batch of scripts."""
    def __init__(self, master, terminate, barrier):
        Thread.__init__(self)
        self.master = master
        self.script_received = Event()
        self.terminate = terminate
        self.scripts = []
        self.barrier = barrier

    @staticmethod
    def append_data(device, location, script_data):
        """Safely gets data from a device using the global lock."""
        device.lock[device].acquire()
        data = device.get_data(location)
        device.lock[device].release()
        if data is not None:
            script_data.append(data)

    @staticmethod
    def set_data(device, location, result):
        """Safely sets data on a device using the global lock."""
        device.lock[device].acquire()
        device.set_data(location, result)
        device.lock[device].release()

    def run(self):
        """Main worker loop."""
        while True:
            self.script_received.wait()
            self.script_received.clear()
            if self.terminate.is_set():
                break

            # Process the assigned batch of scripts.
            if self.scripts:
                for (script, location) in self.scripts:
                    script_data = []
                    # Block Logic: Gather data from neighbors and self.
                    if self.master.neighbours is not None:
                        for device in self.master.neighbours:
                            self.append_data(device, location, script_data)
                    self.append_data(self.master.device, location, script_data)

                    # Execute the script and disseminate results.
                    if script_data:
                        result = script.run(script_data)
                        if self.master.neighbours is not None:
                            for device in self.master.neighbours:
                                self.set_data(device, location, result)
                        self.set_data(self.master.device, location, result)

            # --- Local Sync Point ---
            # Signal completion to the master by waiting on the local barrier.
            self.barrier.wait()
