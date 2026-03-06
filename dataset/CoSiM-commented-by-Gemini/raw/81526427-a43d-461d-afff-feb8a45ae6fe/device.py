"""
This module implements a highly complex and unusual multi-threaded device simulation.

Its architecture is notable for a number of advanced but fragile patterns:
- A two-barrier system (`barrier` and `barrier2`) is used to synchronize the start
  and end of the script processing phase within each timepoint.
- The supervisor thread, which assigns scripts, is tightly coupled with the
  device threads, as it directly participates in the first barrier synchronization.
  This is a significant anti-pattern as it mixes the controller/worker roles.
- A dynamic, non-persistent thread-per-task model is used within each device's
  main thread, with a potential race condition in how the number of threads is tracked.
- The setup of shared resources is order-dependent and not robustly centralized.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    A reusable barrier for synchronizing a fixed number of threads, implemented
    using a two-phase protocol with semaphores.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier protocol."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """Represents a device, managing its state and synchronization."""
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.max_threads = 8
        self.thread = DeviceThread(self)
        self.barrier_lock = Lock()
        self.location_locks = {loc: None for loc in sensor_data.keys()}
        # The simulation uses two distinct barriers for synchronization.
        self.barrier = None
        self.barrier2 = None
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        A complex, order-dependent method to initialize and distribute shared
        resources. It assumes it will be called by all devices and relies on
        the call order to establish the shared state correctly.
        """
        # All devices share the lock from the first device in the list.
        self.barrier_lock = devices[0].barrier_lock
        with self.barrier_lock:
            # Attempt to discover and merge location locks from other devices.
            loc_list = [dev.location_locks for dev in devices]
            for loc_dict in loc_list:
                for key, value in loc_dict.items():
                    if key not in self.location_locks or self.location_locks[key] is None:
                         if value is not None:
                            self.location_locks[key] = value

            # Create any location locks that weren't discovered.
            for key in self.location_locks:
                if self.location_locks[key] is None:
                    self.location_locks[key] = Lock()

            # The first device to run this creates the shared barriers.
            if self.barrier is None:
                self.barrier = ReusableBarrierSem(len(devices))
                self.barrier2 = ReusableBarrierSem(len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script. WARNING: This method involves the caller (supervisor)
        in the device synchronization, which is a design anti-pattern.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # When the supervisor signals the end of assignments, it blocks
            # itself on the barrier with all the device threads.
            self.barrier.wait()
            self.timepoint_done.set()

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
    """The main controller thread for a device."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.num_threads = 1 # Potential race condition: not protected by a lock.

    def run_scripts(self, script, location, neighbours):
        """The target function for worker threads, executing a single script."""
        # Acquire location-specific lock to ensure exclusive access.
        self.device.location_locks[location].acquire()
        
        script_data = []
        # Data aggregation is safe due to the location lock.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        if script_data:
            result = script.run(script_data)
            # Writing data is safe due to the location lock.
            for device in neighbours:
                device.set_data(location, result)
            self.device.set_data(location, result)

        self.device.location_locks[location].release()

    def run(self):
        """The main simulation loop, containing complex synchronization logic."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            
            # --- First Synchronization Point ---
            # All device threads (and the supervisor) wait here before work begins.
            if neighbours is None:
                self.device.barrier.wait()
                break
            self.device.barrier.wait()
            
            # --- Script Execution Phase ---
            child_threads = []
            for (script, location) in self.device.scripts:
                # Dynamically create threads up to a max limit.
                if self.num_threads < self.device.max_threads:
                    self.num_threads += 1
                    arguments = (script, location, neighbours)
                    child = Thread(target=self.run_scripts, args=arguments)
                    child_threads.append(child)
                    child.start()
                else:
                    # If pool is "full", run the task in the controller thread itself.
                    self.run_scripts(script, location, neighbours)
            
            for child in child_threads:
                child.join()
                self.num_threads -= 1
            self.device.scripts = [] # Clear scripts for next timepoint.
            
            # --- Second and Third Synchronization Points ---
            # All device threads wait at the second barrier after work is done.
            self.device.barrier2.wait()
            # Then they wait for the event that was set by the supervisor.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
