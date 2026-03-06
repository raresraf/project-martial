"""
This module provides a device simulation framework using a fixed-size thread pool
per device and a complex, multi-layered synchronization strategy.

Each `Device` instance immediately starts a pool of 8 worker `DeviceThread`s.
Work is distributed statically, where each worker processes a subset of scripts
based on its ID modulo the pool size.

The synchronization is notably intricate and involves three main barriers per cycle:
1. A local `barrier_worker` to synchronize the workers within a single device.
2. A global `timepoint_done` Event to wait for the supervisor's signal.
3. A global `barrier` to synchronize all worker threads from all devices.
This design is complex and potentially fragile due to race conditions and the
on-the-fly initialization of shared resources.
"""


from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier for synchronizing a fixed number of threads.
    NOTE: The implementation appears to have a bug where phase1 resets the counter
    for phase2 and vice-versa, which is incorrect. A correct implementation
    would have each phase reset its own counter after releasing the threads.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """Executes the first phase of the barrier protocol."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Should reset self.count_threads1
        self.threads_sem1.acquire()

    def phase2(self):
        """Executes the second phase of the barrier protocol."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Should reset self.count_threads2

        self.threads_sem2.acquire()

class Device(object):
    """Represents a device, managing a fixed pool of worker threads."""
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_event = Event()
        self.devices = [] # Seems unused.
        self.neighbours = []

        # --- Synchronization Primitives ---
        self.barrier_worker = ReusableBarrier(8) # For workers of this device only.
        self.barrier = None # Global barrier for ALL workers from ALL devices.
        self.location_lock = [] # Shared list of location-specific locks.

        # Create and start the fixed pool of worker threads immediately.
        self.threads = [DeviceThread(self, i) for i in range(8)]
        for thr in self.threads:
            thr.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources for the simulation."""
        if self.device_id == 0:
            # The global barrier must accommodate every worker thread from every device.
            barrier = ReusableBarrier(len(devices) * 8)
            self.barrier = barrier
            
            location_max = 0
            for device in devices:
                device.barrier = barrier
                for location in device.sensor_data:
                    if location > location_max:
                        location_max = location

            # Create and distribute the shared list of location locks.
            self.location_lock = [Lock() for _ in range(location_max + 1)]
            for device in devices:
                device.location_lock = self.location_lock
            
            # Release all worker threads from all devices to start the simulation.
            for device in devices:
                device.setup_event.set()

    def assign_script(self, script, location):
        """
        Assigns a script for execution. Note: on-the-fly lock initialization here
        is not thread-safe.
        """
        if script is not None:
            self.scripts.append((script, location))
            # This on-demand, racy initialization is a fragile pattern.
            if self.location_lock[location] is None:
                self.location_lock[location] = Lock()
            self.script_received.set() # This event seems unused.
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. Not thread-safe by itself."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data. Not thread-safe by itself."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to complete."""
        for thr in self.threads:
            thr.join()

class DeviceThread(Thread):
    """A worker thread that processes a static subset of the device's scripts."""
    def __init__(self, device, idd):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.idd = idd # Unique ID (0-7) for this worker within its device pool.

    def run(self):
        """The main, complex execution loop of the worker thread."""
        # Wait until the master setup_devices call is complete.
        self.device.setup_event.wait()

        while True:
            # --- Intra-Device Sync ---
            # One worker (idd 0) gets the neighbor list for the whole device pool.
            if self.idd == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # All workers in this device's pool wait here to ensure they all have
            # the updated neighbor list before proceeding.
            self.device.barrier_worker.wait()

            if self.device.neighbours is None:
                break # Termination signal.

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            # Another intra-device barrier.
            self.device.barrier_worker.wait()

            # --- Work Execution Phase ---
            i = 0
            for (script, location) in self.device.scripts:
                # Static work partitioning: each worker handles a subset of scripts.
                if i % 8 == self.idd:
                    with self.device.location_lock[location]:
                        script_data = []
                        # Data aggregation and processing, protected by the location lock.
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data:
                            result = script.run(script_data)
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            self.device.set_data(location, result)
                i += 1

            # --- Global Sync ---
            if self.idd == 0: # Only one worker should clear the event.
                self.device.timepoint_done.clear()
            
            # All workers from all devices synchronize here before the next timepoint.
            self.device.barrier.wait()
