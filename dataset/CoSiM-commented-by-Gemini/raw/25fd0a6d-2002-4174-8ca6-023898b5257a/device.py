"""
@file device.py
@brief A device simulation using per-device thread pools and a multi-level barrier system.
@details This script models a network of devices where each device maintains its own fixed-size
pool of worker threads. Synchronization is attempted through a combination of a local barrier for
threads within a single device and a global barrier for synchronizing time steps across all devices.
A global lock is used to serialize access during the data gathering phase.

@warning This implementation contains several logical flaws:
1. The per-location locking is not effective as Lock objects are local to each device, not shared.
2. A global lock serializes the data gathering step, negating the benefits of parallelism.
3. The synchronization logic involving leader threads and multiple barriers is overly complex.
"""

from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """
    Represents a single device, which manages its own thread pool to execute scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbours_received = Event()
        self.script_received = Event()
        self.num_scripts = 0
        self.num_threads = 0
        self.scripts = []
        self.threads = []
        # A barrier for synchronizing threads within this single device.
        self.barrier = None
        # A barrier for synchronizing all devices between time steps.
        self.time_barrier = None
        # A global lock that serializes critical sections across all devices.
        self.global_lock = None
        self.neighbours = None
        # A list of locks for data locations. NOTE: This is local to the device, which is a design flaw.
        self.location_locks = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources and the per-device thread pools.
        Device 0 sets up global objects, while each device sets up its own local pool.
        """
        # Block invariant: The primary device (ID 0) creates and distributes global sync objects.
        if devices[0].device_id == self.device_id:
            time_barrier = ReusableBarrier(len(devices))
            global_lock = Lock()
            for device in devices:
                device.time_barrier = time_barrier
                device.global_lock = global_lock

        # Each device creates its own pool of worker threads.
        self.num_threads = max(min(8, 100 / len(devices)), 1)
        self.barrier = ReusableBarrier(self.num_threads)

        for i in xrange(0, self.num_threads):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(0, self.num_threads):
            self.threads[i].start()

    def assign_script(self, script, location):
        """Assigns a script from the supervisor to this device's script list."""
        if script is not None:
            self.scripts.append((script, location))
            self.num_scripts = len(self.scripts)
        else:
            # A None script signals that all scripts for the time step have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        data = self.sensor_data[location] if location in self.sensor_data else None
        return data

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def acquire_lock(self, location):
        """
        Acquires a lock for a data location.
        @warning This implementation is flawed. The locks are local to each device instance,
        so they do not provide mutual exclusion between different devices.
        """
        my_lock = None
        for (loc, lock) in self.location_locks:
            if location == loc:
                my_lock = lock
        if my_lock is None:
            my_lock = Lock()
            self.location_locks.append((location, my_lock))

        my_lock.acquire()

    def release_lock(self, location):
        """
        Releases a lock for a data location.
        @warning This method is bugged. If the lock doesn't exist, it creates a new one
        and immediately tries to release it, which will raise a ThreadError.
        """
        my_lock = None
        for (loc, lock) in self.location_locks:
            if location == loc:
                my_lock = lock
        if my_lock is None:
            my_lock = Lock()
            self.location_locks.append((location, my_lock))

        my_lock.release()


    def shutdown(self):
        """Waits for all threads in the device's pool to complete."""
        for i in xrange(0, self.num_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread within a device's thread pool. It executes a subset of the device's scripts.
    """

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        Main loop for the worker thread, coordinating with other threads and devices.
        """
        while True:
            
            # The first thread of each device acts as a "leader" for its device for this time step.
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                # Signal to sibling threads that neighbor information is ready.
                self.device.neighbours_received.set()


            # All threads wait until the neighbor data has been fetched.
            self.device.neighbours_received.wait()

            neighbours = self.device.neighbours
            # Pre-condition: Check for the simulation termination signal.
            if neighbours is None:
                break

            # Wait until all scripts for the time step have been assigned.
            self.device.script_received.wait()

            
            # Manually partition the device's scripts among its worker threads.
            scripts = []
            if self.device.num_scripts <= self.device.num_threads:
                if self.thread_id < self.device.num_scripts:
                    scripts = [self.device.scripts[self.thread_id]]
            else:
                workload_size = self.device.num_scripts / self.device.num_threads
                offset1 = self.thread_id * workload_size
                offset2 = (self.thread_id + 1) * workload_size
                scripts = self.device.scripts[offset1:offset2]
                # The leader thread picks up any remaining scripts from uneven division.
                if self.thread_id == 0:
                    offset1 = self.device.num_threads * workload_size
                    offset2 = self.device.num_scripts
                    scripts += self.device.scripts[offset1:offset2]

            
            peers = []
            for device in neighbours:
                if device != self.device:
                    peers.append(device)
            peers.append(self.device)

            
            # Block Logic: Execute the assigned portion of scripts.
            for (script, location) in scripts:
                script_data = []

                
                # Flawed Synchronization: A global lock is used, serializing this entire section
                # across all threads and all devices, defeating the purpose of a parallel simulation.
                self.device.global_lock.acquire()

                # The intended fine-grained locking is non-functional because lock objects are not shared.
                for device in peers:
                    device.acquire_lock(location)
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                self.device.global_lock.release()

                
                if script_data != []:
                    result = script.run(script_data)

                
                # Update data and release the non-functional local locks.
                for device in peers:
                    device.set_data(location, result)
                    device.release_lock(location)

            
            # --- Intra-Device Barrier ---
            # All threads within this device synchronize here to ensure they've all finished their work.
            self.device.barrier.wait()

            # The leader thread handles cleanup and inter-device synchronization.
            if self.thread_id == 0:
                self.device.script_received.clear()
                self.device.neighbours_received.clear()
                # --- Inter-Device Barrier ---
                # All device leaders wait here, ensuring all devices have finished the time step.
                self.device.time_barrier.wait()
            
            # --- Intra-Device Barrier (Release) ---
            # The leader, after passing the global barrier, signals its sibling threads (waiting
            # at the first barrier) to proceed to the next time step by meeting them at the second phase.
            self.device.barrier.wait()



class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads over two phases.
    """
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()