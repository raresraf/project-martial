"""
This module implements a complex, multi-threaded simulation of a distributed device network.

Each device in the network is itself a parallel entity, spawning multiple internal
threads to process its workload. The simulation proceeds in synchronized time steps,
managed by a barrier. A key feature is the dynamic, fine-grained locking mechanism
for data "locations," where the lock for a specific location is created by one
device and then shared across all other devices to ensure system-wide consistency.

The workload (scripts) for each device is statically partitioned among its internal
threads.

Classes:
    ReusableBarrier: An attempted, but flawed, implementation of a reusable barrier.
    Device: Represents a parallel device node in the network.
    DeviceThread: An execution thread for a part of a Device's workload.

@warning The `ReusableBarrier` class in this module has a critical implementation
         flaw and is not thread-safe for reuse. It is susceptible to race
         conditions that can lead to deadlocks.
"""

from threading import Event, Thread, Lock, Condition


class ReusableBarrier(object):
    """
    An attempted implementation of a reusable barrier using a Condition variable.

    @warning This implementation is NOT thread-safe. It is prone to a classic
             race condition known as the "lost wakeup" problem. A thread might
             pass the barrier, loop, and re-enter `wait()` before all other threads
             have woken up, leading to a deadlock. A correct reusable barrier
             requires a two-phase signaling mechanism.
    """
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread arrives, notifies all waiting threads.
            self.cond.notify_all()
            # Resets for reuse, but this is where the race condition occurs.
            self.count_threads = self.num_threads
        else:
            # Threads wait until notified by the last thread.
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a single device which parallelizes its own work across 8 threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.threads = []
        self.bariera = None # Romanian: "barrier"
        self.lock = Lock()
        self.call_neigh = 1 # Flag to coordinate neighbor fetching.
        self.rupe = 0 # Romanian: "break", a flag to signal shutdown.
        self.numara = 0 # Romanian: "count", used to assign IDs to internal threads.
        self.neighbours = []
        self.numara_lock = Lock()
        self.call_neigh_lock = Lock()
        self.global_lock = None # A lock to protect the creation of location-specific locks.
        self.devices = []
        self.location_dict = {} # Maps a location to its specific lock.

        i = 0
        # Each device has 8 internal worker threads.
        while i < 8:
            self.threads.append(DeviceThread(self))
            i += 1



    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the startup of all devices. Device 0 acts as the master.
        """
        if self.device_id == 0:
            # Device 0 creates the barrier for all threads (8 per device).
            self.bariera = ReusableBarrier(8*len(devices))
            self.global_lock = Lock()

            # Propagate the barrier and global lock to all other devices.
            for dev in devices:
                dev.bariera = self.bariera
                dev.global_lock = self.global_lock
        self.devices = devices

        
        for i in xrange(8):
            self.threads[i].start()


    def assign_script(self, script, location):
        """Assigns a script for the device to execute."""
        self.script_received.clear()

        if script is not None:
            self.scripts.append([script, location])
        else:
            # A None script signals the end of script assignment for this timepoint.
            self.timepoint_done.set()
        self.script_received.set()

    def get_data(self, location):
        """Gets data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its internal threads."""
        for i in xrange(8):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    An internal worker thread for a Device.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.lock = Lock()
        self.id_th = 0 # The unique ID (0-7) of this thread within its device.

    def run(self):
        """The main simulation loop for a single worker thread."""
        
        # Assign a unique ID (0-7) to this thread.
        self.device.numara_lock.acquire()
        self.id_th = self.device.numara
        self.device.numara += 1
        self.device.numara_lock.release()

        while True:
            # --- Neighbor Fetching Coordination ---
            # This locked section ensures that the neighbor list is fetched exactly once
            # per device per time step.
            self.device.call_neigh_lock.acquire()
            if self.device.call_neigh == 1:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                if self.device.neighbours is None:
                    # `None` neighbors is the shutdown signal.
                    self.device.rupe = 1
                    self.device.call_neigh_lock.release()
                    break
                self.device.call_neigh = 0
            self.device.call_neigh_lock.release()

            # Check for the shutdown signal.
            self.device.call_neigh_lock.acquire()
            if self.device.rupe == 1:
                self.device.call_neigh_lock.release()
                break
            self.device.call_neigh_lock.release()
            
            # Wait until all scripts for the timepoint have been assigned.
            self.device.timepoint_done.wait()
            
            # --- Work Partitioning ---
            # Statically assign scripts to threads. This thread processes scripts
            # at indices `id_th`, `id_th + 8`, `id_th + 16`, ...
            for i in xrange(self.id_th, len(self.device.scripts), 8):
                [script, location] = self.device.scripts[i]

                # --- Location Lock Creation and Distribution ---
                # The global lock protects the creation of new location-specific locks.
                self.device.global_lock.acquire()
                if location not in self.device.location_dict:
                    # The first thread to encounter a new location creates the lock for it.
                    self.device.location_dict[location] = Lock()
                    # It then propagates this same lock object to all other devices.
                    for j in xrange(len(self.device.devices)):
                        self.device.devices[j].location_dict[location] = \
                        self.device.location_dict[location]
                self.device.global_lock.release()

                # Acquire the specific lock for the location being processed.
                self.device.location_dict[location].acquire()
                
                # --- Data Gathering, Computation, and Propagation ---
                script_data = []
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)



                if script_data != []:
                    result = script.run(script_data)

                    
                    for device in self.device.neighbours:
                        if device.get_data(location) is not None:
                            device.set_data(location, result)
                    
                    if self.device.get_data(location) is not None:
                        self.device.set_data(location, result)

                self.device.location_dict[location].release()

            # --- Two-Phase Barrier Synchronization ---
            # This manual two-phase barrier ensures correct synchronization.
            # Phase 1: All threads wait here after finishing their script processing.
            self.device.bariera.wait()
            # This code runs only after all threads have completed the work above.
            self.device.call_neigh = 1
            self.device.timepoint_done.clear()
            # Phase 2: All threads wait here before starting the next time step.
            # This prevents any thread from looping back and starting the next
            # timepoint before all threads have finished the cleanup phase.
            self.device.bariera.wait()