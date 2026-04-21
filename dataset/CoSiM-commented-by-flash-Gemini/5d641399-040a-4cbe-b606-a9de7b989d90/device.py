
"""
@file device.py
@brief Distributed device simulation framework with dynamic worker batching.

Functional Intent: Implements a multi-threaded execution environment for autonomous 
devices in a simulated network. Features an orchestration layer that manages 
time-discrete processing steps, discovers neighborhood topology, and executes 
scripts in parallel using a throttled worker thread pool. Ensures global 
consistency via barrier synchronization and fine-grained data-location locking.

Domain: Production Systems, Distributed Computing, Parallel Processing.
"""


from threading import Event, Thread, Lock
import supervisor
from barrier import ReusableBarrierSem


class Device(object):
    """
    @brief Represents a physical node in the distributed simulation.
    
    Functional Utility: Manages local sensor state and coordinates task execution. 
    Interfaces with a global supervisor and utilizes a dedicated manager thread 
    to handle parallel script processing across a local worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device and starts its management thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Synchronization: Events for coordinating script assignment and cluster setup.
        self.script_received = Event()
        self.ready = Event()
        
        self.scripts = [] # Logic: Queue of assigned scripts for the current epoch.
        self.locations = [] # Logic: Shared lock pool for location-specific data protection.
        self.get_data_lock = Lock() # Logic: Local mutex for internal sensor state.
        
        self.devices = None
        self.barrier = None
        
        # Control: Spawns the dedicated orchestration thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Orchestrates the cluster-wide initialization of shared resources.
        
        Algorithm: Designated master initialization (Device 0).
        Logic: The master device creates a global barrier and a pool of shared 
        mutexes for protecting distinct data addresses, then distributes them 
        to all peers to ensure synchronization symmetry.
        """
        self.devices = devices
        # Synchronization: Global rendezvous for epoch alignment.
        barrier = ReusableBarrierSem(len(devices))
        
        # Pre-condition: Only the first device performs setup to avoid object duplication.
        if self.device_id == 0:
            # Block Logic: Data-plane protection.
            # Invariant: Initializes 150 locks to cover the expected range of data locations.
            i = 0
            while i < 150:
                self.locations.append(Lock())
                i = i + 1

            # Side Effect: Propagates synchronization objects to all peer devices.
            for dev in devices:
                dev.barrier = barrier
                dev.locations = self.locations
                dev.ready.set()

    def assign_script(self, script, location):
        """
        @brief Binds a transformation script to a specific data location for this step.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signaling None indicates that task assignment is complete for the current epoch.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of local sensor data.
        """
        # Synchronization: Critical section for protecting the internal data map.
        with self.get_data_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates local sensor state.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the orchestration thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Orchestration thread responsible for lifecycle management and task scheduling.
    
    Logic: Manages the transition between neighborhood discovery, local thread 
    spawning, and global barrier points.
    """

    def __init__(self, device):
        Thread.__init__(self, name="%d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core operational loop for the device manager.
        
        Algorithm: Dynamic task batching.
        Logic: 
        1. Waits for cluster-wide readiness.
        2. Identifies neighbors and waits for task assignments.
        3. Spawns worker threads in throttled batches to prevent thread explosion.
        4. Synchronizes at the global barrier after local tasks are completed.
        """
        self.device.ready.wait()

        while True:
            # Block Logic: Topology discovery.
            neigh = self.device.supervisor.get_neighbours()
            if neigh is None:
                break

            self.device.script_received.wait()
            self.device.script_received.clear()

            rem_scripts = len(self.device.scripts)
            threads = []
            
            # Task Preparation: Pre-instantiates worker threads for each script.
            i = 0
            while i < rem_scripts:
                threads.append(MyThread(self.device, neigh, self.device.scripts, i))
                i = i + 1

            # Block Logic: Throttled parallel execution.
            # Optimization: Limits concurrency to 8 active workers to balance overhead and throughput.
            if rem_scripts < 8:
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            else:
                pos = 0
                while rem_scripts != 0:
                    if rem_scripts > 8:
                        # Logic: Processes exactly 8 workers in the current batch.
                        for i in range(pos, pos + 8):
                            threads[i].start()
                        for i in range(pos, pos + 8):
                            threads[i].join()
                        pos = pos + 8
                        rem_scripts = rem_scripts - 8
                    else:
                        # Logic: Processes the remaining workers in the final batch.
                        for i in range(pos, pos + rem_scripts):
                            threads[i].start()
                        for i in range(pos, pos + rem_scripts):
                            threads[i].join()
                        rem_scripts = 0

            # Synchronization: Global consensus point before proceeding to the next time-step.
            self.device.barrier.wait()


class MyThread(Thread):
    """
    @brief Worker thread responsible for executing a single data-processing script.
    
    Algorithm: Neighborhood state reduction and update.
    """

    def __init__(self, device, neigh, scripts, index):
        Thread.__init__(self, name="%d" % device.device_id)
        self.device = device
        self.neigh = neigh
        self.scripts = scripts
        self.index = index

    def run(self):
        """
        @brief Worker execution cycle.
        
        Logic: 
        1. Identifies its specific script and target location.
        2. Acquires a global lock for the specific data location.
        3. Aggregates data from all neighborhood peers.
        4. Executes the script and broadcasts results back to peers.
        """
        (script, loc) = self.scripts[self.index]
        
        # Synchronization: Ensures atomic access to the data location across the distributed swarm.
        self.device.locations[loc].acquire()
        
        info = []
        # Block Logic: Distributed data gathering.
        for neigh_iter in self.neigh:
            aux_data = neigh_iter.get_data(loc)
            if aux_data is not None:
                info.append(aux_data)
        
        aux_data = self.device.get_data(loc)
        if aux_data is not None:
            info.append(aux_data)
        
        if info != []:
            # Functional Intent: Executes user-defined processing logic.
            result = script.run(info)
            
            # Block Logic: Result propagation.
            for neigh_iter in self.neigh:
                neigh_iter.set_data(loc, result)
                self.device.set_data(loc, result)
        
        self.device.locations[loc].release()
