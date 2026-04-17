"""
@cdb52e67-c579-4815-8f5f-86f78a2f9f67/device.py
@brief Concurrent sensor simulation with batch-oriented worker thread execution and global location locking.
* Algorithm: Fixed-size batch task scheduling (8 threads per batch) with multi-level barrier synchronization and per-location mutual exclusion.
* Functional Utility: Orchestrates simulation timepoints across a network of devices by managing a pool of transient worker threads that perform distributed data processing.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    """
    @brief Encapsulates a sensor node with its local data, shared synchronization primitives, and management thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the main coordinator thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.devices = None
        self.barrier = None
        self.thread = DeviceThread(self)
        self.locations = [] # Intent: Shared list of locks protecting specific sensor locations.
        self.data_lock = Lock() # Intent: Serializes local data updates.
        self.get_lock = Lock()  # Intent: Serializes local data retrieval.
        self.setup = Event()    # Intent: Signals completion of shared resource configuration.
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization and distribution of shared simulation resources.
        Invariant: Root node (ID 0) initializes a pool of 100 location locks and a cluster-wide barrier.
        """
        self.devices = devices
        barrier = ReusableBarrierSem(len(devices))
        if self.device_id == 0:
            # Logic: Pre-allocates a standard set of locks for potential sensor locations.
            for _ in range(100):
                self.locations.append(Lock())
            for dev in devices:
                dev.barrier = barrier
                dev.locations = self.locations
                dev.setup.set()

    def assign_script(self, script, location):
        """
        @brief Receives a processing task for the current simulation phase.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signals that script delivery for this timepoint is complete.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Synchronized retrieval of sensor data for a specific location.
        """
        with self.get_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Synchronized update of sensor data for a specific location.
        """
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device's management thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Main coordination thread implementing batch-based task dispatching and phased synchronization.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator thread for a specific device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core lifecycle loop of the device node.
        Algorithm: Chunked worker thread execution (8 at a time) to manage resource usage and synchronization overhead.
        """
        # Block Logic: Ensures global setup is complete before starting simulation phases.
        self.device.setup.wait()
        while True:
            threads = []
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Logic: Exit simulation.
                break
            
            # Block Logic: Wait for scripts to be assigned for the current timepoint.
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            # Dispatch Phase Preparation: Instantiate a worker thread for each assigned script.
            i = 0
            for _ in self.device.scripts:
                threads.append(MyThread(self.device, self.device.scripts, neighbours, i))
                i = i + 1
            
            scripts_rem = len(self.device.scripts)
            start = 0
            
            # Block Logic: Chunked Execution Strategy.
            # Functional Utility: Limits active concurrency to 8 threads per device to prevent context-switching thrashing.
            if len(self.device.scripts) < 8:
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            else:
                while True:
                    if scripts_rem == 0:
                        break
                    if scripts_rem >= 8:
                        # Logic: Processes exactly 8 workers in parallel and waits for completion.
                        for i in xrange(start, start + 8):
                            threads[i].start()
                        for i in xrange(start, start + 8):
                            threads[i].join()
                        start = start + 8
                        scripts_rem = scripts_rem - 8
                    else:
                        # Logic: Processes the remaining tail of workers.
                        for i in xrange(start, start + scripts_rem):
                            threads[i].start()
                        for i in xrange(start, start + scripts_rem):
                            threads[i].join()
                        break
            
            # Synchronization Phase: Align all devices across the cluster.
            self.device.barrier.wait()

class MyThread(Thread):
    """
    @brief Worker thread dedicated to executing a single sensor script unit.
    """
    
    def __init__(self, device, scripts, neighbours, indice):
        """
        @brief Initializes worker with task parameters and context.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours
        self.indice = indice

    def run(self):
        """
        @brief Main execution logic for a single script unit.
        Algorithm: Resource-locked execution with distributed data aggregation and result propagation.
        """
        (script, location) = self.scripts[self.indice]
        
        # Pre-condition: Acquire global location lock for cluster-wide consistency.
        self.device.locations[location].acquire()
        script_data = []
        
        # Distributed Aggregation Phase: Accumulate data from neighborhood and local node.
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        # Execution Phase: Processes collected data and updates state across the neighborhood.
        if script_data != []:
            result = script.run(script_data)
            for device in self.neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)
        
        # Post-condition: Release location lock.
        self.device.locations[location].release()
