"""
@bcc6451f-dc54-41c8-b0d0-42257f16922c/device.py
@brief Distributed sensor processing simulation using a partitioned thread pool and condition-based barriers.
* Algorithm: Static work partitioning (Modulo-8) for worker threads with per-location mutual exclusion.
* Functional Utility: Manages script execution cycles across a device cluster, ensuring synchronized transitions between timepoints.
"""

from threading import Event, Thread, Lock, Condition

class Device(object):
    """
    @brief Core device node that manages local sensor data and executes assigned scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and bootstraps the main control thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = ReusableBarrierCond(0)
        self.dict = {} # Intent: Maps sensor locations to their respective synchronization locks.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collective resource initialization and distribution.
        Pre-condition: Root device (ID 0) performs the allocation of shared barriers and location locks.
        """
        if self.device_id == 0:
            idroot = 0
            self.barrier = ReusableBarrierCond(len(devices))
            
            # Logic: Aggregates all sensor locations across the cluster and assigns a Lock to each.
            for j in xrange(len(devices)):
                if devices[j].device_id == 0:
                    idroot = j
                for location in devices[j].sensor_data:
                    self.dict[location] = Lock()
            
            # Logic: Distributes the shared barrier and global lock map to all devices.
            for k in xrange(len(devices)):
                devices[k].barrier = devices[idroot].barrier
                for j in xrange(len(self.dict)):
                    devices[k].dict[j] = self.dict[j]

    def assign_script(self, script, location):
        """
        @brief Queues a script for execution and signals the arrival of work.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: Signals that all scripts for the current phase have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves data for a specific sensor location.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the data for a specific sensor location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Terminates the device's main management thread.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief Management thread that orchestrates simulation phases and spawns worker threads.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main control loop for the device lifecycle.
        Algorithm: Barrier-synchronized phases for neighbor discovery and parallel script execution.
        """
        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # Synchronization Phase: Ensures all devices have completed neighbor discovery.
            self.device.barrier.wait()
            if neighbours is None:
                break
            
            # Block Logic: Waits for the current timepoint's scripts to be fully assigned.
            self.device.timepoint_done.wait()
            if self.device.scripts is None:
                break
            
            threadsnew = []
            # Block Logic: Static Work Partitioning.
            # Strategy: Distributes assigned scripts among 8 worker threads using round-robin (modulo) logic.
            for j in xrange(8):
                lis = []
                k = 0
                for (script, loc) in self.device.scripts:
                    if k % 8 == j:
                        lis.append((script, loc))
                    k = k + 1

                threadsnew.append(MyThread(self.device, neighbours, lis))
            
            # Logic: Parallel execution of the current batch of scripts.
            for thread in threadsnew:
                thread.start()
            for thread in threadsnew:
                thread.join()
            
            # Post-condition: Reset phase state for the next timepoint.
            self.device.timepoint_done.clear()

class MyThread(Thread):
    """
    @brief Worker thread responsible for executing a subset of assigned scripts.
    """
    
    def __init__(self, device, neighbours, lis):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.lis = lis

    def run(self):
        """
        @brief Executes the partitioned script list.
        Invariant: Acquires the global lock for each target location to ensure atomic state updates across neighbors.
        """
        for (script, location) in self.lis:
            self.device.dict[location].acquire()
            script_data = []
            
            # Distributed Aggregation Logic: Collects sensor readings from peer nodes and self.
            for device in self.neighbours:
                # Logic: Attempts to retrieve data from current neighbor node.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
            
            # Execution and Propagation: Computes new state and broadcasts to the neighborhood.
            if script_data != []:
                result = script.run(script_data)
                
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)
            
            self.device.dict[location].release()

class ReusableBarrierCond():
    """
    @brief Implementation of a reusable synchronization barrier using condition variables.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        @brief Barrier wait logic that resets automatically for subsequent phases.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Logic: Last thread resets the count and notifies all peers.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
