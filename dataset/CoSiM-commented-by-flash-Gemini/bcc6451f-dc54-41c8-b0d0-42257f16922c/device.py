"""
@bcc6451f-dc54-41c8-b0d0-42257f16922c/device.py
@brief Distributed sensor network simulation with static task partitioning.
This implementation employs a fixed-size thread pool (8 threads) to process assigned 
computational tasks. Tasks are distributed among threads using an interleaved 
partitioning strategy, and global consistency is maintained through a network-wide 
barrier and a centralized map of location-specific locks.

Domain: Parallel Task Partitioning, Distributed Mutex, Load Balancing.
"""

from threading import Event, Thread, Lock, Condition

class Device(object):
    """
    Coordinator entity for a network node.
    Functional Utility: Manages local state, synchronizes with the network, and 
    orchestrates the partitioning of tasks among internal workers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device.
        @param device_id: Unique integer identifier.
        @param sensor_data: Initial sensor values.
        @param supervisor: Topology manager.
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
        # Dictionary of locks for every unique sensor location in the network.
        self.dict = {}

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared global resources.
        Logic: The leader (Device 0) creates a reusable barrier and pre-allocates 
        a dictionary of Locks for every sensor location detected in the group.
        """
        if self.device_id == 0:
            idroot = 0
            
            self.barrier = ReusableBarrierCond(len(devices))
            # Block Logic: Discovers all sensor locations and creates a global lock map.
            for j in xrange(len(devices)):
                if devices[j].device_id == 0:
                    idroot = j
                for location in devices[j].sensor_data:
                    self.dict[location] = Lock()

            # Propagation: Distributes the shared barrier and lock map to all peer devices.
            for k in xrange(len(devices)):
                devices[k].barrier = devices[idroot].barrier
                for j in xrange(len(self.dict)):
                    devices[k].dict[j] = self.dict[j]

    def assign_script(self, script, location):
        """Registers a computational task for the current simulation step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main device orchestration thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Main orchestration thread for the node.
    Functional Utility: Manages the lifecycle of the node and partitions scripts 
    into buckets for parallel processing.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Execution loop for the device manager.
        Algorithm: Iterative timepoint processing with interleaved task distribution.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            
            # Global Barrier: Synchronizes topology discovery across all devices.
            self.device.barrier.wait()
            if neighbours is None:
                break
            
            # Wait for supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            if self.device.scripts is None:
                break

            threadsnew = []
            
            # Block Logic: Interleaved Task Partitioning.
            # Logic: Distributes scripts across 8 worker threads using modulo indexing 
            # to balance computational load.
            for j in xrange(8):
                lis = []
                k = 0
                for (script, loc) in self.device.scripts:
                    if k % 8 == j:
                        lis.append((script, loc))
                    k = k + 1

                threadsnew.append(MyThread(self.device, neighbours, lis))

            # Spawns and joins the worker pool for the current timepoint.
            for thread in threadsnew:
                thread.start()
            for thread in threadsnew:
                thread.join()
            
            self.device.timepoint_done.clear()

class MyThread(Thread):
    """
    Worker thread responsible for a partition of tasks.
    Functional Utility: Executes a subset of scripts assigned by the parent DeviceThread.
    """
    
    def __init__(self, device, neighbours, lis):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.lis = lis

    def run(self):
        """
        Execution loop for the worker.
        Logic: Processes each task in the bucket while maintaining atomic access 
        to shared sensor locations.
        """
        for (script, location) in self.lis:
            # Atomic acquisition of the target sensor location lock.
            self.device.dict[location].acquire()
            script_data = []
            
            # Aggregate neighborhood state.
            for device in self.neighbours:
                # Logic Note: Implementation accesses the parent device's data repeatedly.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Include local data.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Apply computational logic and propagate results to the neighborhood graph.
                result = script.run(script_data)
                
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)
            
            # Release the lock for other threads or devices.
            self.device.dict[location].release()

class ReusableBarrierCond():
    """
    Reusable barrier implementation using threading.Condition.
    Functional Utility: Implements a gate-based synchronization point for a fixed 
    number of threads.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
        
    def wait(self):
        """Blocks the calling thread until the threshold is reached."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Threshold reached: notify all and reset for future cycles.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
