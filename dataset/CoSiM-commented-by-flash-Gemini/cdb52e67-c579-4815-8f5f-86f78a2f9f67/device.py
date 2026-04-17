"""
@cdb52e67-c579-4815-8f5f-86f78a2f9f67/device.py
@brief Distributed sensor network simulation with synchronous batch task processing.
This module implements a parallel processing architecture where computational scripts 
are executed in discrete batches of 8 threads. This 'Stop-and-Wait batching' strategy 
ensures that system resources are bounded while maintaining high-throughput 
parallelism. Consistency is enforced via network-wide spatial locks and a reusable 
barrier for temporal synchronization.

Domain: Parallel Systems, Batch Processing, Distributed Mutex.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data state and coordinates global synchronization 
    resources (barrier and spatial lock pool).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.devices = None
        self.barrier = None
        self.thread = DeviceThread(self)
        self.locations = []
        self.data_lock = Lock()
        self.get_lock = Lock()
        self.setup = Event()
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) initializes a reusable barrier and a pool 
        of 100 spatial locks for the entire group.
        """
        self.devices = devices
        barrier = ReusableBarrierSem(len(devices))
        if self.device_id == 0:
            # Atomic Resource Allocation: pre-populates a global lock pool.
            for _ in range(100):
                self.locations.append(Lock())
            
            # Propagation: Distributes resources to all peer devices.
            for dev in devices:
                dev.barrier = barrier
                dev.locations = self.locations
                dev.setup.set()

    def assign_script(self, script, location):
        """Registers a task and signals completion of the simulation step assignment."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data using a dedicated mutex."""
        with self.get_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Thread-safe update of local sensor state."""
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main management thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Main orchestration thread for the node.
    Functional Utility: Implements a synchronous batching distributor that spawns 
    and joins parallel worker threads in fixed-size windows.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop.
        Algorithm: Iterative batch processing with stop-and-wait synchronization.
        """
        # Block until global resources are initialized.
        self.device.setup.wait()
        while True:
            threads = []
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for supervisor signal.
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            # Prepare worker objects for all assigned scripts.
            i = 0
            for _ in self.device.scripts:
                threads.append(MyThread(self.device, self.device.scripts, neighbours, i))
                i = i + 1
            
            scripts_rem = len(self.device.scripts)
            start = 0
            # Block Logic: Batch Parallelization logic.
            # Groups workers into sets of 8, waiting for each set to finish before next.
            if len(self.device.scripts) < 8:
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            else:
                while True:
                    if scripts_rem == 0:
                        break
                    # window-based execution.
                    if scripts_rem >= 8:
                        for i in xrange(start, start + 8):
                            threads[i].start()
                        for i in xrange(start, start + 8):
                            threads[i].join()
                        start = start + 8
                        scripts_rem = scripts_rem - 8
                    else:
                        # final partial window.
                        for i in xrange(start, start + scripts_rem):
                            threads[i].start()
                        for i in xrange(start, start + scripts_rem):
                            threads[i].join()
                        break
            
            # Global temporal consensus.
            self.device.barrier.wait()

class MyThread(Thread):
    """
    Worker thread responsible for a single computational script.
    Functional Utility: Executes a script while maintaining global spatial consistency.
    """
    
    def __init__(self, device, scripts, neighbours, indice):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours
        self.indice = indice

    def run(self):
        """
        Execution logic.
        Logic: Implements atomic read-modify-write via global spatial locks.
        """
        (script, location) = self.scripts[self.indice]
        # Acquire global mutex for the specific sensor location.
        self.device.locations[location].acquire()
        script_data = []
        
        # Aggregate neighborhood and local data.
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Apply computational logic.
            result = script.run(script_data)
            
            # Propagate updates to the neighborhood graph.
            for device in self.neighbours:
                device.set_data(location, result)
                self.device.set_data(location, result)
        
        # Release the spatial mutex.
        self.device.locations[location].release()
