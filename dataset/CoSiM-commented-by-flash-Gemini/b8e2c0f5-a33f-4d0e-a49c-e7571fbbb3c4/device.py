"""
@file device.py
@brief Distributed sensor unit simulation with master-worker architecture and cyclic synchronization.
@details Implements a peer-to-peer network of sensing devices that perform synchronized data 
aggregation. Uses a master thread for cycle coordination and a pool of 8 worker threads 
for concurrent script execution, synchronized via a shared queue and a global cyclic barrier.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty

class ReusableBarrierCond():
    """
    @brief implementation of a reusable cyclic barrier using Condition variables.
    Functional Utility: Synchronizes a fixed group of threads across recurring execution cycles.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()                  
                                                 
    def wait(self):
        """
        @brief Blocks the calling thread until the threshold of arriving threads is met.
        """
        self.cond.acquire()                      
        self.count_threads -= 1
        if self.count_threads == 0:
            # Release all threads and reset the counter for the next cycle.
            self.cond.notify_all()               
            self.count_threads = self.num_threads
        else:
            # Passive wait for the release signal.
            self.cond.wait()
        self.cond.release()                     


class Device(object):
    """
    @brief Logic controller for an autonomous sensing entity in a distributed cluster.
    Functional Utility: Manages local data buffers, coordinates a local worker pool, 
    and shares synchronization resources with network peers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of local sensor location readings.
        @param supervisor entity providing topology discovery and coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Synchronization Primitives.
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        
        # Architecture: Primary management thread.
        self.myThread = myThread(self)
        self.myThread.start()
        
        self.neighbours = []
        self.barrier = None
        self.locks = []
        # Task Distribution: Synchronized queue for worker pool.
        self.queue = Queue()
        self.programEnded = False
        
        # Architecture: Spawns 8 persistent worker threads.
        self.deviceThreads = []
        for _ in xrange(8):
            worker = DeviceThread(self)
            self.deviceThreads.append(worker)
            worker.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of shared cluster resources.
        Logic: Designates device 0 as the master allocator for global barriers and locks.
        """
        devicesNumber = len(devices)
        if self.device_id == 0:
            barrier = ReusableBarrierCond(devicesNumber)
            # Allocation: Creates a registry of 24 locks for sensor locations.
            locks = [Lock() for _ in xrange(24)]
            
            # Propagation: Shares resources with all peer devices.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        @brief Schedules a processing script for the current unit of time.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Protocol: Signals end of the script submission phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieval of local sensor readings.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        @brief update of local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device lifecycle threads.
        """
        self.myThread.join()
        for deviceThread in self.deviceThreads:
            deviceThread.join()


class DeviceThread(Thread):
    """
    @brief worker thread responsible for executing data aggregation scripts.
    Functional Utility: Implements distributed Map-Reduce operation.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Continuous consumer loop for the device task queue.
        """
        while not self.device.programEnded:
            # Sync: Wait for the Master thread to signal the start of a workload.
            self.device.script_received.wait()

            try:
                # Non-blocking acquisition: Workers compete for tasks in the shared queue.
                script, location = self.device.queue.get(block=False)
                
                /**
                 * Block Logic: Critical section for distributed state update.
                 * Invariant: Exclusive access to the location across the cluster.
                 */
                self.device.locks[location].acquire()
            
                script_data = []
                # Map Phase: Aggregates state from topological neighbors.
                for peer in self.device.neighbours:
                    data = peer.get_data(location)
                    if data is not None:
                        script_data.append(data)
                        
                # Local state inclusion.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Computation Phase.
                    result = script.run(script_data)
                    
                    # Reduce/Update: Propagates result back to participants.
                    for peer in self.device.neighbours:
                        peer.set_data(location, result)
                    self.device.set_data(location, result)

                self.device.locks[location].release()
                
                # Notify Master of task completion.
                self.device.queue.task_done()
            
            except Empty:
                # Logic: No more tasks in queue for this cycle.
                pass
            
class myThread(Thread):
    """
    @brief Orchestrator thread managing timepoint cycles and task injection.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Master %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main coordination loop: discovery -> injection -> sync.
        """
        while True:
            # Discovery: Fetches current network topology.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            
            if self.device.neighbours is None:
                # Termination: Signals shutdown to all workers.
                self.device.programEnded = True
                self.device.script_received.set()
                break
            
            # Sync: Wait for local task assignments to conclude.
            self.device.timepoint_done.wait()
            
            # Task Distribution: Populates the work queue.
            for script_task in self.device.scripts:
                self.device.queue.put(script_task)
                
            # Activation: Triggers the worker pool to start processing.
            self.device.script_received.set()
            
            # Synchronization: Wait for all local tasks to be finalized.
            self.device.queue.join()
            
            # Reset and Global Sync.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
