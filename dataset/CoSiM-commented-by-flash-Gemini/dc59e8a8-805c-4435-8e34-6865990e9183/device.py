"""
@dc59e8a8-805c-4435-8e34-6865990e9183/device.py
@brief Distributed sensor network simulation with on-demand task threading.
This module implements a processing model where each computational script is 
executed in a dedicated transient thread. Global consistency is maintained via 
a static class-level pool of spatial locks, and temporal synchronization across 
the network is enforced through a robust two-phase semaphore-based barrier. 
Note: the current implementation serializes thread execution by joining 
workers immediately after startup.

Domain: Concurrent Task Spawning, Two-Phase Barriers, Static Spatial Locking.
"""

from threading import Event, Thread, Semaphore, Lock


class ReusableBarrier(object):
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Uses a double-gate mechanism with semaphores to ensure 
    perfect temporal alignment across a fixed set of threads.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: total count of threads in the synchronization group.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)
        

    def wait(self):
        """Executes the two-phase arrival and exit protocol."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement and semaphore release."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Release the gate for the group.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for next phase/reuse.
                count_threads[0] = self.num_threads
        threads_sem.acquire()
        


class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data state and provides a central interface 
    for simulation step coordination.
    """
    
    # Shared network-wide barrier.
    barrier = None

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Main lifecycle management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource factory.
        Logic: Node 0 initializes the shared barrier for the entire group.
        """
        Device.barrier = ReusableBarrier(len(devices))

    def assign_script(self, script, location):
        """Registers a computational task and signals the orchestration thread."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal end of assignments for the current step.
            self.timepoint_done.set()
        self.script_received.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Simulated node manager.
    Functional Utility: Orchestrates simulation phases and manages the execution 
    of assigned computational scripts.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for the node orchestration.
        Algorithm: Iterative sequence of task processing followed by global barrier.
        """
        while True:
            # Topology refresh.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            script_index = 0
            script_threads = []
            length_scripts_threads = 0
            
            # Block Logic: Task Processing phase.
            while True:
                if script_index < len(self.device.scripts):
                    # logic note: tries to maintain a pool of 8 active threads.
                    if length_scripts_threads < 8:
                        # Functional Utility: Spawns and joins worker.
                        thread = self.call_threads(neighbours, script_index)
                        if thread.is_alive():
                            script_threads.append((thread, True))
                            length_scripts_threads += 1
                        script_index += 1
                    else:
                        # Cleanup Logic for active threads.
                        local_index = 0
                        while local_index < len(script_threads):
                            if (script_threads[local_index][0].isAlive()
                                    and script_threads[local_index][1] is True):
                                script_threads[local_index][1] = False
                                length_scripts_threads -= 1
                            local_index += 1
                elif self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
                    self.device.script_received.clear()
                    break
                else:
                    # Wait for more tasks to arrive.
                    self.device.script_received.wait()
                    self.device.script_received.clear()

            # Global temporal consensus.
            Device.barrier.wait()

    def call_threads(self, neighbours, index):
        """
        Worker invocation helper.
        Logic: Spawns a worker thread and waits for it to complete. 
        Note: This results in sequential task execution within the node.
        """
        thread = MyThread(self.device, neighbours, self.device.scripts[index])
        thread.start()
        # Synchronization: immediate join prevents true parallelism between scripts.
        thread.join()
        return thread




class MyThread(Thread):
    """
    Transient worker thread for computational tasks.
    Functional Utility: Implements a spatial locking protocol using a static 
    class-level lock repository.
    """
    
    # Static Lock Repository: shared across all worker instances.
    locations_locks = {}

    def __init__(self, device, neighbours, (script, location)):
        Thread.__init__(self)
        self.location, self.script = location, script
        self.device, self.neighbours = device, neighbours

        # On-Demand initialization of spatial mutexes.
        if location not in MyThread.locations_locks:
            MyThread.locations_locks[location] = Lock()

    def run(self):
        """
        Execution logic.
        Logic: Atomically acquires the spatial lock for the location, gathers 
        neighborhood data, and propagates results.
        """
        MyThread.locations_locks[self.location].acquire()
        
        script_data = []
        # Aggregate neighborhood and local data.
        for device in self.neighbours:
            if device.get_data(self.location) is not None:
                script_data.append(device.get_data(self.location))
        
        if self.device.get_data(self.location) is not None:
            script_data.append(self.device.get_data(self.location))

        if script_data != []:
            # Process results.
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
            
        # Release spatial mutex.
        MyThread.locations_locks[self.location].release()
