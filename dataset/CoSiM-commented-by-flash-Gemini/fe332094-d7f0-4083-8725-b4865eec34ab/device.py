"""
@fe332094-d7f0-4083-8725-b4865eec34ab/device.py
@brief Distributed sensor network simulation with round-robin task distribution and spatial locking.
This module implements a coordinated parallel processing framework where node managers 
partition computational tasks among a persistent local pool of 8 worker threads 
(DeviceThread). Load balancing is achieved through round-robin assignment of scripts 
to worker buffers. Consistency is guaranteed through a network-wide pool of 
spatial locks, while temporal synchronization across simulation cycles is enforced 
by a monitor-based reusable barrier.

Domain: Load Balancing, Parallel Worker Pools, Distributed Spatial Mutex.
"""

from threading import Thread, Condition, Lock

class ReusableBarrier(object):
    """
    Monitor-based reusable barrier implementation.
    Functional Utility: Provides a temporal rendezvous point for a fixed group 
    of threads using a condition variable to signal threshold arrival.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: participants count.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads 
        self.cond = Condition()               

    def wait(self):
        """Blocks the caller until the arrival threshold is reached."""
        self.cond.acquire()                   
        self.count_threads -= 1
        if self.count_threads == 0:
            # Threshold reached: wake all participants and reset for next cycle.
            self.cond.notify_all()            
            self.count_threads = self.num_threads 
        else:
            self.cond.wait()                 
        self.cond.release()                  


class Device(object):
    """
    Representation of a node in the sensor network simulation.
    Functional Utility: Manages local data state, coordinates global synchronization 
    resource distribution, and supervises a pool of persistent worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id 
        self.sensor_data = sensor_data 
        self.supervisor = supervisor
        
        # Local Barrier: coordinates the internal pool + the assignment phase.
        self.scripts_received = ReusableBarrier(9)
        # Worker-local task buffers.
        self.scripts = {} 
        self.devices = None
        self.timepoint_done = None
        # Spatial lock repository populated during setup.
        self.semafor = {}
        self.thread_list = []
        # Internal barrier for local thread group synchronization.
        self.neighbours_barrier = ReusableBarrier(8)
        # Round-robin assignment pointer.
        self.contor = 0

        # Spawns persistent worker pool.
        for i in range(8):
            self.thread_list.append(DeviceThread(self, i))
            self.scripts.update({i:[]})

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource factory.
        Logic: Coordinator node (ID 0) initializes the global step barrier and 
        pre-allocates a mutex for every sensor location detected in the entire group.
        """
        if self.device_id == 0:
            self.devices = devices
            
            # Global Barrier: sized for every worker thread in the entire network.
            self.timepoint_done = ReusableBarrier(8 * len(self.devices))
            
            # Block Logic: Spatial Lock discovery.
            for device in self.devices:
                for location in device.sensor_data:
                    if location not in self.semafor:
                        # Allocation: ensure a lock exists for every spatial node.
                        self.semafor.update({location: Lock()})
                
                if device.device_id != 0:
                    # Propagation: distribute resources to all peer nodes.
                    device.initialize_device(self.timepoint_done, self.semafor, self.devices)
            
            # Activate coordinator threads.
            for thread in self.thread_list:
                thread.start()

    def initialize_device(self, timepoint_done, semafor, devices):
        """Injection helper for distributed synchronization resources."""
        self.timepoint_done = timepoint_done
        self.semafor = semafor
        self.devices = devices
        
        # Activate worker pool.
        for thread in self.thread_list:
            thread.start()

    def assign_script(self, script, location):
        """
        Workload Distribution.
        Logic: Uses round-robin assignment to populate worker buffers.
        """
        if script is not None:
            # Map script to a specific worker.
            self.scripts[self.contor%8].append((script, location))
            self.contor += 1
        else:
            # Signal end of step workload assignment.
            self.scripts_received.wait()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        value = None
        if location in self.sensor_data:
            value = self.sensor_data[location]
        return value

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully joins all persistent worker threads."""
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    """
    Persistent worker thread implementation with role-based coordination.
    Functional Utility: Executes a partition of computational scripts while 
    participating in multi-stage network synchronization.
    """

    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id
        self.neighbours = None

    def initialize_neighbours(self, neighbours):
        """Updates the cached neighborhood topology."""
        self.neighbours = neighbours

    def run(self):
        """
        Main worker execution loop.
        Algorithm: Iterative multi-phase synchronization: 
        Topology Discovery -> Task Distribution -> Parallel Process -> Consensus.
        """
        while True:
            # Phase 1: Role-Based Topology discovery.
            # Logic: only thread 0 per node fetches the graph and propagates to siblings.
            if self.thread_id == 0:
                self.neighbours = self.device.supervisor.get_neighbours()
                for thread in self.device.thread_list:
                    if thread.thread_id != 0:
                        thread.initialize_neighbours(self.neighbours)

            # Local Barrier: ensure all sibling threads share the same topology.
            self.device.neighbours_barrier.wait()

            if self.neighbours is None:
                # Termination Consensus.
                self.device.timepoint_done.wait()
                break

            # Phase 2: Wait for workload assignment completion.
            self.device.scripts_received.wait()

            # Block Logic: Computational Execution.
            # Threads process their local round-robin partitions.
            for (script, location) in self.device.scripts[self.thread_id]:
                
                # Critical Section: Spatial mutual exclusion across the entire network.
                self.device.semafor[location].acquire()
                script_data = []
                
                # Aggregate neighborhood data.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Include local sensor state.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                    # Optimization: skip processing if only local data exists.
                    if len(script_data) == 1:
                        self.device.semafor[location].release()
                        continue

                if script_data != []:
                    # Compute result and propagate to neighbors and self.
                    result = script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
                
                # Release spatial lock.
                self.device.semafor[location].release()

            # Global Consensus rendezvous: ensure network-wide simulation alignment.
            self.device.timepoint_done.wait()
