"""
@eba165cb-a0ad-4529-bf32-979d0e1756c3/device.py
@brief Distributed sensor network simulation with balanced task partitioning and spatial locking.
This module implements a coordinated parallel processing framework where node managers 
partition computational tasks into balanced chunks for a local pool of transient 
worker threads. Global spatial consistency is enforced via a network-wide pool of 
mutexes, and temporal synchronization is achieved through a multi-node setup 
rendezvous and global simulation barriers.

Domain: Load Balancing, Parallel Task Partitioning, Spatial Mutual Exclusion.
"""

from threading import Event, Thread, Lock
import Barrier


class Device(object):
    """
    Representation of a node in the sensor network simulation.
    Functional Utility: Manages local data, coordinates the discovery of shared 
    synchronization resources, and provides the interface for task assignment.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        # Rendezvous event to ensure network-wide setup completion.
        self.setup_done = Event()
        self.devices = []
        self.barrier = None
        self.locks = None
        self.timepoint_done = Event()
        # Primary lifecycle management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) initializes the shared network barrier 
        and the global spatial lock repository, distributing them to all peers.
        """
        for device in devices:
            if self.device_id != device.device_id:
                self.devices.append(device)

        if self.device_id == 0:
            # Atomic setup of shared resources.
            self.barrier = Barrier.Barrier(len(devices))
            self.locks = {}
            
            # Propagation: share resources with all members of the network.
            for device in devices:
                device.barrier = self.barrier
                device.locks = self.locks
                
        # Signal completion of local setup phase.
        self.setup_done.set()

    def assign_script(self, script, location):
        """
        Task registration.
        Logic: On-demand initialization of spatial mutexes to ensure every targeted 
        sensor location has a corresponding lock.
        """
        if script is not None:
            # Logic Note: bitwise inversion on dictionary has_key is unusual but used for existence check.
            if ~((self.locks).has_key(location)):
                self.locks[location] = Lock()
                
            self.scripts.append((script, location))
        else:
            # Finalize assignment phase for current timepoint.
            self.script_received.set()
            

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        res = None
        if location in self.sensor_data:
            res = self.sensor_data[location]
        return res

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully joins the node manager thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Node-level task manager.
    Functional Utility: Orchestrates simulation phases and balances workloads 
    across transient parallel worker threads.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    @staticmethod
    def split(script_list, number):
        """
        Balanced Partitioning Heuristic.
        Algorithm: Round-robin distribution of tasks into N buckets.
        @return: A nested list of task chunks.
        """
        res = [[] for i in range(number)]
        i = 0
        while i < len(script_list):
            part = script_list[i]
            res[i%number].append(part)
            i = i + 1
        return res

    def run_scripts(self, scripts, neighbours):
        """
        Worker execution logic.
        Algorithm: Iterative computational processing with spatial mutual exclusion.
        """
        for (script, location) in scripts:
            # Critical Section: Spatial mutual exclusion across the entire network.
            with self.device.locks[location]:
                script_data = []
                
                # Aggregate neighborhood state.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Include local sensor data.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Compute results and propagate to peers and self.
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)


    def run(self):
        """
        Main execution loop for node orchestration.
        Algorithm: Multi-node setup sync -> Iterative Step Coordination -> Parallel Fork-Join.
        """
        # Block Logic: Network-wide setup synchronization.
        self.device.setup_done.wait()
        for device in self.device.devices:
            device.setup_done.wait()
            

        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until assignment phase is complete.
            self.device.script_received.wait()
            
            if len(self.device.scripts) != 0:
                # Workload Balancing: partition scripts into 8 chunks.
                scripts_list = self.split(self.device.scripts, 8)
                
                thread_list = []
                # Fork: spawn transient worker for each chunk.
                for scripts in scripts_list:
                    new_thread = Thread(target=self.run_scripts,
                                                     args=(scripts, neighbours))
                    thread_list.append(new_thread)
                    
                for thread in thread_list:
                    thread.start()
                    
                # Join: wait for all parallel chunks to finalize before consensus.
                for thread in thread_list:
                    thread.join()
                    

            # Reset assignment signal for next cycle.
            self.device.script_received.clear()
            
            # Global Consensus Point.
            self.device.barrier.wait()
