"""
@c2698b84-2144-4301-b933-37fed4755b5e/device.py
@brief Distributed sensor network simulation using transient script-specific threads.
This module implements a parallel processing model where individual computational 
tasks are assigned to temporary worker threads. It utilizes condition-variable 
barriers to synchronize simulation timepoints and employs a conditional update 
strategy for sensor state propagation.

Domain: Concurrent Task Spawning, Condition-Based Barriers, Distributed Simulation.
"""

from threading import Event, Thread, Condition

class ReusableBarrierCond():
    """
    Barrier implementation utilizing threading.Condition for thread rendezvous.
    Functional Utility: Synchronizes a predefined number of threads at a single 
    execution point, resetting automatically for subsequent reuse.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks the calling thread until all participants have arrived at the barrier."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Threshold reached: wake all waiting threads and reset the counter.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Representation of a physical or logical sensor node.
    Functional Utility: Acts as a central repository for local sensor data and 
    orchestrates the spawning of processing threads for assigned scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.devices = None
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) initializes a network-wide barrier and 
        distributes it to all peer devices.
        """
        self.devices = devices
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(self.devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """Queues a computational task and signals the local management thread."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for the specified location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates the local sensor state for the specified location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully terminates the device's management and worker threads."""
        self.thread.join()

class ScriptsThread(Thread):
    """
    Transient worker thread responsible for executing a subset of scripts.
    Functional Utility: Implements neighborhood data aggregation and 
    conditional state updates.
    """

    def __init__(self, device, scripts, neighbours):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        """
        Main execution loop for the script worker.
        Logic: Aggregates data from neighbors and applies the script logic, 
        propagating the result based on a comparison check.
        """
        for (script, location) in self.scripts:
            script_data = []
            
            # Aggregate data from the neighborhood.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    # Functional Utility: Basic filtering to ensure data relevance.
                    if data != self.device.get_data(location):
                        script_data.append(data)
                
            # Include local state.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Apply computational logic.
                result = script.run(script_data)

                # Propagation Logic: Updates neighbors if the result exceeds current values.
                # Note: Logic assumes result is comparable and uses it as a location index.
                for device in self.neighbours:
                    if result > device.get_data(result):
                        device.set_data(location, result)
                    
                if result > self.device.get_data(result):
                    self.device.set_data(location, result)
        
        # Local consensus point for script workers.
        self.device.thread.barrier.wait()

class DeviceThread(Thread):
    """
    Management thread for the node's task lifecycle.
    Functional Utility: Orchestrates simulation timepoints and manages the 
    partitioning of scripts among transient worker threads.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = None
        self.list_of_threads = []


    def run(self):
        """
        Main simulation loop for the device manager.
        Algorithm: Iterative worker spawning with double-barrier coordination.
        """
        while True:
            # Refresh topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for simulation start signal.
            self.device.timepoint_done.wait()
            now_thread = 0
            now_script = 0
            
            # Block Logic: Dynamic worker spawning and task assignment.
            for script in self.device.scripts:
                # Logic Note: Implements a 8-thread partition but contains potential indexing errors.
                if now_script == 8:
                    now_script = 0
                else:
                    if now_script < 8:
                        self.list_of_threads.append(ScriptsThread(self.device, [script], neighbours))
                    else:
                        # Attempt to append to existing worker task list.
                        self.list_of_threads[now_thread].scripts.add(script)
                now_thread += 1
                now_script += 1
            
            # Create a barrier for the current set of worker threads.
            self.barrier = ReusableBarrierCond(len(self.list_of_threads))
            
            # Lifecycle: Launch and reclaim workers.
            for thread in self.list_of_threads:
                thread.start()

            for thread in self.list_of_threads:
                thread.join()
            
            # Phase Reset.
            self.list_of_threads = []
            self.device.timepoint_done.clear()
            # Global Consensus.
            self.device.barrier.wait()
            self.list_of_threads = []
