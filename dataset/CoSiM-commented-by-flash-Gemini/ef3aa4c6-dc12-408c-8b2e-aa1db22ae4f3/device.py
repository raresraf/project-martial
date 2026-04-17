"""
@ef3aa4c6-dc12-408c-8b2e-aa1db22ae4f3/device.py
@brief Distributed sensor network simulation with monitor-based spatial mutual exclusion.
This module implements a coordinated parallel processing framework where transient 
worker threads (ScriptThread) execute computational tasks. It utilizes a monitor 
pattern—implemented via a condition variable and a shared list—to enforce 
network-wide mutual exclusion for specific sensor locations. Node managers 
(DeviceThread) orchestrate simulation steps using a fork-join model and temporal 
barriers.

Domain: Monitor-Based Synchronization, Distributed Mutex, Parallel Fork-Join.
"""

from threading import Event, Thread, Lock, Condition
import barrier

class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data state and coordinates global synchronization 
    resources (barrier and spatial monitor) across the group.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Primary orchestration thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        
        # Temporal Rendezvous: initialized during setup.
        self.bariera = barrier.ReusableBarrierCond(1)
        # Mutex for protecting local data state.
        self.data_lock = Lock()
        # Mutex for protecting the script assignment list.
        self.script_lock = Lock()
        
        # Spatial Synchronization: Monitor components.
        self.locationcondition = Condition()
        self.locationlist = []

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource factory.
        Logic: Coordinator node (ID 0) initializes the shared barrier and 
        distributes the spatial monitor (Condition + Registry) to all peer nodes.
        """
        if self.device_id is 0:
            # Atomic setup of shared rendezvous points.
            self.bariera = barrier.ReusableBarrierCond(len(devices))
            for device in devices:
                device.bariera = self.bariera
                device.locationcondition = self.locationcondition
                device.locationlist = self.locationlist

    def assign_script(self, script, location):
        """Registers a task and signals completion of the simulation step assignment."""
        self.script_lock.acquire()
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()
        self.script_lock.release()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        self.data_lock.acquire()
        value = self.sensor_data[location] if location in self.sensor_data else None
        self.data_lock.release()
        return value

    def set_data(self, location, data):
        """Updates local sensor state."""
        self.data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.data_lock.release()

    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Node-level simulation manager.
    Functional Utility: Orchestrates simulation phases and implements a fork-join 
    parallel processing model for computational scripts.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for node orchestration.
        Algorithm: Iterative sequence: 
        Wait -> Fork Sub-threads -> Join Sub-threads -> Global Barrier.
        """
        while True:
            # Topology refresh.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block until workload assignment for the current timepoint is complete.
            self.device.timepoint_done.wait()
            
            self.device.script_lock.acquire()

            # Phase 1: Fork.
            # Logic: Spawns a transient thread for each script in the batch.
            nodes = []
            for (script, location) in self.device.scripts:
                nodes.append(ScriptThread(self.device, script, location,\
                             neighbours, self.device.locationlist,\
                             self.device.locationcondition))
            for j in xrange(len(self.device.scripts)):
                nodes[j].start()

            # Phase 2: Join.
            # Logic: Wait for all parallel sub-tasks to finalize.
            for j in xrange(len(self.device.scripts)):
                nodes[j].join()
            
            # Phase Reset.
            self.device.timepoint_done.clear()
            self.device.script_lock.release()
            
            # Global Consensus point.
            self.device.bariera.wait()

class ScriptThread(Thread):
    """
    Transient worker thread implementation.
    Functional Utility: Implements the monitor-based 'Check-and-Reserve' protocol 
    to ensure spatial consistency across the neighborhood graph.
    """

    def __init__(self, device, script, location, neighbours, locationlist,\
                 locationcondition):
        Thread.__init__(self, name="Service Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.locationlist = locationlist
        self.locationcondition = locationcondition

    def run(self):
        """
        Execution logic.
        Algorithm: Spatial Mutual Exclusion -> Data Aggregation -> Propagate -> Release.
        """
        # Phase 1: Spatial lock acquisition via monitor pattern.
        busy = True
        while busy:
            self.locationcondition.acquire()
            if self.location in self.locationlist:
                # Target location is currently being processed: wait for notification.
                self.locationcondition.wait()
            else:
                # Claim the location by adding to the shared registry.
                self.locationlist.append(self.location)
                busy = False
            self.locationcondition.release()

        # Phase 2: Domain logic.
        script_data = []
        # Aggregate neighborhood state.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            # Include local sensor data.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        if script_data != []:
            # Process results and distribute to peers.
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        
        # Phase 3: Spatial lock release.
        # Logic: Removes location from registry and wakes any threads waiting on this resource.
        self.locationcondition.acquire()
        self.locationlist.remove(self.location)
        self.locationcondition.notify_all()
        self.locationcondition.release()
