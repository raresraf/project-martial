"""
@file device.py
@brief Distributed sensor processing system with parallel script execution and cyclic synchronization.
@details Implements a peer-to-peer network of sensing units that perform synchronized data 
aggregation. Uses a master coordination thread and dynamic worker threads, with task 
partitioning and cluster-wide barriers for consistency.
"""

from threading import Event, Thread, Lock
import Barrier

class Device(object):
    """
    @brief Controller for an autonomous sensing entity in a distributed cluster.
    Functional Utility: Manages local data buffers, coordinates parallel task execution, 
    and shares synchronization references (barriers and locks) with network peers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of local sensor location-reading pairs.
        @param supervisor entity providing topology discovery and cluster coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.setup_done = Event()
        self.devices = []
        self.barrier = None
        self.locks = None
        self.timepoint_done = Event()
        # Lifecycle: Spawns the main device orchestrator thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of shared synchronization resources.
        Logic: Designates device 0 as the allocator for the global barrier and lock registry.
        """
        for device in devices:
            if self.device_id != device.device_id:
                self.devices.append(device)

        if self.device_id == 0:
            # Allocation: Shared cyclic barrier for all devices in the cluster.
            self.barrier = Barrier.Barrier(len(devices))
            # Allocation: Global lock registry for synchronizing access to sensor locations.
            self.locks = {}
            
            # Propagation: Shares allocated resources with all peer devices.
            for device in devices:
                device.barrier = self.barrier
                device.locks = self.locks

        # Signal completion of the setup phase.
        self.setup_done.set()

    def assign_script(self, script, location):
        """
        @brief Schedules a processing script for the current unit of time.
        Logic: Ensures a shared lock exists for the target location.
        """
        if script is not None:
            # Atomic: Creates a lock for the location if it is newly discovered.
            if location not in self.locks:
                self.locks[location] = Lock()
            self.scripts.append((script, location))
        else:
            # Protocol: Signals end of the script submission phase.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief retrieval of local sensor readings.
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
        @brief Gracefully terminates the device orchestrator thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Orchestrator thread managing timepoint cycles and task partitioning.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    @staticmethod
    def split(script_list, number):
        """
        @brief Load-balancing utility: partitions tasks into N buckets.
        @return A list of sub-lists, distributed round-robin.
        """
        res = [[] for _ in range(number)]
        for i, task in enumerate(script_list):
            res[i % number].append(task)
        return res

    def run_scripts(self, scripts, neighbours):
        """
        @brief execution logic for a partition of tasks.
        Architecture: Implements distributed Map-Reduce for each script in the set.
        """
        for (script, location) in scripts:
            /**
             * Block Logic: Critical section for distributed aggregation.
             * Invariant: Exclusive access to the location across the cluster.
             */
            with self.device.locks[location]:
                script_data = []
                
                # Map Phase: Aggregates state from neighborhood.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Local state inclusion.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Computation: execution of the aggregation script.
                    result = script.run(script_data)
                    
                    # Reduce/Update: Propagates result to participants.
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

    def run(self):
        """
        @brief Main cycle loop: discovery -> partitioning -> parallel execution -> barrier.
        """
        # Sync: ensures all cluster devices have finished setup before starting.
        self.device.setup_done.wait()
        for device in self.device.devices:
            device.setup_done.wait()

        while True:
            # Discovery: Fetches current topological neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Sync: Wait for script assignment phase to conclude.
            self.device.script_received.wait()

            if self.device.scripts:
                /**
                 * Block Logic: Parallel processing via task partitioning.
                 * Logic: Divides local workload into 8 buckets and executes 
                 * each bucket in a separate ephemeral worker thread.
                 */
                partitioned_scripts = self.split(self.device.scripts, 8)
                thread_list = []
                
                for bucket in partitioned_scripts:
                    if bucket:
                        new_thread = Thread(target=self.run_scripts, args=(bucket, neighbours))
                        thread_list.append(new_thread)
                        new_thread.start()

                # Sync: Blocks until all local workers finish their partitions.
                for thread in thread_list:
                    thread.join()

            # State Reset and Global Synchronization.
            self.device.script_received.clear()
            self.device.barrier.wait()
