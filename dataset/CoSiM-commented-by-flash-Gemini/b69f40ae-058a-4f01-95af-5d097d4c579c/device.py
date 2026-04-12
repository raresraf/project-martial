"""
@file device.py
@brief Distributed sensor processing system with persistent thread pools and cyclic synchronization.
@details Implements a peer-to-peer network of sensing units that perform synchronized data 
aggregation. Uses a pool of persistent worker threads per device, coordinated via 
reusable barriers and location-specific locks to ensure cluster-wide data consistency.
"""

from threading import Event, Thread, Lock, Condition

class ReusableBarrier(object):
    """
    @brief implementation of a cyclic barrier using Condition variables.
    Functional Utility: Synchronizes a fixed group of threads across recurring execution windows.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        @brief Blocks the calling thread until all expected participants arrive.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Release all waiting threads and reset for the next cycle.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    @brief Logic controller for an autonomous sensing entity in a distributed network.
    Functional Utility: Manages local data buffers, coordinates a persistent worker pool, 
    and shares cluster-wide synchronization references.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of local sensor readings.
        @param supervisor Entity providing topological neighbor discovery services.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.threads = []
        self.bariera = None
        self.lock = Lock()
        
        # State: Flags for coordinating neighbor discovery across local workers.
        self.call_neigh = 1
        self.rupe = 0
        self.numara = 0
        self.neighbours = []
        
        # Synchronization: Primitives for internal thread state management.
        self.numara_lock = Lock()
        self.call_neigh_lock = Lock()
        self.global_lock = None
        self.devices = []
        self.location_dict = {}

        # Architecture: Pre-spawns 8 persistent worker threads.
        for _ in range(8):
            self.threads.append(DeviceThread(self))

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of shared cluster resources.
        Logic: Designates device 0 as the allocator for the cluster barrier and 
        the global lock registry.
        """
        if self.device_id == 0:
            # Allocation: Barrier sized for all workers in the cluster (8 * N).
            self.bariera = ReusableBarrier(8 * len(devices))
            self.global_lock = Lock()

            # Propagation: Shares allocated resources with all peer devices.
            for dev in devices:
                dev.bariera = self.bariera
                dev.global_lock = self.global_lock
        
        self.devices = devices

        # Lifecycle: Activates the persistent worker pool.
        for i in xrange(8):
            self.threads[i].start()

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing script for the current unit of time.
        """
        self.script_received.clear()
        if script is not None:
            self.scripts.append([script, location])
        else:
            # Protocol: Signals end of the script submission phase.
            self.timepoint_done.set()
        self.script_received.set()

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
        @brief Gracefully terminates the worker thread pool.
        """
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    @brief Persistent worker thread managing task execution cycles.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.lock = Lock()
        self.id_th = 0

    def run(self):
        """
        @brief main loop: discovery -> execution -> synchronization -> reset.
        """
        # Initialization: Assigns a unique rank to the worker within the local pool.
        with self.device.numara_lock:
            self.id_th = self.device.numara
            self.device.numara += 1

        while True:
            # Phase 1: Coordinated Neighbor Discovery.
            with self.device.call_neigh_lock:
                if self.device.call_neigh == 1:
                    # Logic: Only one worker per device fetches neighbors to avoid redundancy.
                    self.device.neighbours = self.device.supervisor.get_neighbours()
                    if self.device.neighbours is None:
                        # Termination: Signals shutdown to all workers.
                        self.device.rupe = 1
                        break
                    self.device.call_neigh = 0

            # Exit condition check.
            if self.device.rupe == 1:
                break

            # Phase 2: Parallel Script execution.
            # Sync: Wait for the local assignment phase to conclude.
            self.device.timepoint_done.wait()

            /**
             * Block Logic: Script execution with static partitioning.
             * Logic: Each worker processes a subset of the assigned scripts based 
             * on its internal rank (id_th) and pool size (8).
             */
            for i in xrange(self.id_th, len(self.device.scripts), 8):
                [script, location] = self.device.scripts[i]

                # Synchronization Management: ensures location locks are shared across the cluster.
                with self.device.global_lock:
                    if location not in self.device.location_dict:
                        loc_lock = Lock()
                        for j in xrange(len(self.device.devices)):
                            self.device.devices[j].location_dict[location] = loc_lock

                # Critical Section: exclusive access to the target sensor location.
                with self.device.location_dict[location]:
                    script_data = []
                    # Map Phase: Aggregates state from neighborhood.
                    for peer in self.device.neighbours:
                        data = peer.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    # Inclusion: Local data.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data:
                        # Computation: execution of aggregation logic.
                        result = script.run(script_data)
                        
                        # Reduce Phase: propagates result back to all participants.
                        for peer in self.device.neighbours:
                            if peer.get_data(location) is not None:
                                peer.set_data(location, result)
                        
                        if self.device.get_data(location) is not None:
                            self.device.set_data(location, result)

            # Phase 3: Global Synchronization and Cycle Reset.
            # Barrier 1: Aligns all workers in the cluster at the end of the timepoint.
            self.device.bariera.wait()
            
            # State Management: designated worker (usually rank 0 arrived here) resets flags.
            self.device.call_neigh = 1
            if self.id_th == 0:
                self.device.timepoint_done.clear()
            
            # Barrier 2: Ensures all workers have finished Phase 3 before starting Barrier 1 of next cycle.
            self.device.bariera.wait()
