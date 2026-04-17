"""
@f0ab588a-c7c5-4464-928f-bdd6d83e32f5/device.py
@brief Distributed sensor network simulation with tightly-coupled persistent worker pools.
This module implements a highly synchronized parallel processing framework where 
all worker threads across the network are temporally aligned via a global 
barrier. Each node manages a pool of 8 persistent threads that consume computational 
tasks from a shared node-level stack. Consistency is guaranteed through a 
statically-allocated, lazily-expanded pool of global spatial locks and 
fine-grained node-level mutexes for sensor data protection.

Domain: Tightly-Coupled Parallelism, Shared-Stack Task Distribution, Static Resource Pools.
"""

from threading import Event, Thread, Lock, Condition



class ReentrantBarrier(object):
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
    Representation of a node in the sensor network.
    Functional Utility: Manages local data state, coordinates the discovery 
    and expansion of global synchronization resources, and supervises worker threads.
    """

    # Static Network-Wide Synchronization Resources.
    barrier = None
    devices_lock = Lock()
    locations = []
    nrloc = 0


    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        # Mutex for local sensor data protection.
        self.sensor_data_lock = Lock()

        self.supervisor = supervisor
        # General node-level mutex.
        self.gen_lock = Lock()

        self.script_lock = Lock()
        self.script_event = Event()
        self.scripts = []
        # Transient stack of tasks for the current simulation step.
        self.working_scripts = []

        self.neighbour_request = False
        self.neighbour = None
        self.timepoint_done = False
        self.reinit_barrier = None

        # Degree of local parallelism.
        self.threads_num = 8
        self.threads = []
        for i in xrange(self.threads_num):
            self.threads.append(DeviceThread(self, i))


    def __str__(self):
        return "Device %d" % self.device_id


    def setup_devices(self, devices):
        """
        Global synchronization resource factory and initialization.
        Logic: Configures a local group barrier and participates in the 
        lazy expansion of the global spatial lock pool.
        """
        with self.gen_lock:
            # Local Barrier: coordinates threads within this specific node.
            self.reinit_barrier = ReentrantBarrier(self.threads_num)

        with Device.devices_lock:
            # Block Logic: Global Spatial Lock Pool Expansion.
            # Logic: ensures the static pool has enough mutexes for all sensor locations.
            Device.nrloc = max(Device.nrloc, (max(self.sensor_data.keys())+1) if self.sensor_data else 0)
            while Device.nrloc != len(Device.locations):
                Device.locations.append(Lock())

            # Block Logic: Global Network Barrier Setup.
            if Device.barrier is None:
                # Sized to synchronize EVERY thread in the network group.
                Device.barrier = ReentrantBarrier((len(devices) * self.threads_num))

        # Activate the persistent worker pool.
        for i in xrange(self.threads_num):
            self.threads[i].start()


    def assign_script(self, script, location):
        """Registers a computational task into the node's local and working buffers."""
        with self.script_lock:
            if script is not None:
                self.scripts.append((script, location))
                self.working_scripts.append((script, location))
            else:
                # Signal end of step workload.
                self.timepoint_done = True
            
            # Wake any workers waiting for tasks.
            self.script_event.set()


    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        with self.sensor_data_lock:
            return self.sensor_data[location] \
                    if location in self.sensor_data else None


    def set_data(self, location, data):
        """Updates local sensor state."""
        with self.sensor_data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data


    def shutdown(self):
        """Gracefully joins all local worker threads."""
        for i in xrange(self.threads_num):
            self.threads[i].join()



class DeviceThread(Thread):
    """
    Worker thread implementation with role-based coordination.
    Functional Utility: Participates in a network-wide tightly-coupled execution 
    cycle, consuming tasks from a shared node-level stack.
    """


    def __init__(self, device, thread_nr):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.t_num = thread_nr


    def run_script(self, script, location):
        """
        Computational logic for a single task.
        Logic: Implements atomic read-modify-write via the static spatial lock pool.
        """
        with Device.locations[location]:
            script_data = []
            
            # Aggregate neighborhood state.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Include local state.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Apply computational script and propagate results.
                result = script.run(script_data)

                for device in self.device.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)


    def run(self):
        """
        Main worker execution loop.
        Algorithm: Multi-stage synchronization sequence: 
        Network Barrier -> Step Initialization -> Topology Discovery -> Parallel Execution.
        """
        while True:
            # Phase 1: Network-wide temporal alignment.
            Device.barrier.wait()
            
            # Phase 2: State Reset.
            # Logic: one thread per node refreshes the task stack and step metadata.
            with self.device.script_lock:
                if len(self.device.working_scripts) == 0:
                    self.device.working_scripts = list(self.device.scripts)
                    self.device.timepoint_done = False
                    self.device.neighbour_request = False

            # Local Barrier: ensure node initialization is complete.
            self.device.reinit_barrier.wait()
            
            # Phase 3: Role-Based Topology Discovery.
            # Logic: uses a 'first-come-first-served' pattern to pick a coordinator.
            Device.devices_lock.acquire()
            if self.device.neighbour_request == False:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbour_request = True
            Device.devices_lock.release()

            # Exit Logic.
            if self.device.neighbours is None:
                break

            # Block Logic: Workload Consumption.
            # Threads pull tasks from the shared 'working_scripts' stack until empty.
            while True:
                self.device.script_lock.acquire()
                if len(self.device.working_scripts) != 0:
                    # Atomic task acquisition.
                    (script, location) = self.device.working_scripts.pop()
                    self.device.script_lock.release()

                elif self.device.timepoint_done == True:
                    # Current workload drained: prepare for consensus.
                    self.device.script_lock.release()
                    break

                else:
                    # Task buffer empty but workload not finished: wait for signal.
                    self.device.script_event.clear()
                    self.device.script_lock.release()
                    self.device.script_event.wait()
                    continue
                
                if script is not None:
                    # Execute computational logic.
                    self.run_script(script, location)
