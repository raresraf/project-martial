"""
This module implements a multi-threaded simulation of a distributed system of devices.
It features a custom two-phase reusable barrier for temporal synchronization and 
a lazy initialization strategy for location-specific locks to ensure data consistency 
during concurrent script execution.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier():
    """
    A software-defined synchronization barrier designed for cyclic execution.
    
    Implements a two-phase (double-lock) protocol using semaphores to ensure that 
    all threads reach the synchronization point before any are permitted to 
    transition to the next cycle. This prevents threads from 'racing ahead' and 
    re-entering the first phase of the next cycle before others have finished the 
    current one.
    """
    def __init__(self, num_threads):
        """
        Args:
            num_threads (int): Total number of participants in the synchronization.
        """
        self.num_threads = num_threads
        # State: Atomic counters for each phase.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Signaling: Semaphores to block threads until the arrival quota is met.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Main synchronization entry point.
        Blocks the calling thread through two distinct phases to ensure full participation.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the barrier protocol.

        Logic: 
        1. Decrement the local participant counter atomically.
        2. If last thread to arrive, release the semaphore N times to wake all participants.
        3. Reset the phase counter for the next simulation cycle.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Release Block: Wake up all waiting threads simultaneously.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Cycle Reset: Prepare state for the next timepoint.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a discrete computational node in the distributed environment.
    
    Each device manages its own sensor data and can execute assigned logic scripts.
    It utilizes a lazy lock discovery mechanism to share synchronization primitives 
    with peer nodes that access the same physical data locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Args:
            device_id (int): Node identifier.
            sensor_data (dict): Mapping of data location IDs to values.
            supervisor: Registry for neighborhood discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()

        # Lifecycle: Spawns the main control thread for this node.
        self.thread = DeviceThread(self)
        self.barrier = None
        self.list_thread = []
        self.thread.start()
        
        # Concurrency control: Cache for location-specific data locks.
        self.location_lock = [None] * 100

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Bootstrap Logic: Initializes and distributes the global synchronization barrier.
        Ensures all devices in the cluster reference the same ReusableBarrier instance.
        """
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Task Enqueueing: Sets up a processing script for the current timepoint.

        Lock Management: Implements a distributed lazy-initialization pattern.
        If a lock for the requested 'location' doesn't exist locally, it searches 
        neighboring devices to reuse their existing lock, ensuring that any two nodes 
        accessing the same location are synchronized by the same Lock object.
        """
        ok = 0
        if script is not None:
            self.scripts.append((script, location))
            # Block Logic: Distributed Lock Discovery.
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        # Shared reference: Sync with peers already tracking this location.
                        self.location_lock[location] = device.location_lock[location]
                        ok = 1
                        break
                if ok == 0:
                    # Origin: Create new lock if this is the first device to access the location.
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            # Termination: Signals end of the current simulation step.
            self.timepoint_done.set()

    def get_data(self, location):
        """Atomic access to local sensor state."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Atomic update of local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Graceful termination of the control thread."""
        self.thread.join()

class NewThread(Thread):
    """
    Worker Thread: Executes a single algorithmic script on aggregated sensor data.
    """
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Execution Lifecycle:
        1. Acquire the shared lock for the specific data location.
        2. Pull current data from all topological neighbors.
        3. Execute the script's logic.
        4. Push resulting values back to neighbors and local storage.
        """
        script_data = []
        # Critical Section: Ensure exclusive access to the distributed data location.
        self.device.location_lock[self.location].acquire()
        
        # Block Logic: Data Aggregation.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Computation.
        if script_data != []:
            # Action: Transform the collected set of sensor readings.
            result = self.script.run(script_data)
            
            # Action: Distributed state update.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
            
        self.device.location_lock[self.location].release()

class DeviceThread(Thread):
    """
    Main Orchestrator: Manages the temporal progression of a single Node.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Temporal Loop:
        1. Poll for neighbor topology.
        2. Wait for work assignment.
        3. Execute assigned scripts in parallel.
        4. Participate in the global synchronization barrier.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Terminal condition: Supervisor signaled end of simulation.
                break

            # Wait Logic: Wait for the trigger to start processing the current timepoint.
            self.device.timepoint_done.wait()

            # Concurrency Logic: Parallel script execution.
            for (script, location) in self.device.scripts:
                thread = NewThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            # Execution Barrier (Local): Join all internal workers before moving to global sync.
            for thread_elem in self.device.list_thread:
                thread_elem.start()
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            self.device.list_thread = []

            # Global Synchronization: Align with the entire network before starting next iteration.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
