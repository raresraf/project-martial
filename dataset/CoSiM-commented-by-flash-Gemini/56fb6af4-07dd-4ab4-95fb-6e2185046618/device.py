
"""
@56fb6af4-07dd-4ab4-95fb-6e2185046618/device.py
@brief Event-driven device simulation with condition-based barrier synchronization.
This module implements a simulation architecture for autonomous devices that 
orchestrate parallel computation threads. It uses a condition-variable based 
reusable barrier for temporal synchronization and an event-based signaling 
system for inter-thread communication, ensuring consistent state transitions 
across distributed sensor data mappings.

Domain: Concurrent Simulation, Synchronization Primitives, Parallel Task Execution.
"""

from threading import Event, Thread, Lock, Semaphore, Condition


class ReusableBarrierCond(object):
    """
    Functional Utility: Implements a re-usable N-thread barrier using condition variables.
    Logic: Threads decrement a counter upon arrival. The last thread to arrive 
    notifies all waiting threads and resets the counter for the next cycle.
    """

    def __init__(self, num_threads):
        """
        Constructor: Initializes the barrier with the target thread count.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Execution Logic: Blocks until all participants reach the rendezvous point.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Block Logic: Final arrival release.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Block Logic: Wait for remaining threads.
            self.cond.wait()
        self.cond.release()


class SignalType(object):
    """
    Functional Utility: Enumeration of internal control signals for thread coordination.
    """
    SCRIPT_RECEIVED = 1
    TIMEPOINT_DONE = 2
    TERMINATION = 3


class Device(object):
    """
    Functional Utility: Represent a simulated hardware unit with local sensor data.
    Logic: Manages device-specific threads and events. It coordinates the 
    assignment of computational scripts and ensures data consistency across 
    spatial locations using per-location locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Constructor: Initializes device identity, data storage, and the coordination thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        
        self.devices_barrier = None
        self.signal_received = Event()
        self.signal_type = None
        self.timepoint_work_done = Event()
        self.signal_sent = Event()
        self.data_locks = {}
        self.scripts_lock = Lock()

        
        self.thread = DeviceThread(self)
        self.thread.start()

        
        # Block Logic: Resource lock initialization.
        # Creates a dedicated lock for each unique sensor location.
        for location in sensor_data:
            self.data_locks[location] = Lock()

        self.devices_lock = {}

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Functional Utility: Cluster-wide initialization for synchronization resources.
        Logic: The leader device (id 0) initializes the shared barrier and 
        location locks across all participants.
        """
        if self.device_id == 0:
            devices_barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                device.devices_barrier = devices_barrier
                for location in device.sensor_data:
                    self.devices_lock[location] = Lock()

                device.devices_lock = self.devices_lock

    def assign_script(self, script, location):
        """
        Functional Utility: Assigns a new computational task to the device.
        Logic: Enqueues the script and signals the main device thread. It 
        waits for confirmation to ensure transactional assignment.
        """
        if script is not None:
            
            with self.scripts_lock:
                self.scripts.append((script, location))

            
            self.signal_type = SignalType.SCRIPT_RECEIVED
            self.signal_received.set()
            
            # Block Logic: Await thread reception.
            self.signal_sent.wait()
            self.signal_sent.clear()

        else:
            # Block Logic: Signal temporal boundary.
            self.signal_type = SignalType.TIMEPOINT_DONE
            self.signal_received.set()
            
            self.signal_sent.wait()
            self.signal_sent.clear()
            
            self.timepoint_work_done.wait()
            self.timepoint_work_done.clear()

    def get_data(self, location):
        """
        Functional Utility: Atomic retrieval of local sensor data.
        """
        if location in self.sensor_data:
            with self.data_locks[location]:
                return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Functional Utility: Atomic update of local sensor data.
        """
        if location in self.sensor_data:
            with self.data_locks[location]:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Functional Utility: Graceful termination sequence.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Functional Utility: Primary coordinator thread for a Device instance.
    Logic: Orchestrates temporal steps by interacting with the supervisor 
    to resolve network topology and signals worker threads to execute 
    enqueued scripts before hitting the global barrier.
    """

    def __init__(self, device):
        """
        Constructor: Binds the coordinator to its device and spawns worker threads.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        
        self.neighbours = None

        
        self.signal_received = Event()
        self.signal_type = None
        self.scripts_index = 0

        self.new_timepoint = Semaphore(0)
        self.signal_lock = Lock()

        
        # Block Logic: Computation worker initialization.
        self.num_threads = 8
        self.timepoint_computation_done = [Event() for _ in range(self.num_threads)]
        self.threads = [ComputationThread(self, count) for count in range(self.num_threads)]
        for count in range(self.num_threads):
            self.threads[count].start()

        self.neighbour_locks = [Lock() for _ in range(self.num_threads)]

    def acquire_neighbours(self):
        """
        Block Logic: Critical section acquisition for topology data.
        """
        for lock in self.neighbour_locks:
            lock.acquire()

    def release_neighbours(self):
        """
        Block Logic: Critical section release for topology data.
        """
        for lock in self.neighbour_locks:
            lock.release()

    def run(self):
        """
        Execution Logic: Infinite simulation control loop.
        """
        while True:
            
            self.acquire_neighbours()
            
            # Retrieves the current set of devices in the spatial neighborhood.
            self.neighbours = self.device.supervisor.get_neighbours()

            
            if self.neighbours is None:
                # Block Logic: Termination protocol.
                self.signal_type = SignalType.TERMINATION
                self.device.signal_sent.set()
                self.release_neighbours()

                for computation_thread_done in self.timepoint_computation_done:
                    computation_thread_done.wait()
                    computation_thread_done.clear()
                break

            self.release_neighbours()

            # Block Logic: Inner temporal signal processing loop.
            while True:
                
                self.device.signal_received.wait()
                self.device.signal_received.clear()

                
                self.signal_type = self.device.signal_type
                
                self.signal_received.set()
                self.device.signal_sent.set()

                
                if self.signal_type == SignalType.TIMEPOINT_DONE:
                    # Block Logic: Await parallel computation completion.
                    for computation_thread_done in self.timepoint_computation_done:
                        computation_thread_done.wait()
                        computation_thread_done.clear()

                    self.scripts_index = 0
                    self.device.timepoint_work_done.set()
                    break

            # Block Logic: Global temporal boundary synchronization.
            self.device.devices_barrier.wait()

        
        for computation_thread in self.threads:
            computation_thread.join()


class ComputationThread(Thread):
    """
    Functional Utility: Specialized worker thread for high-concurrency script execution.
    Logic: Continuously waits for device signals and iteratively picks up tasks 
    from the device's script queue using an index-based consumption model.
    """

    def __init__(self, device_thread, thread_id):
        """
        Constructor: Links the worker to the device coordination thread.
        """
        Thread.__init__(self, name="Computing Thread %d" % thread_id)

        self.device_thread = device_thread
        self.thread_id = thread_id

    def run(self):
        """
        Execution Logic: Parallel script consumption loop.
        """
        while True:
            
            self.device_thread.signal_received.wait()
            self.device_thread.neighbour_locks[self.thread_id].acquire()

            if self.device_thread.signal_type == SignalType.TERMINATION:
                self.device_thread.neighbour_locks[self.thread_id].release()
                self.device_thread.timepoint_computation_done[self.thread_id].set()
                break

            # Block Logic: Task execution loop.
            while True:
                self.device_thread.device.scripts_lock.acquire()

                # Checks if all scripts for the current step have been consumed.
                if len(self.device_thread.device.scripts) == self.device_thread.scripts_index:
                    self.device_thread.device.scripts_lock.release()
                    self.device_thread.timepoint_computation_done[self.thread_id].set()
                    break

                # Atomic task acquisition.
                index = self.device_thread.scripts_index
                (script_todo, location) = self.device_thread.device.scripts[index]
                self.device_thread.scripts_index += 1

                self.device_thread.device.scripts_lock.release()

                script_data = []
                
                # Block Logic: Neighborhood data reconciliation.
                for device in self.device_thread.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device_thread.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Executes the script over the aggregated data.
                    result = script_todo.run(script_data)

                    # Propagates the result back to neighbors and self.
                    for device in self.device_thread.neighbours:
                        device.set_data(location, result)

                    self.device_thread.device.set_data(location, result)

            if self.device_thread.signal_type == SignalType.SCRIPT_RECEIVED:
                self.device_thread.signal_received.clear()

            self.device_thread.neighbour_locks[self.thread_id].release()
