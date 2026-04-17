"""
@cf589e3b-9095-4448-a0ab-1cef55de8ec3/device.py
@brief Distributed sensor network simulation with round-robin task distribution.
This module implements a parallel processing architecture using a pool of persistent 
worker threads (ThreadAux). Computational tasks are assigned to workers in a 
round-robin fashion during the dispatch phase. Temporal synchronization is achieved 
through sequential events and two-phase barriers, while a static network-wide pool 
of locks ensures spatial mutual exclusion for sensor data updates.

Domain: Parallel Worker Pools, Round-Robin Dispatch, Event-Driven Synchronization.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Uses a double-gate mechanism with semaphores to ensure 
    perfect temporal alignment across a fixed set of threads.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: Number of participants in the rendezvous.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Executes the two-phase synchronization rendezvous."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """Arrival gate logic."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Release all participants.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Exit gate logic to prevent 'overtaking' the next barrier cycle."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()



class Device(object):
    """
    Simulated network node coordinating local data and parallel tasks.
    Functional Utility: Manages persistent worker threads and provides static 
    resources for network-wide synchronization.
    """
    
    # Static Network Resources: shared across all Device instances.
    bar1 = ReusableBarrier(1)
    event1 = Event()
    locck = []

    def __init__(self, device_id, sensor_data, supervisor):
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = []

        # Sequence of Events: used to trigger workers across simulation steps.
        self.event = []
        for _ in xrange(11):
            self.event.append(Event())

        self.nr_threads_device = 8
        self.nr_thread_atribuire = 0
        
        # Internal barrier for local thread group synchronization.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1)

        self.thread = DeviceThread(self)
        self.thread.start()

        self.threads = []
        # Spawns a pool of persistent worker threads.
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource factory.
        Logic: Coordinator node (ID 0) initializes the shared barrier and 
        a static pool of 30 locks for the entire group.
        """
        self.devices = devices
        
        if self.device_id == 0:
            # Pre-allocates a fixed pool of spatial mutexes.
            for _ in xrange(30):
                Device.locck.append(Lock())
            Device.bar1 = ReusableBarrier(len(devices))
            
            # Signal start of simulation.
            Device.event1.set()

    def assign_script(self, script, location):
        """
        Task Dispatch Logic.
        Algorithm: Round-robin assignment of computational scripts to the worker pool.
        """
        if script is not None:
            # Route the script to a specific worker's local buffer.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            
            # Increment assignment pointer.
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%\
            self.nr_threads_device
        else:
            # All assignments for the current timepoint are complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe access to local sensor data."""
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all local management and worker threads."""
        self.thread.join()
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    """
    Main orchestration thread for node phases.
    Functional Utility: Coordinates simulation timepoints and signals the 
    persistent worker threads using an event sequence.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.contor = 0

    def run(self):
        """
        Main execution loop.
        Logic: Controls the progression of timepoints through event signals 
        and local/global barrier synchronization.
        """
        # Block until global setup is complete.
        Device.event1.wait()

        while True:
            # Fetch topology.
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                # Termination signal: notify all workers to exit.
                self.device.event[self.contor].set()
                break

            # Wait for supervisor to finish script assignments for the step.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Trigger worker pool for the current step.
            self.device.event[self.contor].set()
            self.contor += 1

            # Synchronization Point 1: Wait for local worker group.
            self.device.bar_threads_device.wait()

            # Synchronization Point 2: Network-wide consensus.
            Device.bar1.wait()

class ThreadAux(Thread):
    """
    Persistent worker thread implementation.
    Functional Utility: Continuously processes a subset of assigned scripts 
    while observing network-wide spatial locks.
    """
    
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device
        self.script_loc = {}
        self.contor = 0

    def run(self):
        """
        Worker execution loop.
        Algorithm: Event-driven task processing with spatial mutual exclusion.
        """
        while True:
            # Rendezvous point: wait for the DeviceThread to signal the next phase.
            self.device.event[self.contor].wait()
            self.contor += 1

            # Check for termination.
            neigh = self.device.thread.neighbours
            if neigh is None:
                break

            # Block Logic: Sequential processing of assigned scripts.
            for script in self.script_loc:
                location = self.script_loc[script]
                
                # Critical Section: Network-wide spatial lock for the location.
                Device.locck[location].acquire()
                script_data = []

                # Neighborhood aggregation.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Local state integration.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Compute result and propagate.
                    result = script.run(script_data)
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Finalize spatial transaction.
                Device.locck[location].release()

            # Signal phase completion to the local manager.
            self.device.bar_threads_device.wait()
