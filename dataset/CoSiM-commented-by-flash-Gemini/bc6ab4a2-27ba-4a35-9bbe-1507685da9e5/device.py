"""
@bc6ab4a2-27ba-4a35-9bbe-1507685da9e5/device.py
@brief Distributed sensor network simulation using on-demand thread spawning for tasks.
This implementation utilizes a dynamic worker model where each computational script 
is executed by a transient thread. Concurrency is throttled by a local semaphore, 
while global consistency is maintained via a shared barrier and a pool of location-specific locks.

Domain: Concurrent Programming, Monitor Patterns, Dynamic Thread Management.
"""

from threading import Thread, Lock, Event, Condition, Semaphore

class ReusableBarrier():
    """
    Implements a classic reusable barrier synchronization primitive.
    Functional Utility: Synchronizes a fixed number of threads, allowing them to 
    wait for each other at a specific execution point before proceeding.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: Total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Blocks the calling thread until the required number of threads have called wait().
        Logic: Uses a monitor pattern with notify_all() to wake all waiting threads.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread to arrive wakes everyone else.
            self.cond.notify_all()
            # Reset for next reuse.
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Main entity representing a node in the sensor network.
    Functional Utility: Manages script assignments and orchestrates the lifecycle 
     of local worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device state.
        @param device_id: Unique integer identifier.
        @param sensor_data: Local sensor state dictionary.
        @param supervisor: Topology controller.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_event = Event()

        self.lock_location = []
        self.lock_n = Lock()
        self.barrier = None

        self.thread_script = []
        self.num_thread = 0
        # Functional Utility: Limits the maximum number of concurrent script threads to 8.
        self.sem = Semaphore(value=8)

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes global synchronization resources.
        Logic: The first device (ID 0) acts as a coordinator to create a shared 
        barrier and a pool of 25 location locks for the entire network.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # Pre-allocates locks for potential sensor locations.
            for _ in xrange(25):
                self.lock_location.append(Lock())

            for dev in devices:
                dev.barrier = barrier
                dev.lock_location = self.lock_location
                dev.setup_event.set()

    def assign_script(self, script, location):
        """Queues a script for execution in the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of sensor data for a specific location."""
        return self.sensor_data[location] if location in \
            self.sensor_data else None

    def set_data(self, location, data):
        """Updates the local sensor value."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main device management thread."""
        self.thread.join()

    def shutdown_script(self):
        """
        Cleanup logic for transient script threads.
        Logic: Iteratively joins all threads spawned during the last timepoint.
        """
        for i in xrange(self.num_thread):
            self.thread_script[i].join()

        for i in xrange(self.num_thread):
            del self.thread_script[-1]

        self.num_thread = 0

class NewThreadScript(Thread):
    """
    Transient worker thread for executing a single script.
    Functional Utility: Performs isolated data aggregation and script execution.
    """
    
    def __init__(self, parent, neighbours, location, script):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.parent = parent
        self.location = location
        self.script = script

    def run(self):
        """
        Execution logic for the script thread.
        Logic: Uses a location-specific lock to ensure atomic updates across the network.
        """
        with self.parent.lock_location[self.location]:
            script_data = []
            
            # Aggregate neighborhood state.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Include local state.
            data = self.parent.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Apply script logic and propagate results to the neighborhood.
                result = self.script.run(script_data)

                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.parent.set_data(self.location, result)
            # Signals the semaphore that a worker slot has become available.
            self.parent.sem.release()

class DeviceThread(Thread):
    """
    Management thread for the Device lifecycle.
    Functional Utility: Orchestrates the progression of timepoints and worker spawning.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation loop for the device.
        Algorithm: Iterative timepoint processing with double-barrier synchronization.
        """
        # Block until the global setup is complete.
        self.device.setup_event.wait()

        while True:
            # Retrieve current neighborhood topology.
            with self.device.lock_n:
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    break

            # Barrier Point: Wait until the supervisor indicates the start of a timepoint.
            self.device.timepoint_done.wait()

            # Block Logic: Spawns a new thread for each assigned script.
            # Semaphore throttling ensures we don't overwhelm system resources.
            for (script, location) in self.device.scripts:
                self.device.sem.acquire()
                self.device.thread_script.append(NewThreadScript \
                    (self.device, neighbours, location, script))

                self.device.num_thread = self.device.num_thread + 1
                self.device.thread_script[-1].start()

            # Barrier 1: Ensure all devices have finished spawning their workers.
            self.device.barrier.wait()
            
            # Cleanup: Join all script threads for the current timepoint.
            self.device.shutdown_script()
            
            self.device.timepoint_done.clear()
            
            # Barrier 2: Ensure all devices have completed cleanup before next timepoint.
            self.device.barrier.wait()
