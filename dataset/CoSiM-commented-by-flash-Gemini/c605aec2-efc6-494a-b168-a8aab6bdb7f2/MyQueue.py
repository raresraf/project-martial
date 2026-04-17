"""
@c605aec2-efc6-494a-b168-a8aab6bdb7f2/MyQueue.py
@brief Distributed sensor processing simulation using a worker thread pool and semaphore-based reusable barriers.
* Algorithm: Two-phase commit style barrier synchronization with concurrent task execution via thread-safe queues.
* Functional Utility: Manages a pool of background workers to process sensor scripts, coordinating cluster-wide state consistency.
"""

from Queue import Queue 
from threading import Thread

class MyQueue():
    """
    @brief Custom worker pool implementation for asynchronous task execution.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the thread pool and bootstraps worker threads.
        """
        self.queue = Queue(num_threads)
        self.threads = []
        self.device = None

        # Logic: Spawns a fixed number of worker threads that consume tasks from the shared queue.
        for _ in xrange(num_threads):
            thread = Thread(target=self.run)
            self.threads.append(thread)
        
        for thread in self.threads:
            thread.start()
    
    def run(self):
        """
        @brief Worker thread main loop.
        Algorithm: Producer-Consumer consumption with distributed data aggregation and result propagation.
        """
        while True:
            # Logic: Blocks until a task (neighbors, script, location) is available.
            neighbours, script, location = self.queue.get()

            # Logic: Poison pill handling for graceful thread termination.
            if neighbours is None and script is None:
                self.queue.task_done()
                return
        
            script_data = []
            
            # Distributed Data Aggregation: Collects readings from neighborhood nodes.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Logic: Includes local sensor reading in the processing batch.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Functional Utility: Runs the analysis script and broadcasts results back to peers.
            if script_data != []:
                result = script.run(script_data)
                
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                self.device.set_data(location, result)
            
            # Post-condition: Notifies the queue that a task unit has been completed.
            self.queue.task_done()
    
    def finish(self):
        """
        @brief Orchestrates a clean shutdown of the thread pool.
        """
        # Logic: Ensures all pending tasks are processed before terminating threads.
        self.queue.join()

        # Logic: Injects termination signals for each worker thread.
        for _ in xrange(len(self.threads)):
           self.queue.put((None, None, None))

        for thread in self.threads:
            thread.join()


from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    @brief Implementation of a two-phase reusable synchronization barrier using semaphores.
    * Algorithm: Dual-phase arrival/departure logic to prevent thread overruns in consecutive cycles.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes barrier state with thread count and phase-specific semaphores.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Intent: Array wrapper for mutable integer sharing.
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
 
    def wait(self):
        """
        @brief Blocks calling thread through both phases of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single synchronization phase.
        Invariant: The last thread to arrive releases the entire group of waiting threads.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()

class Device(object):
    """
    @brief High-level representation of a sensor node.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and starts the coordination thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        # Logic: Per-location locks for atomic data access during multi-threaded updates.
        self.location_locks = {location: Lock() for location in self.sensor_data}
        self.scripts_available = False
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Cluster-wide initialization of shared synchronization primitives.
        Invariant: Root device (ID 0) establishes the global barrier for all nodes.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Queues a script for execution in the current simulation phase.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Synchronized acquisition of sensor data.
        Pre-condition: Acquisition of location-specific lock ensures data consistency.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]     
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Synchronized update of sensor data.
        Post-condition: Release of location-specific lock completes the atomic operation.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
        else:
            return None

    def shutdown(self):
        """
        @brief Gracefully terminates the device coordination thread.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief Coordination thread managing temporal phases and task offloading.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = MyQueue(8) # Domain: Static worker pool size of 8 threads.

    def run(self):
        """
        @brief Main execution lifecycle for the device coordination thread.
        Algorithm: Phased execution loop with event-driven task submission.
        """
        self.queue.device = self.device
        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Internal phase loop to handle multi-stage script arrival.
            while True:
                # Logic: Wait for new scripts or completion of the current timepoint.
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    if self.device.scripts_available:
                        self.device.scripts_available = False

                        # Dispatch Phase: Submits currently assigned scripts to the worker pool.
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
                    else:
                        # Logic: Current timepoint concluded.
                        self.device.timepoint_done.clear()
                        self.device.scripts_available = True
                        break
            
            # Synchronization Phase: Wait for all local and global workers to align.
            self.queue.queue.join()
            self.device.barrier.wait()

        # Termination Phase: Clean up worker threads.
        self.queue.finish()
