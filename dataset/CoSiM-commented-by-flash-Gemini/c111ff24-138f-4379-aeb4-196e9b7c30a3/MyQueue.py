"""
@c111ff24-138f-4379-aeb4-196e9b7c30a3/MyQueue.py
@brief multi-threaded simulation with decoupled task queuing and two-phase synchronization.
This module defines a parallel processing framework using an internal worker pool 
managed via a custom queue wrapper. It employs a two-phase barrier synchronization 
and a stateful locking protocol where data acquisition and update operations are 
intrinsically linked to manage shared sensor state.

Domain: Parallel Task Scheduling, Multi-phase Barriers, Stateful Synchronization.
"""

from Queue import Queue 
from threading import Thread

class MyQueue():
    """
    Management wrapper for an asynchronous task queue and its associated workers.
    Functional Utility: Encapsulates the lifecycle of worker threads and provides 
    a clean interface for submitting and finalizing parallel computational tasks.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the queue and spawns workers.
        @param num_threads: Number of persistent worker threads to maintain.
        """
        self.queue = Queue(num_threads)
        self.threads = []
        self.device = None

        # Spawns a pool of workers that will listen to the internal Queue.
        for _ in xrange(num_threads):
            thread = Thread(target=self.run)
            self.threads.append(thread)
        
        for thread in self.threads:
            thread.start()
    
    def run(self):
        """
        Main worker execution loop.
        Logic: Consumes jobs consisting of (neighbours, script, location) and 
        applies the computational logic using the sticky lock protocol.
        """
        while True:
            # Block until a task is available.
            neighbours, script, location = self.queue.get()

            # Exit logic: Check for the poison pill signal.
            if neighbours is None and script is None:
                self.queue.task_done()
                return
        
            script_data = []
            # Aggregate data from neighbors using the parent device's sticky lock protocol.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    # Logic Note: get_data() acquires a location lock.
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Include local sensor data.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Apply the script's domain logic.
                result = script.run(script_data)
                
                # Propagate results back and release held locks.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        # Logic Note: set_data() releases the location lock.
                        device.set_data(location, result)
                
                self.device.set_data(location, result)
            
            # Acknowledge completion for queue synchronization.
            self.queue.task_done()
    
    def finish(self):
        """
        Gracefully shuts down all workers.
        Logic: Waits for pending tasks, then dispatches poison pills to each worker.
        """
        # Block until all current tasks are acknowledged.
        self.queue.join()

        # Signal termination to each thread in the pool.
        for _ in xrange(len(self.threads)):
           self.queue.put((None, None, None))

        # Reclaim thread resources.
        for thread in self.threads:
            thread.join()


from threading import Thread, Event, Lock, Semaphore
from MyQueue import MyQueue

class ReusableBarrier():
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Uses a double-gate mechanism with semaphores to prevent 
    threads from 'overtaking' the barrier reset phase.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: Total number of participants in the synchronization.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
 
    def wait(self):
        """
        Executes a two-phase synchronization.
        Logic: Threads must clear two distinct gates to ensure total network order.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        Internal barrier phase logic using a shared counter and semaphore gate.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                # Last thread releases the gate for all others.
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        # Wait at the gate.
        threads_sem.acquire()

class Device(object):
    """
    Representation of a networked processing node.
    Functional Utility: Manages sensor data and provides synchronized access 
    through a collection of location-specific locks.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        # Pre-allocate locks for local sensor data consistency.
        self.location_locks = {location: Lock() for location in self.sensor_data}
        self.scripts_available = False
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource distribution.
        Logic: Coordinator node (ID 0) initializes and shares the ReusableBarrier.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Queues a task for the current simulation timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data and acquires the location lock.
        Functional Utility: Part 1 of a stateful transactional update.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]     
        else:
            return None

    def set_data(self, location, data):
        """
        Updates data and releases the location lock.
        Functional Utility: Part 2 of a stateful transactional update.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
        else:
            return None

    def shutdown(self):
        """Joins the main lifecycle thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    Node-level lifecycle manager.
    Functional Utility: Coordinates between high-level supervisor signals 
    and the internal parallel worker pool.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = MyQueue(8)

    def run(self):
        """
        Main execution loop.
        Algorithm: Iterative timepoint processing with queue-based parallelization.
        """
        self.queue.device = self.device
        while True:
            # Fetch current topology state.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Event-driven task submission.
            while True:
                # Wait for either new scripts or timepoint completion.
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    if self.device.scripts_available:
                        self.device.scripts_available = False
                        # Feed tasks into the parallel worker pool.
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
                    else:
                        # Timepoint complete: reset state for next cycle.
                        self.device.timepoint_done.clear()
                        self.device.scripts_available = True
                        break
            
            # Synchronize: Wait for all local tasks to finish and reach global consensus.
            self.queue.queue.join()
            self.device.barrier.wait()

        # Final cleanup and worker termination.
        self.queue.finish()
