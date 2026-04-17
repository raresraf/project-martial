"""
@c0c64b39-17a6-4543-b4a8-c33cb364a1ab/device.py
@brief Distributed sensor network simulation with internal worker pool and stateful locking.
This module implements a multi-threaded node architecture where computational tasks 
are managed via an internal producer-consumer queue. It features a unique, stateful 
locking protocol where data acquisition and update operations are coupled to 
synchronize access to shared sensor locations.

Domain: Concurrent Task Queues, Stateful Locking, Distributed State Synchronization.
"""

from threading import Event, Thread, Condition, Lock
from Queue import Queue

class Device(object):
    """
    Core network node representation.
    Functional Utility: Manages local sensor state and provides a high-level 
    interface for data access that incorporates automatic synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device.
        @param device_id: Unique integer identifier.
        @param sensor_data: Initial sensor values.
        @param supervisor: Topology controller.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []       
        self.locks = {}         
                                    
        self.no_more_scripts = Event()  
                                            
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization resources.
        Logic: The leader (Device 0) creates a global barrier which is then 
        propagated to all other nodes in the network.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))

        for device in devices:
            if device is not self:
                device.set_barrier(self.barrier)

    def assign_script(self, script, location):
        """Registers a task for processing and signals when the list is complete."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.no_more_scripts.set()

    def get_data(self, location):
        """
        Retrieves sensor data and acquires a lock on the location.
        Functional Utility: Implements a 'sticky' lock that remains held until set_data is called.
        @return: Sensor value or None if location is not managed here.
        """
        if location in self.sensor_data:
            # Atomic acquisition: caller is now the exclusive owner of this location's state.
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates sensor data and releases the corresponding location lock.
        Functional Utility: Completes the transactional update initiated by get_data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Release: permits other devices or threads to access this location.
            self.locks[location].release()

    def set_barrier(self, barrier):
        """Injects the shared network barrier."""
        self.barrier = barrier

    def shutdown(self):
        """Gracefully terminates the main thread and its worker children."""
        for thread in self.thread.child_threads:
            if thread.is_alive():
                thread.join()
        self.thread.join()


class DeviceThread(Thread):
    """
    Producer thread for the node's internal task queue.
    Functional Utility: Manages a pool of 8 worker threads and dispatches tasks 
    as they are received from the supervisor.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue()        
        self.child_threads = []     
        self.max_threads = 8        

    def run(self):
        """
        Main execution loop for the producer.
        Algorithm: Iterative task distribution with queue-based synchronization.
        """
        # Pre-condition: Initialize a mutex for every local sensor location.
        for location, data in self.device.sensor_data.iteritems():
            self.device.locks[location] = Lock()

        # Spawns the worker pool.
        for i in xrange(self.max_threads):
            thread = Thread(target=process_scripts, args=(self.queue,))
            self.child_threads.append(thread)
            thread.start()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            
            if neighbours is None:
                # Poison Pill: Signals all workers to terminate.
                for i in xrange(len(self.child_threads)):
                    self.queue.put(None)
                self.queue.join()
                break

            done_scripts = 0
            # Enqueue assigned scripts for parallel processing.
            for (script, location) in self.device.scripts:
                job = {'script': script, 'location': location, 
                       'device': self.device, 'neighbours': neighbours}
                self.queue.put(job)     
                done_scripts += 1       

            # Wait for the supervisor to finalize the task list for the current timepoint.
            self.device.no_more_scripts.wait()
            self.device.no_more_scripts.clear()
            
            # Process any remaining scripts assigned after the initial wait.
            if done_scripts < len(self.device.scripts):
                for (script, location) in self.device.scripts[done_scripts:]:
                    job = {'script': script, 'location': location, 
                           'device': self.device, 'neighbours': neighbours}
                    self.queue.put(job)     

            # Wait for all tasks in the current timepoint to be finished by the pool.
            self.queue.join()

            # Global Synchronization: Consensus point before moving to the next timepoint.
            self.device.barrier.wait()

def process_scripts(queue):
    """
    Worker function for the task pool.
    Logic: Continuously consumes jobs from the queue, aggregating data 
    from the neighborhood and executing the task logic.
    """
    while True:
        job = queue.get()
        
        if job is None:
            # Exit signal received.
            queue.task_done()
            break
        
        script = job['script']
        location = job['location']
        mydevice = job['device']
        neighbours = job['neighbours']

        script_data = []
        # Aggregation: Uses the stateful locking protocol (get_data/set_data).
        for device in neighbours:
            if device is not mydevice:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        
        # Include local data state.
        data = mydevice.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Perform computation.
            result = script.run(script_data)

            # Propagation: Updates the neighborhood and releases locks.
            for device in neighbours:
                if device is not mydevice:
                    device.set_data(location, result)
            
            mydevice.set_data(location, result)
        
        queue.task_done()



class ReusableBarrier(object):
    """
    Reusable barrier implementation using a monitor pattern.
    Functional Utility: Coordinates a fixed group of threads to synchronize 
    at periodic intervals.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
    def wait(self):
        """Blocks the calling thread until the barrier count reaches zero."""
        self.cond.acquire()     
        self.count_threads -= 1 
        if self.count_threads == 0: 
            # Final thread arrived: release all waiting threads.
            self.cond.notify_all()  
            # Reset state for immediate reuse in the next cycle.
            self.count_threads = self.num_threads   
        else:
            self.cond.wait()    
        self.cond.release()     
