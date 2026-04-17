"""
@c0c64b39-17a6-4543-b4a8-c33cb364a1ab/device.py
@brief Distributed sensor processing simulation using producer-consumer worker threads and cross-device barriers.
* Algorithm: Multi-threaded task offloading via a centralized queue with location-specific data synchronization.
* Functional Utility: Orchestrates simulation phases across multiple devices, ensuring data consistency during concurrent script execution.
"""

from threading import Event, Thread, Condition, Lock
from Queue import Queue

class Device(object):
    """
    @brief Encapsulates a sensor node with local data, locks, and coordination logic.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and starts its management thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []       
        self.locks = {}         # Intent: Maps locations to Lock objects for synchronized data access.
        self.no_more_scripts = Event() # Intent: Signals the end of script assignment for the current phase.
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collective resource configuration for the device cluster.
        Invariant: Root device (ID 0) initializes the shared barrier for all nodes.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))

        for device in devices:
            if device is not self:
                device.set_barrier(self.barrier)

    def assign_script(self, script, location):
        """
        @brief Receives a script for execution or signals completion of assignments.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.no_more_scripts.set()

    def get_data(self, location):
        """
        @brief Synchronized getter for sensor data at a specific location.
        Pre-condition: Acquisition of the location-specific lock ensures atomicity.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Synchronized setter for sensor data at a specific location.
        Post-condition: Releases the location lock, signaling completion of an update cycle.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def set_barrier(self, barrier):
        """
        @brief Links the device to the cluster-wide synchronization barrier.
        """
        self.barrier = barrier

    def shutdown(self):
        """
        @brief Gracefully terminates all device-owned threads.
        """
        for thread in self.thread.child_threads:
            if thread.is_alive():
                thread.join()
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief Management thread that initializes worker pools and orchestrates simulation timepoints.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue()        # Intent: Task queue for worker threads.
        self.child_threads = []     # Intent: Pool of active worker threads.
        self.max_threads = 8        # Domain: Resource scaling - fixed worker pool size.

    def run(self):
        """
        @brief Main execution lifecycle for the device node.
        Algorithm: Iterative task dispatching and phased barrier synchronization.
        """
        # Logic: Initializes a unique lock for every local sensor data point.
        for location, data in self.device.sensor_data.iteritems():
            self.device.locks[location] = Lock()

        # Block Logic: Spawns the worker thread pool for parallel script processing.
        for i in xrange(self.max_threads):
            thread = Thread(target=process_scripts, args=(self.queue,))
            self.child_threads.append(thread)
            thread.start()

        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            if neighbours is None:
                # Logic: Shutdown signal - inject termination markers for each worker.
                for i in xrange(len(self.child_threads)):
                    self.queue.put(None)
                self.queue.join()
                break

            # Block Logic: Initial task dispatch.
            done_scripts = 0
            for (script, location) in self.device.scripts:
                job = {'script': script, 'location': location, 'device': self.device, 'neighbours': neighbours}
                self.queue.put(job)     
                done_scripts += 1       

            # Block Logic: Waits for final script batch assignment.
            self.device.no_more_scripts.wait()
            self.device.no_more_scripts.clear()
            
            # Logic: Dispatches remaining scripts received between phase start and assignment completion.
            if done_scripts < len(self.device.scripts):
                for (script, location) in self.device.scripts[done_scripts:]:
                    job = {'script': script, 'location': location, 'device': self.device, 'neighbours': neighbours}
                    self.queue.put(job)     

            # Synchronization Phase: Wait for all local workers to finish the current timepoint tasks.
            self.queue.join()

            # Cross-Device Synchronization Phase: Align all nodes before moving to the next timepoint.
            self.device.barrier.wait()

def process_scripts(queue):
    """
    @brief Worker function that executes tasks from the device queue.
    Algorithm: Producer-Consumer pattern with persistent execution loop.
    """
    while True:
        job = queue.get()
        
        # Logic: Poison pill handling for graceful worker termination.
        if job is None:
            queue.task_done()
            break
        
        script = job['script']
        location = job['location']
        mydevice = job['device']
        neighbours = job['neighbours']

        # Distributed Data Aggregation Phase: Collect readings from neighborhood.
        script_data = []
        for device in neighbours:
            if device is not mydevice:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        
        data = mydevice.get_data(location)
        if data is not None:
            script_data.append(data)

        # Logic: Execute processing logic and propagate state transitions cluster-wide.
        if script_data != []:
            result = script.run(script_data)
            
            for device in neighbours:
                if device is not mydevice:
                    device.set_data(location, result)
            
            mydevice.set_data(location, result)
        
        queue.task_done()

class ReusableBarrier(object):
    """
    @brief Custom implementation of a thread synchronization barrier.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
    def wait(self):
        """
        @brief Blocks calling threads until the required count is reached.
        """
        self.cond.acquire()     
        self.count_threads -= 1 
        if self.count_threads == 0: 
            # Logic: Last thread to arrive releases the entire group.
            self.cond.notify_all()  
            self.count_threads = self.num_threads   
        else:
            self.cond.wait()    
        self.cond.release()
