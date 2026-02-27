"""
This file appears to be a concatenation of several different Python scripts
related to a multi-threaded device simulation. It includes a primary simulation
architecture using Device, DeviceThread, and ScriptThread classes, but also
contains a separate, unrelated ReusableBarrier implementation and a test thread
at the end. The code has inconsistencies, such as importing a ReusableBarrier
while also defining one, and contains apparent bugs in the thread termination logic.
"""

from threading import Event, Thread, Lock
# The code imports a ReusableBarrier, but also defines one at the end of the file.
from reusable_barrier_semaphore import ReusableBarrier
import Queue
NUMBER_OF_THREADS = 8

class Device(object):
    """
    Represents a single device node in the simulation. It holds sensor data
    and scripts, and relies on a master DeviceThread to manage execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.data_lock = Lock() # A lock to protect this device's own sensor_data.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Crude method to share a single barrier instance across all devices.
        Any device can create the barrier, which is then propagated to all others.
        """
        barrier = ReusableBarrier(len(devices))
        if self.barrier is None:
            self.barrier = barrier
        for device in devices:
            if device.barrier is None:
                device.barrier = barrier

    def assign_script(self, script, location):
        """Assigns a script to the device or signals the end of a timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's master thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    A master thread for a Device. It acts as a "producer", putting script
    execution tasks onto a shared queue for a pool of worker threads.
    """
    # A class-level dictionary to hold locks for each data location.
    location_locks = {}

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = []
        self.scripts_queue = Queue.Queue()

    def run(self):
        """
        Initializes a pool of worker threads and then enters the main loop,
        which queues up tasks and synchronizes with other devices.
        """
        # Create a pool of worker threads (consumers).
        for _ in range(NUMBER_OF_THREADS):
            self.threads.append(ScriptThread(self.scripts_queue))
        for script_thread in self.threads:
            script_thread.start()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown. Put poison pills in the queue for workers.
                for _ in self.threads:
                    # The stop flag is incorrectly used here (should be True to stop).
                    self.scripts_queue.put(MyObjects(None, None, None, None, False, None))
                break

            self.device.timepoint_done.wait() # Wait for all scripts to be assigned.

            # For each script, create a task and put it on the queue for the workers.
            for (script, location) in self.device.scripts:
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()

                self.scripts_queue.put(MyObjects(self.device, location, script,
                                                 neighbours, True, self.location_locks))
            self.device.timepoint_done.clear()

            self.device.barrier.wait() # Sync with other devices.

        for script_thread in self.threads:
            script_thread.join()

class ScriptThread(Thread):
    """
    A worker thread that consumes tasks from a queue and executes them.
    """
    def __init__(self, queue):
        Thread.__init__(self, name="Script Thread")
        self.queue = queue

    def run(self):
        """
        Continuously fetches tasks from the queue and executes them until a
        shutdown signal is received.
        """
        while True:
            my_objects = self.queue.get(block=True, timeout=None)

            # This is a likely bug. It will cause the thread to exit after the
            # first valid task, as 'stop' is passed as True.
            if my_objects.stop == False:
                break

            # Acquire a location-specific lock to prevent race conditions on data.
            with my_objects.location_locks[my_objects.location]:
                script_data = []
                # Gather data from neighbors.
                for device in my_objects.neighbours:
                    data = device.get_data(my_objects.location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from the parent device, using its own lock.
                with my_objects.device.data_lock:
                    data = my_objects.device.get_data(my_objects.location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    result = my_objects.script.run(script_data)
                    # Broadcast result back to all devices, using their individual locks.
                    for device in my_objects.neighbours:
                        with device.data_lock:
                            device.set_data(my_objects.location, result)
                    with my_objects.device.data_lock:
                        my_objects.device.set_data(my_objects.location, result)

class MyObjects():
    """A simple data class to bundle task information for the work queue."""
    def __init__(self, device, location, script, neighbours, stop, location_locks):
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.stop = stop
        self.location_locks = location_locks

# The following appears to be a separate, unrelated file that was concatenated.

from threading import *

class ReusableBarrier():
    """
    Another definition of a ReusableBarrier. It is functionally identical to
    ReusableBarrierSem defined earlier.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
                                                 
class MyThread(Thread):
    """
    A simple test thread for a ReusableBarrier, seemingly unrelated to the
    main device simulation. Note the use of Python 2 syntax.
    """
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        # Uses Python 2 'xrange' and 'print' statement.
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",
