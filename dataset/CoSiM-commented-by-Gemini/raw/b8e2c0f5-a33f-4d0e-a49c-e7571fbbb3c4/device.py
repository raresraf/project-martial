"""
This module implements a producer-consumer based simulation of a distributed device network.

This version uses a sophisticated threading model where each `Device` has a dedicated
"producer" thread (`myThread`) and a pool of "consumer" worker threads (`DeviceThread`).

The simulation proceeds in synchronized time steps, orchestrated by the producer threads.
In each step:
1. The producer thread (`myThread`) receives scripts and places them into a shared `Queue`.
2. The worker threads (`DeviceThread`) consume scripts from the queue, execute them,
   and process the data.
3. The producer thread waits for all scripts in the queue to be processed using `queue.join()`.
4. All producer threads from all devices synchronize at a global barrier before starting
   the next time step.

@warning This implementation contains several design flaws:
         1. The `ReusableBarrierCond` is not thread-safe and is prone to deadlocks.
         2. The worker threads use an inefficient busy-wait loop instead of blocking
            on the queue directly.
         3. The locking mechanism assumes `location` is a valid integer index.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty


class ReusableBarrierCond():
    """
    An attempted implementation of a reusable barrier using a Condition variable.

    @warning This implementation is NOT thread-safe. It is prone to a classic
             race condition known as the "lost wakeup" problem, which can lead
             to a deadlock.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()                  
                                                 
    
    def wait(self):
        self.cond.acquire()                      
        self.count_threads -= 1;
        if self.count_threads == 0:
            self.cond.notify_all()               
            self.count_threads = self.num_threads
        else:
            self.cond.wait();                    
        self.cond.release();                     


class Device(object):
    """
    A device node, containing a producer thread, a pool of worker threads,
    and a shared work queue.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # The main producer/coordinator thread for this device.
        self.myThread = myThread(self)
        self.myThread.start()
        
        self.neightbours = []
        self.barrier = None
        self.locks = [] # A shared list of locks for data locations.
        self.queue = Queue() # Work queue for this device.
        self.programEnded = False;
        
        # A pool of consumer/worker threads.
        self.deviceThreads = []
        for i in xrange(8):
            worker = DeviceThread(self)
            self.deviceThreads.append(worker)
            worker.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects. Device 0 acts
        as the master.
        """
        devicesNumber = len(devices)
        if self.device_id == 0:
            # The barrier synchronizes the main `myThread` of each device.
            barrier = ReusableBarrierCond(devicesNumber)
            # A list of locks, indexed by location ID.
            locks = [Lock() for _ in xrange(24)]
            
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """Adds a script to the device's list for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its producer and worker threads."""
        self.myThread.join()
        for deviceThread in self.deviceThreads:
            deviceThread.join()


class DeviceThread(Thread):
    """
    A consumer/worker thread. It processes scripts from a shared queue.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop for a worker thread.

        @warning This loop implements an inefficient busy-wait pattern. The thread
                 wakes on an event, then immediately tries a non-blocking `get()`.
                 A more efficient design would use a blocking `get()` on the queue.
        """
        while not self.device.programEnded:
            # Wait for the producer thread to signal that new items may be in the queue.
            self.device.script_received.wait()

            try:
                # Consume one script from the queue.
                script, location = self.device.queue.get(block = False)
                
                # Lock based on location to ensure data consistency.
                self.device.locks[location].acquire()
            
                # --- Data gathering and processing logic ---
                script_data = []
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if len(script_data) > 0:
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                self.device.locks[location].release()
                
                # Signal to the queue that this task is complete.
                self.device.queue.task_done()
            
            except Empty:
                # The queue was empty, loop and wait again.
                pass
            
class myThread(Thread):
    """
    A producer/coordinator thread for a single device. It manages the time steps
    and distributes work to its pool of worker threads.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device myThread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main time-step loop for the device."""
        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            
            if self.device.neighbours is None:
                # Shutdown signal received.
                self.device.programEnded = True;
                self.device.script_received.set() # Wake up workers to let them terminate.
                break;
            
            # Wait for the supervisor to finish assigning all scripts for this timepoint.
            self.device.timepoint_done.wait()
            
            # --- Producer Step ---
            # Place all assigned scripts onto the shared work queue.
            for script in self.device.scripts:
                self.device.queue.put(script)
            
            # Wake up any waiting worker threads.
            self.device.script_received.set()
            
            # --- Synchronization with Workers ---
            # Block until the workers have processed all items in the queue.
            self.device.queue.join()
            
            # --- Cleanup for next timepoint ---
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            
            # --- Synchronization with other Devices ---
            # Wait at the main barrier to ensure all devices start the next
            # timepoint together.
            self.device.barrier.wait()