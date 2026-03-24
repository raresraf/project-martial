"""
This module simulates a network of distributed devices using a thread pool
architecture for script execution within each device.

Key Architectural Points:
- Thread Pool: Each `Device` maintains a fixed-size pool of persistent worker
  threads (`DeviceThread`).
- Master/Worker Setup: Device 0 acts as a master, creating and distributing
  shared synchronization resources like a global barrier and a dictionary of locks.
- Task Queuing: Scripts are assigned to idle worker threads via a queueing
  mechanism within each device.
- Synchronization: The system uses a complex web of barriers and locks to
  coordinate threads within a device and across the entire network.
- Python 2 Context: The use of `from Queue import Queue` suggests this code
  is written for Python 2.
"""


from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierCond
from Queue import Queue

THREAD_NR = 8

class Device(object):
    """
    Represents a device node, which manages a thread pool for script execution.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device, creating and starting its pool of worker threads.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.setup_finished = Event()
        self.dataLock = Lock()
        self.shared_lock = Lock()
        self.thread_queue = Queue(0)
        self.wait_get_neighbours = ReusableBarrierCond(THREAD_NR)
        self.thread_pool = []
        self.neighbours = []

        # Create and start the fixed-size pool of worker threads.
        for i in range(0, THREAD_NR):
            thread = DeviceThread(self, i)
            self.thread_pool.append(thread)
            self.thread_queue.put(thread) # Add the thread to the idle queue.
            thread.start()

    def __str__(self):
        """String representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global barrier and location locks, managed by device 0.
        """
        
        if self.device_id == 0:
            # Device 0 is the master and creates the shared resources.
            self.reusable_barrier = ReusableBarrierCond(len(devices) * THREAD_NR)
            self.location_locks = {}
            # Distribute the shared resources to other devices.
            for device in devices:
                if device.device_id != self.device_id:
                    device.set_location_locks(self.location_locks)
                    device.set_barrier(self.reusable_barrier)

            # Signal that setup is complete.
            self.setup_finished.set()

    def set_barrier(self, reusable_barrier):
        """Sets the global barrier for this device."""
        self.reusable_barrier = reusable_barrier
        self.setup_finished.set()

    def set_location_locks(self, location_locks):
        """Sets the shared location lock dictionary for this device."""
        self.location_locks = location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to an idle worker thread.
        """
        
        if script is not None:
            # A 'None' script indicates the end of the assignment phase.
            self.scripts.append((script, location))
            if location not in self.location_locks:
                # Lazily create a lock for a new location.
                self.location_locks[location] = Lock()

            # Get an idle worker thread from the queue and give it the script.
            thread = self.thread_queue.get()
            thread.give_script(script, location)

            
        else:
            # End of timepoint: dispatch any remaining scripts to the workers.
            for (s, l) in self.scripts:
                thread = self.thread_queue.get()
                thread.give_script(s, l)
            
            # Send a poison pill (None script) to each worker to end the timepoint loop.
            for thread in self.thread_pool:
                thread.give_script(None, None)


    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its worker threads."""
        
        for i in range(THREAD_NR):
            self.thread_pool[i].join()


class DeviceThread(Thread):
    """
    A persistent worker thread within a Device's thread pool.
    """
    

    def __init__(self, device, ID):
        """Initializes the worker thread."""
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id = ID
        self.script_queue = Queue(0)

    def give_script(self, script, location):
        """Receives a script to execute from the parent Device."""
        self.script_queue.put((script, location))

    def run(self):
        """
        The main loop for the worker thread.
        """

        while True:
            
            # Wait until the initial device setup is complete.
            self.device.setup_finished.wait()

            
            # Thread 0 of each device is responsible for fetching the neighbor list.
            if self.id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            
            # All workers on this device wait here until the neighbor list is fetched.
            self.device.wait_get_neighbours.wait()

            if self.device.neighbours is None:
                # Supervisor signals termination.
                break

            # Inner loop for processing scripts within a single timepoint.
            while True:
                (script, location) = self.script_queue.get()

                if script is None:
                    # Poison pill received, end of timepoint for this worker.
                    break

                self.device.location_locks[location].acquire()

                script_data = []
                

                # Gather data from neighbors. This locking pattern is dangerous and can
                # easily lead to deadlocks if two devices try to get data from each other.
                for device in self.device.neighbours:
                    device.dataLock.acquire()
                    data = device.get_data(location)
                    device.dataLock.release()

                    if data is not None:
                        script_data.append(data)

                
                # Gather data from the local device.
                self.device.dataLock.acquire()
                data = self.device.get_data(location)
                self.device.dataLock.release()
                
                if data is not None:
                   script_data.append(data)

                self.device.location_locks[location].release()

                if script_data != []:
                    
                    result = script.run(script_data)
                    
                    self.device.location_locks[location].acquire()

                    
                    for device in self.device.neighbours:
                        device.dataLock.acquire()
                        device.set_data(location, result)
                        device.dataLock.release()

                    


                    self.device.dataLock.acquire()
                    self.device.set_data(location, result)
                    self.device.dataLock.release()
                    self.device.location_locks[location].release()

               
                # Mark this worker as idle by putting it back in the device's queue.
                self.device.thread_queue.put(self)

            
            # Wait at the global barrier for all worker threads of all devices to finish.
            self.device.reusable_barrier.wait()