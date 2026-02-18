"""
Models a distributed system of devices using a Master-Worker threading pattern.

Each device in this simulation framework has one 'Master' thread and a pool of
'Worker' threads. The Master thread coordinates with a central supervisor,
receives tasks (scripts), and places them in a local queue. The Worker threads
consume tasks from this queue, executing them in parallel. Synchronization across
all devices is handled by a global barrier, while race conditions for specific
data locations are managed by a set of shared, per-location locks.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty


class ReusableBarrier(object):
    """
    A reusable synchronization barrier implemented using a Condition variable.
    Allows a set of threads to all wait for each other to reach a certain point.
    """

    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread arrives, notify all waiting threads.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Represents a single device, managing its state, a Master thread for coordination,
    and a pool of Worker threads for script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.neighbours = []

        self.barrier = None
        self.locks = []
        self.timepoint_done = Event()
        self.tasks_ready = Event()
        self.tasks = Queue()
        self.simulation_ended = False

        
        self.master = DeviceThreadMaster(self)
        self.master.start()

        
        self.workers = []
        for i in xrange(8):
            worker = DeviceThreadWorker(self, i)
            self.workers.append(worker)
            worker.start()

    def __str__(self):
        
        return "Device [%d]" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources (barrier, locks) to all devices."""
        
        if self.device_id == 0:
            # Device 0 is responsible for creating the shared objects.
            barrier = ReusableBarrier(len(devices))
            locks = [Lock() for _ in xrange(24)] # A fixed set of 24 location locks.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that script assignment for this timepoint is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data (not individually thread-safe)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all threads to shut down the device cleanly."""
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceThreadMaster(Thread):
    """
    The coordination thread for a single Device. It communicates with the
    supervisor and dispatches tasks to its worker threads.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device [%d] Thread Master" % device.device_id)


        self.device = device

    def run(self):
        """Main loop for the Master thread."""
        while True:
            
            self.device.neighbours = self.device.supervisor.get_neighbours()

            
            if self.device.neighbours is None:
                # Supervisor signals simulation shutdown.
                self.device.simulation_ended = True
                self.device.tasks_ready.set() # Wake up workers so they can terminate.
                
                break

            
            # 1. Wait for supervisor to signal that all scripts are assigned for this step.
            self.device.timepoint_done.wait()

            
            # 2. Populate the local task queue for the worker threads.
            for task in self.device.scripts:
                self.device.tasks.put(task)

            
            # 3. Signal to workers that tasks are ready to be processed.
            self.device.tasks_ready.set()

            
            # 4. Wait for all tasks in the queue to be completed by the workers.
            self.device.tasks.join()

            
            # 5. Reset events for the next time step.
            self.device.tasks_ready.clear()
            self.device.timepoint_done.clear()

            
            # 6. Wait at the global barrier to synchronize with all other devices.
            self.device.barrier.wait()


class DeviceThreadWorker(Thread):
    """
    A worker thread that executes scripts from its parent device's task queue.
    """

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device [%d] Thread Worker [%d]" % (device.device_id, thread_id))


        self.device = device
        self.thread_id = thread_id

    def run(self):
        """Main loop for the Worker thread."""
        while not self.device.simulation_ended:
            
            # Wait for the Master to signal that tasks are available.
            self.device.tasks_ready.wait()

            if self.device.simulation_ended:
                break
                
            try:
                
                # Process tasks from the queue until it's empty for this time step.
                script, location = self.device.tasks.get(block=False)

                
                # Acquire the global lock for this location to ensure safe data access.
                self.device.locks[location].acquire()

                script_data = []

                
                # Gather data from neighbors and self.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:


                        script_data.append(data)

                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                
                # Execute script and update data if input data was found.
                if len(script_data) > 0:
                    
                    result = script.run(script_data)

                    
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                
                # Release the location lock.
                self.device.locks[location].release()

                
                # Signal that this one task is done.
                self.device.tasks.task_done()
            except Empty:
                # The queue is empty for this time step, continue waiting for next signal.
                pass
