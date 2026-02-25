"""
This script implements a distributed device simulation using a structured,
multi-threaded approach. Each device manages a pool of worker threads to
execute computational scripts in parallel for each time step of the simulation.

This implementation correctly handles work distribution by pre-partitioning tasks
for each worker, avoiding the race conditions seen in some other versions.
"""

from threading import Event, Thread, Lock
# Assuming ReusableBarrierSem is defined in a 'barrier' module.
from barrier import ReusableBarrierSem
# Assuming Worker is defined in a 'worker' module, though it is also defined below.
from worker import Worker

class Device(object):
    """
    Represents a node in the distributed network. It manages its own data,
    computational tasks (scripts), and a pool of worker threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device, its resources, and starts its main control thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.devices = []
        self.cores = 8 # The number of worker threads to use.
        self.barrier = None
        self.shared_locks = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """Assigns the shared global barrier to this device."""
        self.barrier = barrier

    def set_locks(self, locks):
        """Assigns the shared list of location-based locks to this device."""
        self.shared_locks = locks

    def setup_devices(self, devices):
        """

        Performs one-time global setup for the simulation, executed only by
        the device with ID 0. It creates and distributes the shared barrier
        and location-based locks.
        """
        self.devices = devices
        
        # Ensure only one device (device 0) initializes shared resources.
        if self.device_id == 0:
            lbarrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                dev.set_barrier(lbarrier)

        # Determine the number of locks needed based on the highest location key.
        max_loc = max(self.sensor_data.keys(), key=int)
        
        # If the current device discovers a need for more locks, it creates
        # and distributes them to all other devices. This part could have
        # a race condition if multiple devices call it concurrently, but
        # in a typical setup, it would be called sequentially or by one device.
        if  max_loc + 1 > len(self.shared_locks):
            llocks = []
            for _ in range(max_loc + 1):
                llocks.append(Lock())
            # Distribute the newly created locks.
            for dev in self.devices:
                dev.set_locks(llocks)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed in the upcoming timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A 'None' script signals the end of script assignment for this step.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device. It orchestrates the execution of
    scripts for each timepoint by distributing work to a pool of worker threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def distribute_scripts(self, scripts):
        """
        Partitions the list of scripts among the available worker cores.
        
        Args:
            scripts (list): The list of all (script, location) tuples for the timepoint.
            
        Return:
            A list of lists, where each inner list is the work for one worker.
        """
        worker_scripts = [[] for _ in range(self.device.cores)]
        i = 0
        for script in scripts:
            worker_scripts[i % self.device.cores].append(script)
            i += 1
        return worker_scripts

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.
            
            # Wait for the supervisor to finish assigning scripts for this timepoint.
            self.device.timepoint_done.wait()

            # Create and start worker threads, giving each its own partition of work.
            inner_workers = []
            worker_scripts = self.distribute_scripts(self.device.scripts)
            for worker_scr in worker_scripts:
                inner_thread = Worker(worker_scr, neighbours, self.device)
                inner_workers.append(inner_thread)
                inner_thread.start()

            # Wait for all worker threads to complete their assigned scripts.
            for thr in inner_workers:
                thr.join()

            self.device.timepoint_done.clear()
            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()

class Worker(Thread):
    """
    A worker thread that executes a subset of the computational scripts
    for a single device in a given timepoint.
    """
    def __init__(self, script_loc, neighbours, device):
        """
        Initializes the worker.

        Args:
            script_loc (list): The list of (script, location) tuples assigned to this worker.
            neighbours (list): A list of the parent device's neighbors.
            device (Device): A reference to the parent device.
        """
        Thread.__init__(self)
        self.script_loc = script_loc
        self.neighbours = neighbours
        self.script_data = []
        self.device = device

    def run(self):
        """Executes each script assigned to this worker."""
        for (script, location) in self.script_loc:
            # Acquire the specific lock for the data location to ensure safe updates.
            self.device.shared_locks[location].acquire()
            self.script_data = []
            
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    self.script_data.append(data)

            # Gather data from the parent device.
            data = self.device.get_data(location)
            if data is not None:
                self.script_data.append(data)
            
            # Execute the script and broadcast the results.
            if self.script_data:
                result = script.run(self.script_data)
                
                for dev in self.neighbours:
                    dev.set_data(location, result)
                self.device.set_data(location, result)
             
            # Release the lock for the location.
            self.device.shared_locks[location].release()
