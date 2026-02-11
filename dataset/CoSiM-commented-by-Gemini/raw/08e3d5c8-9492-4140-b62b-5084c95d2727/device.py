"""
This module defines a simulated distributed device network using a producer-consumer
threading model within each device, synchronized globally with barriers.

NOTE: This implementation contains a critical race condition. The master thread
(`DeviceThread`) does not wait for its worker threads (`ParallelScript`) to finish
their work for the current time step before it proceeds to the global `time_bar`.
This breaks the synchronization logic of the Bulk Synchronous Parallel model.
"""

from threading import Event, Thread, Lock, Semaphore
# Note: 'barrier' is a custom module, assumed to contain a ReusableBarrier class.
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single device in the network. Each device manages a local
    master-producer thread and a pool of worker-consumer threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.neighbours = []

        # Global barriers and fine-grained locks, distributed by Device 0.
        self.time_bar = None
        self.script_bar = None
        self.devloc = []

        # Event to signal that scripts have been assigned for the current step.
        self.script_received = Event()
        self.timepoint_done = Event() # This event seems unused.

        # The master thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects (barriers, locks).
        This is a centralized setup performed by device 0.
        """
        if self.device_id == 0:
            # Create two barriers for two-phase synchronization.
            self.time_bar = ReusableBarrierSem(len(devices))
            self.script_bar = ReusableBarrierSem(len(devices))
            for device in devices:
                device.time_bar = self.time_bar
                device.script_bar = self.script_bar

            # Determine the max location index to create enough locks.
            maxim = 0
            for device in devices:
                loc_list = device.sensor_data.keys()
                loc_list.sort()
                if loc_list and loc_list[-1] > maxim:
                    maxim = loc_list[-1]
            
            # Create a list of fine-grained locks for each location.
            self.devloc = [Lock() for _ in range(maxim + 1)]
            for device in devices:
                device.devloc = self.devloc

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of assignments.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal the master thread that scripts are ready.
            self.script_received.set()
            # This wait is misplaced; it blocks the supervisor's thread, not the device's.
            self.script_bar.wait()

    def get_data(self, location):
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class ParallelScript(Thread):
    """
    A worker-consumer thread that executes scripts from a queue.
    """
    def __init__(self, device_thread):
        Thread.__init__(self)
        self.device_thread = device_thread

    def run(self):
        while True:
            # Block until the producer (DeviceThread) signals a task is available.
            self.device_thread.sem_scripts.acquire()
            
            # Pop a task from the shared list (acting as a queue).
            nod = self.device_thread.to_procces.pop(0)
            
            # A 'None' task is a poison pill to terminate the worker.
            if nod is None:
                break
            
            neighbours, script, location = nod[0], nod[1], nod[2]

            # Acquire the specific lock for the data location.
            self.device_thread.device.devloc[location].acquire()
            try:
                script_data = []
                # Gather data from neighbors and self.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.device_thread.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Run the script and propagate the result.
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device_thread.device.set_data(location, result)
            finally:
                # Ensure the lock is always released.
                self.device_thread.device.devloc[location].release()


class DeviceThread(Thread):
    """
    The master-producer thread for a device. It creates a pool of workers,
    dispatches tasks, and synchronizes with other devices.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.sem_scripts = Semaphore(0)
        self.numar_procesoare = 8  # Romanian for "number of processors"
        self.to_procces = [] # A list used as a task queue.
        self.pool = self.create_pool(self) # Create and start the worker pool.

    def create_pool(self, device_thread):
        """Creates and starts the fixed-size pool of worker threads."""
        pool = []
        for _ in xrange(self.numar_procesoare):
            aux_t = ParallelScript(device_thread)
            pool.append(aux_t)
            aux_t.start()
        return pool

    def run(self):
        """The main producer loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # --- Simulation Termination ---
                # Post a poison pill for each worker thread to consume.
                for _ in range(self.numar_procesoare):
                    self.to_procces.append(None)
                    self.sem_scripts.release() # Wake up a worker to terminate.
                for item in self.pool:
                    item.join() # Wait for all workers to finish.
                break
            
            # 1. Wait for the supervisor to signal that scripts are assigned.
            self.device.script_received.wait()
            
            # 2. Producer: Add tasks to the queue and signal workers.
            for (script, location) in self.device.scripts:
                nod = (neighbours, script, location)
                self.to_procces.append(nod)
                self.sem_scripts.release() # Wake up one worker.

            # 3. Global Barrier 1: Wait for all devices to finish producing tasks.
            self.device.script_bar.wait()

            # --- CRITICAL FLAW ---
            # The master thread proceeds to the time barrier immediately, WITHOUT waiting
            # for its own worker threads to finish processing the tasks for this step.
            # This breaks the BSP model, as this device may not have finished its
            # computations before signaling that it has.
            # 4. Global Barrier 2: Synchronization for the end of the time step.
            self.device.time_bar.wait()
            
            # Clear event for the next cycle.
            self.device.script_received.clear()
