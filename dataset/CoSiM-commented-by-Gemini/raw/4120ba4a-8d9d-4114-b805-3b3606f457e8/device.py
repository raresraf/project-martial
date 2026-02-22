"""
Models a distributed sensor network simulation using a producer-consumer model.

This script simulates a network of devices that process sensor data in discrete,
synchronized time steps. The architecture differs from a simple master-worker
model by using a producer-consumer pattern. The main `DeviceThread` acts as a
producer, preparing a queue of scripts, while a pool of `WorkerThread` instances
act as consumers, processing scripts from the shared queue in parallel.

Classes:
    Device: A container for a single node's state, data, threads, and the
            many synchronization primitives needed to coordinate the simulation.
    DeviceThread: A high-level coordinator thread for a device. It synchronizes
                  the start of a time step, produces a work queue, and waits
                  for the workers to finish.
    WorkerThread: A consumer thread that pulls computational tasks from a shared
                  queue, executes them, and disseminates the results.
"""


from threading import Event, Thread, Semaphore
# Note: The `barrier` and `worker` modules are assumed to be in the same directory.
from barrier import ReusableBarrierSem
from worker import WorkerThread


class Device(object):
    """Represents a single device, managing its state, data, and threads."""

    
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance and its synchronization primitives."""

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # --- Synchronization Primitives ---
        self.barrier_set = Event() # Signals that the main barrier is ready.
        self.script_received = Event() # Not used in this implementation.
        self.scripts = [] # Temporary list for incoming scripts for a time step.
        self.timepoint_done = Event() # Signals all scripts for a time step are assigned.
        self.barrier = None # The main barrier to sync all Devices.
        self.data_locks = [] # Shared location-based locks for all devices.
        self.script_queue = [] # The shared work queue for this device's workers.
        self.script_lock = Semaphore(1) # Lock to protect the script_queue.
        self.exit_flag = Event() # Signals all threads to terminate.
        self.tasks_finished = Event() # Signaled by worker 0 when the queue is empty.
        self.start_tasks = Event() # Signals workers to start processing the queue.
        
        # --- Threads ---
        self.thread = DeviceThread(self)
        self.thread_list = [] # Pool of worker threads.
        self.worker_number = 8
        self.worker_barrier = ReusableBarrierSem(self.worker_number)


    def set_flag(self):
        """Sets the event to signal that the main barrier has been created."""
        self.barrier_set.set()

    def set_barrier(self, barrier):
        """Assigns the shared main barrier to this device."""
        self.barrier = barrier

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources and starts all device and worker threads.

        Device 0 acts as a master to create the shared main barrier and the
        set of data locks for all locations. Other devices wait until this is
        done. Then, all devices start their main and worker threads.
        """
        # Master device (id 0) sets up shared resources.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            
            # Determine the highest location index to size the locks dictionary.
            location_index = -1
            for dev in devices:
                for k in dev.sensor_data:
                    if k > location_index:
                        location_index = k

            # Create and distribute the shared barrier and data locks.
            self.data_locks = {loc : Semaphore(1) for loc in range(location_index+1)}
            for dev in devices:
                dev.set_barrier(self.barrier)
                dev.data_locks = self.data_locks
                dev.set_flag() # Signal to other devices that setup is complete.
        else:
            # Worker devices wait for the master to finish setup.
            self.barrier_set.wait()
            
        # All devices start their main thread and worker pool.
        self.thread.start()
        for tid in range(self.worker_number):
            thread = WorkerThread(self, tid)
            self.thread_list.append(thread)
            thread.start()

    def assign_script(self, script, location):
        """Assigns a script for the upcoming time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of assignments for this time step.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all threads associated with this device."""
        for thread in self.thread_list:
            thread.join()
        self.thread.join()


class DeviceThread(Thread):
    """A high-level coordinator thread for a single Device.
    
    This thread acts as a "producer". It orchestrates the start and end of a
    time step, prepares the work queue, and signals the worker threads to start.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            # 1. Synchronize with all other devices at the start of a time step.
            self.device.barrier.wait()

            # 2. Get neighbors for this time step.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # If neighbors is None, it's the signal to shut down.
            if self.device.neighbours is None:
                self.device.exit_flag.set() # Signal workers to exit.
                self.device.start_tasks.set() # Unblock any waiting workers.
                break

            # 3. Wait for the supervisor to finish assigning all scripts for this step.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # 4. Produce the work: copy the assigned scripts to the shared queue.
            self.device.script_queue = list(self.device.scripts)

            # 5. Signal the worker threads to begin consuming from the queue.
            self.device.start_tasks.set()

            # 6. Wait for the workers to signal that they have finished all tasks.
            self.device.tasks_finished.wait()
            self.device.tasks_finished.clear()

# This class is defined in the `worker` module, but its logic is presented here
# for context, as it's tightly coupled with the Device.
from threading import Thread
class WorkerThread(Thread):
    """A worker/consumer thread that processes scripts from a shared queue."""
    
    def __init__(self, device, thread_id):
        Thread.__init__(self)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        iteratii = 0
        while True:
            # --- SYNCHRONIZATION LOGIC ---
            # Workers sync among themselves.
            self.device.worker_barrier.wait()

            # Worker 0 is responsible for signaling completion to the DeviceThread.
            if self.thread_id == 0 and iteratii != 0:
                self.device.tasks_finished.set()

            # Wait for the DeviceThread (producer) to signal that work is ready.
            self.device.start_tasks.wait()

            # Another internal barrier to ensure all workers start together.
            self.device.worker_barrier.wait()
            if self.thread_id == 0:
                self.device.start_tasks.clear() # Reset for the next cycle.
            
            iteratii += 1
            
            # Check for the global exit signal.
            if self.device.exit_flag.is_set():
                break

            # --- WORK CONSUMPTION LOGIC ---
            # Safely pop a task from the shared queue.
            self.device.script_lock.acquire()
            if len(self.device.script_queue) > 0:
                (script, location) = self.device.script_queue.pop(0)
                self.device.script_lock.release()
            else:
                # No more work in the queue.
                self.device.script_lock.release()
                continue

            # Lock the specific data location before processing.
            self.device.data_locks[location].acquire()
            
            # 1. Aggregate data from neighbors and self.
            script_data = []
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # 2. Run the computation.
            if script_data != []:
                result = script.run(script_data)
                
                # 3. Disseminate the result to all neighbors and self.
                for device in self.device.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
                
            # Release the lock for the location.
            self.device.data_locks[location].release()
