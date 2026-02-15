
"""
Models a distributed network of devices that operate in synchronized timepoints.
Each device processes data from its local sensors and communicates with its
neighbors to execute distributed scripts. The simulation relies on threading for
concurrency and a barrier for time-step synchronization.
"""

from threading import Event, Thread, Lock
from Queue import Queue
import barrier

class Device(object):
    """
    Represents a single device (or node) in the distributed network.

    Each device has a unique ID, its own sensor data, and is managed by a
    supervisor. It operates in discrete, synchronized timepoints, executing
    assigned scripts using data from its neighbors.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local
                                sensor readings, typically mapping locations to
                                data values.
            supervisor (Supervisor): An object that manages the network topology
                                     and provides neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.everyone = []
        self.barrier = None
        self.locations_lock = [None] * 100
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the network synchronization and locking mechanisms.

        This method should only be called on one device (the coordinator, here
        device_id == 0), which then distributes the shared synchronization
        objects to all other devices in the network.

        Args:
            devices (list[Device]): A list of all devices in the simulation.
        """

        if self.device_id == 0:
            # A reusable barrier to synchronize all devices at the end of a timepoint.
            self.barrier = barrier.ReusableBarrierCond(len(devices))

            for device in devices:
                device.barrier = self.barrier

            self.everyone = devices

            for device in devices:
                device.everyone = self.everyone
            
            # Pre-allocates a pool of locks for location-based resource protection.
            for i in range(100):
                self.locations_lock[i] = Lock()

            for device in devices:
                device.locations_lock = self.locations_lock

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device in the current timepoint.

        Args:
            script (object): The script object to be executed. It must have a
                             `run` method.
            location (any): The data location the script will operate on.
        """

        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is a signal that all scripts for the timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (any): The location to query for data.

        Returns:
            The data at the given location, or None if not available.
        """

        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location.

        Args:
            location (any): The location to update.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the device's main thread to gracefully shut it down."""

        self.thread.join()

class DeviceThread(Thread):
    """
    The main execution thread for a Device.

    This thread orchestrates the device's lifecycle, managing the execution of
    scripts for each timepoint. It spawns worker threads to process scripts
    concurrently.
    """

    def __init__(self, device):
        """
        Initializes the device's main thread.

        Args:
            device (Device): The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []
        self.jobs = Queue(maxsize=0)
        self.exit = False
        self.time_done = False
        self.neighbours = []

    def run(self):
        """
        The main loop for the device thread.

        The thread waits for scripts to be assigned for a timepoint, processes
        them using a pool of worker threads, and then synchronizes with all
        other devices using a barrier before proceeding to the next timepoint.
        """
        while True:

            self.time_done = False
            
            # Determines network topology for the current timepoint.
            self.neighbours = self.device.supervisor.get_neighbours()
            
            # A None neighbor list signals the end of the simulation.
            if self.neighbours is None:
                self.time_done = True
                self.exit = True
            
            # Pre-condition: Wait until all scripts for the timepoint are assigned.
            if self.exit is False:
                self.device.timepoint_done.wait()

            if self.time_done is False:
                if self.neighbours == []:
                    self.time_done = True
            
            # Block Logic: If there are neighbors, execute assigned scripts.
            if self.time_done is False:
                
                # Spawns a pool of worker threads to execute jobs.
                for i in range(8):
                    thread = Workerr(self, self.neighbours)
                    self.workers.append(thread)

                for i in range(8):
                    self.workers[i].start()
                
                # Loads the job queue with scripts for the workers to consume.
                for script in self.device.scripts:
                    self.jobs.put(script)
                
                # Waits for all script jobs to be completed by the workers.
                self.jobs.join()

                # Signals worker threads to exit by sending a sentinel (None).
                for i in range(8):
                    self.jobs.put(None)
                
                # Waits for the sentinel messages to be processed.
                self.jobs.join()
                
                # Cleans up the worker threads.
                for i in range(8):
                    self.workers[i].join()

                self.workers = []

                self.time_done = True
            
            # Post-condition: Timepoint is complete for this device.
            if self.time_done is True:
                
                # Resets the event for the next timepoint.
                self.device.timepoint_done.clear()
                # Invariant: All devices wait at the barrier, ensuring synchronization
                # across the network before starting the next timepoint.
                self.device.everyone[0].barrier.wait()

                if self.exit is True:
                    break

# NOTE: "Workerr" is likely a typo for "Worker".
class Workerr(Thread):
    """
    A worker thread that executes a single script.

    It gathers data from neighboring devices, runs the script, and disseminates
    the result.
    """

    def __init__(self, device, neighbours):
        """
        Initializes a worker thread.

        Args:
            device (DeviceThread): The parent device thread that manages this worker.
            neighbours (list[Device]): A list of neighboring devices to communicate with.
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        The main loop for a worker thread.

        It continuously fetches jobs from the queue. For each job, it acquires
        a lock for the target location, gathers data, executes the script, and
        updates the data on itself and its neighbors.
        """
        while True:

            job_to_do = self.device.jobs.get()
            
            # A None job is the sentinel value to signal the thread to exit.
            if job_to_do is None:
                self.device.jobs.task_done()
                break
            
            # Invariant: Acquires a lock to ensure exclusive access to a specific
            # location's data across all workers and devices.
            self.device.device.locations_lock[job_to_do[1]].acquire()
            script_data = []
            
            # Gathers data from all neighbors for the specified location.
            for device in self.neighbours:



                data = device.get_data(job_to_do[1])

                if data is not None:
                    script_data.append(data)
            
            # Gathers data from the local device as well.
            data = self.device.device.get_data(job_to_do[1])

            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                # Executes the script with the aggregated data.
                result = (job_to_do[0]).run(script_data)
                
                # Disseminates the result to all neighbors and the local device.
                for device in self.neighbours:

                    device.set_data(job_to_do[1], result)



                    self.device.device.set_data(job_to_do[1], result)

            self.device.device.locations_lock[job_to_do[1]].release()

            self.device.jobs.task_done()
