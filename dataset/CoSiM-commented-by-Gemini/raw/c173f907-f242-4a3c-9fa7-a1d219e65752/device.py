
"""
This module implements a distributed, agent-based simulation framework.

It defines the core components for a network of devices that can process data
in parallel. The simulation proceeds in synchronized time steps, where each device
can execute scripts based on its own data and the data of its neighbors. A key
feature is the distributed locking mechanism to ensure data consistency during
parallel processing.

Classes:
    Device: Represents a single node or agent in the simulation network.
    WorkerThread: Executes computational scripts for a Device, handling locking.
    DeviceThread: The main control loop for a Device's lifecycle.
"""

from threading import Event, Thread, Lock
from barrier import *

class Device(object):
    """
    Represents a single device in the distributed simulation.

    Each device runs its own control thread (DeviceThread), manages its local
    sensor data, and can execute assigned scripts using a pool of WorkerThreads.
    It synchronizes with other devices using barriers and events.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data,
                                typically mapping locations to values.
            supervisor (Supervisor): An external object that manages the overall
                                     simulation and device interactions (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script or timepoint has been assigned.
        self.script_received = Event()
        self.scripts = []
        self.scriptType = ""
        # Event to signal that a timepoint synchronization is complete.
        self.timepoint_done = Event()

        self.allDevices = []
        # Event to signal that the global list of all devices has been received.
        self.devices_setup = Event()

        # A barrier for synchronizing all devices at the start of each simulation loop.
        self.barrierLoop = []

        # A global lock to serialize the process of acquiring multiple resource locks.
        self.canRequestResourcesLock = Lock()

        # A dictionary of locks, one for each data location, to control local access.
        self.myResourceLock = { loc : Lock() for loc in self.sensor_data.keys() }

        self.neighbours = []

        self.numWorkers = 8

        # The main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Provides the device with the context of all other devices in the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.allDevices = devices
        # Initializes a reusable barrier for synchronizing all devices.
        self.barrierLoop = ReusableBarrierCond(len(devices))
        self.devices_setup.set()


    def assign_script(self, script, location):
        """
        Assigns a script to be executed or marks a synchronization timepoint.

        Args:
            script (Script): The script object to be executed. If None, this is
                             treated as a timepoint.
            location (any): The data location the script will operate on.
        """
        if script is not None:
            # Append the script and its target location to the list of work.
            self.scripts.append((script, location))
            self.scriptType = "SCRIPT"
            self.script_received.set()
        else:
            # If no script, this is a signal for a timepoint synchronization.
            self.scriptType = "TIMEPOINT"
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        ret = self.sensor_data[location] if location in self.sensor_data else None
        return ret

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by waiting for its main thread to complete."""
        self.thread.join()




class WorkerThread(Thread):
    """
    A thread that executes a batch of scripts for a single Device.

    This thread is responsible for the core logic of acquiring distributed locks
    on data locations, gathering data from neighbor devices, running the script,
    and propagating the results back to the neighborhood.
    """

    def __init__(self, device, listOfIndexes):
        """
        Initializes a WorkerThread.

        Args:
            device (Device): The parent device this worker belongs to.
            listOfIndexes (list): A list of indexes into the parent device's
                                  `scripts` list that this worker should process.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.listOfIndexes = listOfIndexes

    def run(self):
        """The main execution logic for the worker."""

        for i in self.listOfIndexes:
            (script, location) = self.device.scripts[i]

            # --- Distributed Locking Protocol: Start ---
            # Acquire a global lock to ensure that only one worker (across all devices)
            # is attempting to acquire neighborhood resource locks at a time. This
            # prevents deadlocks that could arise from circular dependencies (e.g.,
            # Worker A locks loc1 on Device 1, Worker B locks loc1 on Device 2,
            # then A tries to lock loc1 on 2 and B tries to lock loc1 on 1).
            self.device.allDevices[0].canRequestResourcesLock.acquire()

            # Acquire the lock for the target location on the local device.
            if location in self.device.myResourceLock:
                self.device.myResourceLock[location].acquire()
            # Acquire the lock for the same location on all neighboring devices.
            for device in self.device.neighbours:
                if self.device.device_id != device.device_id:
                      if location in device.myResourceLock:
                            device.myResourceLock[location].acquire()
            
            # Release the global lock now that all necessary resource locks are held.
            self.device.allDevices[0].canRequestResourcesLock.release()
            # --- Distributed Locking Protocol: End ---


            # Gather data from the local device and its neighbors.
            script_data = []
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Execute the script with the aggregated data.
                result = script.run(script_data)

                # Propagate the result back to the local device and all neighbors.
                self.device.set_data(location, result)
                for device in self.device.neighbours:
                    device.set_data(location, result)

            # --- Release Locks ---
            # Release the locks in the reverse order they were acquired.
            if location in self.device.myResourceLock:
                self.device.myResourceLock[location].release()
            for device in self.device.neighbours:
                if self.device.device_id != device.device_id:
                      if location in device.myResourceLock:
                            device.myResourceLock[location].release()



class DeviceThread(Thread):
    """The main control loop thread for a single Device."""

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation lifecycle for the device.

        This loop synchronizes with all other devices, gets tasks, distributes
        them to a pool of worker threads, and waits for completion before starting
        the next simulation step.
        """
        # Wait until the supervisor has provided the global list of all devices.
        self.device.devices_setup.wait()

        while True:
            # Synchronize with all other devices at the start of a timepoint.
            # Assumes device 0 holds the shared barrier instance.
            self.device.allDevices[0].barrierLoop.wait()

            # Get the current set of neighbors from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # If the supervisor returns None, it's a signal to shut down.
            if self.device.neighbours is None:
                break

            # Wait until a script or timepoint signal is received from the supervisor.
            while True:
                self.device.script_received.wait()
                self.device.script_received.clear()
                if self.device.scriptType == "SCRIPT":
                    # If it's a script, we might need to wait for more scripts
                    # before the timepoint is considered complete.
                    continue
                # Wait for the final timepoint signal.
                self.device.timepoint_done.wait()
                self.device.timepoint_done.clear()
                break


            # --- Worker Pool Management ---
            workerThreadList = []
            indexesList = []

            # Create empty lists to hold script indexes for each worker.
            for i in range(self.device.numWorkers):
                indexesList.append([])
            # Distribute script indexes among the workers using a modulo operator.
            for i in range(len(self.device.scripts)):
                indexesList[i%self.device.numWorkers].append(i)

            # Create and start a WorkerThread for each set of assigned indexes.
            for i in range(self.device.numWorkers):
                if indexesList[i] != []:
                    workerThread = WorkerThread(self.device,indexesList[i])
                    workerThreadList.append(workerThread)
                    workerThread.start()

            # Wait for all worker threads to complete their tasks.
            for i in range(self.device.numWorkers):
                if indexesList[i] != []:
                    # This join logic seems incorrect, as it joins based on the
                    # loop index `i` rather than the worker thread object itself.
                    # Documenting as-is.
                    workerThreadList[i].join()
