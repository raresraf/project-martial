"""
Models a device in a distributed simulation environment, using a thread-per-script model.

This script defines a `Device` that participates in a simulation coordinated
by a supervisor. Each device runs a main `DeviceThread` which, in turn, spawns
a `ScriptRunner` thread for each assigned script. The system uses a shared
barrier for synchronization between time steps and a somewhat complex mechanism
for managing locks associated with different data locations.
"""


from threading import Event, Thread, Lock
import barrier
import runner


class Device(object):
    """
    Represents a single device in the simulation.

    Each device manages its own state, including sensor data and assigned scripts,
    and is responsible for executing these scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id: Unique identifier for the device.
            sensor_data: Initial sensor data for the device.
            supervisor: The central supervisor of the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()


        self.thread = DeviceThread(self)
        self.barr = None
        self.devices = []
        self.runners = []
        self.locks = [None] * 50
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for all devices in the simulation.

        Args:
            devices: A list of all devices in the simulation.
        """
        if self.barr is None:
            barr = barrier.ReusableBarrierSem(len(devices))
            self.barr = barr
            for dev in devices:
                if dev.barr is None:
                    dev.barr = barr
        
        for dev in devices:
            if dev is not None:
                self.devices.append(dev)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location.

        Also handles the lazy initialization and sharing of location locks.

        Args:
            script: The script to be executed.
            location: The location for script execution.
        """
        ok = 0
        if script is not None:
            self.scripts.append((script, location))
            
            # This block attempts to find an existing lock for the location from other
            # devices, or creates a new one if none is found. This is a decentralized
            # and potentially inefficient way to manage shared locks.
            if self.locks[location] is None:
                for device in self.devices:
                    if device.locks[location] is not None:
                        self.locks[location] = device.locks[location]
                        ok = 1
                        break
                if ok == 0:
                    self.locks[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data for a given location.

        Args:
            location: The location to retrieve data from.

        Returns:
            The data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        """
        Sets data for a given location.

        Args:
            location: The location to set data at.
            data: The data to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread of execution for a device.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The parent Device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread.

        It waits for a timepoint to be marked as done, then spawns ScriptRunner
        threads for each script. The logic for batching these threads is complex
        and potentially buggy.
        """
        while True:
            # Get neighbors for the current simulation step.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break


            self.device.timepoint_done.wait()

            # Create and run a ScriptRunner for each assigned script.
            for (script, location) in self.device.scripts:
                run = runner.ScriptRunner(self.device, script, location,
                                          neighbours)
                self.device.runners.append(run)

                n = len(self.device.runners)
                x = n / 8
                r = n % 7
                
                # The lock is acquired here, which serializes the execution for this location,
                # largely defeating the purpose of the threading within this block.
                self.device.locks[location].acquire()
                # The logic for starting threads in batches seems overly complicated and
                # may not function as intended.
                for i in xrange(0, x):
                    for j in xrange(0, 8):
                        self.device.runners[i * 8 + j].start()
                
                if n >= 8:
                    for i in xrange(len(self.device.runners) - r,
                                    len(self.device.runners)):
                        self.device.runners[i].start()
                
                else:
                    for i in xrange(0, n):
                        self.device.runners[i].start()
                for i in xrange(0, n):
                    self.device.runners[i].join()
                


                self.device.locks[location].release()
                
                self.device.runners = []

            # Clear the event and wait at the barrier for the next timepoint.
            self.device.timepoint_done.clear()
            
            self.device.barr.wait()


from threading import Thread


class ScriptRunner(Thread):
    """
    A thread dedicated to executing a single script.
    """

    def __init__(self, device, script, location, neighbours):
        """
        Initializes the ScriptRunner.

        Args:
            device: The parent Device object.
            script: The script to be executed.
            location: The location for script execution.
            neighbours: A list of neighboring devices.
        """
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script.

        Gathers data from the parent device and its neighbors, runs the script,
        and then distributes the result.
        """
        script_data = []
        
        # Gather data from neighbors.
        for device in self.neighbours:
            
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the parent device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Run the script.
            result = self.script.run(script_data)
            
            # Distribute the result to neighbors.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Set the result for the parent device.
            self.device.set_data(self.location, result)