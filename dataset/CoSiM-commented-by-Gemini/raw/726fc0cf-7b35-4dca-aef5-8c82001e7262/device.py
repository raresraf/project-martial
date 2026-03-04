"""
A device simulation framework using a custom reusable barrier and a thread-per-script model.

This script defines a simulation of distributed devices. It features a custom
`ReusableBarrierSem` for synchronization. Each device's main thread (`DeviceThread`)
spawns a new thread (`MyScriptThread`) for each script it needs to execute. The
synchronization logic within the `DeviceThread` is notable for its use of a
double barrier wait in each cycle.
"""


from threading import Event, Semaphore, Lock, Thread



class ReusableBarrierSem(object):
    """
    A reusable barrier implemented using semaphores.

    This barrier allows a group of threads to wait for each other to reach a
    certain point before proceeding. It uses a two-phase protocol to allow for
    multiple uses.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier.

        Args:
            num_threads: The number of threads that will synchronize on this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)
    def wait(self):
        """Waits at the barrier until all threads have called this method."""
        self.phase1()
        self.phase2()
    def phase1(self):
        """The first phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    def phase2(self):
        """The second phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a device in the simulated network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id: Unique identifier for the device.
            sensor_data: The device's local sensor data.
            supervisor: The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        
        self.my_lock = Lock()
        self.barrier = ReusableBarrierSem(0)
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for all devices.

        Args:
            devices: A list of all devices in the simulation.
        """
        if self.device_id == 0:
            # Device 0 creates the barrier and other devices use its instance.
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        Args:
            script: The script to be executed.
            location: The location for the script's execution.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()


    def get_data(self, location):
        """
        Retrieves data for a given location.

        Args:
            location: The location from which to retrieve data.

        Returns:
            The data at the location, or None.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets data for a given location.

        Args:
            location: The location to set data at.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()



class MyScriptThread(Thread):
    """
    A thread dedicated to executing a single script.
    """

    def __init__(self, script, location, device, neighbours):
        """
        Initializes the script execution thread.

        Args:
            script: The script to be executed.
            location: The location for execution.
            device: The parent Device object.
            neighbours: A list of neighboring devices.
        """
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script, gathers data, and distributes the result.
        """
        script_data = []

        # Gather data from neighbors and the parent device.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)



        if script_data != []:
            # Run the script.
            result = self.script.run(script_data)

            # Distribute the result to neighbors, using a per-device lock.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            # Set the result for the parent device.
            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()

class DeviceThread(Thread):
    """
    The main control thread for a device.
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
        The main execution loop for the device.

        This loop has an unusual synchronization pattern, waiting at the barrier
        twice per cycle.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # First barrier wait of the cycle.
            self.device.barrier.wait()


            self.device.script_received.wait()
            script_threads = []
            
            # Create a thread for each script.
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            for thread in script_threads:
                thread.start()
            for thread in script_threads:
                thread.join()
            
            
            self.device.timepoint_done.wait()
            # Second barrier wait of the cycle.
            self.device.barrier.wait()
            self.device.script_received.clear()