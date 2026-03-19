"""
This module provides a framework for simulating a distributed network of devices.

It defines a `Device` class that operates in a multi-threaded environment,
communicating with neighboring devices to process sensor data in synchronized
time-steps. The simulation uses thread barriers for synchronization and an
event-based system for script assignment.
"""


from threading import Event, Thread, Condition

class ReusableBarrierCond():
    """
    A reusable barrier implementation using a Condition variable.

    This barrier blocks a specified number of threads until all of them have
    reached the barrier. Once all threads are waiting, they are all released,
    and the barrier resets for the next use.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Causes a thread to wait at the barrier.

        The thread will block until `num_threads` have called this method.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # If this is the last thread to arrive, wake up all waiting threads.
        if self.count_threads == 0:
            self.cond.notify_all()
            # Reset the barrier for the next use.
            self.count_threads = self.num_threads
        else:
            # Wait until notified by the last thread.
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device runs in its own thread, manages its own sensor data, and
    executes assigned scripts in synchronized time-steps with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor data.
            supervisor (object): A supervisor object that manages the network topology.
        """
        self.device_id = device_id
        self.devices = None
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been assigned.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that a timepoint computation is complete.
        self.timepoint_done = Event()
        self.barrier = None
        # The main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()



    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for all devices in the simulation.

        This method should be called on a single master device (e.g., device_id 0)
        to initialize and distribute a common barrier to all devices.
        """
        self.devices = devices
        # The device with ID 0 acts as the master for setting up the shared barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(self.devices))
            for device in devices:
                device.barrier = self.barrier


    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        If a script is provided, it is added to the device's script queue.
        If the script is None, it signals that the current timepoint is done.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class ScriptsThread(Thread):
    """
    A worker thread responsible for executing a script on sensor data.
    """

    def __init__(self, device, scripts, neighbours):
        """
        Initializes the script-executing thread.

        Args:
            device (Device): The parent device instance.
            scripts (list): A list of scripts to execute.
            neighbours (list): A list of neighboring devices.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours


    def run(self):
        """
        The main execution logic for the script thread.

        It gathers data from its device and neighbors, runs the script,
        and then updates data based on the script's result.
        """
        for (script, location) in self.scripts:
            script_data = []
            
            # Gather data from all neighboring devices.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    if data != self.device.get_data(location):
                        script_data.append(data)
                
            # Include the device's own data.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Execute the script with the collected data.
                result = script.run(script_data)

                # Intent: Update data on neighbors if the script result is greater.
                # NOTE: The logic `device.get_data(result)` is likely a bug. It uses the
                # script's output `result` as a key to fetch data, which is unusual.
                # It likely intended to use `device.get_data(location)`.
                for device in self.neighbours:
                    if result > device.get_data(result):
                        device.set_data(location, result)
                    
                if result > self.device.get_data(result):
                    self.device.set_data(location, result)
            
        # Synchronize with other script threads before finishing.
        self.device.thread.barrier.wait()

class DeviceThread(Thread):
    """
    The main control thread for a Device, managing its lifecycle and script execution.
    """

    def __init__(self, device):
        """
        Initializes the main device thread.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = None
        self.list_of_threads = []


    def run(self):
        """
        The main loop for the device.

        Waits for a timepoint signal, processes assigned scripts by spawning
        worker threads, and synchronizes with all other devices at the end of
        the time-step.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            # If the supervisor returns None, the simulation is shutting down.
            if neighbours is None:
                break
            
            # Wait for the signal to begin processing the next time-step.
            self.device.timepoint_done.wait()
            now_thread = 0
            now_script = 0
            
            # This logic seems to be a crude way of batching scripts for worker threads.
            for script in self.device.scripts:
                if now_script == 8:
                    now_script = 0
                else:
                    if now_script < 8:
                        self.list_of_threads.append(ScriptsThread(self.device, [script], neighbours))
                    else:
                        self.list_of_threads[now_thread].scripts.add(script)
                now_thread += 1
                now_script += 1
            
            # Create a barrier to synchronize the completion of all worker threads for this device.
            self.barrier = ReusableBarrierCond(len(self.list_of_threads))
            for thread in self.list_of_threads:
                thread.start()

            # Wait for all local script worker threads to finish.
            for thread in self.list_of_threads:
                thread.join()
            
            self.list_of_threads = []
            # Clear the signal, preparing for the next time-step.
            self.device.timepoint_done.clear()
            # Wait at the global barrier to synchronize with all other devices.
            self.device.barrier.wait()
            self.list_of_threads = []