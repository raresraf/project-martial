"""
A device simulation framework featuring a flawed, manual barrier implementation.

This module defines a device simulation that attempts to synchronize time steps
using a custom barrier mechanism. This barrier relies on threads incrementing a
shared `step` counter and then polling the state of all other devices. This
non-atomic polling is a race condition and is not a reliable method for
synchronization, making the system prone to deadlocks.
"""

import threading



from threading import Event, Thread, Lock, Semaphore, current_thread

class Device(object):
    """
    Represents a single device which participates in a custom, poll-based
    barrier synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's internal sensor data.
            supervisor: The central supervisor managing the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.next_iteration = True  
        self.thread = DeviceThread(self)
        self.set_data_lock = Lock()
        # The `step` counter is this device's view of the current timepoint.
        self.step = 0

        self.all_devices = []
        self.all_devices_count = 0
        # The `new_time` event is used to signal/wait for the barrier.
        self.new_time = Event()
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Provides each device with a list of all other devices."""
        self.all_devices = devices
        self.all_devices_count = len(self.all_devices)

    def assign_script(self, script, location):
        """Assigns a script for execution in the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data. This read is not synchronized with `set_data`,
        creating a potential race condition.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Thread-safely updates sensor data at a specific location."""
        if location in self.sensor_data:
            with self.set_data_lock:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()

    def increment_step(self):
        """Thread-safely increments the device's step counter."""
        with self.set_data_lock:
            self.step += 1
                        
class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating a flawed custom barrier.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
            """The main execution loop, organized into discrete timepoints."""
            while True: 
                self.device.new_time.set()
                
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    break
                
                # Wait for script assignments to complete.
                self.device.timepoint_done.wait()
                # Spawn a helper thread to execute the scripts for this timepoint.
                run_thread = RunScripts(self.device, neighbours)
                run_thread.start()
                run_thread.join()

                # Block Logic: A custom, flawed implementation of a barrier.
                # It attempts to synchronize by having the "last" thread to
                # finish a step signal all others to continue.
                self.device.increment_step()
                count = 0
                # CRITICAL: This loop is a race condition. It iterates over all
                # devices and checks their `step` value, but other threads may be
                # modifying their own `step` concurrently.
                for d in self.device.all_devices:
                    if d.step == self.device.step:
                        count += 1
                
                # If this thread believes it's the last one, it signals others.
                if count == self.device.all_devices_count:
                    for d in self.device.all_devices:
                        d.new_time.set()
                # Otherwise, it waits to be signaled, which may never happen
                # due to the race condition, causing a deadlock.
                else:
                    self.device.new_time.wait()

class RunScripts(Thread):
    """A helper thread to execute all assigned scripts for a device."""
    def __init__(self, device, neighbours):
        """Initializes the RunScripts thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """Executes the device's list of scripts for the current timepoint."""
        self.device.new_time.clear()
        
        # Block Logic: Process each assigned script sequentially.
        for (script, location) in self.device.scripts:
            script_data = []
            
            # Aggregate data from neighbours and self.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
            
            # Invariant: Script runs only if there is data to process.
            if script_data != []:
                result = script.run(script_data)

                # Broadcast the result to all participants.
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)