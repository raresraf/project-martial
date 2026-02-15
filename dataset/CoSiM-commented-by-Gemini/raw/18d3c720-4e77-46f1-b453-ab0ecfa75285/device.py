"""
This module defines a distributed device simulation framework.

The architecture consists of a main controller thread per device that, for each
simulation step, dynamically creates a new pool of worker threads to execute
tasks. Synchronization between devices is managed by a two-phase barrier system.
"""

from threading import Event, Thread, Lock
from reference import CommonReference
import multiprocessing

class Device(object):
    """
    Represents a single device node in the simulation.
    
    Each device has a main controller thread and dynamically creates worker
    threads each time step to process a list of assigned scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device object.
        
        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): A dictionary of the device's local data.
            supervisor (Supervisor): The central supervisor managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals that script assignment is done.
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self) # The main controller thread.
        self.thread.start()
        self.number_of_processors = multiprocessing.cpu_count()

        # An event to ensure the device waits for global setup to complete.
        self.wait_for_reference = Event()
        
        # A shared object containing synchronization barriers.
        self.synch_reference = None

        self.thread_list = [] # Stores dynamically created worker threads.

        # A dictionary of locks to protect access to data locations.
        self.location_locks = {}
        for entry in self.sensor_data:
            self.location_locks[entry] = Lock()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes the common synchronization object.
        
        This method is called by the master device (ID 0) to create a
        `CommonReference` object containing shared barriers and distribute it to
        all other devices in the simulation.
        """
        self.devices = devices
        
        if self.device_id == 0:
            # The master device creates the shared synchronization reference.
            self.synch_reference = CommonReference(len(self.devices))
            for dev in self.devices:
                if dev.device_id != 0:
                    dev.synch_reference = self.synch_reference
            
            # Signal all devices that the setup is complete.
            for dev in self.devices:
                dev.wait_for_reference.set()
        else:
            # Non-master devices wait here until the setup is done.
            self.wait_for_reference.wait()

    def assign_script(self, script, location):
        """
        Assigns a script to the device.
        
        A `None` script signals that all scripts for the current step have
        been assigned, triggering the `script_received` event.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves data from a specific location.
        
        @note This method has a significant flaw: it acquires a lock but
              never releases it, which will lead to deadlocks.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
        return self.sensor_data[location] if location in \
               self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets data at a specific location.
        
        @note This method has a significant flaw: it releases a lock that
              it never acquired.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()

    def shutdown(self):
        """Waits for the device's main controller thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main controller thread for a device, orchestrating the simulation steps."""

    def __init__(self, device):
        """Initializes the controller thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def simple_task(self, neighbours, script, location):
        """
        The target function for a worker thread, executing one script.
        
        It gathers data, runs the script, and distributes the results.
        """
        script_data = []
        
        # Gather data from neighboring devices.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # If any data was found, run the script and distribute the result.
        if script_data:
            result = script.run(script_data)
            for device in neighbours:
                device.set_data(location, result)
            self.device.set_data(location, result)

    def run_tasks(self, neighbours, list_of_tuples):
        """A helper function for a worker thread to run a list of tasks."""
        for (script, location) in list_of_tuples:
            self.simple_task(neighbours, script, location)

    def run(self):
        """The main simulation loop."""
        while True:
            # Get neighbors from the supervisor. If None, simulation is over.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            # --- First Barrier ---
            # Synchronize with all other devices before starting computation.
            self.device.synch_reference.first_barrier.wait()
            
            if self.device in neighbours:
                neighbours.remove(self.device)
            
            # --- Dynamic Thread Creation ---
            # Partition the scripts and create a new set of worker threads for this step.
            self.list_of_thread_lists = [[] for _ in range(self.device.number_of_processors)]
            
            load_factor = 2
            if len(self.device.scripts) <= (load_factor * self.device.number_of_processors):
                # If there are few scripts, create one thread per script.
                for (script, location) in self.device.scripts:
                    thread = Thread(target=self.simple_task, args=(neighbours, script, location))
                    self.device.thread_list.append(thread)
                    thread.start()
                for thread in self.device.thread_list:
                    thread.join()
                del self.device.thread_list[:]
            else:
                # If there are many scripts, partition them among threads.
                i = 0
                for (script, location) in self.device.scripts:
                    self.list_of_thread_lists[i % self.device.number_of_processors].append((script, location))
                    i += 1
                
                for i in range(self.device.number_of_processors):
                    if self.list_of_thread_lists[i]:
                        thread = Thread(target=self.run_tasks, args=(neighbours, self.list_of_thread_lists[i]))
                        self.device.thread_list.append(thread)
                        thread.start()
                
                for thread in self.device.thread_list:
                    thread.join()
                del self.device.thread_list[:]
            
            # --- Second Barrier ---
            # Synchronize with all other devices after computation is complete.
            self.device.synch_reference.second_barrier.wait()

# This class definition seems to be pulled from another file.
from barrier import SimpleBarrier
from threading import Lock

class CommonReference(object):
    """A container for shared synchronization objects (barriers and locks)."""
    
    def __init__(self, number_of_devices):
        """
        Initializes the shared reference with two barriers.
        
        Args:
            number_of_devices (int): The total number of devices in the simulation.
        """
        self.lock = Lock()
        self.first_barrier = SimpleBarrier(number_of_devices)
        self.second_barrier = SimpleBarrier(number_of_devices)
