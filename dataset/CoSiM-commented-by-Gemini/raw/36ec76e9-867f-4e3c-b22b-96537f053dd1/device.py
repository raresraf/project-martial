"""
This module defines a hybrid simulation model for a distributed device network.
It uses a master device (ID 0) to set up a shared barrier, but lock creation
is handled in a decentralized and potentially racy manner. Worker threads are
created for each timepoint, with each worker being responsible for a "bundle"
of scripts assigned in a round-robin fashion.

Note: The file contains a `from worker import Worker` statement, but the `Worker`
class is also defined within this file, suggesting the import is a leftover from
a previous file structure.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
from worker import Worker


class Device(object):
    """
    Represents a device in the network, responsible for managing its own
    control thread and participating in shared synchronization.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []


        self.devices = []
        self.cores = 8 # Number of worker threads to spawn per timepoint.
        self.barrier = None
        self.shared_locks = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """Assigns a shared barrier object to this device."""
        self.barrier = barrier

    def set_locks(self, locks):
        """Assigns a shared list of lock objects to this device."""
        self.shared_locks = locks

    def setup_devices(self, devices):
        """
        Initializes shared resources (barrier and locks) for all devices.
        
        @note The lock creation logic is decentralized and has a potential race
        condition. If multiple devices concurrently decide they need to resize
        and propagate the lock list, their actions could overwrite each other.
        """
        self.devices = devices
        
        # Device 0 is the master for barrier creation.
        if self.device_id == 0:
            lbarrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                dev.set_barrier(lbarrier)

        # Find the highest location ID this device is aware of.
        max_loc = max(self.sensor_data.keys(), key=int)
        
        # Block Logic: Decentralized lock list creation.
        # If this device requires more locks than are currently shared, it creates
        # a new list and propagates it to all other devices.
        if  max_loc+1 > len(self.shared_locks):
            llocks = []
            for _ in range(max_loc+1):
                llocks.append(Lock())
            self.set_locks(llocks)
            for dev in self.devices:
                dev.set_locks(llocks)

    def assign_script(self, script, location):
        """Assigns a script to the device or signals the end of a timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It distributes scripts to worker
    threads and manages synchronization across timepoints.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def distribute_scripts(self, scripts):
        """
        Distributes a list of scripts into bundles for each worker thread
        using a round-robin strategy.
        """
        worker_scripts = []
        for _ in range(self.device.cores):
            worker_scripts.append([])
        i = 0
        for script in scripts:
            worker_scripts[i % self.device.cores].append(script)
            i = i + 1
        return worker_scripts

    def run(self):
        """
        The main simulation loop for the device.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.
            
            # Wait for the supervisor to finish assigning all scripts for this timepoint.
            self.device.timepoint_done.wait()

            # Create worker threads and assign them bundles of scripts.
            inner_workers = []
            worker_scripts = self.distribute_scripts(self.device.scripts)
            for worker_scr in worker_scripts:
                inner_thread = Worker(worker_scr,\
                                      neighbours,\
                                      self.device)
                inner_workers.append(inner_thread)
                inner_thread.start()

            # Wait for all worker threads to complete for this timepoint.
            for thr in inner_workers:
                thr.join()

            self.device.timepoint_done.clear()
            # Synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()



class Worker(Thread):
    """
    A worker thread that executes a pre-assigned bundle of scripts sequentially.
    """
    def __init__(self, script_loc, neighbours, device):
        """
        Initializes the worker with its workload.
        
        Args:
            script_loc (list): A list of (script, location) tuples to execute.
            neighbours (list): A list of neighboring Device objects.
            device (Device): The parent device.
        """
        Thread.__init__(self)
        self.script_loc = script_loc
        self.neighbours = neighbours
        self.script_data = []
        self.device = device

    def run(self):
        """
        Processes each script in the assigned bundle sequentially.
        """
        for (script, location) in self.script_loc:
            # Acquire the lock for the specific location to ensure serial access.
            self.device.shared_locks[location].acquire()
            self.script_data = []
            
            # Collect data from neighbors.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    self.script_data.append(data)
            
            # Collect data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                self.script_data.append(data)
            
            # If any data was collected, run the script and update the network.
            if self.script_data != []:
                result = script.run(self.script_data)
                
                for dev in self.neighbours:
                    dev.set_data(location, result)
                
                self.device.set_data(location, result)
            
            # Release the lock for this location.
            self.device.shared_locks[location].release()
