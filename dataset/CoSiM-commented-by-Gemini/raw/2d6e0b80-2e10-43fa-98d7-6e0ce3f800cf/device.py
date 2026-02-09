"""
A simulation framework for a network of devices using a global lock.

This module defines a `Device` class and a custom `ReusableBarrier`. The system
simulates a network of devices that process data in synchronized time steps.
A key feature of this implementation is the use of a single, global lock for
all script executions, which serializes the data processing across all devices,
ensuring that only one script runs at any given time.
"""

from threading import Lock, Semaphore, Event, Thread

class ReusableBarrier(object):
    """
    Provides a reusable barrier for synchronizing a fixed number of threads.

    This barrier is implemented using a two-phase signaling protocol with
    semaphores. Threads wait at the barrier until all participating threads have
    arrived. The two-phase nature prevents threads from one synchronization
    cycle from proceeding before all threads have left the previous cycle.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock() 
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """
        Causes a thread to wait until all `num_threads` have reached the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the two-phase barrier protocol.

        The last thread to arrive resets the counter for the next wave and
        releases all waiting threads by signaling a semaphore.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Invariant: The last thread to arrive triggers the release.
            if count_threads[0] == 0: 
                i = 0
                while i < self.num_threads:
                    threads_sem.release() 
                    i += 1                
                count_threads[0] = self.num_threads  
        threads_sem.acquire() 
                              
                              

class Device(object):
    """
    Represents a single device in the simulated network.

    Each device manages its own data and runs a control thread. It participates
    in a synchronized setup where a single global lock and a shared barrier
    are distributed for system-wide use.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's internal sensor data.
            supervisor: The central supervisor managing the device network.
        """
        self.barrier = None
        self.lock = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources for all devices.

        The first device in the list acts as a master, creating a shared
        `ReusableBarrier` and a single, global `Lock`. These resources are
        then assigned to all devices in the simulation.

        Args:
            devices (list): A list of all Device objects.
        """
        if devices[0].barrier is None:
            # Block Logic: The master device creates and distributes the shared objects.
            if self.device_id == devices[0].device_id:
                bariera = ReusableBarrier(len(devices))
                # A single lock is created and shared across all devices.
                my_lock = Lock()
                for device in devices:
                    device.barrier = bariera
                    device.lock = my_lock



    def assign_script(self, script, location):
        """
        Receives a script from the supervisor for the current timepoint.

        Args:
            script: The script object to execute.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating serialized execution.

    This thread manages the device's operation within synchronized timepoints.
    It processes all its assigned scripts sequentially, using a global lock that
    ensures only one script is running across the entire system at any moment.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop, organized into discrete timepoints."""
        while True:
            # Get neighbours for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to finish assigning scripts.
            self.device.timepoint_done.wait()

            # Block Logic: Sequentially process each assigned script.
            for (script, location) in self.device.scripts:
                # Pre-condition: Acquire the single global lock. This serializes
                # script execution across all devices in the system.
                self.device.lock.acquire()
                script_data = []
                
                # Aggregate data from neighbours and self.
                for device in neighbours:
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
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                self.device.lock.release()
            
            self.device.timepoint_done.clear()
            # Invariant: All devices must wait at the barrier, ensuring they all
            # complete the current timepoint before any can proceed.
            self.device.barrier.wait()