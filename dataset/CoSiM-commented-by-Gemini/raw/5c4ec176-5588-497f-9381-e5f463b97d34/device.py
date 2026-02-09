"""
@file device.py
@brief Defines a device model for a distributed simulation with no data locking.

This file implements a `Device` class that processes scripts serially within a
single thread. It uses a two-phase, semaphore-based `ReusableBarrierSem` for
synchronization between timepoints.

@warning This implementation lacks any locking mechanism for concurrent data access,
         making it prone to race conditions and data corruption in a multi-device
         environment.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    A reusable barrier implemented using two semaphores.

    This implementation uses a standard two-phase protocol to ensure that all
    threads wait at the barrier before any can proceed, and that they all
    pass the second phase before the barrier can be used again.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        Causes a thread to wait at the barrier. It consists of two phases
        to ensure reusability.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            # The last thread to arrive releases all other threads from this phase.
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset for next use.
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads2 -= 1
            # The last thread to arrive releases all other threads from this phase.
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset for next use.
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a single device in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.devices = None


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared barrier for all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices
        # Block Logic: The root device (ID 0) creates the shared barrier and
        # distributes it to all other devices.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                dev.barrier = self.barrier
                

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script: The script object to run.
            location: The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that script assignment is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location. This operation is not thread-safe.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location. This operation is not thread-safe.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop. It processes scripts serially and then waits
        at a barrier.

        @warning Data access within this loop is not synchronized. When multiple
                 devices run scripts that access shared locations, race conditions
                 will occur, leading to data corruption.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Process all assigned scripts serially within this thread.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Gather data from neighbors (UNSAFE - NO LOCKING).
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    # Propagate result to neighbors (UNSAFE - NO LOCKING).
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

            
            self.device.timepoint_done.clear()
            # Wait at the barrier for all other devices to finish their timepoint.
            self.device.barrier.wait()