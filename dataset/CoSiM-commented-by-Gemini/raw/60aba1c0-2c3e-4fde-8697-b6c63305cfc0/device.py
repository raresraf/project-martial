"""
@file device.py
@brief A distributed device simulation using a ThreadPoolExecutor for concurrent script execution.

This script models a network of devices that operate in synchronized time steps.
Each device uses a single management thread that delegates parallel tasks to a
ThreadPoolExecutor. Synchronization between devices is achieved with a shared
barrier, while internal task management is event-driven.
"""

from threading import Event, Thread, Lock
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
# Assumes the existence of a ReusableBarrierCond in a local 'barrier' module.
from barrier import ReusableBarrierCond

class Device(object):
    """
    Represents a node in the distributed network. It manages its own state,
    data, and a single worker thread that uses a thread pool for parallelism.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary holding the device's local data.
        @param supervisor A reference to the central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Events for coordinating script processing within a timepoint.
        self.script_received = Event()
        self.timepoint_done = Event()

        # Shared barrier for synchronizing all devices between timepoints.
        self.barrier = None
        # Per-location locks to protect this device's data.
        self.locks = {}
        for location in sensor_data:
            self.locks[location] = Lock()

        # Each device runs a single management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared barrier for all devices.
        This is orchestrated by device 0.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current timepoint.

        @param script The script to execute.
        @param location The location associated with the script's data.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of work for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def acquire_lock(self, location):
        """Acquires the lock for a specific location on this device."""
        if location in self.sensor_data:
            self.locks[location].acquire()

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def release_lock(self, location):
        """Releases the lock for a specific location on this device."""
        if location in self.sensor_data:
            self.locks[location].release()

    def shutdown(self):
        """Shuts down the device by joining its worker thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a device, managing a ThreadPoolExecutor to run scripts
    and handling all synchronization logic.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # A pool of worker threads to execute scripts concurrently.
        self.executor = ThreadPoolExecutor(max_workers=8)

    def run_script(self, script, location, neighbours):
        """
        The target function for the ThreadPoolExecutor. It runs a single script.
        
        @warning This function implements a distributed locking scheme where it
        acquires locks on neighboring devices. This is highly prone to deadlock
        if not carefully managed (e.g., by enforcing a global lock acquisition order).

        @param script The script to run.
        @param location The data location for the script.
        @param neighbours A list of neighboring devices.
        """
        script_data = []

        # Block Logic (Critical Section): Acquire locks on all neighbors before self.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                # Acquire a lock on the specified location *on the neighboring device*.
                device.acquire_lock(location)
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Acquire the lock for the same location on the local device.
        self.device.acquire_lock(location)
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # If any data was collected, run the script and disseminate the results.
        if script_data != []:
            result = script.run(script_data)

            # Write result and release locks on neighbors.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)
                    device.release_lock(location)

            # Write result and release lock on self.
            self.device.set_data(location, result)
            self.device.release_lock(location)

    def run(self):
        """Main execution loop, processing timepoints and managing script execution."""
        # The outer loop progresses through discrete timepoints.
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal from supervisor.

            futures = []

            # The inner loop handles task submission for a single timepoint.
            while True:
                # Wait for either a new script or a "timepoint done" signal.
                if self.device.script_received.isSet() or self.device.timepoint_done.wait():
                    if self.device.script_received.isSet():
                        self.device.script_received.clear()
                        
                        # Submit all received scripts to the thread pool.
                        for (script, location) in self.device.scripts:
                            futures.append(self.executor.submit(self.run_script, script,
                                                                location, neighbours))
                    else:
                        # "Timepoint done" was signaled, meaning all scripts for this
                        # timepoint have been submitted. Wait for them to complete.
                        wait(futures, timeout=None, return_when=ALL_COMPLETED)
                        self.device.timepoint_done.clear()
                        self.device.script_received.set() # Reset for next cycle.
                        break # Exit inner loop to synchronize with other devices.

            # Block Logic: All devices wait here, ensuring that none can start the
            # next timepoint until all have finished the current one.
            self.device.barrier.wait()

        # Cleanly shut down the thread pool.
        self.executor.shutdown()
