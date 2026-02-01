# -*- coding: utf-8 -*-
"""
/**
 * @file device.py
 * @brief This module defines a simulated distributed system of devices that can execute scripts and synchronize.
 *
 * @details
 * The system consists of `Device` objects, each managed by a `DeviceThread`. A central `Supervisor`
 * (defined elsewhere) coordinates the devices. The simulation proceeds in synchronized timepoints,
 * where each device executes assigned scripts, gathers data from its neighbors, computes a result,
 * and updates its own data and its neighbors' data.
 *
 * Core components:
 * - **Device**: Represents a single node in the system, holding sensor data and managing its script execution logic.
 * - **DeviceThread**: The main control thread for a device. It spawns `Worker` threads to execute scripts in parallel.
 * - **Worker**: A thread responsible for executing a subset of the scripts assigned to a device. It handles data
 *   gathering from neighbors, script execution, and data propagation.
 * - **ReusableBarrier**: A synchronization primitive that ensures all devices complete their work for a
 *   given timepoint before any device can proceed to the next one. This enforces a lock-step execution model.
 * - **Concurrency**: The system uses `Lock` objects to ensure thread-safe access to shared sensor data
 *   and to serialize operations on the same data location.
 *
 * The overall architecture demonstrates a pattern of distributed computation with parallel processing
 * within each node and global synchronization across all nodes.
 */
"""
from threading import Thread, Semaphore, Lock, Event


class ReusableBarrier(object):
    """
    Implements a reusable, two-phase synchronization barrier for a fixed number of threads.

    This barrier forces a set of threads to wait at a synchronization point until all of them
    have reached it. Once all threads are waiting, they are all released simultaneously. The barrier
    resets itself after each release, making it "reusable" for iterative, multi-phase algorithms.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier for a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will be synchronized by this barrier.
        """
        self.num_threads = num_threads
        # Counters for the two synchronization phases. A list is used to make the counter mutable
        # across different method calls within the same object instance.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # A lock to protect access to the shared counters, preventing race conditions.
        self.count_lock = Lock()
        # Semaphores used to block and release threads for each of the two phases.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to block until all `num_threads` have called this method.
        The implementation uses two phases to prevent race conditions where "fast" threads could
        loop around and re-enter the barrier before "slow" threads have exited from the
        previous synchronization.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single synchronization phase.

        Args:
            count_threads (list[int]): The counter for the current phase.
            threads_sem (Semaphore): The semaphore for blocking threads in this phase.
        """
        # Block Logic: Manages the arrival of threads at the barrier.
        with self.count_lock:
            count_threads[0] -= 1
            # Pre-condition: Checks if the current thread is the last one to arrive.
            if count_threads[0] == 0:
                # If it is the last thread, it unblocks all other waiting threads.
                nr_threads = self.num_threads
                while nr_threads > 0:
                    threads_sem.release()
                    nr_threads -= 1
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # Block the thread until it's released by the last thread.
        threads_sem.acquire()   
                                
class Device(object):
    """
    Represents a single device (or node) in the distributed system simulation.

    Each device has a unique ID, local sensor data, and a reference to the global supervisor.
    It manages its own execution thread and synchronizes with other devices using a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device object.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data, keyed by location.
            supervisor (Supervisor): The central supervisor object managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script is assigned.
        self.script_received = Event()
        self.scripts = []
        # Event to signal the completion of a timepoint's tasks.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        # Shared synchronization primitives.
        self.locks = None
        self.barrier = None

    def __str__(self):
        """
        Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects (barrier and locks) to all devices.

        This method should be called once to ensure all devices in the simulation
        share the same barrier for global synchronization and the same set of locks for
        location-based data access.

        Args:
            devices (list): A list of all `Device` objects in the system.
        """
        # Functional Utility: This setup routine ensures all devices share a single barrier instance for synchronization
        # and a common set of locks for accessing data at specific locations, preventing race conditions.
        if all(element is None for element in [self.barrier, self.locks]):
            barrier = ReusableBarrier(len(devices))
            locks = []
            
            # Determine the maximum location ID to create a lock for each location.
            max_locations = 0
            for device in devices:
                for location in device.sensor_data.keys():
                    if location > max_locations:
                        max_locations = location

            for location in range(max_locations + 1):
                locks.append(Lock())

            
            # Distribute the shared barrier and locks to all devices.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution or signals the end of a timepoint.

        Args:
            script (Script): The script object to execute. If None, it signals the end of the timepoint.
            location (any): The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))

        else:
            # A None script is a sentinel value indicating that all scripts for the timepoint have been assigned.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data for a specific location from the device's local sensor data.

        Args:
            location (any): The key for the data to retrieve.

        Returns:
            The data for the given location, or None if the location does not exist.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the data for a specific location in the device's local sensor data.

        Args:
            location (any): The key for the data to update.
            data (any): The new value for the data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by waiting for its main thread to complete.
        """
        self.thread.join()



class DeviceThread(Thread):
    """
    The main control thread for a single `Device`.

    This thread is responsible for managing the device's lifecycle, including spawning
    worker threads to execute scripts, coordinating with the supervisor, and participating
    in global barrier synchronization.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent `Device` object that this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device.

        This loop represents the device's active participation in the simulation. It repeatedly
        waits for a timepoint, processes assigned scripts by distributing them to worker threads,
        and then synchronizes with all other devices at a barrier before starting the next timepoint.
        """
        # Block Logic: The main loop for the device's lifecycle.
        # Invariant: At the start of each iteration, all devices in the system are at the same synchronization point.
        while True:
            
            # Fetches the list of neighbors for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: A None value for neighbors signals a system-wide shutdown.
            if neighbours is None:
                break

            
            # Waits until the supervisor signals that all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            number_of_scripts = len(self.device.scripts)

            
            if number_of_scripts != 0:

            	# Block Logic: Script distribution to worker threads.
                # Creates a pool of worker threads to process scripts in parallel.
                if number_of_scripts < 8:
                    number_of_threads = number_of_scripts
                else:
                    number_of_threads = 8

                workers_list = []

                
                
                for i in range(number_of_threads):
                    worker = Worker(self.device, neighbours)
                    workers_list.append(worker)

                current_thread = 0
                average_scripts = 0

                
                
                

                if number_of_threads > 0:
                    # Functional Utility: Distributes scripts among worker threads using a simple round-robin like strategy.
                    average_scripts = len(self.device.scripts) / number_of_threads
                aux_average = average_scripts

                
                
                
                for(script, location) in self.device.scripts:
                    if aux_average > 0:
                        workers_list[current_thread].scripts.append((script, location))
                        aux_average -= 1
                    if aux_average == 0:
                        aux_average = average_scripts

                        
                        
                        if current_thread < number_of_threads - 1:
                            current_thread += 1
                        else:
                            current_thread = 0
                
                # Start all worker threads.
                for i in range(number_of_threads):
                    workers_list[i].start()

                
                # Wait for all worker threads to complete their assigned scripts.
                for i in range(number_of_threads):
                    workers_list[i].join()

            self.device.timepoint_done.clear()
            
            # Functional Utility: Waits at the barrier for all other devices to finish their timepoint processing.
            self.device.barrier.wait()

class Worker(Thread):
    """
    A worker thread that executes a subset of a device's scripts for a single timepoint.
    """
    def __init__(self, device, neighbours):
        """
        Initializes a Worker thread.

        Args:
            device (Device): The parent device that this worker belongs to.
            neighbours (list): A list of neighboring devices to interact with.
        """
        Thread.__init__(self)
        self.device = device
        self.scripts = []
        self.neighbours = neighbours

    def run(self):
        """
        Executes the scripts assigned to this worker.

        For each script, it acquires a location-specific lock, gathers data from the parent
        device and its neighbors, runs the script, and propagates the result.
        """
        for (script, location) in self.scripts:
            script_data = []

            
            # Acquire a lock to ensure exclusive access to data at this location.
            self.device.locks[location].acquire()

            
            # Block Logic: Aggregate data from all neighbors for the specified location.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            
            # Include the parent device's own data in the aggregation.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                # Functional Utility: Execute the script with the collected data.
                result = script.run(script_data)
                
                # Block Logic: Propagate the result to all neighbors.
                for device in self.neighbours:
                    device.set_data(location, result)
                
                # Update the parent device's data with the result.
                self.device.set_data(location, result)

            
            
            # Release the lock for the location.
            self.device.locks[location].release()
