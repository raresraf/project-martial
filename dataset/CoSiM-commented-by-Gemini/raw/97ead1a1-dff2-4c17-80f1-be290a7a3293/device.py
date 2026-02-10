"""
@file device.py
@brief This file defines a simulated device for a distributed system using a per-script threading model
       and a custom two-phase semaphore-based barrier.
@details The implementation spawns a new thread for each script assigned to a device during a time step.
         Synchronization between devices is handled by a shared, custom-built two-phase barrier.
         A critical flaw in this design is the lack of any locking mechanism for accessing sensor data,
         making the 'get_data' and 'set_data' operations prone to race conditions.
"""


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier:
    """
    @brief A custom, reusable two-phase barrier implemented with semaphores.
    @details This barrier ensures that a group of threads all wait at a synchronization point before any of them
             are allowed to proceed. The two-phase design prevents threads from one iteration (or "wave")
             from proceeding into the next iteration before all threads have completed the current one.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters for each phase, wrapped in a list to be mutable by reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Semaphores to block threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier.
        @details The thread must pass through two distinct synchronization phases.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes one phase of the two-phase barrier.
        @param count_threads The counter for the current phase.
        @param threads_sem The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive (the "gatekeeper") opens the gate for all other threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # All threads, including the gatekeeper, wait here until the gate is opened.
        threads_sem.acquire()

class Device(object):
    """
    @brief Represents a single device in the simulated network.
    @details Each device runs a main control thread that spawns individual threads for each script.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.barrier = None
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs collective setup, with device 0 creating and sharing the global barrier.
        @param devices A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            bariera = ReusableBarrier(len(devices))
            self.barrier = bariera
            for dev in devices:
                if dev.barrier is None:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device.
        @param script The script to be executed.
        @param location The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script signals that all scripts for the time step have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @warning This method is not thread-safe. Concurrent calls from multiple ScriptThreads
                 can lead to race conditions when accessing the sensor_data dictionary.
        @return The sensor data or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data at a given location.
        @warning This method is not thread-safe. Concurrent calls from multiple ScriptThreads
                 can lead to race conditions, potentially causing data corruption.
        @param location The location to update.
        @param data The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class ScriptThread(Thread):
    """
    @brief A short-lived thread created to execute a single script.
    @details This thread gathers data, runs a script, and disseminates the results. It does not
             use any locks, making its data operations unsafe.
    """
    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.device = device


        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.script_data = []

    def run(self):
        """
        @brief The main execution logic for the script thread.
        """
        # Block Logic: Gathers data from neighbors and the local device without any locking.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                self.script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            self.script_data.append(data)

        if self.script_data != []:
            
            result = self.script.run(self.script_data)
            
            # Block Logic: Writes the result back to the local device and neighbors without locking.
            # This can cause race conditions if multiple scripts target the same location.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)
        self.script_data = []

class DeviceThread(Thread):
    """
    @brief The main control thread for a device.
    @details This thread orchestrates the execution of scripts for each time step by spawning
             a dedicated thread for each script.
    """
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device thread.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A `None` value for neighbors signals termination.
                break
            
            # Invariant: Wait until all scripts for the current time step are assigned.
            self.device.timepoint_done.wait()
            threads = []
            
            
            # Block Logic: Spawns one new thread for each assigned script.
            for (script, location) in self.device.scripts:
                thrScript = ScriptThread(self.device, script, location, neighbours)
                threads.append(thrScript)

            for thread in threads:
                thread.start()
            # Wait for all script threads to complete before proceeding.
            for thread in threads:
                thread.join()
            
            self.device.timepoint_done.clear()
            
            # Invariant: After all local work is done, wait at the global barrier to synchronize
            # with all other devices before starting the next time step.
            self.device.barrier.wait()