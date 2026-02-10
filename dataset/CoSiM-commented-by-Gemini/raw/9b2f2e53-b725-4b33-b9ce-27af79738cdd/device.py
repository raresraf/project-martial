"""
@file device.py
@brief This file defines a simulated device using a multi-level threading model where a main device
       thread spawns worker threads for each script.
@details The design features a custom condition-based reusable barrier and a per-device lock for
         write operations (`set_data`). However, the implementation has critical flaws: the barrier
         is initialized incorrectly in a racy manner, and read operations (`get_data`) are not
         thread-safe, creating a potential for race conditions.
"""

from threading import Event, Thread, Lock, Condition


class ReusableBarrierCond:
    """
    @brief A reusable barrier implementation using a `threading.Condition`.
    @details Allows a group of threads to synchronize at a point. It is reusable after all
             threads have passed.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    @brief Represents a single device in the simulated network.
    @details The device uses a main thread to manage script execution, which in turn spawns
             a new thread for each individual script.
    """
    barrier = None

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()


        self.scripts = []
        self.timepoint_done = Event()
        # A lock to protect write access to this device's sensor_data.
        self.set_data_lock = Lock()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Attempts to set up a shared barrier for all devices.
        @warning The logic here is buggy and racy. It will create multiple barrier objects instead
                 of a single shared one. This will likely cause the simulation to deadlock as
                 different devices will be waiting on different barrier instances.
        @param devices A list of all Device objects.
        """
        
        
        for device in devices:
            if device.device_id == 0:
                self.barrier = ReusableBarrierCond(len(devices))
            else:
                self.barrier = devices[0].barrier

        pass

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @warning This method is not thread-safe. It accesses the sensor_data dictionary directly
                 without acquiring a lock. This can lead to inconsistent reads if another thread
                 is simultaneously modifying the data via `set_data`.
        @return The sensor data or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data at a given location in a thread-safe manner.
        @details This method correctly uses a lock to prevent race conditions during write operations.
        @param location The location to update.
        @param data The new data value.
        """
        with self.set_data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class ComputationThread(Thread):
    """
    @brief A short-lived worker thread that executes a single computational script.
    """

    def __init__(self, device_thread, neighbours, script_data):
        Thread.__init__(self, name="Worker %s" % device_thread.name)
        self.device_thread = device_thread
        self.neighbours = neighbours
        self.script = script_data[0]
        self.location = script_data[1]

    def run(self):
        """
        @brief Gathers data, runs a script, and disseminates the results.
        """
        script_data = []
        
        # Block Logic: Gathers data from neighbors. This is a potential race condition site
        # because the `get_data` method is not thread-safe.
        for device in self.neighbours:


            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        data = self.device_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            


            result = self.script.run(script_data)

            
            # The `set_data` method is thread-safe, so these write operations are protected.
            for device in self.neighbours:
                device.set_data(self.location, result)

            
            self.device_thread.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    @brief The main control thread for a device, which spawns worker threads.
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

            
            # Block Logic: Spawns one new ComputationThread for each assigned script.
            local_threads = []
            for script_data in self.device.scripts:
                worker = ComputationThread(self, neighbours, script_data)
                worker.start()
                local_threads.append(worker)

            # Wait for all local computation threads to finish.
            for worker in local_threads:
                worker.join()

            
            self.device.timepoint_done.clear()

            
            # Invariant: Wait at the (buggily-initialized) barrier to synchronize with other devices.
            self.device.barrier.wait()