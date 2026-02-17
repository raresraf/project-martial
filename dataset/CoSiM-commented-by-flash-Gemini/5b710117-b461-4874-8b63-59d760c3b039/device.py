


"""



@5b710117-b461-4874-8b63-59d760c3b039/device.py



@brief Implements a simulated device in a distributed system, featuring sequential script execution and a custom reusable barrier.



This module defines a `Device` that processes sensor data and executes scripts sequentially



within its `DeviceThread`. Synchronization across devices is managed by a shared



`ReusableBarrier` implemented with `threading.Condition`. Data access within a device



is protected by a shared `Lock`.



"""







from threading import Thread,Event,Condition,Lock







class ReusableBarrier():



    """



    @brief Implements a reusable barrier for synchronizing a fixed number of threads using a Condition object.



    This barrier ensures that all participating threads wait at a synchronization point



    until every thread has reached it, after which all are released simultaneously.



    """



    def __init__(self, num_threads):



        """



        @brief Initializes the reusable barrier.



        @param num_threads: The total number of threads that will participate in this barrier.



        """



        self.num_threads = num_threads



        self.count_threads = self.num_threads    # Counter for threads waiting at the barrier.



        self.cond = Condition()                  # Condition variable for blocking and releasing threads.



                                                 



 



    def wait(self):



        """



        @brief Blocks the calling thread until all `num_threads` have reached this barrier.



        Invariant: All threads are held until `count_threads` reaches zero, then all are notified and proceed.



        """



        self.cond.acquire()                      # Acquire the condition lock.



        self.count_threads -= 1;



        if self.count_threads == 0:



            self.cond.notify_all()               # Last thread to arrive notifies all waiting threads.



            self.count_threads = self.num_threads    # Reset counter for next reuse.



        else:



            self.cond.wait();                    # Threads wait here until notified by the last thread.



        self.cond.release();                     # Release the condition lock.







class Device(object):



    """



    @brief Represents a single device in the distributed system simulation.



    Manages its local sensor data, assigned scripts, and coordinates its operation



    through a dedicated thread, a shared barrier, and a shared lock for data consistency.



    """



    



    # Class-level attributes to be shared across all instances of Device.



    # Note: These are intended to be set once by a coordinating device (e.g., device_id 0).



    lock = None    # Shared Lock to protect access to sensor data across devices.



    barrier = None # Shared ReusableBarrier for global time step synchronization.







    def __init__(self, device_id, sensor_data, supervisor):



        """



        @brief Initializes a Device instance.



        @param device_id: A unique identifier for this device.



        @param sensor_data: A dictionary containing the device's local sensor readings.



        @param supervisor: The supervisor object responsible for managing the overall simulation.



        """



        self.device_id = device_id



        self.sensor_data = sensor_data



        self.supervisor = supervisor



        self.script_received = Event() # Event to signal that a script has been assigned.



        self.scripts = [] # List to store assigned scripts.



        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.







    def __str__(self):



        """



        @brief Provides a string representation of the device.



        @return A string in the format "Device <device_id>".



        """



        return "Device %d" % self.device_id







    def setup_devices(self, devices):



        """



        @brief Sets up the shared `ReusableBarrier` and `Lock` for synchronization among all devices.



        The device with `device_id == 0` is responsible for initializing the class-level



        `Device.barrier` and `Device.lock`. All other devices implicitly use these shared instances.



        @param devices: A list of all Device instances in the simulation.



        Precondition: This method is called once during system setup.



        """



        # Block Logic: Initializes the shared class-level barrier and lock if this is the first device.



        # This ensures that only one instance of the barrier and lock exists across all devices.



        if self.device_id == 0:



            Device.barrier = ReusableBarrier(len(devices))



            Device.lock = Lock() # Initialize shared lock for data access.







        # Block Logic: Creates and starts the dedicated thread for this device, passing the shared barrier and lock.



        self.thread = DeviceThread(self, Device.barrier, Device.lock)



        self.thread.start()











    def assign_script(self, script, location):



        """



        @brief Assigns a script to the device for execution at a specific data `location`.



        Signals that a script has been received, or that a timepoint is done if no script.



        @param script: The script object to assign, or `None` to signal completion.



        @param location: The data location relevant to the script.



        """



        if script is not None:



            self.scripts.append((script, location))



            self.script_received.set()



        else:



            # Block Logic: Signals completion of the timepoint if no script is assigned.



            self.timepoint_done.set()







    def get_data(self, location):



        """



        @brief Retrieves sensor data for a given location.



        Note: This method does not acquire the shared `Device.lock`, which could lead to race conditions



        if `set_data` is called concurrently by another thread, or if this `get_data` is called



        concurrently with `set_data` by different threads without external synchronization.



        @param location: The key identifying the sensor data.



        @return The data associated with the location, or `None` if the location is not found.



        """



        return self.sensor_data[location] if location in self.sensor_data else None







    def set_data(self, location, data):



        """



        @brief Sets or updates sensor data for a specified location.



        Note: This method does not acquire the shared `Device.lock`. It is assumed that external



        mechanisms (e.g., in `DeviceThread`) will handle the locking for data modification.



        @param location: The key for the sensor data to be modified.



        @param data: The new data value to store.



        Precondition: `location` must be a valid key in `self.sensor_data`.



        """



        if location in self.sensor_data:



            self.sensor_data[location] = data







    def shutdown(self):



        """



        @brief Shuts down the device's operational thread, waiting for its graceful completion.



        """



        self.thread.join()



















class DeviceThread(Thread):



    """



    @brief The dedicated thread of execution for a `Device` instance.



    This thread manages the device's operational cycle, including fetching neighbor data,



    executing scripts sequentially, and coordinating with other device threads using



    a shared `ReusableBarrier` and a shared `Lock` for data access.



    Time Complexity: O(T * S * (N * D_access + D_script_run)) where T is the number of timepoints,



    S is the number of scripts per device, N is the number of neighbors, D_access is data access



    time, and D_script_run is script execution time.



    """



    







    def __init__(self, device , barrier , lock):



        """



        @brief Initializes a `DeviceThread` instance.



        @param device: The `Device` instance that this thread is responsible for.



        @param barrier: The shared `ReusableBarrier` for global synchronization.



        @param lock: The shared `Lock` for protecting access to sensor data.



        """



        Thread.__init__(self, name="Device Thread %d" % device.device_id)



        self.device = device



        self.barrier = barrier



        self.lock = lock



        



    def run(self):



        """



        @brief The main loop for the device's operational thread.



        Block Logic:



        1. Continuously synchronizes with all other device threads using the shared barrier.



           Invariant: All active `DeviceThread` instances must reach this barrier before any can



           proceed, ensuring synchronized advancement of the simulation.



        2. Fetches neighbor information from the supervisor.



           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.



        3. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.



        4. Clears the `timepoint_done` event for the next cycle.



        5. Acquires the shared `self.lock` to protect the entire script execution phase for this device.



        6. Processes each assigned script: it collects data from neighbors and itself,



           runs the script, and then updates data on neighbors and itself.



           Invariant: All data modification operations for both self and neighbors are protected by `self.lock`.



        7. Releases the shared `self.lock`.



        """



        while True:



            # Block Logic: Synchronizes all device threads at the start of each timepoint.



            self.barrier.wait()



            neighbours = self.device.supervisor.get_neighbours()



            if neighbours is None:



                break



            



            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).



            self.device.timepoint_done.wait()



            



            # Block Logic: Clears the `timepoint_done` event for the next timepoint cycle.



            self.device.timepoint_done.clear()







            # Block Logic: Acquires the shared lock to protect data access during script execution for this device.



            self.lock.acquire()



            # Block Logic: Processes each script assigned to the device for the current timepoint.



            # Invariant: Each script retrieves data from neighbors and itself, executes, and updates data.



            for (script, location) in self.device.scripts:



                script_data = []



                



                # Block Logic: Collects data from neighboring devices for the specified location.



                for device in neighbours:



                    data = device.get_data(location)



                    if data is not None:



                        script_data.append(data)



                



                # Block Logic: Collects data from its own device for the specified location.



                data = self.device.get_data(location)



                if data is not None:



                    script_data.append(data)







                # Block Logic: Executes the script if any data was collected and propagates the result.



                if script_data != []:



                    



                    result = script.run(script_data)







                    # Block Logic: Updates neighboring devices with the script's result.



                    for device in neighbours:



                        device.set_data(location, result)



                    



                    # Block Logic: Updates its own device's data with the script's result.



                    self.device.set_data(location, result)



            # Block Logic: Releases the shared lock after all scripts for this timepoint have been processed.



            self.lock.release()


