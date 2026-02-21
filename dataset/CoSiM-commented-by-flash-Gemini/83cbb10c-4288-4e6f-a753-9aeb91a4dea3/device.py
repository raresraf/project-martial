


from threading import Event, Thread



import ReusableBarrier







class Device(object):



    """



    Represents a device in a simulated distributed system.



    This version utilizes a globally shared, static reusable barrier for synchronization



    and directly executes scripts without fine-grained locking.



    """



    # A single, static instance of ReusableBarrier shared across all Device objects.



    # Its behavior (e.g., phase-based or condition variable based) is assumed to be



    # implemented in the imported ReusableBarrier module.



    reusable_barrier = ReusableBarrier.ReusableBarrier()







    def __init__(self, device_id, sensor_data, supervisor):



        """



        Initializes a new Device instance.







        Args:



            device_id (int): A unique identifier for the device.



            sensor_data (dict): A dictionary holding the device's sensor data,



                                keyed by location.



            supervisor (Supervisor): The supervisor object responsible for



                                     managing devices and providing neighborhood information.



        """



        # Register this device with the global reusable barrier.



        Device.reusable_barrier.add_thread()



        self.device_id = device_id



        self.sensor_data = sensor_data



        self.supervisor = supervisor



        # Event to signal when new scripts are ready for processing.



        self.script_received = Event()



        # List to store assigned scripts.



        self.scripts = []



        # Event for timepoint completion (its usage seems implicit or minimal in this version).



        self.timepoint_done = Event()



        # The dedicated thread for this device's execution logic.



        self.thread = DeviceThread(self)



        self.thread.start()







    def __str__(self):



        """



        Returns a string representation of the Device.



        """



        return "Device %d" % self.device_id







    def setup_devices(self, devices):



        """



        This method is a placeholder in this implementation, as the global barrier



        is initialized statically and threads are added during device instantiation.



        No additional setup logic is performed here.



        """



        pass







    def assign_script(self, script, location):



        """



        Assigns a script to be executed by the device.



        If `script` is None, it signals that scripts are ready to be processed



        for the current timepoint.







        Args:



            script (Script): The script object to be executed. If None, it signals



                             the readiness of previously assigned scripts.



            location: The data location pertinent to the script.



        """



        if script is not None:



            self.scripts.append((script, location))



        else:



            # Signal that script assignments are complete and scripts can now be processed.



            self.script_received.set()







    def get_data(self, location):



        """



        Retrieves sensor data for a given location.







        Args:



            location: The key for the sensor data.







        Returns:



            The sensor data for the specified location, or None if not found.



        """



        return self.sensor_data[location] if location in self.sensor_data else None







    def set_data(self, location, data):



        """



        Sets or updates sensor data for a given location.







        Args:



            location: The key for the sensor data.



            data: The new data to set.



        """



        if location in self.sensor_data:



            self.sensor_data[location] = data







    def shutdown(self):



        """



        Shuts down the device by joining its execution thread.



        """



        self.thread.join()











class DeviceThread(Thread):



    """



    The dedicated thread of execution for a Device object.



    It manages the device's operational logic within the simulation loop,



    including supervisor interaction, script execution, and synchronization



    with other devices via a global static barrier.



    """



    def __init__(self, device):



        """



        Initializes the DeviceThread.







        Args:



            device (Device): The Device object associated with this thread.



        """



        Thread.__init__(self, name="Device Thread %d" % device.device_id)



        self.device = device







    def run(self):



        """



        The main execution loop for the device thread.



        It continuously processes timepoints, synchronizes with other devices



        using a global barrier, executes scripts sequentially, and updates data.



        """



        while True:



            # Retrieve current neighbor information from the supervisor.



            neighbours = self.device.supervisor.get_neighbours()



            # If no neighbors are returned, it signifies the end of the simulation.



            if neighbours is None:



                break







            # Wait on the global reusable barrier, ensuring all devices are synchronized



            # before starting script execution for the current timepoint.



            Device.reusable_barrier.wait();



            # Wait for the script_received event, signaling that scripts are assigned



            # and ready for processing for this timepoint.



            self.device.script_received.wait()



            # Clear the event for the next cycle.



            self.device.script_received.clear();







            # Iterate through and execute all assigned scripts for the current timepoint.



            # WARNING: This implementation lacks explicit locking mechanisms (e.g., mutexes)



            # for protecting shared data (sensor_data on local and neighboring devices)



            # during read/write operations. This can lead to race conditions and



            # data inconsistency in a concurrent environment if multiple DeviceThreads



            # attempt to modify the same data locations simultaneously.



            for (script, location) in self.device.scripts:



                script_data = []



                



                # Collect data from neighboring devices for the current location.



                for device in neighbours:



                    data = device.get_data(location)



                    if data is not None:



                        script_data.append(data)



                



                # Collect data from the local device for the current location.



                data = self.device.get_data(location)



                if data is not None:



                    script_data.append(data)







                # Execute the script if there is data to process.



                if script_data != []:



                    # Execute the script with the aggregated data.



                    result = script.run(script_data)







                    # Propagate the script's result back to all neighboring devices.



                    for device in neighbours:



                        device.set_data(location, result)



                    



                    # Update the local device's data with the script's result.



                                        self.device.set_data(location, result)
            self.device.scripts = []

