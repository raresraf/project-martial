

"""


This module defines a framework for simulating a network of devices that


process scripts on sensor data in a distributed and concurrent manner.


It includes custom synchronization primitives and a multi-threaded execution model


for each device.


"""





from threading import Condition, Event, Lock, Thread





class ReusableBarrier(object):


    """A reusable barrier for synchronizing a group of threads.





    This implementation allows a fixed number of threads to wait for each other


    at a synchronization point. It can be reused after all threads have passed.


    """


    


    def __init__(self, num_threads):


        """Initializes the barrier for a given number of threads.





        Args:


            num_threads (int): The number of threads to synchronize.


        """


        self.num_threads = num_threads


        self.count_threads = self.num_threads


        self.cond = Condition()





    def reinit(self):


        """Reduces the number of threads required by the barrier.





        Note: This method seems to have a potential race condition and its logic


        for re-initialization is unusual. It appears to be intended for dynamically


        adjusting the barrier size as threads terminate.


        """


        with self.cond:


            self.num_threads -= 1


        self.wait()





    def wait(self):


        """Makes the calling thread wait until all threads have reached the barrier."""


        with self.cond:


            self.count_threads -= 1


            if self.count_threads == 0:


                # The last thread has arrived, notify all waiting threads and reset for reuse


                self.cond.notify_all()


                self.count_threads = self.num_threads


            else:


                self.cond.wait()








class Device(object):


    """Represents a single device in the simulated network."""





    def __init__(self, device_id, sensor_data, supervisor):


        """Initializes a device instance.





        Args:


            device_id (int): A unique identifier for the device.


            sensor_data (dict): A dictionary representing the device's sensor data.


            supervisor (object): A supervisor object to get simulation-wide information (e.g., neighbors).


        """


        self.device_id = device_id


        self.devices = []


        self.sensor_data = sensor_data


        self.supervisor = supervisor


        self.script_received = Event()


        self.start = Event()


        self.scripts = []


        self.locations_lock = {}





        self.scripts_to_process = []


        self.timepoint_done = Event()


        self.nr_script_threats = 0


        self.thread = DeviceThread(self)


        self.thread.start()


        self.script_threats = []


        self.barrier_devices = None


        self.neighbours = None


        self.cors = 8  # Represents the number of cores for parallel script execution


        self.lock = None


        self.results = {}


        self.results_lock = None





    def __str__(self):


        return "Device %d" % self.device_id





    def setup_devices(self, devices):


        """Sets up shared resources for all devices in the simulation.





        This method is intended to be called by a master device (device_id 0) to


        initialize and distribute shared locks and the synchronization barrier.





        Args:


            devices (list): A list of all device objects in the simulation.


        """


        if self.device_id == 0:


            lock = Lock()


            results_lock = Lock()


            barrier = ReusableBarrier(len(devices))


            for device in devices:


                device.lock = lock;


                device.results_lock = results_lock


                device.barrier_devices = barrier


                device.devices = devices


            for device in devices:


                device.start.set()





        for script in self.scripts:


            self.scripts_to_process.append(script)


            


    def assign_script(self, script, location):


        """Assigns a script to be executed by the device.





        Args:


            script (object): The script to be executed.


            location (str): The location associated with the script's data.


        """


        with self.lock:


            for device in self.devices:


                if location not in device.locations_lock:


                    device.locations_lock[location] = Lock()





        if script is not None:


            self.scripts.append((script, location))


            self.scripts_to_process.append((script, location))


            self.script_received.set()


        else:


            # A None script signals the end of a timepoint


            self.timepoint_done.set()


            self.script_received.set()


            


    def get_data(self, location):


        """Retrieves sensor data for a given location.





        Args:


            location (str): The data location.





        Returns:


            The sensor data, or None if the location is not found.


        """


        return self.sensor_data.get(location)


        


    def set_data(self, location, data):


        """Updates sensor data for a given location.





        Args:


            location (str): The data location.


            data: The new data value.


        """


        if location in self.sensor_data:


            self.sensor_data[location] = data





    def shutdown(self):


        """Shuts down the device by joining its main thread."""


        self.thread.join()








class DeviceThread(Thread):


    """The main control thread for a device."""





    def __init__(self, device):


        """Initializes the DeviceThread.





        Args:


            device (Device): The parent device object.


        """


        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device


        self.device.neighbours = None





    def run(self):


        """The main execution loop for the device thread.





        This loop orchestrates the device's behavior in each simulation step,


        including executing scripts and synchronizing with other devices.


        """


        self.device.start.wait()


        while True:


            # Get neighbors for the current simulation step


            self.device.neighbours = self.device.supervisor.get_neighbours()





            if self.device.neighbours is None:


                # Supervisor signals the end of the simulation


                self.device.barrier_devices.reinit()


                break


            


            # Reset scripts for the current timepoint


            self.device.scripts_to_process = list(self.device.scripts)


            self.device.results = {}





            while True:


                # Wait for scripts or a timepoint-done signal


                if not self.device.timepoint_done.is_set():


                    self.device.script_received.wait()


                    self.device.script_received.clear()





                if not self.device.scripts_to_process:


                    if self.device.timepoint_done.is_set():


                        break





                # Process scripts in parallel using ScriptThread workers


                while self.device.scripts_to_process:


                    list_threats = []


                    self.device.script_threats = []


                    self.device.nr_script_threats = 0


                    


                    # Create a batch of script threads up to the number of cores


                    while self.device.scripts_to_process and self.device.nr_script_threats < self.device.cors:


                        script, location = self.device.scripts_to_process.pop(0)


                        list_threats.append((script, location))


                        self.device.nr_script_threats += 1


                        


                    for script, location in list_threats:


                        # Gather data from neighbors and self


                        script_data = [


                            dev.get_data(location) for dev in self.device.neighbours 


                            if dev.get_data(location) is not None


                        ]


                        data = self.device.get_data(location)


                        if data is not None:


                            script_data.append(data)





                        # Start a new thread to execute the script


                        thread_script_d = ScriptThread(self.device, script, location, script_data)


                        self.device.script_threats.append(thread_script_d)


                        thread_script_d.start()





                    for thread in self.device.script_threats:


                        thread.join()





            # Apply results from script execution


            for location, result in self.device.results.items():


                with self.device.locations_lock[location]:


                    for device in self.device.neighbours:


                        device.set_data(location, result)


                    self.device.set_data(location, result)





            self.device.timepoint_done.wait()


            self.device.timepoint_done.clear()





            # Synchronize with all other devices before the next step


            self.device.barrier_devices.wait()








class ScriptThread(Thread):


    """A thread dedicated to executing a single script."""





    def __init__(self, device, script, location, script_data):


        """Initializes the ScriptThread.





        Args:


            device (Device): The parent device object.


            script (object): The script to execute.


            location (str): The data location for the script.


            script_data (list): The data to be processed by the script.


        """


        Thread.__init__(self, name="Device Script Thread %d" % device.device_id)


        self.device = device


        self.location = location


        self.script = script


        self.script_data = script_data





    def run(self):


        """Executes the script and stores the result."""


        if self.script_data:


            result = self.script.run(self.script_data)


            with self.device.results_lock:


                self.device.results[self.location] = result


        self.device.nr_script_threats -= 1

