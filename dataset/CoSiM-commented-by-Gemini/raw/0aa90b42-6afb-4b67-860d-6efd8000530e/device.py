


"""



This module defines a distributed device simulation framework.







It includes a `Device` class representing a network node, a `Barrier` class for thread



synchronization, and a `DeviceThread` for concurrent execution. The framework is



designed for simulating data processing and script execution across a network of devices



that can communicate with their neighbors.



"""







from threading import Event, Thread, Condition











class Barrier():



    """



    A reusable barrier for synchronizing a fixed number of threads.







    This class provides a synchronization point where threads can wait until all



    participating threads have reached the barrier. The number of threads to wait for



    is configurable.



    """



    



    num_threads = 0



    count_threads = 0







    def __init__(self):



        """Initializes the barrier with a condition variable and an event."""



        



        self.cond = Condition()



        self.thread_event = Event()







    def wait(self):



        """



        Causes the calling thread to wait until all threads have reached the barrier.







        When a thread calls wait(), it decrements a counter. If the counter reaches



        zero, all waiting threads are notified. Otherwise, the thread blocks until



        it is notified.



        """



        



        self.cond.acquire()



        Barrier.count_threads -= 1







        if Barrier.count_threads == 0:



            self.cond.notify_all()



            Barrier.count_threads = Barrier.num_threads



        else:



            self.cond.wait()







        self.cond.release()







    @staticmethod



    def add_thread():



        """



        Increments the number of threads that the barrier will wait for.







        This should be called for each new thread that will participate in the barrier



        synchronization.



        """



        



        Barrier.num_threads += 1



        Barrier.count_threads = Barrier.num_threads











class Device(object):



    """



    Represents a device in a distributed network that can process sensor data.







    Each device runs in its own thread and can execute scripts on data collected from



    itself and its neighbors. It synchronizes with other devices using a shared barrier.



    """



    



    barrier = Barrier()







    def __init__(self, device_id, sensor_data, supervisor):



        """



        Initializes a Device instance.







        Args:



            device_id (int): A unique identifier for the device.



            sensor_data (dict): A dictionary of sensor data, keyed by location.



            supervisor (Supervisor): A supervisor object that manages the network.



        """



        



        Device.barrier.add_thread()



        self.device_id = device_id



        self.sensor_data = sensor_data



        self.supervisor = supervisor



        self.script_received = Event()



        self.scripts = []



        self.thread = DeviceThread(self)



        self.thread.start()







    def __str__(self):



        



        return "Device %d" % self.device_id







    def setup_devices(self, devices):



        """



        Placeholder for device setup logic.







        Args:



            devices (list): A list of other devices in the network.



        """



        



        



        pass







    def assign_script(self, script, location):



        """



        Assigns a script to be executed by the device.







        Args:



            script (Script): The script object to execute.



            location (str): The location associated with the script's data.



        """



        



        if script is not None:



            self.scripts.append((script, location))



        else:



            self.script_received.set()







    def get_data(self, location):



        """



        Retrieves sensor data for a given location.







        Args:



            location (str): The location to retrieve data for.







        Returns:



            The sensor data, or None if the location is not found.



        """



        



        return self.sensor_data[location] if location in self.sensor_data else None







    def set_data(self, location, data):



        """



        Updates sensor data for a given location.







        Args:



            location (str): The location to update data for.



            data: The new data value.



        """



        



        if location in self.sensor_data:



            self.sensor_data[location] = data







    def shutdown(self):



        """Shuts down the device's thread."""



        



        self.thread.join()











class DeviceThread(Thread):



    """



    The execution thread for a Device instance.







    This thread contains the main loop where the device synchronizes with other



    devices, waits for scripts, and executes them.



    """



    







    def __init__(self, device):



        """



        Initializes the device thread.







        Args:



            device (Device): The device that this thread will run.



        """



        



        Thread.__init__(self, name="Device Thread %d" % device.device_id)



        self.device = device















    def run(self):



        """



        The main execution loop for the device.







        The loop consists of the following steps:



        1. Wait for all devices to reach the synchronization barrier.



        2. Wait for a script to be assigned.



        3. Execute the script on data from this device and its neighbors.



        4. Update the data on this device and its neighbors with the script's result.



        """







        while True:



            







            neighbours = self.device.supervisor.get_neighbours()



            if neighbours is None:



                break



            



            Device.barrier.wait()



            self.device.script_received.wait()



            self.device.script_received.clear()







            



            for (script, location) in self.device.scripts:



                script_data = []



                



                for device in neighbours:



                    data = device.get_data(location)



                    if data is not None:



                        script_data.append(data)



                



                data = self.device.get_data(location)



                if data is not None:



                    script_data.append(data)







                if script_data != []:



                    



                    result = script.run(script_data)







                    for device in neighbours:



                        device.set_data(location, result)



                    self.device.set_data(location, result)


