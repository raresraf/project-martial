"""
This module simulates a network of devices that process sensor data concurrently.

It defines three main classes:
- Device: Represents a single device in the network, managing its own data and scripts.
- DeviceThread: The main thread for a device, which orchestrates worker threads.
- WorkerThread: A thread that executes scripts using data from its device and neighbours.

The simulation uses threading for concurrency, queues for task management, and a
reusable barrier for synchronization between devices in each simulation step.
"""
import threading
from threading import Thread
from Queue import Queue
from cond_barrier import ReusableBarrier


class Device(object):
    """Represents a device in the simulated network.

    Each device has an ID, its own sensor data, and a reference to a supervisor
    that manages the network topology. It uses a queue to receive scripts for
    execution and a pool of worker threads to process them.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor data.
            supervisor (Supervisor): The supervisor object for the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        # A dictionary of locks to protect access to sensor data for each location.
        self.location_locks = {loc: threading.Lock() for loc in self.sensor_data}
        self.supervisor = supervisor
        self.scripts = []

        # Each device runs in its own thread.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Queue for incoming scripts to be processed.
        self.scripts_queue = Queue()
        
        # This queue seems to be unused in the Device class itself.
        self.workers_queue = Queue()

        # A barrier for synchronizing with other devices.
        self.barrier = None


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the synchronization barrier for all devices.

        This method should be called on one device (e.g., device with ID 0) to
        initialize and distribute a shared barrier to all devices in the network.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = barrier

    def assign_script(self, script, location):
        """Assigns a script to the device for processing.

        Args:
            script (Script): The script object to be executed.
            location (str): The location associated with the script.
        """
        self.scripts_queue.put((script, location))

    def get_data(self, location):
        """Retrieves sensor data for a given location without locking.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            The sensor data, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def get_data_synchronize(self, location):
        """Retrieves sensor data for a given location with locking.

        This method acquires a lock for the specified location before reading
        the data, ensuring thread-safe access.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            The sensor data, or None if the location is not found.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Sets the sensor data for a given location without locking.

        Args:
            location (str): The location to update.
            data: The new sensor data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def set_data_synchronize(self, location, data):
        """Sets the sensor data for a given location with locking.

        This method updates the data and then releases the lock for the
        specified location. It assumes the lock was acquired by a call to
        get_data_synchronize.

        Args:
            location (str): The location to update.
            data: The new sensor data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()

    def shutdown(self):
        """Shuts down the device's thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main execution thread for a device.

    This thread manages a pool of worker threads that execute scripts. It handles
    the main simulation loop for the device, including fetching neighbors,
    distributing scripts to workers, and synchronizing with other devices.
    """

    def __init__(self, device):
        """Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.stop = False

    def run(self):
        """The main loop of the device thread."""
        num_workers = 16
        
        workers = []
        
        workers_queue = Queue()

        # Create and start the worker threads.
        for i in range(num_workers):
            workers.append(WorkerThread(self.device, i, workers_queue))
        for worker in workers:
            worker.start()

        while True:
            # Get the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbours signals the end of the simulation.
                break

            # Update the neighbors list for each worker.
            neighbours = [x for x in neighbours if x != self.device]
            for worker in workers:
                worker.neighbours = neighbours

            # Re-queue existing scripts for the workers to process in this step.
            for script in self.device.scripts:
                workers_queue.put(script)

            # Process new scripts from the device's script queue.
            while True:
                script, location = self.device.scripts_queue.get()
                if script is None:
                    # A None script is a signal to end this processing phase.
                    break
                
                self.device.scripts.append((script, location))
                workers_queue.put((script, location))

            # Wait for all workers to finish processing the scripts for this step.
            workers_queue.join()
            
            # Synchronize with all other devices before starting the next step.
            self.device.barrier.wait()

        # Shutdown sequence: signal all workers to exit.
        for worker in workers:
            workers_queue.put((None, None))
        for worker in workers:
            worker.join()


class WorkerThread(Thread):
    """A thread that executes scripts for a device.

    Worker threads pull scripts from a shared queue, gather data from their
    own device and its neighbors, execute the script, and then propagate the
    results back to the devices.
    """

    def __init__(self, device, worker_id, queue):
        """Initializes a WorkerThread.

        Args:
            device (Device): The device this worker belongs to.
            worker_id (int): A unique identifier for the worker.
            queue (Queue): The queue from which to get scripts.
        """
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.device = device
        self.neighbours = []
        self.worker_id = worker_id
        self.queue = queue

    def run(self):
        """The main loop of the worker thread."""
        while True:
            script, location = self.queue.get()
            if script is None:
                # A None script is the signal to terminate.
                self.queue.task_done()
                break

            # Collect data from neighboring devices and the local device.
            script_data = []
            
            for device in self.neighbours:
                data = device.get_data_synchronize(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data_synchronize(location)
            if data is not None:
                script_data.append(data)

            # Execute the script if any data was collected.
            if script_data != []:
                
                result = script.run(script_data)

                # Propagate the script's result back to all involved devices.
                # This is a form of data consensus or state update.
                for device in self.neighbours:
                    device.set_data_synchronize(location, result)
                
                self.device.set_data_synchronize(location, result)
            self.queue.task_done()
