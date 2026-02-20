"""
A simulation of a distributed system of devices using a thread pool pattern.
This module defines the behavior of concurrent devices that process scripts
and synchronize their states at discrete timepoints. It differs from other
versions by using a queue and a fixed-size thread pool for task execution,
which is a more scalable approach.
"""

from threading import Lock, Thread, Event
from Queue import Queue
from barrier import ReusableBarrier


class Device(object):
    """
    Represents a single device in the simulated network. Each device has its own
    main control thread and a pool of worker threads to execute tasks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's local sensor data.
            supervisor (obj): The central supervisor object for the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = list()
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.lock_dict = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for a group of devices.
        This method relies on the device with ID 0 to act as the master
        and create the shared barrier and lock dictionary for all devices.
        """
        # Block Logic: This setup is performed only by the device with ID 0.
        if self.device_id is 0:
            barrier = ReusableBarrier(len(devices))
            lock_dict = dict()

            for device in devices:
                device.barrier = barrier
                device.lock_dict = lock_dict

    def assign_script(self, script, location):
        """
        Assigns a script to the device. If script is None, it signals the
        end of a timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, managing a pool of worker threads
    that execute scripts from a shared queue.
    """

    def __init__(self, device):
        """
        Initializes the control thread and its associated worker thread pool.
        
        Args:
            device (Device): The parent device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue()
        self.thread_pool = list()

        self.thread_num = 8
        
        # Create and start the worker threads for the pool.
        for _ in range(0, self.thread_num):
            my_thread = Thread(target=self.executor_service)
            my_thread.start()
            self.thread_pool.append(my_thread)

    def run(self):
        """
        The main loop for the device. It coordinates script execution for each
        timepoint and handles synchronization.
        """
        while "Not finished": # This is an infinite loop, broken by supervisor signal.
            neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: If supervisor returns None, the simulation is over.
            if neighbours is None:
                # Signal all worker threads to exit.
                for _ in range(0, self.thread_num):
                    self.queue.put(None)
                self.shutdown()
                self.thread_pool = list()
                break

            # Wait for the supervisor to signal that all scripts for the timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Enqueue all scripts for the current timepoint.
            for (script, location) in self.device.scripts:
                queue_info = [script, location, neighbours]
                self.queue.put(queue_info)

            # Wait for all tasks in the queue to be completed by the worker threads.
            self.queue.join()

            # Synchronize with all other devices at the barrier.
            self.device.barrier.wait()

            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()

    def executor_service(self):
        """
        The target function for the worker threads in the pool.
        It continuously fetches tasks from the queue and executes them.
        """
        while "Not finished": # Infinite loop, broken by a None task.
            tasks = self.queue.get()

            if tasks is None:
                # Shutdown signal received.
                self.queue.task_done()
                break
            else:
                script_t = tasks[0]
                location_t = tasks[1]
                neighbours_t = tasks[2]

            # Block Logic: On-demand lock creation and acquisition.
            # If a lock for the given location doesn't exist, it's created.
            # This ensures that operations on the same location are serialized.
            if self.device.lock_dict.get(location_t) is None:
                self.device.lock_dict[location_t] = Lock()

            self.device.lock_dict[location_t].acquire()

            # Perform the data collection, computation, and update.
            self.data_processing(self.device, script_t, location_t, neighbours_t)

            self.device.lock_dict[location_t].release()
            self.queue.task_done()

    @classmethod
    def data_processing(cls, device, script, location, neighbours):
        """
        A class method that encapsulates the logic for processing a single script.
        """
        # Collect data from neighbors and the local device.
        script_info = list()
        for i in range(0, len(neighbours)):
            data = neighbours[i].get_data(location)
            if data:
                script_info.append(data)

        data = device.get_data(location)
        if data != None:
            script_info.append(data)

        # If data was found, run the script and update the network.
        if script_info:
            result = script.run(script_info)
            send_info = [location, result]

            # Update neighbors.
            for i in range(0, len(neighbours)):
                neighbours[i].set_data(send_info[0], send_info[1])

            # Update local device.
            device.set_data(send_info[0], send_info[1])

    def shutdown(self):
        """Waits for all worker threads in the pool to terminate."""
        for i in range(0, self.thread_num):
            self.thread_pool[i].join()
