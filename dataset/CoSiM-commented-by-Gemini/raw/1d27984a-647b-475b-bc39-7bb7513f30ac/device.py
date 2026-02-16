"""
This module implements a simulation of a distributed device network using Python's
threading capabilities.

It defines `Device` objects that run concurrently, managed by `DeviceThread`s.
The simulation attempts to synchronize using a custom `ReusableBarrier` and uses
a complex, two-level locking strategy to manage access to shared data.

NOTE: This implementation contains several significant concurrency issues, including
a non-reusable barrier and a high potential for deadlock in the Worker threads.
The comments describe the intended logic, not a correct implementation.
"""

from threading import Event, Thread, Condition, Lock


class Device(object):
    """
    Represents a single device in the simulation. Each device has sensor data,
    executes scripts, and communicates with its neighbors.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): A dictionary of the device's local sensor values.
            supervisor (Supervisor): The central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Event to signal that all scripts for a time step have been assigned.
        self.scripts_done = Event()
        # A lock intended to protect this specific device's data.
        self.my_lock = Lock()

        # Shared resources to be populated by the setup_devices method.
        self.locations = None
        self.barrier = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources like locks and the barrier.

        This method uses a master-slave pattern where device 0 creates the
        shared resources, and other devices get a reference to them. This setup
        itself is not thread-safe.
        """
        # Master device (id 0) creates the shared resources.
        if self.device_id is 0:
            # A dictionary mapping a location to a shared Lock for that location.
            self.locations = {}
            self.barrier = ReusableBarrier(len(devices));
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass
                else:
                    self.locations[loc] = Lock()
        
        # Slave devices get references to the shared resources from device 0.
        else:
            self.locations = devices[0].locations
            self.barrier = devices[0].get_barrier()
            # Each device adds its own location locks to the shared dictionary,
            # which is not a thread-safe operation.
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass
                else:
                    self.locations[loc] = Lock()

        # The main thread for this device is started after setup.
        self.thread = DeviceThread(self, self.barrier, self.locations)
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be run, or signals the end of assignments."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the device's main thread to shut down gracefully."""
        self.thread.join()

    def get_barrier(self):
        """Returns the shared barrier instance."""
        return self.barrier

class DeviceThread(Thread):
    """
    The main, long-running thread for a Device, orchestrating its lifecycle.
    """

    def __init__(self, device, barrier, locations):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.locations = locations

    def run(self):
        """
        Main loop: waits for scripts, executes them via Worker threads,
        and synchronizes at the barrier.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for supervisor to signal that all scripts are assigned.
            self.device.scripts_done.wait()
            self.device.scripts_done.clear()

            # Create and start a worker for each assigned script.
            workers = []
            for (script, location) in self.device.scripts:
                w = Worker(self.device, neighbours, script, location, self.locations)
                workers.append(w)
                w.start()

            # Wait for all workers to complete their tasks for this time step.
            for w in workers:
                w.join()

            # Synchronize with all other devices before starting the next time step.
            self.barrier.wait()

class Worker(Thread):
    """
    A short-lived thread to execute one script, with a flawed locking protocol.
    """
    def __init__(self, device, neighbours, script, location, locations):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.locations = locations

    def run(self):
        """
        Executes the script logic. This method's locking strategy is prone to deadlock.
        """
        # --- Start of First-Level Lock ---
        # Acquires a lock specific to the data's location. This is intended
        # to prevent two workers from modifying the same location simultaneously.
        self.locations[self.location].acquire()
        script_data = []

        # --- Start of Second-Level Locking ---
        # Inside the location lock, it acquires a second, device-specific lock
        # for each device it interacts with. This nested locking of different
        # resource types (location and device) in an undefined order is a
        # classic recipe for deadlock.
        for device in self.neighbours:
            device.my_lock.acquire()
            data = device.get_data(self.location)
            device.my_lock.release()
            if data is not None:
                script_data.append(data)
        
        self.device.my_lock.acquire()
        data = self.device.get_data(self.location)
        self.device.my_lock.release()
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = self.script.run(script_data)

            # Disseminate the result, again using the flawed nested locking pattern.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()
            
            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()
            
        # --- End of Second-Level Locking ---
        
        # Release the first-level location lock.
        self.locations[self.location].release()


class ReusableBarrier():
    """
    A non-reusable barrier implementation. This implementation is flawed and
    can lead to deadlocks because it does not use a two-phase commit to
    prevent fast threads from re-entering the barrier before slow threads have left.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Blocks until all `num_threads` have called this method.
        """
        self.cond.acquire()
        self.count_threads -= 1;
        if self.count_threads == 0:
            # Last thread notifies all waiting threads.
            self.cond.notify_all()
            # Counter is reset, but threads may not have all exited the wait() call,
            # creating a race condition for the next barrier use.
            self.count_threads = self.num_threads
        else:
            self.cond.wait(); # Wait to be notified by the last thread.
        self.cond.release();
