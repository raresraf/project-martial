from threading import *

"""
This module implements a complex distributed device simulation framework.

The architecture is notable for its multi-level barrier synchronization and a
unique threading model where each device manages its own internal pool of
threads that actively pull tasks from a shared list.

Key components:
- ReusableBarrier: A custom, two-phase reusable barrier for synchronization.
- Device: Represents a node in the network, managing a set of DeviceThreads.
- DeviceThread: Worker threads within a device. Thread 0 has special duties
  for inter-device coordination.

WARNING: The locking mechanism for data access (`get_data`/`set_data`) appears
to be implemented incorrectly and may lead to deadlocks.
"""

class ReusableBarrier(object):
    """
    A custom, two-phase reusable barrier implemented with Semaphores and a Lock.

    This barrier ensures that a group of threads can synchronize at a point,
    and it can be reused multiple times. The two-phase design prevents race
    conditions where faster threads from a subsequent `wait` call could get
    ahead of slower threads from the previous call.
    """

    def __init__(self, numOfTh):
        """Initializes the barrier for a given number of threads."""
        self.numOfTh = numOfTh
        # State for the two phases, each with a counter and a semaphore.
        self.threads = [{}, {}]
        self.threads[0]['count'] = numOfTh
        self.threads[1]['count'] = numOfTh
        self.threads[0]['sem'] = Semaphore(0)
        self.threads[1]['sem'] = Semaphore(0)
        self.lock = Lock()

    def wait(self):
        """
        Causes the calling thread to wait until all threads have reached the
        barrier. It consists of two internal synchronization phases.
        """
        for i in range(0, 2):
            with self.lock:
                self.threads[i]['count'] -= 1
                if self.threads[i]['count'] == 0:
                    # The last thread to arrive releases all waiting threads.
                    for _ in range(self.numOfTh):
                        self.threads[i]['sem'].release()
                    # Reset the counter for the next use of this phase.
                    self.threads[i]['count'] = self.numOfTh
            # All threads block on the semaphore until released.
            self.threads[i]['sem'].acquire()


class Device(object):
    """
    Represents a device in the simulation, managing its own data and threads.

    Each device contains a pool of `DeviceThread` instances that perform the
    actual work. It handles script assignments from a supervisor and coordinates
    synchronization with other devices.

    Attributes:
        device_id (int): The unique ID for the device.
        sensor_data (dict): A dictionary of sensor readings for the device.
        supervisor (object): The central simulation supervisor.
        scripts (list): A shared list of scripts for the device's threads to execute.
        timepoint_done (Event): An event to signal progression in the simulation time step.
        threads (list): The internal pool of DeviceThread workers.
        tBariera (ReusableBarrier): The main barrier for synchronizing all devices.
        iBariera (ReusableBarrier): An internal barrier for the device's own threads.
        locks (list): A list of locks for accessing sensor data by location index.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device and its internal worker threads."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.scripts = []
        self.timepoint_done = Event()

        self.threads = []
        self.no_threads = 8
        self.tBariera = None
        self.locks = []
        self.sLock = Lock()  # Lock for accessing the shared scripts list.
        self.iBariera = ReusableBarrier(8)  # Internal barrier for worker threads.

        self.etLock = Lock()
        self.lastScripts = []

        if device_id == 0:
            # Device 0 is responsible for creating and distributing shared resources.
            self.init_event = Event()

        for tid in range(self.no_threads):
            thread = DeviceThread(self, tid)
            self.threads.append(thread)


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared synchronization objects for all devices.

        If this is device 0, it creates the main barrier and data locks.
        If not, it waits for device 0 to finish initialization before
        copying the references to the shared objects.
        """
        if self.device_id != 0:
            # Non-master devices wait for the master (device 0) to set up.
            i = 0
            while (i < len(devices) and devices[i].device_id != 0):
                i += 1
            if i < len(devices):
                devices[i].init_event.wait()
                self.tBariera = devices[i].tBariera
                self.locks = devices[i].locks
        else:
            # Device 0 initializes the main barrier and data locks.
            aux = 0
            self.tBariera = ReusableBarrier(len(devices))
            # The size of the locks list is the sum of sensor locations of all devices.
            for d in devices:
                aux += len(d.sensor_data)
            self.locks = [RLock() for _ in range(aux)]
            # Signal that initialization is complete.
            self.init_event.set()

        # Start all worker threads after setup is complete.
        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        """Assigns a script to the device or signals timepoint completion."""
        if script is None:
            self.etLock.acquire()
            self.timepoint_done.set()
        else:
            with self.sLock:
                self.scripts.append((script, location))

    def get_data(self, location):
        """
        Retrieves data for a given location, acquiring a lock.

        WARNING: This method acquires a lock but never releases it, which will
        cause a deadlock if another thread tries to acquire the same lock.
        The `set_data` method is expected to release it, which is an unsafe
        and error-prone pattern.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets data for a given location and releases a lock.

        WARNING: This method releases a lock that it did not acquire. It relies
        on a preceding `get_data` call to have acquired the lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """Waits for all worker threads to join."""
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """

    A worker thread within a device.

    Thread 0 has special responsibilities, including communication with the
    supervisor and participating in the top-level barrier. All threads share
    the work of processing scripts from the device's script list.
    """

    def __init__(self, device, thread_id):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        Main execution loop for the thread.

        The logic here is complex and appears to contain duplicated code blocks.
        Threads steal tasks from `device.scripts` and execute them, synchronizing
        internally with `iBariera` and externally via thread 0 with `tBariera`.
        """
        if self.thread_id == 0:
            # Thread 0 is responsible for fetching the list of neighbors.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device in self.device.neighbours:
                self.device.neighbours.remove(self.device)

        while True:
            if self.thread_id == 0:
                # Thread 0 merges leftover scripts from the previous iteration.
                with self.device.sLock:
                    self.device.scripts += self.device.lastScripts
                    self.device.lastScripts = []
            # All threads within the device synchronize here.
            self.device.iBariera.wait()
            neighbours = self.device.neighbours
            # A None neighbor list signals termination.
            if neighbours is None:
                break

            # --- Start of script processing loop (appears duplicated) ---
            while len(self.device.scripts) != 0:
                script = None
                with self.device.sLock:
                    if len(self.device.scripts) != 0:
                        script, location = self.device.scripts.pop(0)
                        self.device.lastScripts.append((script, location))

                if script:
                    # Data gathering and script execution logic.
                    script_data = []
                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None: script_data.append(data)
                    data = self.device.get_data(location)
                    if data is not None: script_data.append(data)

                    if script_data:
                        result = script.run(script_data)
                        for device in neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)
            # --- End of first script processing block ---

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()

            # --- Start of second script processing loop (identical to the first) ---
            # This block appears to be a copy-paste of the one above.
            while len(self.device.scripts) != 0:
                script = None
                with self.device.sLock:
                    if len(self.device.scripts) != 0:
                        script, location = self.device.scripts.pop(0)
                        self.device.lastScripts.append((script, location))

                if script:
                    script_data = []
                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None: script_data.append(data)
                    data = self.device.get_data(location)
                    if data is not None: script_data.append(data)

                    if script_data:
                        result = script.run(script_data)
                        for device in neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)
            # --- End of second script processing block ---

            # Internal synchronization before the main barrier.
            self.device.iBariera.wait()

            if self.thread_id == 0:
                # Thread 0 participates in the main, inter-device barrier.
                self.device.tBariera.wait()

                # Thread 0 fetches neighbors for the next round.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                if self.device.neighbours and self.device in self.device.neighbours:
                    self.device.neighbours.remove(self.device)

                # Reset for the next time step.
                self.device.timepoint_done.clear()
                self.device.etLock.release()
