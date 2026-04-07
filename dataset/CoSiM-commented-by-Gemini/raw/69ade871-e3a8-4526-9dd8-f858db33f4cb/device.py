"""
This module defines a multi-threaded Device class for use in a distributed
simulation or sensor network framework.

Architectural Intent:
The `Device` class represents a single node in a network of devices. Each device
runs a pool of worker threads (`DeviceThread`) to process computational
`scripts` on its own and its neighbors' data. The system is designed for
timestep-based simulations, where all devices synchronize at the end of each
step using a shared barrier.

Concurrency Model:
- A `Queue` (`script_queue`) is used to dispatch scripts to worker threads.
- `Lock` primitives are used to protect shared state, such as sensor data,
  thread counters, and script lists.
- A custom two-phase `Barrier` class ensures that all threads of a device, and
  all devices in the simulation, synchronize correctly between timesteps.
- An asynchronous data-access mechanism (`get_data_with_listener`) allows
  threads to request data and be notified via a callback-like system if the
  data is temporarily locked, preventing deadlocks and busy-waiting.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

class Device(object):
    """
    Represents a single computational device in a simulated network.

    Manages state, concurrency, and communication with neighboring devices. Each
    device operates a pool of threads to execute scripts on sensor data.
    """

    # Statically defines the number of worker threads per device instance.
    _CORE_COUNT = 8

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding the initial sensor values.
            supervisor (object): A supervisor object that provides network topology
                                 (e.g., neighbors).
        """
        self.device_id = device_id

        self.sensor_data = sensor_data
        self.busy_data = {}

        # Initialize tracking for asynchronous data listeners.
        for key in sensor_data:
            self.busy_data[key] = {"busy": False, "queue": []}

        self.device_barrier = None
        self.supervisor = supervisor
        self.scripts = []
        self.listeners = 0
        self.script_queue = Queue()
        self.processing_finished = Event()
        self.devices_done = Event()
        self.thread_barrier = Barrier(Device._CORE_COUNT)
        self.working_threads = 0
        self.received_stop = False


        self.neighbours = []
        self.thread_lock = Lock()
        self.data_lock = Lock()
        self.script_lock = Lock()

        
        self.threads = []

        # Block Logic: Spawn and start the pool of worker threads.
        for i in xrange(Device._CORE_COUNT):


            self.threads.append(DeviceThread(self))
            self.threads[i].start()



    def __str__(self):
        """Returns the string representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Establishes a shared barrier among a list of devices.

        Functional Utility: This method implements a simple leader election
        protocol (lowest device_id becomes leader) to create a single shared
        barrier instance for all devices in the simulation, ensuring they can
        synchronize as a group.
        """

        
        device = self

        # Elect the device with the lowest ID as the barrier master.
        for dev in devices:
            if dev.device_id < device.device_id:
                device = dev

        
        # The leader creates and distributes the shared barrier.
        if device == self:
            self.device_barrier = Barrier(len(devices))

            for dev in devices:
                dev.device_barrier = self.device_barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be processed by the device's worker threads.

        A `None` script is treated as a stop signal for the dispatch queue.
        """

        self.script_lock.acquire()


        if script is not None:
            self.scripts.append((script, location))
            self.script_queue.put((script, location, False))
        else:
            # A None script indicates that no more new scripts will be added.
            self.received_stop = True

            
            
            self.script_queue.put(None)

        self.script_lock.release()

    def increase_listeners(self):
        """
        Increments the count of listeners waiting for data.

        This is part of the asynchronous data access pattern. A thread that
        fails to acquire data becomes a "listener".
        """

        self.script_lock.acquire()

        self.listeners = self.listeners + 1

        self.script_lock.release()

    def decrease_listeners(self):
        """Decrements the count of listeners waiting for data."""
        self.script_lock.acquire()

        self.listeners = self.listeners - 1

        self.script_lock.release()

    def should_stop_thread(self):
        """
        Determines if worker threads should terminate for the current timestep.

        Functional Intent: A thread should stop if all scripts have been
        dispatched, the script queue is empty, and no threads are waiting
        asynchronously for data. This is the condition for ending a timestep loop.
        """

        self.script_lock.acquire()

        val = self.script_queue.empty() and \
              self.listeners == 0 and \
              (self.received_stop is True)

        if val is True:
            # If stopping, put a sentinel value back on the queue to unblock
            # any other threads that may be waiting on `get()`.
            
            self.script_queue.put(None)


        self.script_lock.release()

        return val

    def thread_start_timestep(self):
        """
        Performs per-timestep initialization for a worker thread.

        The first thread to enter this method for a given timestep is responsible
        for fetching the list of neighboring devices from the supervisor.
        """

        self.thread_lock.acquire()

        
        
        # Block Logic: The first thread entering the timestep performs initialization.
        if self.working_threads == 0:
            self.neighbours = self.supervisor.get_neighbours()

            
            if self.neighbours is not None:
                if self in self.neighbours:
                    self.neighbours.remove(self)

        self.working_threads = self.working_threads + 1

        self.thread_lock.release()

    def decr_working_threads(self):
        """Atomically decrements the count of active worker threads."""

        self.thread_lock.acquire()

        self.working_threads = self.working_threads - 1
        val = self.working_threads

        self.thread_lock.release()

        return val

    def get_working_threads(self):
        """Atomically retrieves the count of active worker threads."""

        self.thread_lock.acquire()

        val = self.working_threads

        self.thread_lock.release()

        return val

    def finish(self):
        """
        Signals the end of a device's work for the current timestep.

        This function blocks on a global barrier, ensuring all devices in the
        simulation have completed their work before proceeding to the next step.
        """
        self.device_barrier.wait()

        self.start_timestep()



    
    def start_timestep(self):
        """
        Resets the device's state for the beginning of a new timestep.

        This clears the script queue and re-populates it with all assigned
        scripts, allowing the simulation to run another cycle.
        """

        self.thread_lock.acquire()

        self.script_lock.acquire()

        
        # Clear any residual items from the previous timestep's queue.
        while self.script_queue.empty() is False:
            self.script_queue.get()

        
        # Re-queue all persistent scripts for the new timestep.
        for script in self.scripts:
            self.script_queue.put((script[0], script[1], False))

        self.received_stop = False

        self.script_lock.release()

        self.devices_done.set()
        self.devices_done.clear()

        self.thread_lock.release()

    def get_data(self, location):
        """Provides simple, thread-safe read access to sensor data."""
        result = None

        self.data_lock.acquire()

        if location in self.sensor_data:
            result = self.sensor_data[location]

        self.data_lock.release()

        return result

    def get_data_with_listener(self, location, listener):
        """
        Implements an asynchronous, non-blocking data fetch mechanism.

        Functional Intent: If the requested data `location` is currently locked
        ("busy"), instead of blocking, this method adds the `listener` (a tuple
        of script and callback queue) to a wait queue for that location and
        returns immediately. If the data is available, it marks it as busy and
        returns the data.
        """

        result = None

        self.data_lock.acquire()

        if location in self.sensor_data:

            # If another thread is working on this data, queue the listener.
            if self.busy_data[location]["busy"] is True:
                
                self.busy_data[location]["queue"].append(listener)
                result = False
            else:
                # Otherwise, lock the data and return it.
                self.busy_data[location]["busy"] = True
                result = self.sensor_data[location]

        self.data_lock.release()

        return result

    def set_data(self, location, data):
        """Provides simple, thread-safe write access to sensor data."""

        self.data_lock.acquire()

        self.sensor_data[location] = data

        self.data_lock.release()

    def set_data_with_listener(self, location, data):
        """
        Updates data and notifies a waiting listener, if one exists.

        Functional Intent: After updating the data value, this method marks the
        data location as "not busy". It then checks if any listeners are queued
        for this location. If so, it dequeues one and posts a notification to its
        callback queue, effectively re-awakening the waiting thread.
        """

        self.data_lock.acquire()

        self.sensor_data[location] = data

        if location not in self.busy_data:
            
            self.busy_data[location] = {"busy": False, "queue": []}
        else:
            # Release the lock on the data location.
            self.busy_data[location]["busy"] = False

            # If there are waiting threads, notify the next one in the queue.
            if len(self.busy_data[location]["queue"]) > 0:
                listener = self.busy_data[location]["queue"].pop(0)

                listener[1].put((listener[0], location, True))

        self.data_lock.release()

    def shutdown(self):
        """Waits for all worker threads to complete and join."""


        for i in xrange(len(self.threads)):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread that executes scripts for a single Device.
    """

    def __init__(self, device):
        """Initializes the thread and associates it with a parent device."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    
    
    def _get_script_data(self, script, location, neighbours):
        """
        Acquires data from the parent device and its neighbors.

        Functional Intent: This method attempts to lock and retrieve data from
        a specific `location` on both the parent device and all of its neighbors
        in an all-or-nothing fashion. If any device's data is busy, it releases
        all previously acquired locks and returns False, indicating a retry is needed.

        Returns:
            - A list of data values on success.
            - `False` if any of the required data locations were busy.
            - `None` if the data location does not exist on any device.
        """
        data = []
        locked_devices = []

        res = self.device.get_data_with_listener(
            location,
            (script, self.device.script_queue)
        )

        if res is False:
            # If the home device's data is busy, fail immediately.
            return False
        elif res is not None:
            data.append(res)
            locked_devices.append((self.device, res))

        # Block Logic: Attempt to acquire data locks from all neighbors.
        for neighbour in neighbours:
            res = neighbour.get_data_with_listener(
                location,
                (script, self.device.script_queue)
            )

            if res is False:
                # If any neighbor is busy, release all acquired locks and fail.
                
                for dev in locked_devices:
                    dev[0].set_data_with_listener(location, dev[1])

                return False
            elif res is not None:
                data.append(res)
                locked_devices.append((neighbour, res))

        if len(data) == 0:
            return None

        return data

    def _update_devices(self, location, value, neighbours):
        """Updates a data value on the parent device and all its neighbors."""
        self.device.set_data_with_listener(location, value)

        for neighbour in neighbours:
            neighbour.set_data_with_listener(location, value)

    def _loop(self, neighbours):
        """
        The main processing loop for a single timestep.

        This loop constitutes a state machine:
        1. Dequeue a script.
        2. Attempt to acquire all necessary data (`_get_script_data`).
        3. If data acquisition fails, register as a listener and wait.
        4. If it succeeds, run the script and update device values.
        5. Synchronize on a barrier before finishing the timestep.
        """

        # Pre-condition: `neighbours` contains the list of neighbors for this step.
        # Invariant: The loop continues as long as there is potential work to be done.
        while self.device.should_stop_thread() is False:

            res = self.device.script_queue.get()

            if res is None:
                continue

            
            (script, location, is_listener) = res

            if is_listener is True:
                # This work item is a notification that previously-locked data is now free.
                self.device.decrease_listeners()

            data = self._get_script_data(script, location, neighbours)

            if data is False:
                # Data was locked; become a listener and wait to be re-awakened.
                self.device.increase_listeners()
            elif data is not None:
                new_value = script.run(data)


                self._update_devices(location, new_value, neighbours)

        # Synchronize with other local threads before coordinating with other devices.
        self.device.thread_barrier.wait()

        val = self.device.decr_working_threads()

        # Block Logic: The last thread to finish is responsible for signaling
        # completion for the entire device.
        if val == 0:
            self.device.finish()
        else:
            self.device.devices_done.wait()

        self.device.thread_barrier.wait()


    def run(self):
        """The main entry point for the thread's lifecycle."""
        # This outer loop manages the progression between timesteps.
        while True:
            self.device.thread_start_timestep()

            neighbours = self.device.neighbours

            if neighbours is None:
                break

            self._loop(neighbours)

        self.device.thread_barrier.wait()

        val = self.device.decr_working_threads()

        if val == 0:
            self.device.finish()

class Barrier(object):
    """
    A reusable two-phase barrier implementation.

    Functional Intent: This barrier ensures that no thread proceeds to a subsequent
    phase of computation until all threads have completed the current phase. It
    uses a two-phase mechanism (controlled by two semaphores) to prevent race
    conditions where faster threads could loop around and re-enter the barrier
    before slower threads have left it from the previous wait.
    """

    def __init__(self, thread_count):
        """
        Initializes the barrier for a fixed number of threads.
        """
        self.lock = Lock()
        self.sem1 = Semaphore(0)
        self.sem2 = Semaphore(0)
        self.thread_count = thread_count
        self.thread_count1 = [thread_count]
        self.thread_count2 = [thread_count]


    def wait(self):
        """
        Blocks until all participating threads have called this method.
        """
        # First phase: ensures all threads have arrived at the barrier.
        self._phase(self.thread_count1, self.sem1)
        # Second phase: ensures all threads have been released from the first
        # phase before any can proceed.
        self._phase(self.thread_count2, self.sem2)

    def _phase(self, thread_count, sem):
        """Executes a single phase of the barrier synchronization."""
        self.lock.acquire()

        thread_count[0] = thread_count[0] - 1
        value = thread_count[0]

        self.lock.release()

        # The last thread to arrive resets the count and releases all other threads.
        if value == 0:
            thread_count[0] = self.thread_count
            for _ in xrange(self.thread_count):
                sem.release()

        sem.acquire()
