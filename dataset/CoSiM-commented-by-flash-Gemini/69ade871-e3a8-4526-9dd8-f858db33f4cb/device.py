
"""
@69ade871-e3a8-4526-9dd8-f858db33f4cb/device.py
@brief Distributed device simulation with listener-based data synchronization.
This module implements a simulation environment for network-connected devices 
that execute parallel computational scripts. It features a listener-based 
locking mechanism for sensor data access and a custom two-phase semaphore 
barrier for global temporal synchronization, ensuring consistent state across 
concurrently executing worker threads.

Domain: Concurrent Simulation, Distributed Locking, Barrier Synchronization.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

class Device(object):
    """
    Functional Utility: Represent a simulated device with local sensor state.
    Logic: Manages a pool of DeviceThreads, a script queue, and coordinate-specific 
    busy-tracking for sensor data. It utilizes events and barriers to ensure 
    lockstep progression through simulation timesteps.
    """

    
    _CORE_COUNT = 8

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Constructor: Initializes device state, busy-tracking metadata, and worker pool.
        """
        
        self.device_id = device_id

        self.sensor_data = sensor_data
        self.busy_data = {}

        # Block Logic: Spatial busy-tracking initialization.
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

        # Block Logic: Worker pool dispatch.
        for i in xrange(Device._CORE_COUNT):


            self.threads.append(DeviceThread(self))
            self.threads[i].start()



    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Functional Utility: Global synchronization setup.
        Logic: Elects a leader device (lowest ID) to initialize a cluster-wide 
        barrier, ensuring all devices synchronize on the same temporal steps.
        """

        
        device = self

        for dev in devices:
            if dev.device_id < device.device_id:
                device = dev

        
        if device == self:
            self.device_barrier = Barrier(len(devices))

            for dev in devices:
                dev.device_barrier = self.device_barrier

    def assign_script(self, script, location):
        """
        Functional Utility: Enqueues a new computational task for execution.
        Logic: Uses a sentinel (None) in the script queue to signal the end 
         of assignments for the current timestep.
        """

        self.script_lock.acquire()


        if script is not None:
            self.scripts.append((script, location))
            self.script_queue.put((script, location, False))
        else:
            self.received_stop = True

            
            
            self.script_queue.put(None)

        self.script_lock.release()

    def increase_listeners(self):
        """
        Functional Utility: Atomic increment for wait-queue tracking.
        """

        self.script_lock.acquire()

        self.listeners = self.listeners + 1

        self.script_lock.release()

    def decrease_listeners(self):
        """
        Functional Utility: Atomic decrement for wait-queue tracking.
        """
        self.script_lock.acquire()

        self.listeners = self.listeners - 1

        self.script_lock.release()

    def should_stop_thread(self):
        """
        Functional Utility: Termination check for worker threads.
        Logic: Returns true if the queue is empty, no listeners are waiting, 
        and a stop signal has been received.
        """

        self.script_lock.acquire()

        val = self.script_queue.empty() and \
              self.listeners == 0 and \
              (self.received_stop is True)

        if val is True:
            
            
            self.script_queue.put(None)


        self.script_lock.release()

        return val

    def thread_start_timestep(self):
        """
        Functional Utility: Temporal step initialization for workers.
        Logic: The first thread to arrive retrieves the neighborhood topology 
        from the supervisor.
        """

        self.thread_lock.acquire()

        
        
        if self.working_threads == 0:
            self.neighbours = self.supervisor.get_neighbours()

            
            if self.neighbours is not None:
                if self in self.neighbours:
                    self.neighbours.remove(self)

        self.working_threads = self.working_threads + 1

        self.thread_lock.release()

    def decr_working_threads(self):
        """
        Functional Utility: Atomic decrement for worker thread accounting.
        """

        self.thread_lock.acquire()

        self.working_threads = self.working_threads - 1
        val = self.working_threads

        self.thread_lock.release()

        return val

    def get_working_threads(self):
        """
        Functional Utility: Atomic retrieval of active worker count.
        """

        self.thread_lock.acquire()

        val = self.working_threads

        self.thread_lock.release()

        return val

    def finish(self):
        """
        Functional Utility: Finalizes current timestep across all devices.
        Logic: Blocks on global device barrier then resets local temporal state.
        """
        self.device_barrier.wait()

        self.start_timestep()



    
    def start_timestep(self):
        """
        Functional Utility: Prepares for the next temporal simulation step.
        Logic: Clears stale scripts and re-populates the queue with active scripts.
        """

        self.thread_lock.acquire()

        self.script_lock.acquire()

        
        while self.script_queue.empty() is False:
            self.script_queue.get()

        
        for script in self.scripts:
            self.script_queue.put((script[0], script[1], False))

        self.received_stop = False

        self.script_lock.release()

        self.devices_done.set()
        self.devices_done.clear()

        self.thread_lock.release()

    def get_data(self, location):
        """
        Functional Utility: Atomic retrieval of local sensor data.
        """
        result = None

        self.data_lock.acquire()

        if location in self.sensor_data:
            result = self.sensor_data[location]

        self.data_lock.release()

        return result

    def get_data_with_listener(self, location, listener):
        """
        Functional Utility: Conflict-aware data retrieval.
        Logic: If data is busy, adds the caller to a wait-queue. Otherwise, 
        marks it as busy and returns the data.
        """

        result = None

        self.data_lock.acquire()

        if location in self.sensor_data:

            if self.busy_data[location]["busy"] is True:
                
                self.busy_data[location]["queue"].append(listener)
                result = False
            else:
                
                self.busy_data[location]["busy"] = True
                result = self.sensor_data[location]

        self.data_lock.release()

        return result

    def set_data(self, location, data):
        """
        Functional Utility: Atomic local sensor update.
        """

        self.data_lock.acquire()

        self.sensor_data[location] = data

        self.data_lock.release()

    def set_data_with_listener(self, location, data):
        """
        Functional Utility: Synchronized data update with release signaling.
        Logic: Updates data, clears busy flag, and notifies the next waiting 
        listener in the queue.
        """

        self.data_lock.acquire()

        self.sensor_data[location] = data

        if location not in self.busy_data:
            
            self.busy_data[location] = {"busy": False, "queue": []}
        else:
            
            self.busy_data[location]["busy"] = False

            if len(self.busy_data[location]["queue"]) > 0:
                listener = self.busy_data[location]["queue"].pop(0)

                listener[1].put((listener[0], location, True))

        self.data_lock.release()

    def shutdown(self):
        """
        Functional Utility: Orchestrates graceful worker thread termination.
        """


        for i in xrange(len(self.threads)):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    Functional Utility: Core execution loop for a device worker.
    Logic: Continuously consumes tasks from the script queue, coordinates 
    multi-device data aggregation, and executes computational scripts.
    """

    def __init__(self, device):
        """
        Constructor: Binds the worker thread to its parent device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    
    
    def _get_script_data(self, script, location, neighbours):
        """
        Block Logic: Neighborhood data aggregation protocol.
        Logic: Attempts to acquire locks and data from self and all neighbors 
        atomically. If any lock fails, it performs a rollback (release) of 
        previously acquired locks to prevent deadlocks.
        """
        data = []
        locked_devices = []

        res = self.device.get_data_with_listener(
            location,
            (script, self.device.script_queue)
        )

        if res is False:
            
            return False
        elif res is not None:
            data.append(res)
            locked_devices.append((self.device, res))

        for neighbour in neighbours:
            res = neighbour.get_data_with_listener(
                location,
                (script, self.device.script_queue)
            )

            if res is False:
                
                
                # Block Logic: Lock rollback on failure.
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
        """
        Block Logic: Results propagation.
        """
        self.device.set_data_with_listener(location, value)

        for neighbour in neighbours:
            neighbour.set_data_with_listener(location, value)

    def _loop(self, neighbours):
        """
        Execution Logic: Task processing loop for a temporal step.
        """

        while self.device.should_stop_thread() is False:

            res = self.device.script_queue.get()

            if res is None:
                continue

            
            (script, location, is_listener) = res

            if is_listener is True:

                self.device.decrease_listeners()

            data = self._get_script_data(script, location, neighbours)

            if data is False:
                
                self.device.increase_listeners()
            elif data is not None:
                # Core script execution.
                new_value = script.run(data)


                self._update_devices(location, new_value, neighbours)

        self.device.thread_barrier.wait()

        val = self.device.decr_working_threads()

        if val == 0:
            self.device.finish()
        else:
            self.device.devices_done.wait()

        self.device.thread_barrier.wait()


    def run(self):
        """
        Execution Logic: Main temporal simulation loop.
        """
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
    Functional Utility: Two-phase reusable barrier implementation.
    Logic: Uses two sequential semaphore gates (sem1, sem2) to ensure 
    that no thread can enter the next synchronization phase until all 
    threads have cleared the current one.
    """

    def __init__(self, thread_count):
        self.lock = Lock()
        self.sem1 = Semaphore(0)
        self.sem2 = Semaphore(0)
        self.thread_count = thread_count
        self.thread_count1 = [thread_count]
        self.thread_count2 = [thread_count]


    def wait(self):
        """
        Blocks until the required number of threads call wait().
        """
        self._phase(self.thread_count1, self.sem1)
        self._phase(self.thread_count2, self.sem2)

    def _phase(self, thread_count, sem):
        """
        Block Logic: Atomic phase transition.
        """
        self.lock.acquire()

        thread_count[0] = thread_count[0] - 1
        value = thread_count[0]

        self.lock.release()

        if value == 0:
            # Last thread release mechanism.
            thread_count[0] = self.thread_count
            for _ in xrange(self.thread_count):
                sem.release()

        sem.acquire()
