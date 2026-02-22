"""
This file models a distributed sensor network simulation.

The architecture is intended to be a producer-consumer model, where a main
`DeviceThread` (producer) for each device populates a shared `Queue` with tasks,
and a pool of `MyWorker` threads (consumers) process them.

@warning This script contains several critical concurrency flaws that make it
         non-functional and incorrect. These include an unsafe barrier, incorrect
         barrier synchronization in the main loop, and broken data aggregation logic.
"""


from threading import Event, Thread, Lock, Condition
from Queue import Queue

class ReusableBarrier(object):
    """
    A synchronization barrier implemented with a Condition variable.

    @warning This is a non-reusable, single-phase barrier. It is NOT safe for use
             inside a loop. It is vulnerable to a "stray wakeup" race condition
             where a fast thread can loop and re-enter `wait()` before slow
             threads have exited, corrupting the barrier's state and causing a
             deadlock.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """Represents a single device node in the network."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.queue = Queue()
        self.setup = Event()
        self.threads = []
        self.locations_lock = []
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources using a master device."""
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # Creates a hardcoded number of 25 locks.
            for _ in range(25):
                lock = Lock()
                self.locations_lock.append(lock)

            # Assign shared resources to all devices.
            for device in devices:
                device.barrier = barrier
                device.locations_lock = self.locations_lock
                # Signal other devices that setup is complete.
                device.setup.set()

    def assign_script(self, script, location):
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class DeviceThread(Thread):
    """
    The main producer/orchestrator thread for a device.

    @warning The synchronization in the `run` loop is incorrect. It waits on the
             barrier *before* the consumer threads have finished their work
             (there is no `queue.join()`). This creates a major race condition where
             the producer can loop and start the next time step before the current
             one is complete.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        # Wait for shared resources to be initialized.
        self.device.setup.wait()

        # Create and start a pool of consumer threads.
        for _ in range(8):
            thread = MyWorker(self.device)
            thread.start()
            self.device.threads.append(thread)

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Shutdown: put a "poison pill" for each worker and wait.
                for _ in range(len(self.device.threads) * 8): # This is likely a bug, over-populating the queue
                    self.device.queue.put(None)
                for thread in self.device.threads:
                    thread.join()
                break

            # Wait for all scripts for the time step to be assigned.
            self.device.timepoint_done.wait()
            
            # FLAW: Waits at the barrier before work is consumed.
            self.device.barrier.wait()

            # Puts all tasks on the queue for the workers.
            for (script, location) in self.device.scripts:
                self.device.queue.put((neighbours, location, script))

            self.device.timepoint_done.clear()
            
            # FLAW: Waits at barrier again, without ensuring tasks are done.
            self.device.barrier.wait()

class MyWorker(Thread):
    """
    A consumer thread that processes tasks from a shared queue.

    @warning The data aggregation logic is broken. It iterates through all
             neighbors but only uses the data from the *last* neighbor in the list,
             ignoring all others.
    """
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

    def run(self):
        while True:
            # Block until a task is available.
            elem = self.device.queue.get()
            
            # "Poison pill" for termination.
            if elem is None:
                break
            
            self.device.locations_lock[elem[1]].acquire()
            script_data = []
            data = None
            
            # FLAW: This loop overwrites `data` in each iteration.
            for device in elem[0]:
                data = device.get_data(elem[1])
            
            # Only the data from the LAST neighbor is ever used.
            if data is not None:
                script_data.append(data)
            
            data = self.device.get_data(elem[1])
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = elem[2].run(script_data)
                for device in elem[0]:
                    device.set_data(elem[1], result)
                self.device.set_data(elem[1], result)

            self.device.locations_lock[elem[1]].release()
            self.device.queue.task_done()
