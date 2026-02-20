"""
This module defines a device simulation using a queue-based worker model.

A master device (ID 0) creates a shared barrier. At each time step, the main
device thread creates a new pool of worker threads. These workers pull tasks
from a shared queue until they receive a termination signal. This recreation
of threads in every step is a notable and inefficient design choice.
"""

from Queue import Queue
from threading import Semaphore, Lock
from threading import Event, Thread


class Device(object):
    """
    Represents a device in the network. It manages a queue of tasks for its
    worker threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.read_data = sensor_data
        self.supervisor = supervisor
        self.active_queue = Queue()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.time = 0

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources. Device 0 acts as master and creates the
        shared barrier for all other devices.
        """
        if self.device_id == 0:
            self.new_round = ReusableBarrierSem(len(devices))
            self.devices = devices
            for device in self.devices:
                device.new_round = self.new_round
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to a temporary list. When a None script is received
        (signaling end of timepoint), all accumulated scripts are pushed to the
        active work queue, followed by 'poison pills' to terminate the workers.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            for (script, location) in self.scripts:
                self.active_queue.put((script, location))
            # Add 8 "poison pills" to the queue to stop the 8 worker threads.
            for x in range(8):
                self.active_queue.put((-1, -1))

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.read_data[location] if location in self.read_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a specific location.
        @note This method is not thread-safe and can lead to race conditions
        when called concurrently by multiple worker threads.
        """
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device.
    @note This thread's design is highly inefficient, as it creates, starts,
    and joins a new pool of worker threads in every single iteration of its
    main loop.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers_number = 8

    def run(self):
        """
        Main simulation loop. Re-creates the worker pool at each time step.
        """
        neighbours = self.device.supervisor.get_neighbours()
        while True:
            self.workers = []
            self.device.neighbours = neighbours
            if neighbours is None:
                break # End of simulation.

            # Create and start a new set of workers for the current timepoint.
            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            # Wait for all workers to finish their tasks for this timepoint.
            for worker in self.workers:
                worker.join()
            
            # Synchronize with all other devices.
            self.device.new_round.wait()
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):
    """
    A worker thread that pulls tasks from a shared queue and executes them.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Continuously fetches tasks from the queue until a 'poison pill' is received.
        @note The data collection and update phases lack explicit locks, which can
        cause race conditions when multiple workers access the same location.
        """
        while True:
            script, location = self.device.active_queue.get()
            # The tuple (-1, -1) is a "poison pill" signaling termination.
            if script == -1:
                break
            
            script_data = []
            matches = []
            # Collect data from neighbors.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    matches.append(device)
                    script_data.append(data)
            # Collect data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
                matches.append(self.device)
            
            # Only run the script if data was gathered from more than one source.
            if len(script_data) > 1:
                result = script.run(script_data)
                # Update devices only if the new result is greater than the old value.
                for device in matches:
                    old_value = device.get_data(location)
                    if old_value < result:
                        device.set_data(location, result)


class ReusableBarrierSem():
    """
    A reusable two-phase barrier implemented with semaphores.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the caller until all threads have reached the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """Executes the first phase of the barrier protocol."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Executes the second phase of the barrier protocol."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
