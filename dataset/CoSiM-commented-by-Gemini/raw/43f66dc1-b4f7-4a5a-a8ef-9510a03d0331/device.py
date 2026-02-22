"""
This file models a distributed sensor network simulation.

The architecture uses a "thread-per-location" model. For each time step, the
main `DeviceThread` groups all assigned computational scripts by their target
location. It then spawns a single worker thread (`ParallelScript`) for each
location, which is responsible for executing all scripts associated with it.

@warning This script contains a critical race condition in its setup phase.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """A reusable barrier for synchronizing a fixed number of threads.
    
    This is a correct, two-phase implementation using semaphores that is safe
    for use in loops.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters are stored in a list to be mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all `num_threads` have called this method."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier protocol."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """Represents a single device node in the network."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Not used in this implementation.
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.big_lock = [] # Will hold a list of locks, one for each location.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources.

        @warning This method has a race condition. Every device that calls this
                 will re-create the `barrier` and `big_lock` list and then
                 overwrite the references in every other device. The last device
                 to complete this method "wins", and all devices will share the
                 resources it created. This initialization is not deterministic.
        """
        barrier = ReusableBarrier(len(devices))
        lock1 = Lock() # This lock is created but never used.

        num_locations = {}
        for device in devices:
            for location in device.sensor_data.keys():
                num_locations[location] = 1

        big_lock = [Lock() for _ in range(len(num_locations))]

        for device in devices:
            device.lock1 = lock1
            device.barrier = barrier
            device.big_lock = big_lock

    def assign_script(self, script, location):
        """Assigns a script for the upcoming time step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data[location] \
            if location in self.sensor_data else None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class ParallelScript(Thread):
    """
    A worker thread responsible for executing all scripts for a single location.

    @note This implementation is inefficient. The `run` method re-aggregates
          all neighbor data for every single script in its list, even though
          the data is the same for all scripts operating on the same location.
          A better approach would be to aggregate data once, then run all scripts.
    """
    def __init__(self, device, scripts, location, neighbours):
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts # A list of scripts for this location.
        self.location = location
        self.neighbours = neighbours

    def run(self):
        # This loop iterates through all scripts assigned to this worker's location.
        for script in self.scripts:
            # The lock is acquired and released for each script, which is inefficient.
            self.device.big_lock[self.location].acquire()

            # Data aggregation is repeated for each script.
            script_data = []
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Compute and disseminate result.
            if script_data != []:
                result = script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)
                
            self.device.big_lock[self.location].release()


class DeviceThread(Thread):
    """The main orchestrator thread for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal.

            # 1. Wait for the supervisor to assign all scripts for this time step.
            self.device.timepoint_done.wait()

            threads = []
            scripts = {} # Used to group scripts by location.

            # 2. Group all assigned scripts by their location.
            for (script, location) in self.device.scripts:
                if scripts.has_key(location):
                    scripts[location].append(script)
                else:
                    scripts[location] = [script]

            # 3. Create one worker thread for each location.
            for location in scripts.keys():
                new = ParallelScript(self.device, scripts[location],
                                     location, neighbours)
                threads.append(new)

            # 4. Start and wait for all worker threads to complete.
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # 5. Synchronize with all other devices before the next time step.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
