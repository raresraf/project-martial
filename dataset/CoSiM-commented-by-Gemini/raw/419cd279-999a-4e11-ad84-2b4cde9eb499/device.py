"""
This file models a distributed sensor network simulation.

The concurrency model uses a fixed-size pool of peer threads (`DeviceThread`)
for each device. One thread per device is designated as the "first" or "leader"
thread to handle coordination tasks for that device's time step. Work is
distributed among a device's threads using a shared index counter, a form of
work stealing.

Global resources like data locks and a main barrier are stored as class-level
(`static`) variables in the `Device` class.

Classes:
    Barrier: A synchronization primitive to make threads wait for each other.
    Device: Represents a node, holding state and managing a pool of worker threads.
    DeviceThread: A thread that performs the core work of the simulation.
"""


from threading import Event, Thread, Condition, Lock


class Barrier(object):
    """
    A synchronization barrier implemented with a Condition variable.

    @warning This is a non-reusable, single-phase barrier. It is NOT safe for use
             inside a loop. A race condition can occur where a fast thread loops
             around and re-enters `wait()` before a slow thread has woken up from
             the `cond.wait()` call, leading to a deadlock or incorrect behavior.
             A correct reusable barrier requires two phases or other protection
             against this "stray wakeup" problem.
    """

    def __init__(self, num_threads=0):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        
        self.cond = Condition()

    def wait(self):
        """Blocks until all `num_threads` have called this method."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread arrives, wakes up all others, and resets the counter.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Wait to be notified by the last thread.
            self.cond.wait()
        
        self.cond.release()

class Device(object):
    """
    Represents a device node in the network.

    This class uses class-level (`static`) variables for resources shared
    across all device instances, such as the main barrier and location locks.
    Each device instance manages a pool of `DeviceThread` workers.
    """
    # Class-level variables shared by all Device instances.
    bariera_devices = Barrier()
    locks = []

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device and its pool of worker threads."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # State for the current time step.
        self.scripts = []
        self.locations = []
        self.nr_scripturi = 0
        self.script_crt = 0 # Shared index for the work-stealing queue.

        # Synchronization for this device's worker pool.
        self.timepoint_done = Event()
        self.neighbours = []
        self.event_neighbours = Event()
        self.lock_script = Lock()
        self.bar_thr = Barrier(8)

        # Create and start a pool of 8 threads, designating one as the leader.
        self.thread = DeviceThread(self, 1) # The "first" or leader thread.
        self.thread.start()
        self.threads = []
        for _ in range(7):
            tthread = DeviceThread(self, 0) # Worker threads.
            self.threads.append(tthread)
            tthread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes the shared, class-level resources."""
        # This setup is only performed once by the first device to call it.
        Device.bariera_devices = Barrier(len(devices))
        if Device.locks == []:
            # Create a lock for each possible data location.
            for _ in range(self.supervisor.supervisor.testcase.num_locations):
                Device.locks.append(Lock())

    def assign_script(self, script, location):
        """Assigns a script for the upcoming time step."""
        if script is not None:
            self.scripts.append(script)
            self.locations.append(location)
            self.nr_scripturi += 1
        else:
            # A None script signals that all assignments for this step are done.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all threads associated with this device."""
        self.thread.join()
        for tthread in self.threads:
            tthread.join()


class DeviceThread(Thread):
    """A worker thread for a device."""

    def __init__(self, device, first):
        """
        Initializes the thread.
        Args:
            device (Device): The parent device instance.
            first (int): 1 if this is the leader thread for the device, 0 otherwise.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.first = first # Flag to identify the leader thread.

    def run(self):
        """Main execution loop for the thread."""
        while True:
            # --- Coordination Phase (Leader Only) ---
            if self.first == 1:
                # The leader thread fetches neighbors and resets the work counter.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.script_crt = 0
                # Signal to other threads of this device that neighbors are ready.
                self.device.event_neighbours.set()

            # All threads of this device wait for the neighbor list to be ready.
            self.device.event_neighbours.wait()

            # A None neighbor list is the global shutdown signal.
            if self.device.neighbours is None:
                break

            # All threads wait for the supervisor to signal that scripts are assigned.
            self.device.timepoint_done.wait()

            # --- Work-Stealing and Execution Phase ---
            while True:
                # Atomically get the index of the next script to process.
                self.device.lock_script.acquire()
                index = self.device.script_crt
                self.device.script_crt += 1
                self.device.lock_script.release()

                # If the index is out of bounds, all work for this device is done.
                if index >= self.device.nr_scripturi:
                    break

                location = self.device.locations[index]
                script = self.device.scripts[index]

                # Acquire the global lock for this specific location.
                Device.locks[location].acquire()

                # Aggregate data from neighbors and self.
                script_data = []
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Run computation and disseminate results.
                if script_data != []:
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the location lock.
                Device.locks[location].release()

            # --- Synchronization Phase ---
            # 1. All 8 threads of this device synchronize locally.
            self.device.bar_thr.wait()
            
            # The leader thread resets events for the next time step.
            if self.first == 1:
                self.device.event_neighbours.clear()
                self.device.timepoint_done.clear()

            # 2. Local barrier again to prevent the leader from racing ahead.
            self.device.bar_thr.wait()
            
            # 3. The leader thread waits at the global barrier to sync with all other devices.
            if self.first == 1:
                Device.bariera_devices.wait()
