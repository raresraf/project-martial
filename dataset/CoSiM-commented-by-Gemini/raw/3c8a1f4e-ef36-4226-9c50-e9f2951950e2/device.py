
"""
@brief A flawed, event-driven distributed device simulation.
@file device.py

This module implements a highly complex and flawed simulation of distributed
devices. Each device consists of a main control thread (`DeviceThread`) and a
pool of worker threads (`ThreadAux`). Synchronization is attempted via a mix of
class-level (static) shared barriers and events, and instance-level barriers.

WARNING:
This architecture is fundamentally broken and illustrates numerous anti-patterns.
1.  **Buggy `ReusableBarrier`**: The barrier implementation is not safe. It holds
    a lock while releasing waiting threads, which can lead to deadlocks.
2.  **Critical Data Race**: Worker threads (`ThreadAux`) read the `neighbours`
    list directly from the main thread instance without any synchronization.
    This is a severe race condition that will lead to unpredictable behavior.
3.  **Hardcoded Limits**: The simulation is arbitrarily limited to 30 lockable
    locations and 11 time steps due to hardcoded list sizes for locks and events,
    making the design inflexible and brittle.
4.  **Misuse of Class Variables**: Key synchronization primitives (a barrier, an
    event, and all locks) are stored as class variables, meaning all device
    instances are coupled through shared global state. This prevents running
    multiple simulations and is extremely poor design.
"""


from threading import Event, Thread, Lock, Semaphore, Lock

class ReusableBarrier(object):
    """
    A flawed implementation of a reusable two-phase barrier using semaphores.

    WARNING: This implementation is not safe. It holds `counter_lock` while
    releasing the semaphores, an anti-pattern that serializes thread wakeup and
    can cause deadlocks.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads

        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks all threads until all have called wait. Two-phase approach."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread wakes up all others for this phase.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset for next use.
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread wakes up all others for this phase.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset for next use.
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()



class Device(object):
    """
    Represents a device node, which manages a main thread and a pool of workers.
    
    This class heavily relies on shared, static (class-level) variables for
    synchronization, which is a major design flaw.
    """
    
    # Class-level variables shared across ALL device instances.
    bar1 = ReusableBarrier(1) # Global barrier for all devices.
    event1 = Event()          # Global startup event.
    locck = []                # Global list of locks for all locations.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device, its control thread, and its worker thread pool.
        """
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.devices = []

        # A hardcoded list of 11 events, presumably one for each time step.
        # This severely limits the simulation duration.
        self.event = []
        for _ in xrange(11):
            self.event.append(Event())

        self.nr_threads_device = 8
        self.nr_thread_atribuire = 0 # Used for round-robin script assignment.
        
        # An internal barrier for this device's main thread and its workers.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1)

        # Each device has one main thread and a pool of worker threads.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the globally shared state using Device 0 as a master.
        """
        self.devices = devices
        
        if self.device_id == 0:
            # Master device initializes the global locks (hardcoded to 30).
            for _ in xrange(30):
                Device.locck.append(Lock())
            # Master re-initializes the global barrier for all devices.
            Device.bar1 = ReusableBarrier(len(devices))
            # Master sets the global start event, releasing all waiting devices.
            Device.event1.set()

    def assign_script(self, script, location):
        """
        Assigns a script to one of the worker threads in a round-robin fashion.
        """
        if script is not None:
            # Script is added to a specific worker's dictionary.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%\
            self.nr_threads_device
        else:
            # A `None` script signals that all scripts for the time step are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main thread and all worker threads."""
        self.thread.join()
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    """
    The main control thread for a single device. It coordinates its workers.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.contor = 0 # Time step counter, used to index the event list.

    def run(self):
        # Wait for the global start signal from the master device.
        Device.event1.wait()

        while True:
            # Fetch the list of neighbors for the current time step.
            self.neighbours = self.device.supervisor.get_neighbours()

            # If supervisor signals end of simulation, signal workers and exit.
            if self.neighbours is None:
                self.device.event[self.contor].set()
                break

            # Wait for supervisor to confirm all scripts for this step are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Unblock this device's worker threads for the current time step.
            self.device.event[self.contor].set()
            self.contor += 1

            # Wait for this device's own worker threads to finish their work.
            self.device.bar_threads_device.wait()

            # Wait for ALL other devices to complete the current time step.
            Device.bar1.wait()

class ThreadAux(Thread):
    """
    A worker thread that executes assigned scripts.
    """
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device
        self.script_loc = {} # Holds scripts assigned to this specific thread.
        self.contor = 0 # Time step counter.

    def run(self):
        while True:
            # Wait for the main device thread to signal the start of a time step.
            self.device.event[self.contor].wait()
            self.contor += 1

            # CRITICAL RACE CONDITION: Reads `neighbours` from the parent thread
            # without any synchronization. It might get a stale value from a
            # previous iteration or a partially updated one.
            neigh = self.device.thread.neighbours
            if neigh is None:
                break # Exit if simulation has ended.

            for script in self.script_loc:
                location = self.script_loc[script]
                
                # Acquire a global lock based on location index.
                Device.locck[location].acquire()
                script_data = []

                # Gather data from neighbors.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # If data was found, execute script and propagate results.
                if script_data != []:
                    result = script.run(script_data)
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                Device.locck[location].release()

            # Signal to the main device thread that this worker is done.
            self.device.bar_threads_device.wait()
