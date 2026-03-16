"""
This module provides a threaded simulation framework for a network of devices.

It aims to simulate devices operating in synchronized time steps, but contains
several critical flaws in its synchronization and locking logic that will
lead to deadlocks and race conditions.
"""

from threading import Event, Thread, Condition, Lock
from Queue import Queue

class Device(object):
    """Represents a device in the simulation.

    Manages its own data and relies on a DeviceThread to orchestrate script
    execution and synchronization.

    NOTE: The locking mechanism in get_data and set_data is fundamentally flawed
    and will cause deadlocks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Locks are initialized later by the DeviceThread, which is unusual.
        self.locks = {}
        # An event to signal that script assignment for a time step is complete.
        self.no_more_scripts = Event()
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the shared barrier for all devices."""
        if self.device_id == 0:
            # Device 0 creates the barrier.
            self.barrier = ReusableBarrier(len(devices))

        # The barrier is then distributed to all other devices.
        for device in devices:
            if device is not self:
                device.set_barrier(self.barrier)


    def assign_script(self, script, location):
        """Assigns a script to be run, or signals the end of assignments."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A 'None' script is a sentinel that unblocks the main device thread.
            self.no_more_scripts.set()

    def get_data(self, location):
        """
        Retrieves data for a location.

        CRITICAL FLAW: This method acquires a lock but never releases it,
        guaranteeing a deadlock on the second access to the same location.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets data for a location.

        CRITICAL FLAW: This method releases a lock it did not acquire,
        which will raise a ThreadError. It is meant to be the pair to get_data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def set_barrier(self, barrier):
        """Receives the shared barrier object from device 0."""
        self.barrier = barrier

    def shutdown(self):
        """Attempts to shut down all threads."""
        for thread in self.thread.child_threads:
            if thread.is_alive():
                thread.join()
        self.thread.join()


class DeviceThread(Thread):
    """Main control thread for a device, managing a pool of worker threads."""

    def __init__(self, device):
        """Initializes the thread and its worker queue."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue()
        self.child_threads = []
        self.max_threads = 8


    def run(self):
        """Main simulation loop, orchestrating time steps."""
        # Unusually, this child thread initializes the locks for its parent device.
        for location, data in self.device.sensor_data.iteritems():
            self.device.locks[location] = Lock()

        # Create and start the pool of worker threads.
        for i in xrange(self.max_threads):
            thread = Thread(target=process_scripts, args=(self.queue,))
            self.child_threads.append(thread)
            thread.start()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            # Supervisor signals simulation end by returning None.
            if neighbours is None:
                # Terminate worker threads by sending sentinel values.
                for i in xrange(len(self.child_threads)):
                    self.queue.put(None)
                self.queue.join()
                break

            done_scripts = 0
            # Enqueue all scripts that have been assigned so far.
            for (script, location) in self.device.scripts:
                job = {'script': script, 'location': location, 'device': self.device, 'neighbours': neighbours}
                self.queue.put(job)
                done_scripts += 1

            # Wait for the signal that no more scripts will be assigned this step.
            self.device.no_more_scripts.wait()
            self.device.no_more_scripts.clear()
            
            # Complex logic to handle scripts assigned after the initial queuing.
            if done_scripts < len(self.device.scripts):
                for (script, location) in self.device.scripts[done_scripts:]:
                    job = {'script': script, 'location': location, 'device': self.device, 'neighbours': neighbours}
                    self.queue.put(job)

            # Wait for all workers to complete their tasks for this time step.
            self.queue.join()
            # Synchronize with all other devices before the next time step.
            self.device.barrier.wait()

def process_scripts(queue):
    """The target function for worker threads."""
    while True:
        job = queue.get()
        # Sentinel value check to terminate the thread.
        if job is None:
            queue.task_done()
            break
        
        script, location, mydevice, neighbours = job['script'], job['location'], job['device'], job['neighbours']
        script_data = []
        
        # This block will deadlock due to the flawed get_data/set_data implementation.
        for device in neighbours:
            if device is not mydevice:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        
        data = mydevice.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = script.run(script_data)
            # Disseminate the result to all participating devices.
            for device in neighbours:
                if device is not mydevice:
                    device.set_data(location, result)
            mydevice.set_data(location, result)
        
        queue.task_done()


class ReusableBarrier(object):
    """
    A custom implementation of a reusable barrier.

    NOTE: This is a single-phase barrier, which is not thread-safe for reuse
    in loops as it is vulnerable to race conditions. A correct implementation
    requires two phases.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks until all threads have called this method."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread arrives, notifies all waiting threads.
            self.cond.notify_all()
            # Resets for next use, but this is where the race condition can occur.
            self.count_threads = self.num_threads
        else:
            # Wait to be notified by the last thread.
            self.cond.wait()
        self.cond.release()
