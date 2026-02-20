"""
This module implements a device simulation using a worker pool model with a
custom, manually implemented task queue.

A master `DeviceThread` hands out work to a pool of `ScriptThread` workers,
which compete to process scripts from a shared list. The setup of shared
synchronization primitives is flawed and likely to cause deadlocks.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a device in the network, which manages a pool of worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.loopBarrier = None
        self.locationSemaphores = None
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for the devices.
        @note This setup logic is critically flawed. It creates local barrier
        and semaphore objects and assigns them, but there is no master/slave
        logic or guarantee that all devices will share the same objects. This
        will lead to synchronization failures.
        """
        loopBarrier = ReusableBarrierCond(len(devices))
        locationSemaphores = {}
        for device in devices :
            device.loopBarrier = loopBarrier
            device.locationSemaphores = locationSemaphores

    def assign_script(self, script, location):
        """
        Assigns a script to be processed and lazily initializes a semaphore
        for the script's location if one does not already exist.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Lazy initialization of location-specific semaphores.
            if self.locationSemaphores.get(location) is None:
                self.locationSemaphores[location] = Semaphore()
        else:
            # When all scripts are assigned, set the event to start processing.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The master thread for a device. It creates a pool of workers and provides
    them with tasks via a custom, thread-safe `getWork` method.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workLock = Lock() # Lock to protect access to the script list index.
        self.lastScriptGiven = 0;

    def run(self):
        """
        The main simulation loop. For each timepoint, it creates workers,
        waits for them to process all scripts, and then synchronizes.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            self.lastScriptGiven = 0;

            # Wait until the supervisor has assigned all scripts.
            self.device.script_received.wait()
            
            workers = []
            workLock = Lock() # This lock is created but not used; the master's lock is used.

            # Create a pool of workers.
            for i in range(0, min(8, len(self.device.scripts))):
                worker = ScriptThread(self, neighbours, workLock)
                workers.append(worker)

            for worker in workers:
                worker.start()

            for worker in workers:
                worker.join()

            self.device.script_received.clear()
            # Wait at the global barrier for all devices to finish the timepoint.
            self.device.loopBarrier.wait()

    def getWork(self):
        """
        A thread-safe method for workers to get the next available script.
        This acts as a manual task queue.
        """
        script = None
        # This critical section is protected by the master thread's lock.
        if (self.lastScriptGiven < len(self.device.scripts)):
            script = self.device.scripts[self.lastScriptGiven]
            self.lastScriptGiven += 1

        return script

class ScriptThread(Thread) :
    """
    A worker thread that repeatedly requests and executes scripts from its
    master `DeviceThread` until no more work is available for the timepoint.
    """
    def __init__(self, master, neighbours, workLock):
        Thread.__init__(self)
        self.master = master
        self.neighbours = neighbours
        self.workLock = workLock # This lock is passed but the master's lock is used inside getWork.

    def run(self) :
        # Acquire lock, get a task, release lock. This is the main work-pulling loop.
        self.master.workLock.acquire()
        scriptLocation = self.master.getWork()
        self.master.workLock.release()

        while scriptLocation is not None:
            (script, location) = scriptLocation
            script_data = []
            
            # Acquire the semaphore for the specific data location.
            self.master.device.locationSemaphores.get(location).acquire()

            # --- Critical Section for this location ---
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.master.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)
                for device in self.neighbours:
                    device.set_data(location, result)
                self.master.device.set_data(location, result)
            # --- End Critical Section ---

            self.master.device.locationSemaphores.get(location).release()
            
            # Get the next piece of work.
            self.master.workLock.acquire()
            scriptLocation = self.master.getWork()
            self.master.workLock.release()
