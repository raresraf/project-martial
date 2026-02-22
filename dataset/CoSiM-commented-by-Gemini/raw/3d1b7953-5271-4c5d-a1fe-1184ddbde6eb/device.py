
"""
@brief A device simulation with master-worker setup and flawed concurrency.
@file device.py

This module implements a distributed device simulation. It uses a master-worker
pattern for initialization, where Device 0 is responsible for setting up shared
resources (a barrier and location locks). For script execution, it uses an
inefficient model where new worker threads (`ScriptThread`) are created and
destroyed for every time step.

WARNING: This implementation contains severe performance and concurrency flaws.
1.  **Startup Race Condition**: In `setup_devices`, there is no mechanism to make
    worker devices wait for the master (Device 0) to finish initializing the
    shared `barrier` and `location_locks`. Since all threads are started in
    `__init__`, worker threads can start their `run` loop and try to access these
    uninitialized resources, leading to a crash.
2.  **Inefficient Threading**: In `DeviceThread.run`, a new list of threads is
    created, started, and joined for every single time step. This introduces
    massive overhead.
3.  **Flawed `ReusableBarrier`**: The barrier implementation holds a lock while
    releasing waiting threads, a classic anti-pattern that can cause deadlocks.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    A flawed implementation of a reusable two-phase barrier using semaphores.

    WARNING: This implementation is not safe. It holds `count_lock` while
    releasing the semaphores, which is an anti-pattern that can serialize
    thread wakeup and create deadlocks.
    """
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Use a list for pass-by-reference.
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all threads reach the barrier. Consists of two phases."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            
            if count_threads[0] == 0:
                # Last thread wakes up others and resets the counter.
                for _ in range(self.num_threads):
                    threads_sem.release()
                
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """Represents a device node in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start() # RACE CONDITION: Thread starts before setup is complete.
        self.location_locks = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources using Device 0 as a master.
        
        RACE CONDITION: Worker devices do not wait for this method to complete
        on the master device, so they may access `barrier` and `location_locks`
        before they are initialized.
        """
        if 0 == self.device_id:
            # Block for master device.
            # Create the shared barrier.
            self.barrier = ReusableBarrier(len(devices))
            
            # Find all unique locations across all devices.
            locations = []
            for device in devices:
                for location in device.sensor_data:
                    if location not in locations:
                        locations.append(location)
            
            # Create a list of locks, one for each location.
            self.location_locks = []
            for _ in range(len(locations)):
                self.location_locks.append(Lock())
            
            # Distribute shared resources to all other devices.
            for device in devices:
                device.barrier = self.barrier
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """Adds a script to the device's workload for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that script assignment is done for this time step.
            self.timepoint_done.set()

    def get_data(self, location):
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class ScriptThread(Thread):
    """
    A short-lived worker thread that executes a single script.
    """
    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """Executes the script, safely locking the specific location."""
        with self.device.location_locks[self.location]:
            script_data = []
            # Gather data from neighbors and self.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            # If data is available, run the script and propagate results.
            if script_data != []:
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    The main control thread for a device, which inefficiently spawns new
    worker threads for each time step.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main simulation loop for the device."""
        while True:
            # Fetch neighbors for the current time step.
            vecini = self.device.supervisor.get_neighbours()
            if vecini is None:
                break # End of simulation.
            
            # Wait for supervisor to assign all scripts for this time step.
            self.device.timepoint_done.wait()
            threads = []
            
            # --- INEFFICIENCY WARNING ---
            # Spawns a new thread for each script in every time step.
            if len(vecini) != 0:
                for (script, locatie) in self.device.scripts:
                    thread = ScriptThread(self.device, script, locatie, vecini)
                    threads.append(thread)
                    thread.start()
                
                # Wait for all of this step's worker threads to complete.
                for thread in threads:
                    thread.join()
            
            # Reset event and wait at the global barrier for all other devices.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
