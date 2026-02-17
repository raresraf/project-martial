# -*- coding: utf-8 -*-
"""Defines components for a simulated distributed sensor network.

This module provides classes for simulating a network of devices that process
sensor data concurrently. The simulation uses a barrier to operate in synchronized
time steps. This implementation features slightly different synchronization
mechanisms compared to other versions.
"""

from threading import Event, Thread, Lock, Condition, Semaphore


class ReusableBarrier(object):
    """A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a two-phase protocol with semaphores to ensure
    that threads wait for each other at the barrier point in each cycle without
    lapping.
    """
    def __init__(self, num_threads):
        """Initializes the barrier."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads reach the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one of the two synchronization phases."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread to arrive releases all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """Represents a device node in the distributed sensor simulation.

    Manages its own sensor data, scripts, and a worker thread. Device 0 acts
    as a coordinator for initializing shared resources.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.general_barrier = None
        self.lock_till_init = Semaphore()
        self.lock_for_certain_place = None
        self.dictionary_lock = Condition(Lock())
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self, 0)

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and starts the device threads.

        Device 0 creates and distributes the shared barrier and lock dictionary.
        It uses a Semaphore in an unconventional way to manage startup, where the
        main thread blocks until each worker thread signals its readiness.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            # Coordinator device (ID 0) initializes shared resources.
            self.lock_for_certain_place = {}
            self.general_barrier = ReusableBarrier(len(devices))

            for i in range(len(devices)):
                devices[i].general_barrier = self.general_barrier
                devices[i].lock_for_certain_place = self.lock_for_certain_place
            # The main thread waits for each worker to release its semaphore.
            for i in range(len(devices)):
                if not devices[i].device_id == 0:
                    devices[i].lock_till_init.acquire()

        elif not self.device_id == 0:
            # Worker device signals that it's ready.
            self.lock_till_init.release()
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be run in the next time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates the sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's worker thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The worker thread responsible for executing a device's logic."""
    def __init__(self, device, th_id):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Note: th_id is passed but not used.

    def run(self):
        """The main execution loop for the device thread."""
        while True:
            # Phase 1: Synchronize all threads.
            self.device.general_barrier.wait()
            
            # Note: All threads fetch neighbors, which is redundant work.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Phase 2: Wait for signal to start processing scripts.
            self.device.timepoint_done.wait()
            current_scripts = self.device.scripts
            
            # Phase 3: Execute assigned scripts for the time step.
            for (script, location) in current_scripts:
                # --- Critical Section for Location Data ---
                self.device.dictionary_lock.acquire()

                if not location in self.device.lock_for_certain_place:
                    self.device.lock_for_certain_place[location] = Condition(Lock())
                loc_cond = self.device.lock_for_certain_place[location]
                loc_cond.acquire()
                
                self.device.dictionary_lock.release()

                script_data = []
                # Gather data from neighbors.
                # Note: The `i = i + 1` line has no effect in a Python for-loop.
                for i in range(len(neighbours)):
                    if neighbours[i].get_data(location) is not None:
                        script_data.append(neighbours[i].get_data(location))
                        i = i + 1
                
                # Gather this device's own data.
                if self.device.get_data(location) is not None:
                    script_data.append(self.device.get_data(location))

                if script_data != []:
                    result = script.run(script_data)
                    
                    # Disseminate the result to all neighbors and self.
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
                
                loc_cond.release()
                # --- End of Critical Section ---
            
            # Phase 4: Reset for the next cycle.
            self.device.timepoint_done.clear()
