"""
@d321d653-7f33-4e70-9af8-0842de7fbd18/device.py
@brief Distributed sensor processing simulation using transient worker threads and a shared location-lock registry.
* Algorithm: Dynamic task spawning where each script is executed in a dedicated `ScriptThread`, with on-demand lock initialization and two-phase semaphore barriers.
* Functional Utility: Orchestrates simulation phases across a device cluster by managing neighbor data aggregation and synchronized state updates.
"""

from threading import Lock, Thread, Event

class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings and coordination thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and prepare the main coordinator thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.locks = None # Intent: Map of location-specific locks shared across devices.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization setup.
        Invariant: The device with the minimum ID acts as the leader, initializing and distributing the barrier and locks.
        """
        leader = devices[0]
        for device in devices:
            if device.device_id < leader.device_id:
                leader = device

        if self.device_id == leader.device_id:
            self.barrier = ReusableBarrier(len(devices))
            self.locks = {} # Shared lock registry.
            for device in devices:
                device.barrier = self.barrier
                device.locks = self.locks

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing task for the current simulation phase.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: Signals completion of script batch delivery.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Standard data retrieval for sensor locations.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Standard data update for sensor locations.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device's coordination thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Coordinator thread managing the lifecycle of simulation phases.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator thread.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main execution lifecycle for simulation phases.
        Algorithm: Iterative worker thread spawning and collective barrier synchronization.
        """
        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits for the current timepoint batch to be fully assigned.
            self.device.timepoint_done.wait()

            threads = []
            # Dispatch Phase: Spawns one thread per assigned script.
            for (script, location) in self.device.scripts:
                script_thread = ScriptThread(self.device, script, location, neighbours)
                threads.append(script_thread)
                threads[-1].start()

            # Logic: Joins all local script threads before proceeding.
            for thread in threads:
                thread.join()

            # Post-condition: Reset phase state and align at the global barrier.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()


class ScriptThread(Thread):
    """
    @brief worker thread implementing the execution of a single sensor script.
    """

    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes the worker with its task parameters and context.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        @brief Main execution logic for a single script unit.
        Algorithm: Resource-locked execution with distributed data aggregation and propagation.
        """
        # Logic: On-demand lock creation for the target location.
        # Note: This pattern assumes consistent device.locks reference across threads.
        if self.location not in self.device.locks:
            self.device.locks[self.location] = Lock()

        # Pre-condition: Acquire location lock for atomic distributed update.
        with self.device.locks[self.location]:
            script_data = []
            
            # Distributed Aggregation Phase: Collect readings from neighbors and self.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Execution and Propagation Phase.
            if script_data != []:
                result = self.script.run(script_data)

                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)


from threading import *

class ReusableBarrier():
    """
    @brief Implementation of a two-phase synchronization barrier using semaphores.
    * Algorithm: Dual-stage arrival pattern to ensure strict thread alignment.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Intent: Shared mutable counter.
        self.count_threads2 = [self.num_threads] # Intent: Shared mutable counter.
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Synchronizes the calling thread through both phases of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single synchronization stage.
        Invariant: The last thread to arrive releases the entire group.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
