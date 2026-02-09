"""
@file device.py
@brief Implements a device model with a two-phase, map-reduce style update mechanism.

This file defines a `Device` that uses a pool of `CoreThread` workers to execute
scripts. The simulation follows a two-phase protocol for each timepoint:
1.  **Compute Phase:** Worker threads read data and compute script results in parallel.
2.  **Write Phase:** After all devices have finished computing, the main threads
    write the results back.
Two barrier synchronizations are used to separate these phases.

@note The initial data gathering (`get_data`) is not protected by locks, which
      can lead to race conditions.
"""

from threading import Event, Thread, Condition

class ReusableBarrierCond(object):
    """
    A reusable barrier implemented using a Condition variable.

    @note This implementation may be subject to race conditions, as a notified
          thread could re-enter `wait()` before the notifying thread has released
          the lock, potentially causing a deadlock.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()                  
    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.cond.acquire()                      
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()               
            self.count_threads = self.num_threads
        else:
            self.cond.wait()  
        self.cond.release()                     

class CoreThread(Thread):
    """
    A worker thread that executes a batch of scripts serially.
    
    It performs the 'compute' phase of the timepoint, running scripts and
    storing the results internally without writing them back to the devices.
    """
    def __init__(self):
        Thread.__init__(self)
        self.threads = []
        self.results = []
    def append_script(self, script, location, data):
        """Adds a script and its associated data to this thread's execution batch."""
        self.threads.append((script, location, data))
    def run(self):
        """
        Executes all scripts in the batch and stores the (script, location, result)
        tuples in an internal list.
        """
        self.results = [(script, location, script.run(data)) \
        for (script, location, data) in self.threads]

class Device(object):
    """
    Represents a single device in the simulation.
    """
    # A class-level barrier shared by all device instances.
    barrier = ReusableBarrierCond(0)
    barrier_set = False
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared barrier and starts the main device thread.
        """
        # Block Logic: The first device to call this method creates the shared barrier.
        if not Device.barrier_set:
            Device.barrier = ReusableBarrierCond(len(devices))
            Device.barrier_set = True

        self.thread = DeviceThread(self)
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that script assignment is complete for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. This access is not synchronized."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This access is not synchronized."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating the two-phase update.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = []

    def run(self):
        """The main simulation loop, implementing the 'compute-then-write' logic."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for the supervisor to finish assigning scripts.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            
            # --- Start Compute Phase ---
            self.threads = [CoreThread() for i in range(8)]
            
            # Block Logic: Distribute assigned scripts to the 8 CoreThread workers
            # in a round-robin fashion.
            count = 0
            for (script, location) in self.device.scripts:
                script_data = []
                # Gather data from neighbors and self (UNSYNCHRONIZED).
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                if script_data != []:
                    self.threads[count].append_script(script, location, script_data)
                    count = (count+1) % 8

            # Start all worker threads that have scripts assigned.
            for i in range(8):
                if self.threads[i].threads != []:
                    self.threads[i].start()
            
            # Wait for all worker threads to finish their computation.
            for i in range(8):
                if self.threads[i].threads != []:
                    self.threads[i].join()
            
            # --- First Barrier ---
            # All devices wait here, ensuring computation is complete everywhere
            # before anyone starts writing results.
            Device.barrier.wait()

            # --- Start Write Phase ---
            # Block Logic: Iterate through the results from all worker threads and
            # write them back to the relevant devices.
            for i in range(8):
                for (script, location, result) in self.threads[i].results:
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
            
            # --- Second Barrier ---
            # All devices wait here, ensuring all data writes are complete
            # before the next timepoint begins.
            Device.barrier.wait()