"""
This module implements a distributed device simulation using a multi-phase,
thread-spawning approach for concurrency.

Key architectural features:
- A self-contained, two-phase reusable barrier (`ReusableBarrierSem`) is used
  for synchronization.
- A "master" device (device 0) is responsible for creating and distributing the
  shared barrier instance to all other devices.
- Each device's main control thread (`DeviceThread`) follows a rigid, multi-phase
  procedure for each time step: it first gathers all necessary data, then
  dynamically creates a new set of worker threads (`Node`) for computation,
  waits for them to finish, and finally broadcasts the results.
- This model uses an inefficient anti-pattern of creating and destroying threads
  in a loop for every time step.
- Data access is synchronized implicitly by the procedural phases and the final
  barrier, rather than by explicit locks during reads/writes.

Note: This script appears to be written for Python 2.
"""


from threading import enumerate, Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    A reusable barrier implemented with Semaphores for thread synchronization.

    It operates in two phases to ensure that threads from a new iteration cannot
    proceed until all threads from the previous iteration have completed both phases.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a specific number of threads.
        :param num_threads: The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()       # Lock to protect access to the counters.
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase.

    def wait(self):
        """Blocks the calling thread until all participating threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive releases all threads waiting on the semaphore.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire() # All threads wait here.

    def phase2(self):
        """The second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            self.count_threads1 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a single device node in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs collective setup of the shared barrier.

        Device 0 creates the barrier, and all other devices get a reference to it.
        :param devices: A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    self.barrier = device.barrier

    def assign_script(self, script, location):
        """Assigns a script to be run in the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        Gets data for a location. This method is not thread-safe and relies
        on the caller for synchronization.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Sets data for a location. This method is not thread-safe and relies
        on the caller for synchronization.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class Node(Thread):
    """
    A one-shot worker thread that executes a single script.

    Its join() method is overridden to conveniently return the script's result.
    """
    def __init__(self, script, script_data):
        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        self.result = None
         
    def run(self):
        """Executes the script and stores the result."""
        self.result = self.script.run(self.script_data)

    def join(self):
        """Joins the thread and returns the script and its result."""
        Thread.join(self)
        return (self.script, self.result)


class DeviceThread(Thread):
    """
    The main control thread for a device, organizing work into distinct phases
    for each time step.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop, structured into sequential phases."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # --- Wait for script assignment ---
            self.device.script_received.wait()
            self.device.script_received.clear()

            # --- Phase 1: Gather Data ---
            # All data is read before any computation begins. This is a form of
            # implicit synchronization, avoiding the need for locks during reads.
            thread_list=[]
            scripts_result = {}
            scripts_data = {}
            for (script, location) in self.device.scripts:
                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                scripts_data[script] = script_data
                if script_data:
                    # --- Phase 2: Create Worker Threads ---
                    # A new thread is created for each script to be executed.
                    nod = Node(script, script_data)
                    thread_list.append(nod)

            # --- Phase 3: Execute Threads and Collect Results ---
            for nod in thread_list:
                nod.start()
            for nod in thread_list:
                key, value = nod.join()
                scripts_result[key] = value

            # --- Phase 4: Update Data ---
            # All data is written after all computations are complete.
            for (script, location) in self.device.scripts:
                if scripts_data.get(script):
                    for device in neighbours:
                        device.set_data(location, scripts_result[script])
                    self.device.set_data(location, scripts_result[script])
            
            # --- Phase 5: Synchronize with Barrier ---
            # Wait for all other devices to complete the time step.
            self.device.barrier.wait()
