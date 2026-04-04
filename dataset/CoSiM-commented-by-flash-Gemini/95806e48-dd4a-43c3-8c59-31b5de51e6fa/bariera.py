

"""
@95806e48-dd4a-43c3-8c59-31b5de51e6fa/bariera.py
@brief Implements multi-threaded simulation for distributed sensor devices using custom reentrant barriers.

This module defines core components for simulating a network of sensor devices.
It features two custom reentrant barrier implementations: `BarieraReentrantaCond`
(using condition variables) and `BarieraReentrantaSem` (using semaphores).
Each device (`Device`) operates with multiple worker threads (`ThreadDispozitiv`)
that execute scripts, manage local sensor data, and interact with neighbors.
Synchronization is managed both within a device and across devices using these barriers.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate under the guidance of a supervisor.

Original language of code comments and identifiers: Romanian. Semantic documentation: English.

Classes:
- BarieraReentrantaCond: A reentrant barrier implementation using a Condition variable.
- BarieraReentrantaSem: A reentrant barrier implementation using Semaphores.
- Device: Represents a single simulated sensor device.
- ThreadDispozitiv: A worker thread responsible for executing scripts for a device.

Domain: Distributed Systems Simulation, Concurrent Programming, Parallel Processing, Custom Synchronization Primitives.
"""

from threading import Condition, Semaphore, Lock

class BarieraReentrantaCond(object):
    """
    @brief A reentrant barrier implementation using a Condition variable.

    This barrier allows multiple threads to wait until all have reached a common
    point. It can be reused after all threads have passed through.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the Reentrant Condition Barrier.

        @param num_threads: The total number of threads that will participate in this barrier.
        """
        self.num_threads = num_threads
        # Current count of threads waiting at the barrier.
        self.count_threads = self.num_threads
        # Condition variable used for thread synchronization.
        self.cond = Condition()

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all
               participating threads have called `wait()`.
        """
        # Critical Section: Acquire the condition variable lock to modify shared state.
        self.cond.acquire()
        # Decrement the count of threads yet to reach the barrier.
        self.count_threads -= 1
        if self.count_threads == 0:
            # Block Logic: If this is the last thread, notify all waiting threads.
            self.cond.notify_all()
            # Reset the thread count for reuse of the barrier.
            self.count_threads = self.num_threads
        else:
            # Block Logic: If not the last thread, wait for notification.
            self.cond.wait()
        # Release the condition variable lock.
        self.cond.release()

class BarieraReentrantaSem(object):
    """
    @brief A reentrant barrier implementation using Semaphores (a two-phase barrier).

    This barrier allows multiple threads to wait until all have reached a common
    point and can be reused. It employs two semaphores to manage the two phases
    of synchronization, ensuring proper reentrancy.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the Reentrant Semaphore Barrier.

        @param num_threads: The total number of threads that will participate in this barrier.
        """
        self.num_threads = num_threads
        # Count of threads for the first phase.
        self.count_threads1 = self.num_threads
        # Count of threads for the second phase.
        self.count_threads2 = self.num_threads
        # Lock to protect access to thread counters.
        self.counter_lock = Lock()
        # Semaphore for the first phase of the barrier.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second phase of the barrier.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all
               participating threads have called `wait()`.
        """
        # Block Logic: Executes the first phase of the barrier synchronization.
        self.prima_faza()
        # Block Logic: Executes the second phase of the barrier synchronization.
        self.a_doua_faza()

    def prima_faza(self):
        """
        @brief Implements the first phase of the two-phase semaphore barrier.

        Threads decrement a counter. The last thread to reach zero releases
        all other waiting threads for this phase.
        """
        with self.counter_lock:
            # Decrement the count of threads for the first phase.
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Block Logic: If this is the last thread, release all `num_threads` waiting on `threads_sem1`.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the count for `count_threads1` for the next use of the barrier.
                self.count_threads1 = self.num_threads

        # Block Logic: All threads acquire from `threads_sem1`, ensuring they wait until all are ready.
        self.threads_sem1.acquire()

    def a_doua_faza(self):
        """
        @brief Implements the second phase of the two-phase semaphore barrier.

        Threads decrement a second counter. The last thread to reach zero releases
        all other waiting threads for this phase. This ensures the barrier is reentrant.
        """
        with self.counter_lock:
            # Decrement the count of threads for the second phase.
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Block Logic: If this is the last thread, release all `num_threads` waiting on `threads_sem2`.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the count for `count_threads2` for the next use of the barrier.
                self.count_threads2 = self.num_threads

        # Block Logic: All threads acquire from `threads_sem2`, ensuring they wait until all are ready for the second phase.
        self.threads_sem2.acquire()

from threading import Event, Thread, Lock
from bariera import BarieraReentrantaCond, BarieraReentrantaSem

class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device (`Dispozitiv` in Romanian) manages its own sensor data (`sensor_data`),
    interacts with a supervisor (`supervisor`), and executes assigned scripts (`scripturi`)
    in a multi-threaded environment. It uses `ThreadDispozitiv` as worker threads and
    coordinates synchronization with custom barriers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data, keyed by location.
        @param supervisor: A reference to the central supervisor managing all devices.
        """
        # Unique identifier for this device.
        self.device_id = device_id
        # Dictionary storing sensor data, keyed by location.
        self.sensor_data = sensor_data
        # Reference to the central supervisor.
        self.supervisor = supervisor
        # Counter for rotating script assignment among worker threads (0-7).
        self.nr_thread = 0
        # List to hold references to the device's worker threads (`ThreadDispozitiv`).
        self.threaduri = []
        # List to store references to neighboring devices (`vecini`).
        self.vecini = []
        # List of Locks (`lista_loc`), where each lock protects data for a specific location.
        self.lista_loc = []
        # List of lists, where each inner list stores scripts for a specific worker thread (8 worker threads).
        self.scripturi = [[] for _ in range(8)]
        # Barrier (`bar_sync_threaduri`) for synchronizing the device's own worker threads.
        self.bar_sync_threaduri = BarieraReentrantaCond(8)
        # Event (`initializare_gata`) to signal that device initialization (especially shared resources) is complete.
        self.initializare_gata = Event()
        # Event (`timepoint_gata`) to signal that all scripts for the current timepoint have been assigned.
        self.timepoint_gata = Event()
        # Global barrier (`bariera_disp`) for inter-device synchronization at timepoint boundaries.
        self.bariera_disp = None
    
    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Dispozitiv <device_id>".
        """
        return "Dispozitiv %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up global synchronization resources for all devices.

        This method designates device 0 as the master to initialize the global
        inter-device barrier (`bariera_disp`) and the array of location-specific
        locks (`lista_loc`). Slave devices wait for the master's initialization
        to complete and receive references to these shared resources. It also
        starts the device's worker threads (`ThreadDispozitiv`).

        @param devices: A list of all Device instances in the simulation.
        """
        maxim_locatii = 0       
        lista_disp = devices # `lista_disp` is a local copy of all devices.
        if self.device_id == 0: 
            # Block Logic: Master device (device 0) initializes the global inter-device barrier.
            self.bariera_disp = BarieraReentrantaSem(len(devices))

            # Block Logic: Determines the maximum location index across all devices to size `lista_loc`.
            for loc in self.sensor_data.keys():
                maxim_locatii = max(maxim_locatii, loc)
            # Inline: Removes itself from the local list for iteration purposes.
            lista_disp.remove(self)     
            
            for dispozitiv in lista_disp:
                for loc in dispozitiv.sensor_data.keys():
                    maxim_locatii = max(maxim_locatii, loc)
                # Assigns the master's inter-device barrier to other devices.
                dispozitiv.bariera_disp = self.bariera_disp
            
            # Block Logic: Populates `lista_loc` with `Lock` objects, one for each location up to `maxim_locatii`.
            for _ in range(maxim_locatii + 1):
                self.lista_loc.append(Lock())
            
            # Block Logic: Distributes the initialized `lista_loc` (list of locks) to all other devices.
            for dispozitiv in lista_disp:
                dispozitiv.lista_loc = self.lista_loc
            
            # Signals that master initialization is complete.
            self.initializare_gata.set()
        else:   
            # Block Logic: Slave device initialization and synchronization.
            # Inline: Removes itself from the local list for iteration purposes.
            lista_disp.remove(self)             
            for dispozitiv in lista_disp:       
                if dispozitiv.device_id == 0:   
                    # Block Logic: Waits for device 0 (master) to complete its initialization.
                    dispozitiv.initializare_gata.wait()
                    break

        # Block Logic: Spawns and starts 8 worker threads (`ThreadDispozitiv`) for the current device.
        for th_curr in range(8):
            thrd = ThreadDispozitiv(self, th_curr, self.bar_sync_threaduri,
                                    self.bariera_disp)
            self.threaduri.append(thrd)
            self.threaduri[-1].start()


    def assign_script(self, script, zona):
        """
        @brief Assigns a script (`script`) to be executed for a specific location (`zona`).

        Scripts are assigned in a round-robin fashion to the device's 8 worker threads.
        If `script` is None, it signals that all scripts for the current timepoint
        have been assigned, and sets `timepoint_gata`.

        @param script: The script object to be executed.
        @param zona: The location pertinent to the script execution.
        """
        # Block Logic: If a script is provided, assign it to the next available worker thread.
        if script is not None:
            self.scripturi[self.nr_thread].append((script, zona))
            # Inline: Increments `nr_thread` in a round-robin manner (0 to 7).
            self.nr_thread = (self.nr_thread + 1) % 8
        else:
            # Pre-condition: `script` is None, indicating no more scripts for the current timepoint.
            # Invariant: The `timepoint_gata` event is set, signaling workers to begin processing.
            self.timepoint_gata.set()

    def get_data(self, zona):
        """
        @brief Retrieves sensor data for a given location (`zona`).

        @param zona: The location for which to retrieve data.
        @return: The sensor data for the specified location, or None if the location is not found.
        """
        return self.sensor_data[zona] if zona in self.sensor_data else None

    def set_data(self, zona, info):
        """
        @brief Sets or updates sensor data for a given location (`zona`).

        @param zona: The location for which to set data.
        @param info: The new data value to be set.
        """
        if zona in self.sensor_data:
            self.sensor_data[zona] = info

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's worker threads.

        Waits for all `ThreadDispozitiv` instances to complete their execution.
        """
        for thrd in self.threaduri:
            thrd.join()


class ThreadDispozitiv(Thread):
    """
    @brief A worker thread for a `Device` instance, responsible for executing scripts.

    Each `ThreadDispozitiv` (`Device Thread` in English) processes a subset of
    scripts assigned to its parent `Device`. It uses both an internal barrier
    (`bar_sync_threaduri`) for intra-device synchronization and a global barrier
    (`bariera_disp`) for inter-device synchronization.
    """

    def __init__(self, device, nr_thread, bar_sync_threaduri, bar_sync_div):
        """
        @brief Initializes a ThreadDispozitiv worker thread.

        @param device: The parent `Device` instance.
        @param nr_thread: The unique identifier for this worker thread (0-7).
        @param bar_sync_threaduri: The internal barrier for synchronizing worker threads within the same device.
        @param bar_sync_div: The global barrier for synchronizing across different devices.
        """
        self.device = device
        # This worker thread's identifier.
        self.nr_thread = nr_thread
        # Internal barrier for worker threads within the same device.
        self.bar_sync_threaduri = bar_sync_threaduri
        # Global barrier for synchronization across all devices.
        self.bar_sync_div = bar_sync_div

        Thread.__init__(self, name="Dispozitiv %d Thread %d" % (device.device_id, nr_thread))

    def run(self):
        """
        @brief The main execution loop for the ThreadDispozitiv.

        Pre-condition: The device and its synchronization mechanisms are properly set up.
        Invariant: The thread continuously processes assigned scripts for its portion of the workload,
                   synchronizing with other worker threads and devices at appropriate points.
        """
        while True:
            # Block Logic: Only worker thread 0 is responsible for calling the global inter-device barrier
            # and fetching neighbors from the supervisor. This reduces contention.
            if self.nr_thread == 0:
                self.bar_sync_div.wait() # Waits for all devices to reach this point.
                # Fetches the current neighbors for the device.
                self.device.vecini = self.device.supervisor.get_neighbours()

            # Block Logic: All worker threads within the same device synchronize using an internal barrier.
            # This ensures that `self.device.vecini` is updated before all threads proceed.
            self.bar_sync_threaduri.wait() 

            if self.device.vecini is None:
                # Pre-condition: `self.device.vecini` is None, indicating a termination signal from the supervisor.
                # Invariant: The loop breaks, leading to thread termination.
                break

            # Block Logic: Waits for the parent device to signal that all scripts for the current
            # timepoint have been assigned.
            self.device.timepoint_gata.wait() 

            # Block Logic: Iterates through the scripts assigned to this specific worker thread.
            for (script, zona) in self.device.scripturi[self.nr_thread]:
                date_script = [] # `date_script` is `script_data` in English.
                
                # Critical Section: Acquire the location-specific lock (`lista_loc[zona]`)
                # to ensure exclusive access to data for this `zona`.
                self.device.lista_loc[zona].acquire()
                # Block Logic: Gathers data from neighboring devices (`vecini`) for the current `zona`.
                for dispozitiv in self.device.vecini:
                    info = dispozitiv.get_data(zona) # `info` is `data` in English.
                    if info is not None:
                        date_script.append(info)
                
                # Gathers data from its own `sensor_data` for the current `zona`.
                info = self.device.get_data(zona)
                
                if info is not None:
                    date_script.append(info)

                if date_script != []:
                    # Executes the script with the collected data.
                    result = script.run(date_script)
                    
                    # Block Logic: Updates data on neighboring devices (`vecini`) with the script's `result`.
                    for dispozitiv in self.device.vecini:
                        dispozitiv.set_data(zona, result)
                    
                    # Updates its own data with the script's `result`.
                    self.device.set_data(zona, result)

                # Releases the location-specific lock.
                self.device.lista_loc[zona].release()
                
            # Block Logic: All worker threads within the same device synchronize using an internal barrier.
            # This ensures all scripts assigned to the current device are processed before proceeding.
            self.bar_sync_threaduri.wait()

            # Block Logic: Only worker thread 0 resets `timepoint_gata` for the next cycle.
            if self.nr_thread == 0: 
                self.device.timepoint_gata.clear()
