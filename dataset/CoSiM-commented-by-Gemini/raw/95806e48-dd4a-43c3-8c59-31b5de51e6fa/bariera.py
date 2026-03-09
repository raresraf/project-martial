"""
This module implements a distributed device simulation using a multi-layered
synchronization strategy and a static round-robin work distribution.

The implementation includes two custom reusable barrier classes: one based on
`Condition` variables and a more robust two-phase version based on `Semaphore`s.
The simulation designates a master device (ID 0) for global setup and a master
thread (ID 0) within each device for per-timepoint coordination.
Variable and method names are in Romanian.
"""
from threading import Condition, Semaphore, Lock, Event, Thread


class BarieraReentrantaCond(object):
    """
    A reusable barrier implemented using a `threading.Condition` object.

    WARNING: This implementation can be susceptible to race conditions in certain
    scenarios, as a woken thread might loop and re-enter the wait cycle before
    all other threads have been woken from the previous cycle.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Causes a thread to wait until all threads have reached the barrier."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # The last thread to arrive notifies all waiting threads.
            self.cond.notify_all()
            # Reset the counter for the next use.
            self.count_threads = self.num_threads
        else:
            # Wait to be notified by the last thread.
            self.cond.wait()
        self.cond.release()

class BarieraReentrantaSem(object):
    """
    A robust two-phase reusable barrier implemented using `threading.Semaphore`s.
    This prevents threads from starting a new wait cycle before all threads have
    exited the previous one. The method names are in Romanian.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait through two phases to ensure reusability."""
        self.prima_faza() # "first phase"
        self.a_doua_faza() # "second phase"

    def prima_faza(self):
        """The first synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def a_doua_faza(self):
        """The second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a device in the simulation. Manages a pool of worker threads
    (`ThreadDispozitiv`) and coordinates with a master device for setup.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.nr_thread = 0 # (thread_number) Counter for round-robin script assignment.
        self.threaduri = [] # (threads) The pool of worker threads.
        self.vecini = [] # (neighbors) List of neighboring devices.
        self.lista_loc = [] # (location_list) Shared list of locks for data locations.
        # A list of lists, one for each worker thread, to hold assigned scripts.
        self.scripturi = [[] for _ in range(8)] # (scripts)
        self.bar_sync_threaduri = BarieraReentrantaCond(8) # Local barrier for own threads.
        self.initializare_gata = Event() # (initialization_done)
        self.timepoint_gata = Event() # (timepoint_done)
        self.bariera_disp = None # (device_barrier) Global barrier for all devices.

    def __str__(self):
        return "Dispozitiv %d" % self.device_id # (Device)

    def setup_devices(self, devices):
        """
        Initializes the simulation environment. Device 0 acts as the master.
        """
        maxim_locatii = 0 # (max_locations)
        lista_disp = devices # (device_list)
        if self.device_id == 0: # Master device setup logic.
            self.bariera_disp = BarieraReentrantaSem(len(devices))

            for loc in self.sensor_data.keys():
                maxim_locatii = max(maxim_locatii, loc)
            lista_disp.remove(self)     
            
            for dispozitiv in lista_disp: # (device)
                for loc in dispozitiv.sensor_data.keys():
                    maxim_locatii = max(maxim_locatii, loc)
                dispozitiv.bariera_disp = self.bariera_disp
            
            for _ in range(maxim_locatii + 1):
                self.lista_loc.append(Lock())
            
            for dispozitiv in lista_disp:
                dispozitiv.lista_loc = self.lista_loc
            
            self.initializare_gata.set()
        else: # Worker device setup logic.
            lista_disp.remove(self)             
            for dispozitiv in lista_disp:       
                if dispozitiv.device_id == 0:
                    # Wait for the master device to finish setting up sync objects.
                    dispozitiv.initializare_gata.wait()
                    break

        # After setup, all devices start their own pool of 8 worker threads.
        for th_curr in range(8):
            thrd = ThreadDispozitiv(self, th_curr, self.bar_sync_threaduri,
                                    self.bariera_disp)
            self.threaduri.append(thrd)
            self.threaduri[-1].start()

    def assign_script(self, script, zona): # (zone)
        """
        Assigns scripts to worker threads in a round-robin fashion.
        """
        if script is not None:
            # Each worker thread has its own list of scripts.
            self.scripturi[self.nr_thread].append((script, zona))
            self.nr_thread = (self.nr_thread + 1) % 8
        else:
            # A None script signals that the timepoint processing can start.
            self.timepoint_gata.set()

    def get_data(self, zona): # (zone)
        return self.sensor_data[zona] if zona in self.sensor_data else None

    def set_data(self, zona, info): # (zone, info)
        if zona in self.sensor_data:
            self.sensor_data[zona] = info

    def shutdown(self):
        for thrd in self.threaduri:
            thrd.join()


class ThreadDispozitiv(Thread): # (DeviceThread)
    """
    A persistent worker thread. Each device runs a pool of these. Thread 0
    acts as a local coordinator for its device.
    """
    def __init__(self, device, nr_thread, bar_sync_threaduri, bar_sync_div):
        self.device = device
        self.nr_thread = nr_thread # (thread_number) This thread's ID (0-7).
        self.bar_sync_threaduri = bar_sync_threaduri # Local barrier for device's threads.
        self.bar_sync_div = bar_sync_div # Global barrier for all devices.
        Thread.__init__(self, name="Dispozitiv %d Thread %d" % (device.device_id, nr_thread))

    def run(self):
        while True:
            # --- Per-Timepoint Coordination ---
            # Pre-condition 1: The local master thread (ID 0) waits for all devices
            # to be ready, then fetches the list of neighbors for this timepoint.
            if self.nr_thread == 0:
                self.bar_sync_div.wait() # Global barrier
                self.device.vecini = self.device.supervisor.get_neighbours()

            # Pre-condition 2: All local threads wait here to ensure the neighbor list
            # is populated before they proceed.
            self.bar_sync_threaduri.wait() # Local barrier

            if self.device.vecini is None:
                break # End of simulation.

            # Pre-condition 3: All local threads wait for the supervisor to finish
            # assigning scripts for this timepoint.
            self.device.timepoint_gata.wait() 

            # --- Work Execution ---
            # Each thread processes only the scripts assigned to its own list.
            for (script, zona) in self.device.scripturi[self.nr_thread]:
                date_script = [] # (script_data)
                
                # --- Critical Section ---
                self.device.lista_loc[zona].acquire()
                for dispozitiv in self.device.vecini:
                    info = dispozitiv.get_data(zona)
                    if info is not None:
                        date_script.append(info)
                
                info = self.device.get_data(zona)
                if info is not None:
                    date_script.append(info)

                if date_script != []:
                    result = script.run(date_script)
                    for dispozitiv in self.device.vecini:
                        dispozitiv.set_data(zona, result)
                    self.device.set_data(zona, result)
                self.device.lista_loc[zona].release()
                # --- End Critical Section ---
            
            # Post-condition 1: All local threads wait here, ensuring all work for
            # this device is complete before the timepoint can end.
            self.bar_sync_threaduri.wait() # Local barrier

            # Post-condition 2: The local master thread resets the event for the next timepoint.
            if self.nr_thread == 0: 
                self.device.timepoint_gata.clear()