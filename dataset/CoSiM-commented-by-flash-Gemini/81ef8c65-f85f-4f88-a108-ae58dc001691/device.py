


from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    Implements a reusable barrier synchronization mechanism using semaphores.
    This barrier allows a fixed number of threads (`num_threads`) to wait for each other
    at a synchronization point, and then allows them to proceed. It is "reusable"
    because it can be used multiple times without reinitialization.
    It uses a two-phase approach to prevent threads from "slipping" through the barrier
    if they arrive too early for the next cycle.
    """


    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any of them can proceed.
        """
        self.num_threads = num_threads
        # Counter for threads in phase 1.
        self.count_threads1 = self.num_threads
        # Counter for threads in phase 2.
        self.count_threads2 = self.num_threads
        # Lock to protect access to the thread counters.
        self.counter_lock = Lock()
        # Semaphore for threads waiting in phase 1. Initialized to 0, so all threads
        # will block until released.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for threads waiting in phase 2. Initialized to 0.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have also called `wait()`. Once all threads have arrived, they are all released.
        This method executes both phase1 and phase2 of the barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        The first phase of the barrier. Threads decrement a counter and the last thread
        to arrive releases all waiting threads for phase 1.
        """
        with self.counter_lock: # Protect access to the counter.
            self.count_threads1 -= 1
            # Conditional Logic: If this is the last thread to arrive at phase 1.
            if self.count_threads1 == 0:
                # Release all `num_threads` from the first semaphore.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the counter for the next use of phase 1.
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire() # Block until released by the last thread in phase 1.

    def phase2(self):
        """
        The second phase of the barrier. Threads decrement a counter and the last thread
        to arrive releases all waiting threads for phase 2. This phase ensures reusability.
        """
        with self.counter_lock: # Protect access to the counter.
            self.count_threads2 -= 1
            # Conditional Logic: If this is the last thread to arrive at phase 2.
            if self.count_threads2 == 0:
                # Release all `num_threads` from the second semaphore.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the counter for the next use of phase 2.
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire() # Block until released by the last thread in phase 2.


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.timepoint_done = Event()
        self.scripts = []

        self.barrier_worker = ReusableBarrier(8)
        self.setup_event = Event()
        self.devices = []
        self.locks = None
        self.neighbours = []
        self.barrier = None
        self.threads = []

        for i in range(8):
            self.threads.append(DeviceThread(self, i))

        for thr in self.threads:
            thr.start()

        self.location_lock = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices)*8)
            self.barrier = barrier
            location_max = 0
            for device in devices:
                device.barrier = barrier
                for location, data in device.sensor_data.iteritems():
                    if location > location_max:
                        location_max = location
                device.setup_event.set()
            self.setup_event.set()

            self.location_lock = [None] * (location_max + 1)

            for device in devices:
                device.location_lock = self.location_lock
                device.setup_event.set()
            self.setup_event.set()

    def assign_script(self, script, location):
        
        busy = 0
        if script is not None:
            self.scripts.append((script, location))
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        busy = 1
                        break

                if busy == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for thr in self.threads:
            thr.join()


class DeviceThread(Thread):
    

    def __init__(self, device, idd):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.idd = idd

    def run(self):
        
        
        self.device.setup_event.wait()

        while True:
            
            if self.idd == 0:
                neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours = neighbours

            self.device.barrier_worker.wait()

            if self.device.neighbours is None:
                break

            self.device.timepoint_done.wait()
            self.device.barrier_worker.wait()

            i = 0
            
            for (script, location) in self.device.scripts:
                if i % 8 == self.idd:
                    with self.device.location_lock[location]:
                        script_data = []
                        
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data != []:
                            
                            result = script.run(script_data)

                            
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            
                            self.device.set_data(location, result)
                i = i + 1

            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
