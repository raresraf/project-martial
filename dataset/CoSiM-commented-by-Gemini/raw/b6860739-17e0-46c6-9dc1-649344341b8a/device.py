from threading import Event, Thread, Lock, Semaphore
# The ReusableBarrier is assumed to be in a 'barrier.py' file.
from barrier import ReusableBarrier
from Queue import Queue


class Device(object):
    """
    Represents a device that internally manages a fixed pool of worker threads
    and uses a producer-consumer queue for script execution.

    Architectural Role: This model is fundamentally different from previous versions.
    Each `Device` instance is a multi-threaded system in itself, spawning 8 worker
    threads (`DeviceThread`) upon initialization. Scripts are not executed by a
    main control loop but are placed on a thread-safe `Queue`, from which the
    worker threads consume tasks. Synchronization between devices is handled
    manually using semaphores.

    Warning: This implementation lacks any locking mechanism for `get_data` and
    `set_data`, making it highly susceptible to race conditions when threads
    from different devices access shared data.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # A thread-safe queue to hold scripts for the worker threads.
        self.scripts = Queue()
        self.lock_device_method = Lock() # This lock is initialized but never used.
        self.neighbours = None
        # A barrier for synchronizing the 8 internal worker threads of this device.
        self.local_barrier = ReusableBarrier(8)
        # A semaphore used for a manual, all-device barrier implementation.
        self.synch_sem = Semaphore(0)
        self.devices = []
        # An event to delay thread execution until `setup_devices` is complete.
        self.startup_event = Event()

        # Create and start a fixed pool of 8 worker threads for this device.
        self.vector_of_threads = []
        for i in range(0, 8):
            dev_thread = DeviceThread(self, i)
            self.vector_of_threads.append(dev_thread)
            dev_thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Stores the list of all devices and signals worker threads to start."""
        self.devices = devices
        self.startup_event.set()

    def assign_script(self, script, location):
        """
        Adds a script to the queue. A `None` script acts as a termination signal.
        
        Functional Utility: This method acts as the "producer". When a `None`
        script is received (signaling the end of a time-step), it places 8
        "poison pills" on the queue, one for each worker thread, to gracefully
        stop their current work loop.
        """
        if script is not None:
            self.scripts.put((script, location))
        else:
            # Add a "poison pill" for each worker thread.
            for _ in range(0, 8):
                self.scripts.put((None, None))

    def get_data(self, location):
        """Retrieves data. Not thread-safe."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates data. Not thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads of this device to terminate."""
        for dev_thread in self.vector_of_threads:
            dev_thread.join()

    def synchronize_devices(self):
        """
        Implements a manual, semaphore-based barrier across all devices.
        
        Functional Utility: This method is called by one thread per device (thread 0)
        to ensure all devices have reached the same point before proceeding with
        the next time-step.
        """
        # Release the semaphore of every other device.
        for device in self.devices:
            device.synch_sem.release()
        # Wait to acquire the semaphore N times, once for each device.
        for _ in self.devices:
            self.synch_sem.acquire()

class DeviceThread(Thread):
    """
    A worker thread that consumes and executes scripts from a shared queue.
    """
    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id # The ID of this thread within the device's pool (0-7).

    def run(self):
        """The main consumer loop for the worker thread."""
        # Wait until the initial device setup is complete.
        self.device.startup_event.wait()
        
        while True:
            # Block Logic: Global and Local Synchronization.
            # Invariant: Thread 0 of each device is responsible for initiating a
            # global synchronization and fetching the neighbor list for its device.
            if self.thread_id == 0:
                self.device.synchronize_devices()
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # All 8 threads of this device wait here. This ensures that global
            # sync is complete and neighbors are fetched before any thread proceeds.
            self.device.local_barrier.wait()
            
            # A `None` neighbor list is the signal to shut down the entire simulation.
            if self.device.neighbours is None:
                break

            # Block Logic: Consume scripts from the queue until a poison pill is received.
            while True: 
                # `get` is a blocking call; the thread waits here if the queue is empty.
                (script, location) = self.device.scripts.get()
                
                # A `None` script is the "poison pill" signaling the end of this time-step's work.
                if script is None:
                    break
                
                script_data = []
                # Data gathering phase. This is not thread-safe across devices.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Script execution and data propagation phase.
                if script_data:
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

            # After consuming a poison pill, wait here for all 7 other threads
            # of this device to also finish their work for the time-step.
            self.device.local_barrier.wait()