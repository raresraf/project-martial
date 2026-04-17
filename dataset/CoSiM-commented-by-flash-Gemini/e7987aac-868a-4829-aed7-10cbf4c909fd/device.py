"""
@e7987aac-868a-4829-aed7-10cbf4c909fd/device.py
@brief Distributed sensor network simulation with interleaved task distribution.
This module implements a coordinated parallel processing framework using a fixed 
pool of persistent worker threads (DeviceThread). Tasks are distributed among workers 
using an interleaved indexing strategy to ensure balanced workload distribution. 
Consistency is maintained through a network-wide pool of binary semaphores 
providing spatial mutual exclusion, and temporal synchronization is achieved 
via specialized simulation-step barriers.

Domain: Parallel Systems, Interleaved Workload Balancing, Spatial Mutual Exclusion.
"""

from threading import Event, Thread, Semaphore
from reusable_barrier import TimePointsBarrier, ClassicBarrier

class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data, coordinates synchronization resource 
    allocation, and orchestrates the internal parallel worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.threads = []
        self.neighbours = []
        self.num_threads = 8

        # Global Synchronization resources populated during setup.
        self.locations_semaphore = None
        self.devices_barrier = None
        self.neighbours_barrier = None
        self.all_devices = None

    def set_neighbours(self, new_neighbours):
        """Updates the cached neighborhood topology."""
        self.neighbours = new_neighbours


    def set_devices_barrier(self, barrier):
        """Injects the shared network-wide barrier."""
        self.devices_barrier = barrier


    def set_locations_semaphore(self, locations_semaphore):
        """Injects the global pool of spatial semaphores."""
        self.locations_semaphore = locations_semaphore

    def get_locations(self, location_list):
        """Helper to aggregate all spatial locations managed by this node."""
        for location in self.sensor_data:
            location_list.append(location)


    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization initialization.
        Logic: Coordinator node (ID 0) discovers all unique spatial locations and 
        pre-allocates a binary semaphore for each to ensure network-wide mutual exclusion.
        """
        if self.device_id == 0:
            barrier = ClassicBarrier(len(devices))
            locations = []
            locations_semaphore = []
            
            # Discovery: aggregate all locations across the entire network.
            for device in devices:
                device.get_locations(locations)
            locations = sorted(list(set(locations)))

            # Atomic Resource Allocation: create a mutex for every spatial location.
            for i in range(0, len(locations)):
                locations_semaphore.append(Semaphore(value=1))

            # Propagation: distribute resources to all peers.
            for device in devices:
                device.set_devices_barrier(barrier)
                device.set_locations_semaphore(locations_semaphore)

        self.all_devices = devices
        # Functional Utility: specialized barrier for simulation timepoints.
        self.neighbours_barrier = TimePointsBarrier(self.num_threads, self)

        # Spawns persistent worker pool.
        for i in range(self.num_threads):
            current_thread = DeviceThread(self, i)
            current_thread.start()
            self.threads.append(current_thread)

    def assign_script(self, script, location):
        """Registers a task and signals completion of the simulation step assignment."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def has_data(self, location):
        """Checks if a sensor location is managed by this node."""
        return location in self.sensor_data

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully joins all local worker threads."""
        for i in range(self.num_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    Worker implementation for the node.
    Functional Utility: Executes computational scripts using an interleaved 
    distribution model and maintains spatial mutual exclusion for data updates.
    """

    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        Main worker execution loop.
        Algorithm: Iterative sequence: 
        Step Rendezvous -> Task Fulfillment (Interleaved) -> Global Barrier.
        """
        while True:
            # Barrier Point: Wait for topology discovery and simulation step start.
            self.device.neighbours_barrier.wait()

            if self.device.neighbours is None:
                break

            # Wait for supervisor to finalize script assignments.
            self.device.timepoint_done.wait()

            if len(self.device.neighbours) != 0:
                devices_with_date = []
                
                # Block Logic: Interleaved Task Distribution.
                # Algorithm: Each thread handles a subset of the script list 
                # based on its unique thread ID (modulo logic).
                for index in range(
                        self.thread_id,
                        len(self.device.scripts),
                        self.device.num_threads):
                    (script, location) = self.device.scripts[index]
                    
                    script_data = []
                    # Critical Section: Spatial mutual exclusion for the location.
                    self.device.locations_semaphore[location].acquire()

                    # Aggregate neighborhood state.
                    for device in self.device.neighbours:
                        if device.has_data(location):
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                                devices_with_date.append(device)

                    # Include local state.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                        devices_with_date.append(self.device)

                    if script_data != []:
                        # Apply computational logic and propagate results to all contributors.
                        result = script.run(script_data)
                        for device in devices_with_date:
                            device.set_data(location, result)
                        devices_with_date = []

                    # Release spatial lock.
                    self.device.locations_semaphore[location].release()
