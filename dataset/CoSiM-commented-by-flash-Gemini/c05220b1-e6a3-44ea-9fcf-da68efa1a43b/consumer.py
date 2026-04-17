"""
@c05220b1-e6a3-44ea-9fcf-da68efa1a43b/consumer.py
@brief multi-threaded electronic marketplace simulation with pessimistic concurrency.
This implementation provides a framework for simulated trade where Producers supply goods 
and Consumers execute transaction sequences. It employs a high-contention locking 
strategy and manual state tracking for individual items ('Disponibil' vs 'Indisponibil') 
to manage distributed inventory.

Domain: Concurrent Systems, Resource Contention, Pessimistic Locking.
"""

from threading import Thread, Lock
import time


class Consumer(Thread):
    """
    Simulation entity representing a shopper.
    Functional Utility: Sequentially processes a collection of shopping carts. 
    Note: Uses a class-level lock that may serialize all instances of Consumer.
    """
    
    cart_id = -1
    name = ''
    # Shared lock used to synchronize consumer threads.
    my_lock = Lock()

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer.
        @param carts: Nested list of shopping operations.
        @param marketplace: Shared trading hub.
        @param retry_wait_time: Interval for retrying unavailable items.
        """
        Thread.__init__(self)
        self.carts = carts
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        self.name = kwargs['name']

    def run(self):
        """
        Main transactional loop.
        Logic: Executes all operations in all assigned carts under a global mutex.
        """
        self.my_lock.acquire()
        for i in range(len(self.carts)):
            # Secure a session for the current cart.
            self.cart_id = self.marketplace.new_cart()
            for j in range(len(self.carts[i])):
                if self.carts[i][j]['type'] == 'add':
                    for k in range(self.carts[i][j]['quantity']):
                        verify = False
                        # Retry loop for item acquisition.
                        while not verify:
                            verify = self.marketplace.add_to_cart(self.cart_id,
                                                                  self.carts[i][j]['product']
                                                                  )
                            if not verify:
                                time.sleep(self.retry_wait_time)

                elif self.carts[i][j]['type'] == 'remove':
                    for k in range(self.carts[i][j]['quantity']):
                        self.marketplace.remove_from_cart(self.cart_id, self.carts[i][j]['product'])
            
            # Commit the order and print results.
            list_1 = self.marketplace.place_order(self.cart_id)
            for k in range(len(list_1) - 1, -1, -1):
                print(self.name + ' bought ' + str(list_1[k][0]))
                # Functional Utility: cleanup step to restore marketplace state.
                self.marketplace.remove_from_cart(self.cart_id, list_1[k][0])
        self.my_lock.release()


from threading import Lock


class Marketplace:
    """
    Central hub for coordinating supply and demand.
    Functional Utility: Manages producer queues and consumer carts using explicit 
    state transitions and fine-grained locking during search operations.
    """
    
    id_producer = 0
    id_cart = 0
    queues = []
    carts = []
    # Locks used to protect specific operational phases (add vs remove).
    my_Lock1 = Lock()
    my_Lock2 = Lock()
    done = 0

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        @param queue_size_per_producer: Capacity per producer buffer.
        """
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """Registers a new producer and returns its index."""
        self.queues.append([])
        self.id_producer = self.id_producer + 1
        return self.id_producer - 1

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace.
        @return: True if successful, False if producer capacity is reached.
        """
        if len(self.queues[producer_id]) >= self.queue_size_per_producer:
            return False
        # State: Items are published with a 'Disponibil' status.
        self.queues[producer_id].append([product, "Disponibil"])
        return True

    def new_cart(self):
        """Initializes a new consumer session context."""
        self.carts.append([])
        self.id_cart = self.id_cart + 1
        return self.id_cart - 1

    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from a producer queue to a consumer cart.
        Logic: Scans all queues for a matching 'Disponibil' item. 
        Note: Performs lock acquire/release for every item check.
        """
        verify = 0
        for i in range(len(self.queues)):
            for j in range(len(self.queues[i])):
                self.my_Lock1.acquire()
                if product == self.queues[i][j][0] \
                        and self.queues[i][j][1] == 'Disponibil' \
                        and verify == 0:
                    # Mark as claimed and update session state.
                    self.carts[cart_id].append([product, i])
                    self.queues[i][j][1] = 'Indisponibil'
                    verify = 1
                    self.my_Lock1.release()
                    break
                self.my_Lock1.release()
                if verify == 1:
                    break
        if verify == 1:
            return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Restores a product from a cart back to its producer's available pool.
        """
        for i in range(len(self.carts[cart_id])):
            if product == self.carts[cart_id][i][0]:
                for j in range(len(self.queues[self.carts[cart_id][i][1]])):
                    self.my_Lock2.acquire()
                    if self.queues[self.carts[cart_id][i][1]][j][0] == product \
                            and self.queues[self.carts[cart_id][i][1]][j][1] == 'Indisponibil':
                        # Restore status to 'Disponibil'.
                        self.queues[self.carts[cart_id][i][1]][j][1] = 'Disponibil'
                        self.carts[cart_id].remove(self.carts[cart_id][i])
                        self.my_Lock2.release()
                        return True
                    self.my_Lock2.release()
        return False

    def place_order(self, cart_id):
        """Returns the final contents of a consumer session."""
        return self.carts[cart_id]


from threading import Thread
import time


class Producer(Thread):
    """
    Simulated manufacturing unit.
    Functional Utility: Continuously cycles through a product catalog and 
    attempts to fulfill production quotas.
    """
    
    producer_id = -1

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        super().__init__(daemon=kwargs['daemon'])
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products

    def run(self):
        """
        Main production and distribution cycle.
        """
        self.producer_id = self.marketplace.register_producer()
        while True:
            for i in range(len(self.products)):
                for j in range(self.products[i][1]):
                    verify = self.marketplace.publish(self.producer_id, self.products[i][0])
                    # simulate production duration.
                    time.sleep(self.products[i][2])
                    if not verify:
                        # backoff if the marketplace is congested.
                        time.sleep(self.republish_wait_time)
                        break


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """Base class for items in the marketplace."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Specialization for beverages."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Specialization for beverages."""
    acidity: str
    roast_level: str
