"""
@cea5fa2c-54d0-456c-aa5e-328e498c2da2/consumer.py
@brief multi-threaded electronic marketplace with a two-phase reservation protocol.
This module implements a sophisticated trading system where products undergo a 
reservation-and-commit lifecycle. Producers publish goods to per-producer supply 
lists, where they can be atomically 'reserved' by consumers. This reservation 
locks the item to a specific session (cart), preventing other consumers from 
claiming it until the transaction is either committed (place_order) or reversed 
(remove_from_cart).

Domain: Transactional Systems, Reservation Patterns, Concurrent Inventory Management.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Simulation entity representing a shopper.
    Functional Utility: Manages automated shopping sessions by iterating through 
    a task list and performing state-persistent operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread.
        @param carts: Nested list of shopping operations.
        @param marketplace: Central transaction hub.
        @param retry_wait_time: Interval for retrying failed reservations.
        """
        Thread.__init__(self, name=kwargs["name"], kwargs=kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Main shopper execution loop.
        Algorithm: Iterative task fulfillment with backoff-and-retry.
        """
        for lista_cumparaturi in self.carts:
            # Atomic creation of a new transaction session.
            cart_id = self.marketplace.new_cart()
            
            failed = True
            # Block Logic: Workload fulfillment loop.
            # Logic: Continues attempting operations until the entire cart task list is empty.
            while failed:
                failed = False
                for operatie in lista_cumparaturi:
                    cantitate = operatie["quantity"]
                    for _ in range(cantitate):
                        if operatie["type"] == "add":
                            if self.marketplace.add_to_cart(cart_id, operatie["product"]):
                                # State Persistence: decrements required quantity upon success.
                                operatie["quantity"] = operatie["quantity"] - 1
                            else:
                                # Functional Utility: Triggers backoff on resource contention.
                                failed = True
                                break
                        elif operatie["type"] == "remove":
                            if self.marketplace.remove_from_cart(cart_id, operatie["product"]):
                                operatie["quantity"] = operatie["quantity"] - 1
                            else:
                                failed = True
                                break
            
            if failed:
                # Synchronous backoff to allow marketplace supply to refresh.
                sleep(self.retry_wait_time)
            else:
                # Commit: finalizes the reservation session.
                self.marketplace.place_order(cart_id)


from threading import Lock
from threading import current_thread


class PublishedProduct:
    """
    State wrapper for a marketplace item.
    Functional Utility: Extends a product data model with a concurrency flag 
    to track reservation status.
    """
    
    def __init__(self, product):
        self.product = product
        self.reserved = False

    def __eq__(self, obj):
        """Equality check incorporating both identity and reservation state."""
        ret = isinstance(obj, PublishedProduct) and self.reserved == obj.reserved
        return ret and obj.product == self.product

class ProductsList:
    """
    Thread-safe container for per-producer inventory.
    Functional Utility: Manages the atomic transitions between available, 
    reserved, and removed states for individual items.
    """
    
    def __init__(self, maxsize):
        """
        Initializes the supply line.
        @param maxsize: Maximum buffer capacity.
        """
        self.lock = Lock()
        self.list = []
        self.maxsize = maxsize

    def put(self, item):
        """Adds a new available product to the supply line."""
        with self.lock:
            if self.maxsize == len(self.list):
                return False
            self.list.append(item)
        return True

    def rezerva(self, item):
        """
        Performs an atomic reservation of an item.
        Logic: Scans for a matching non-reserved product and sets its reserved flag.
        @return: True if reserved successfully, False otherwise.
        """
        item = PublishedProduct(item)
        with self.lock:
            if item in self.list:
                # Transition: Available -> Reserved.
                self.list[self.list.index(item)].reserved = True
                return True
        return False

    def anuleaza_rezervarea(self, item):
        """Reverses a reservation, restoring item availability."""
        item = PublishedProduct(item)
        item.reserved = True
        with self.lock:
            # Transition: Reserved -> Available.
            self.list[self.list.index(item)].reserved = False

    def remove(self, item):
        """Permanently removes a reserved product from the supply list."""
        product = PublishedProduct(item)
        product.reserved = True
        with self.lock:
            # Final state change: removal from global inventory.
            self.list.remove(product)
            return item

class Cart:
    """
    Transactional session storage for a consumer.
    Functional Utility: Tracks the associations between reserved products and 
    their originating producers.
    """

    def __init__(self):
        self.products = []

    def add_product(self, product, producer_id):
        """Stores a reservation-producer pairing."""
        self.products.append((product, producer_id))

    def remove_product(self, product):
        """Removes a pairing and returns the associated producer ID for restoration."""
        for item in self.products:
            if item[0] == product:
                self.products.remove(item)
                return item[1]
        return None

    def get_products(self):
        """Retrieves all currently reserved items in the session."""
        return self.products

class Marketplace:
    """
    Central coordinator managing the two-phase transaction protocol.
    Functional Utility: Mediates between supply (Producers) and demand (Consumers) 
    using atomic reservation and multi-cart session management.
    """
    
    def __init__(self, queue_size_per_producer):
        self.print_lock = Lock()
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_queues = {}
        self.generator_id_producator = 0
        self.generator_id_producator_lock = Lock()

        self.carts = {}
        self.cart_id_generator = 0
        self.cart_id_generator_lock = Lock()

    def register_producer(self):
        """Registers a new supply entity and initializes its thread-safe inventory list."""
        with self.generator_id_producator_lock:
            id_producator = self.generator_id_producator
            self.generator_id_producator += 1
            self.producer_queues[id_producator] = ProductsList(self.queue_size_per_producer)
        return id_producator

    def publish(self, producer_id, product):
        """Publishes a new unit to a specific producer's line."""
        return self.producer_queues[producer_id].put(PublishedProduct(product))

    def new_cart(self):
        """Allocates a new unique session identifier and cart instance."""
        with self.cart_id_generator_lock:
            current_cart_id = self.cart_id_generator
            self.cart_id_generator += 1
            self.carts[current_cart_id] = Cart()
            return current_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Phase 1: Attempt to reserve a product from any available producer.
        Logic: Iteratively polls all supply lines until a reservation is secured.
        """
        producers_num = 0
        with self.generator_id_producator_lock:
            producers_num = self.generator_id_producator

        for producer_id in range(producers_num):
            if self.producer_queues[producer_id].rezerva(product):
                self.carts[cart_id].add_product(product, producer_id)
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Phase 1 Reversal: Cancels a reservation and restores item availability.
        """
        producer_id = self.carts[cart_id].remove_product(product)
        if producer_id is None:
            return False
        self.producer_queues[producer_id].anuleaza_rezervarea(product)
        return True

    def place_order(self, cart_id):
        """
        Phase 2: Commit. Permanently removes reserved items from the global supply.
        """
        lista = list()
        for (produs, producer_id) in self.carts[cart_id].get_products():
            # Atomically remove from the supply line.
            lista.append(self.producer_queues[producer_id].remove(produs))
            with self.print_lock:
                print(f"{current_thread().getName()} bought {produs}")
        return lista


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Manufacturing simulation thread.
    Functional Utility: continuously publishes goods into the marketplace hub.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, name=kwargs["name"], daemon=kwargs["daemon"], kwargs=kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        Main production cycle.
        Logic: Observes individual production times and handles hub backpressure.
        """
        producer_id = self.marketplace.register_producer()
        while True:
            for (product, cantitate, production_time) in self.products:
                # Simulate manufacturing delay.
                sleep(production_time)
                for _ in range(cantitate):
                    # Busy-wait until the marketplace accepts the item.
                    while not self.marketplace.publish(producer_id, product):
                        sleep(self.republish_wait_time)
