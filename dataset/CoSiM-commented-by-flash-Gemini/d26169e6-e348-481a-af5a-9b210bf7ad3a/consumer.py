"""
@d26169e6-e348-481a-af5a-9b210bf7ad3a/consumer.py
@brief multi-threaded electronic marketplace with hierarchical concurrency control.
This module implements a coordinated trading hub where Producers supply goods 
and Consumers execute transactions. The system employs a nested locking model 
(Global lock for session management and Producer-specific locks for inventory) 
to ensure safe item transfers between supply queues and consumer carts.

Domain: Concurrent Systems, Hierarchical Locking, Producer-Consumer Simulation.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Simulation entity representing a customer.
    Functional Utility: Executes a predefined sequence of shopping operations 
    across multiple carts using a polling-based retry strategy for item acquisition.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread.
        @param carts: Nested list of shopping task batches.
        @param marketplace: Central trading hub.
        @param retry_wait_time: delay between failed acquisition attempts.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        Main execution loop for shopper actions.
        Logic: Processes each cart sequentially, attempting to fulfill 'add' 
        requests by polling the marketplace until success.
        """
        for cart in self.carts:
            # Atomic creation of a new transaction context.
            id_cart = self.marketplace.new_cart()
            for purchase in cart:
                if purchase["type"] == 'add':
                    for _ in range(purchase["quantity"]):
                        # Functional Utility: Poll-wait loop for item availability.
                        cart_new_product = self.marketplace.add_to_cart(id_cart,
                                                                        purchase["product"])
                        while not cart_new_product:
                            sleep(self.retry_wait_time)
                            cart_new_product = self.marketplace.add_to_cart(id_cart,
                                                                            purchase["product"])
                else:
                    # Transaction Reversal: remove items from cart back to marketplace.
                    for _ in range(purchase["quantity"]):
                        self.marketplace.remove_from_cart(id_cart, purchase["product"])
            
            # Commit: finalize order and print manifest.
            order = self.marketplace.place_order(id_cart)
            for buy in order:
                print(self.name + ' bought ' + str(buy))


import threading


class Marketplace:
    """
    Central hub coordinating supply and demand through transactional transfers.
    Functional Utility: Manages inventory buffers and consumer sessions using 
    hierarchical locking to minimize system-wide contention.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        @param queue_size_per_producer: Capacity limit per supply line.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.contor_producer = -1
        self.contor_consumer = -1
        # Inventory buffers for each producer.
        self.product_queue = [[]]
        # Session storage for each consumer.
        self.cart_queue = [[]]
        # Mapping to track item-producer origins for consistent transaction reversal.
        self.producer_cart = [[]]
        # Global mutex for session and producer registration.
        self.lock = threading.Lock()
        # Per-producer mutexes for inventory updates.
        self.producer_locks = []

    def register_producer(self):
        """Allocates a new supply line and associated mutex."""
        with self.lock:
            self.contor_producer += 1
            tmp = self.contor_producer
            self.product_queue.append([])
            self.producer_cart.append([])
            self.producer_locks.append(threading.Lock())
        return tmp

    def publish(self, producer_id, product):
        """
        Accepts a product from a producer into the local supply buffer.
        Logic: verifies capacity under a producer-specific lock.
        @return: True if accepted, False otherwise.
        """
        self.producer_locks[producer_id].acquire()
        if self.queue_size_per_producer > len(self.product_queue[producer_id]):
            self.product_queue[producer_id].append(product)
            self.producer_locks[producer_id].release()
            return True
        self.producer_locks[producer_id].release()
        return False

    def new_cart(self):
        """Initializes a new consumer session context."""
        self.lock.acquire()
        self.contor_consumer += 1
        self.cart_queue.append([])
        tmp = self.contor_consumer
        self.lock.release()
        return tmp

    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from a producer's buffer to a consumer cart.
        Logic: Performs a global search across all buffers and atomically moves 
        the item if found, updating the origin map.
        """
        # Block Logic: Availability Search.
        if any(product in list_products for list_products in self.product_queue):
            for products in self.product_queue:
                for prod in products:
                    if prod == product:
                        # Transactional Transfer.
                        self.lock.acquire()
                        tmp = self.product_queue.index(products)
                        # Functional Utility: tracks origin to support order reversal.
                        self.producer_cart[tmp].append((product, cart_id))
                        self.cart_queue[cart_id].append(product)
                        self.product_queue[tmp].remove(product)
                        self.lock.release()
                        return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Restores a product from a cart back to its originating producer.
        Logic: uses the producer_cart map to identify the target buffer.
        """
        self.cart_queue[cart_id].remove(product)
        # Block Logic: origin resolution.
        for producer in self.producer_cart:
            # Logic Note: Implementation has inconsistencies in mapping tuple format.
            if (cart_id, product) in producer:
                tmp = self.producer_cart.index(producer)
                self.producer_cart.remove((cart_id, product))
                # Restore to original producer buffer.
                self.producer_locks[tmp].acquire()
                self.product_queue[tmp].append(product)
                self.producer_locks[tmp].release()

    def place_order(self, cart_id):
        """Finalizes the session and returns the list of purchased goods."""
        return self.cart_queue[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Simulation thread representing a manufacturing entity.
    Functional Utility: Continuously produces items and handles marketplace 
    backpressure via a retrying publication loop.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        Main production cycle.
        Algorithm: Iterative manufacturing with synchronous publication retries.
        """
        id_producer = self.marketplace.register_producer()
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    # Simulate individual unit production time.
                    sleep(product[2])
                    market_confirm = self.marketplace.publish(id_producer, product[0])
                    # Block Logic: Publication Retry loop.
                    while not market_confirm:
                        sleep(self.republish_wait_time)
                        market_confirm = self.marketplace.publish(id_producer, product[0])


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """Base data model for goods."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Product specialization."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Product specialization."""
    acidity: str
    roast_level: str
