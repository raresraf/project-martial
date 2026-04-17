"""
@f732e1c1-3bcd-41d6-852e-1721374788e1/consumer.py
@brief multi-threaded electronic marketplace with per-producer granular locking.
This module implements a coordinated trading environment where Producers supply goods 
to dedicated buffers and Consumers execute transactional operations. The system 
utilizes specialized mutex locks for each producer line, minimizing global 
contention while ensuring atomic item transfers. A double-check pattern in the 
acquisition logic guarantees consistent state even under high concurrency.

Domain: Concurrent Systems, Granular Locking, Producer-Consumer Simulation.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Consumer entity simulating a shopper.
    Functional Utility: Manages a single shopping session (cart) across multiple 
    operation batches, using a polling-based retry strategy for item acquisition.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread.
        @param carts: Nested list of shopping operation batches.
        @param marketplace: Central trading hub.
        @param retry_wait_time: delay between failed acquisition attempts.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        Main execution loop for shopper actions.
        Logic: Secures a unique cart identifier and sequentially fulfills 'add' 
        and 'remove' tasks until the entire workload is processed.
        """
        # Atomic creation of a new transaction context.
        id_cart = self.marketplace.new_cart()
        for cart in self.carts:
            for operation in cart:
                qty = operation['quantity']
                type_op = operation['type']
                product = operation['product']
                
                # Block Logic: Fulfillment loop.
                # Logic: Continues until the required quantity for the current operation is met.
                while qty > 0:
                    if type_op == 'add':
                        ret_value = self.marketplace.add_to_cart(id_cart, product)
                        if ret_value:
                            qty = qty - 1
                        else:
                            # Functional Utility: Fixed-interval backoff when out of stock.
                            sleep(self.retry_wait_time)

                    if type_op == 'remove':
                        # Transaction reversal: restore item to global supply.
                        self.marketplace.remove_from_cart(id_cart, product)
                        qty = qty - 1
        
        # Commit: finalize the session and print purchased inventory.
        cart_list = self.marketplace.place_order(id_cart)
        for product in cart_list:
            print(self.kwargs['name'] + " bought " + str(product))

from threading import Lock


class Marketplace:
    """
    Central coordinator managing inventory buffers and shopper sessions.
    Functional Utility: Uses multiple synchronization primitives to isolate different 
    categories of shared state (registration, session ID, and per-producer stock).
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace hub.
        @param queue_size_per_producer: Capacity limit per supply line.
        """
        self.queue_size = queue_size_per_producer
        # Registry of producer supply lines.
        self.producers = []
        # Registry of active consumer sessions.
        self.carts = []
        
        # Granular Synchronization.
        self.lock_producers = Lock()
        self.lock_consumer = Lock()
        # Efficiency Logic: dedicated mutex for every supply line to reduce contention.
        self.producers_locks = []

    def register_producer(self):
        """
        Allocates a new unique supply line index.
        Logic: Uses a global registration mutex to ensure atomic indexing and 
        creation of a line-specific lock.
        """
        with self.lock_producers:
            id_new_producer = len(self.producers)
            self.producers.append(list())
            self.producers_locks.append(Lock())

        return id_new_producer

    def publish(self, producer_id, product):
        """
        Accepts a product from a producer into the local supply buffer.
        Logic: verifies producer buffer capacity before publication.
        @return: True if accepted, False otherwise.
        """
        if len(self.producers[producer_id]) == self.queue_size:
            return False

        # Atomic state update.
        self.producers[producer_id].append(product)
        return True

    def new_cart(self):
        """Creates a new unique consumer session."""
        with self.lock_consumer:
            id_new_cart = len(self.carts)
            self.carts.append(list())

        return id_new_cart

    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from a producer's buffer to a consumer cart.
        Logic: Performs a global search and utilizes a double-check pattern 
        with per-producer locks to ensure atomic item acquisition.
        """
        # Block Logic: Multi-producer inventory scan.
        for id_producer in range(len(self.producers)):
            # Heuristic check before acquiring lock.
            if product in self.producers[id_producer]:
                # Critical Section: protect specific supply line during transfer.
                with self.producers_locks[id_producer]:
                    # Double-Check Pattern: ensure item wasn't claimed while waiting for lock.
                    if product in self.producers[id_producer]:
                        # Transactional Transfer.
                        self.producers[id_producer].remove(product)
                        # Association: stores product and origin for possible reversal.
                        self.carts[cart_id].append((product, id_producer))
                        return True
        return False

    def remove_from_cart(self, cart_id, product):
        """Restores a product from a cart back to its originating producer."""
        for (current_product, id_producer) in self.carts[cart_id]:
            if current_product == product:
                # Transaction Reversal.
                self.producers[id_producer].append(product)
                self.carts[cart_id].remove((current_product, id_producer))
                break

    def place_order(self, cart_id):
        """Finalizes the purchase and flushes the session context."""
        list_cart = list()
        for element in self.carts[cart_id]:
            list_cart.append(element[0])
        # session reset.
        self.carts[cart_id].clear()
        return list_cart


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Simulation thread representing a manufacturing entity.
    Functional Utility: Manages continuous product cycles and handles backpressure 
    via periodic retries.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the producer and secures a supply line ID.
        """
        Thread.__init__(self, group=None, target=None, name=kwargs['name'], daemon=kwargs['daemon'])
        self.marketplace = marketplace
        self.products = products
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        Main production cycle.
        Algorithm: Iterative manufacturing with synchronous backoff retries.
        """
        id_producer = self.marketplace.register_producer()
        while True:
            for (product, qty, time) in self.products:
                # Block Logic: Quota fulfillment.
                while qty > 0:
                    ret_value = self.marketplace.publish(id_producer, product)
                    if ret_value:
                        # Simulate manufacturing time.
                        sleep(time)
                        qty = qty - 1
                    else:
                        # Functional Utility: backpressure handling.
                        sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """Core data model for goods."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Specialization."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Specialization."""
    acidity: str
    roast_level: str
