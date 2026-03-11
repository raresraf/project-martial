"""
This module simulates an e-commerce marketplace with producers and consumers
running as concurrent threads.

NOTE: This file appears to be a combination of multiple modules (consumer,
producer, marketplace, models). The documentation treats it as a single file
but notes the likely intended separation of concerns.

WARNING: The implementation has significant thread-safety issues. The Marketplace
class lacks proper locking for its shared data structures, making it prone to
race conditions when accessed by multiple Producer and Consumer threads. The locks
used within the Consumer and Producer classes are local to each thread instance
and do not protect the shared Marketplace state.
"""

from threading import Lock, Thread
import time
import collections
from dataclasses import dataclass

# ==============================================================================
# Likely intended to be in a 'marketplace.py' file
# ==============================================================================

class Marketplace:
    """
    Acts as the central hub for producers and consumers.

    It manages product inventories from multiple producers and handles shopping
    cart operations. This class is NOT thread-safe.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                                           producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0
        self.id_cart = 0
        # Stores products available from each producer. {producer_id: [product, ...]}
        self.products_lists = collections.defaultdict(list)
        # Stores items currently in each shopping cart. {cart_id: [product, ...]}
        self.carts_lists = collections.defaultdict(list)
        # Tracks which producer a bought item came from, to return it correctly.
        self.bought_items = collections.defaultdict(list)
        self.lock_carts = Lock()
        self.lock_producers = Lock()

    def register_producer(self):
        """
        Assigns a unique ID to a new producer. Thread-safe.
        """
        with self.lock_producers:
            self.id_producer += 1
            return self.id_producer

    def publish(self, producer_id, product):
        """
        Allows a producer to list a product for sale.

        WARNING: Not thread-safe. A race condition exists between checking the
        length and appending to the list.
        """
        if producer_id in self.products_lists:
            if len(self.products_lists[producer_id]) >= self.queue_size_per_producer:
                return False # Inventory is full.
        self.products_lists[producer_id].append(product)
        return True

    def new_cart(self):
        """Creates a new shopping cart and returns its unique ID. Thread-safe."""
        with self.lock_carts:
            self.id_cart += 1
        return self.id_cart

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a shopping cart by searching all producer inventories.

        WARNING: Not thread-safe. Iterating and modifying `products_lists`
        without a global lock can lead to race conditions.
        """
        for key, values in self.products_lists.items():
            for j in values:
                if j == product:
                    self.bought_items[j].append(key)
                    self.carts_lists[cart_id].append(j)
                    self.products_lists[key].remove(j)
                    return True
        return False # Product not found.

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to its original producer.

        WARNING: Not thread-safe. Modifying shared lists without locks.
        """
        self.carts_lists[cart_id].remove(product)
        # Return the product to the producer who originally published it.
        self.products_lists[self.bought_items[product].pop()].append(product)

    def place_order(self, cart_id):
        """Finalizes an order and returns the list of products in the cart."""
        return self.carts_lists[cart_id]

# ==============================================================================
# Likely intended to be in a 'producer.py' file
# ==============================================================================

class Producer(Thread):
    """
    A thread that produces items and publishes them to the marketplace.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id_producer = self.marketplace.register_producer()
        self.lock = Lock() # WARNING: This lock is local and provides no protection.

    def run(self):
        """Continuously produces and publishes products."""
        while True:
            for i in self.products:
                operation_counter = 0
                while operation_counter < i[1]:
                    # This lock protects nothing of consequence, as the race
                    # condition is in the shared marketplace object.
                    with self.lock:
                        if not self.marketplace.publish(self.id_producer, i[0]):
                            time.sleep(self.republish_wait_time)
                        else:
                            operation_counter += 1
                            time.sleep(i[2]) # Wait after successful publication.

# ==============================================================================
# Likely intended to be in a 'consumer.py' file
# ==============================================================================

class Consumer(Thread):
    """
    A thread that simulates a customer shopping in the marketplace.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts # List of shopping operations.
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        self.lock = Lock() # WARNING: This lock is local and provides no protection.

    def run(self):
        """Processes each shopping cart, performing add/remove operations."""
        for i in self.carts:
            id_cart = self.marketplace.new_cart()
            for operation in i:
                operation_counter = 0
                while operation_counter < operation["quantity"]:
                    # This lock does not protect the shared marketplace state
                    # from other consumer threads.
                    with self.lock:
                        if operation["type"] == "add":
                            ret = self.marketplace.add_to_cart(id_cart, operation["product"])
                            if not ret:
                                # Wait and retry if product is not available.
                                time.sleep(self.retry_wait_time)
                            else:
                                operation_counter += 1
                        else: # "remove" operation
                            self.marketplace.remove_from_cart(id_cart, operation["product"])
                            operation_counter += 1
            
            # Finalize the order and print the bought items.
            for product in self.marketplace.place_order(id_cart):
                print("%s bought %s" %(self.kwargs['name'], product))

# ==============================================================================
# Likely intended to be in a 'models.py' or 'products.py' file
# ==============================================================================

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple, immutable data class for a generic product."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class for Tea, inheriting from Product."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class for Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
