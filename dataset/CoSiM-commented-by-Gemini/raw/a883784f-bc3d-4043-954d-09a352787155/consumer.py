"""
This module simulates a multi-threaded producer-consumer marketplace using a
more complex and fine-grained locking mechanism than a single-lock approach.

It defines `Producer` and `Consumer` threads that interact with a central
`Marketplace` class. The `Marketplace` manages inventory and carts by using
multiple locks to separate different kinds of operations. However, the inventory
management is complex and inefficient, and several potential bugs exist.
"""

from threading import Thread, Lock
from time import sleep
import logging
from logging.handlers import RotatingFileHandler

# Note: The original file had circular imports and was split. The classes are
# rearranged here into a logical order for a single-file representation.

class Marketplace:
    """
    The central marketplace managing inventory and carts with multiple locks.

    This implementation attempts to increase concurrency with fine-grained locking,
    but the logic for inventory management is complex and inefficient.
    """

    # --- Class-level logger setup ---
    logger = logging.getLogger('marketplace.log')
    logger.setLevel(logging.INFO)
    logger.addHandler(RotatingFileHandler('marketplace.log', maxBytes=2000, backupCount=10))

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The max number of items each producer
                                           can have listed at once.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = -1
        self.consumer_id = -1
        # The state is managed by a complex dictionary of lists.
        # self.producers maps producer_id -> [[product, status_flag], ...]
        self.producers = {}
        # self.carts maps cart_id -> [[producer_id, product, original_index], ...]
        self.carts = {}
        # --- Fine-grained locking ---
        self.publish_lock = Lock()
        self.register_lock = Lock()
        self.producer_lock = Lock() # A coarse lock protecting the entire producers dictionary.
        self.cart_lock = Lock()

    def register_producer(self):
        """Registers a new producer under lock and returns its ID."""
        with self.register_lock:
            self.logger.info('Start - register_producer')
            self.producer_id += 1
            self.producers[self.producer_id] = []
            self.logger.info('End - register_producer')
            return self.producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product for a given producer.

        Warning: The initial check for queue size is outside the lock, creating
        a potential race condition where multiple threads could pass the check
        before any acquire the lock.
        """
        with self.publish_lock:
            self.logger.info('Start - publish')
            success = False
            producer_inventory = self.producers.get(producer_id)
            if producer_inventory is not None and len(producer_inventory) < self.queue_size_per_producer:
                # The product status is '1' for available.
                producer_inventory.append([product, 1])
                success = True
            self.logger.info('End - publish')
            return success

    def new_cart(self):
        """Creates a new, empty cart for a consumer under lock."""
        with self.cart_lock:
            self.logger.info('Start - new_cart')
            self.consumer_id += 1
            self.carts[self.consumer_id] = []
            self.logger.info('End - new_cart')
            return self.consumer_id

    def add_to_cart(self, cart_id, product):
        """
        Adds an available product to a cart.

        This method has very high complexity as it iterates through all producers
        and all their products to find a matching, available item. The entire
        operation is protected by a single `producer_lock`, blocking all other
        cart operations.
        """
        with self.producer_lock:
            self.logger.info('Start - add_to_cart, params = %d, %s', cart_id, product)
            found_flag = False
            for producer_id, products in self.producers.items():
                for i, prod_entry in enumerate(products):
                    # Check if product matches and is available (flag == 1).
                    if prod_entry[0] == product and prod_entry[1] == 1:
                        # Mark the product as unavailable (in a cart).
                        prod_entry[1] = 0
                        # Store a "pointer" to the item in the cart.
                        self.carts[cart_id].append([producer_id, product, i])
                        found_flag = True
                        break
                if found_flag:
                    break
            self.logger.info('End - add_to_cart')
            return found_flag

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and makes it available again.

        Warning: This method lacks locking, creating a race condition. Multiple
        threads could try to modify `self.carts` and `self.producers` concurrently.
        """
        self.logger.info('Start - remove_from_cart, params = %d, %s', cart_id, product)
        for entry in self.carts[cart_id]:
            if entry[1] == product:
                self.carts[cart_id].remove(entry)
                # Use the "pointer" to find the original item and reset its availability flag.
                self.producers[entry[0]][entry[2]][1] = 1
                break
        self.logger.info('End - remove_from_cart')

    def place_order(self, cart_id):
        """Returns a list of products in the cart for the final order."""
        self.logger.info('Start - place_order, params = %d', cart_id)
        # Returns only the product object from the [producer, product, index] entry.
        return [entry[1] for entry in self.carts.get(cart_id, [])]


class Producer(Thread):
    """A thread that simulates a producer publishing products."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        Continuously registers and publishes products.

        Warning: The `while True` loop causes the producer to repeatedly
        register for a new ID and publish its entire catalog, which is likely a bug.
        A correct implementation would register once.
        """
        while True:
            id_ = self.marketplace.register_producer()
            for product_info in self.products:
                product, quantity, wait_time = product_info
                for _ in range(quantity):
                    # Busy-wait until the product can be published.
                    while self.marketplace.publish(id_, product) is False:
                        sleep(self.republish_wait_time)
                    sleep(wait_time)

class Consumer(Thread):
    """A thread that simulates a consumer filling a cart and placing an order."""
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """Processes each shopping cart configuration."""
        cart_id = self.marketplace.new_cart()
        for cart_config in self.carts:
            for operation in cart_config:
                action = operation["type"]
                product = operation["product"]
                quantity = operation["quantity"]
                
                for _ in range(quantity):
                    if action == "add":
                        # Busy-wait until the item can be added to the cart.
                        while self.marketplace.add_to_cart(cart_id, product) is False:
                            sleep(self.retry_wait_time)
                    else: # "remove"
                        self.marketplace.remove_from_cart(cart_id, product)

        # Finalize and print the order.
        final_cart_contents = self.marketplace.place_order(cart_id)
        for product in final_cart_contents:
            print(f"{self.name} bought {product}", flush=True)

# Example usage function (not part of the core simulation logic).
def unittesting():
    market = Marketplace(10)
    # ... (rest of test function)
