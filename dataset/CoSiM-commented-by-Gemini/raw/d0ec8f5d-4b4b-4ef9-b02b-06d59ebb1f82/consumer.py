
"""
@file consumer.py (and others)
@brief A multi-threaded producer-consumer simulation for an e-commerce marketplace.
@details This module defines a producer-consumer system with a central marketplace.
It includes classes for the Consumer, Marketplace, Producer, and a test suite.

@warning CRITICAL CONCURRENCY FLAWS: This implementation contains several race
conditions. Key methods in the `Marketplace` class, such as `publish` and
`place_order`, access shared data structures without any locking, making them
not thread-safe. The `add_to_cart` method also contains a Time-of-check
to time-of-use (TOCTOU) vulnerability. These flaws can lead to data corruption
and incorrect behavior under concurrent load.

NOTE: This file appears to be a concatenation of multiple Python files.
"""


import threading
import time
import collections
import json
import logging
from logging.handlers import RotatingFileHandler
import unittest

# --- Consumer Logic ---
class Consumer(threading.Thread):
    """
    Represents a consumer that buys products from the marketplace.

    Each consumer thread processes a list of shopping commands and interacts with
    the marketplace to fulfill them. A key feature of this implementation is that
    each consumer instance uses a single, persistent cart for all its operations.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of shopping lists for the consumer to process.
        :param marketplace: The shared Marketplace instance.
        :param retry_wait_time: Time to wait before retrying a failed operation.
        """
        threading.Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        # Functional Utility: Each consumer gets one unique cart ID for its lifetime.
        self.consumert_cart_id = self.marketplace.new_cart()
        self.retry_wait_time = retry_wait_time

    def run(self):
        """The main execution loop for the consumer thread."""
        if self.marketplace is None:
            return

        # Block Logic: Process each shopping list assigned to this consumer.
        for cart_entry in self.carts:
            for elem in cart_entry:
                # Invariant: Loop until the desired quantity of the product is processed.
                while elem['quantity'] > 0:
                    # Perform 'add' or 'remove' operation.
                    if elem['type'] == 'add':
                        valid_op = self.marketplace.add_to_cart(
                                                        self.consumert_cart_id,
                                                        elem['product'])
                    else:
                        valid_op = self.marketplace.remove_from_cart(
                                                        self.consumert_cart_id,
                                                        elem['product'])

                    # Block Logic: Implements a busy-wait retry mechanism.
                    # If an operation fails, wait and try again.
                    if not valid_op:
                        time.sleep(self.retry_wait_time)
                    else:
                        elem['quantity'] = elem['quantity'] - 1

            # After processing a shopping list, place the order.
            products = self.marketplace.place_order(self.consumert_cart_id)
            for product_types in products:
                for product in product_types:
                    print(f'{str(threading.currentThread().getName())} bought {str(product)}')


# --- Marketplace Logic ---
class Marketplace:
    """
    The central marketplace that coordinates producers and consumers.

    @warning This class is not thread-safe. Several methods modify shared state
    without proper synchronization, leading to race conditions.
    """
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        handlers=[RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=5)],
        format='%(asctime)s - %(message)s',
        level=logging.INFO)

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        :param queue_size_per_producer: The maximum number of products a single
                                        producer can have listed at one time.
        """
        self.max_products_allowed = queue_size_per_producer
        self.ticket_nr = 0  # Counter for unique producer IDs.
        self.products_nr = [] # Tracks item count per producer, indexed by producer_id.
        self.carts_nr = 0 # Counter for unique cart IDs.
        self.products = collections.defaultdict(list) # Maps product to a list of producer IDs.
        self.cart_products = {} # Maps cart_id to products within it.

        # Concurrency: Locks for specific operations.
        self.register_producer_lock = threading.Lock()
        self.new_cart_lock = threading.Lock()
        self.add_to_cart_lock = threading.Lock() # Used for both add and remove.

        logging.info('Started Marketplace process.')

    def register_producer(self):
        """Registers a new producer and returns a unique ID."""
        with self.register_producer_lock:
            self.products_nr.append(0)
            producer_id = self.ticket_nr
            self.ticket_nr += 1
            logging.info('Registered producer with ID %s.', producer_id)
            return producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.
        @warning NOT THREAD-SAFE. This method reads and writes shared state
        (`self.products_nr` and `self.products`) without a lock, creating a
        race condition between concurrent producers.
        """
        if self.products_nr[producer_id] >= self.max_products_allowed:
            logging.info('Producer %s published too many products.', producer_id)
            return False

        self.products[product].append(producer_id)
        self.products_nr[producer_id] += 1
        logging.info('Producer %s published product %s.', producer_id, product)
        return True

    def new_cart(self):
        """Creates a new, empty cart and returns its unique ID."""
        with self.new_cart_lock:
            cart_id = self.carts_nr
            self.carts_nr += 1
            logging.info('Created new cart with ID %s.', cart_id)
            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart.
        @warning TOCTOU BUG: The initial check for product existence is performed
        outside the lock. Another thread could remove the last instance of the
        product between the check and when the lock is acquired, leading to an error.
        """
        if product not in self.products or len(self.products[product]) <= 0:
            logging.info('Product %s does not exist on marketplace.', product)
            return False

        with self.add_to_cart_lock:
            # Re-check inside the lock to mitigate the TOCTOU bug, though the first check is still problematic.
            if len(self.products.get(product, [])) <= 0:
                return False

            producer_picked = self.products[product].pop(0)
            self.products_nr[producer_picked] -= 1

            if cart_id not in self.cart_products:
                self.cart_products[cart_id] = {}

            if product in self.cart_products[cart_id]:
                self.cart_products[cart_id][product].append(producer_picked)
            else:
                self.cart_products[cart_id][product] = [producer_picked]

        logging.info('Product %s was added to cart %s.', product, cart_id)
        return True

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the marketplace inventory."""
        if cart_id > self.carts_nr or cart_id not in self.cart_products or 
           product not in self.cart_products[cart_id] or 
           len(self.cart_products[cart_id][product]) <= 0:
            logging.info('Cannot remove product %s from cart %s.', product, cart_id)
            return False

        with self.add_to_cart_lock:
            removed_product_producer_id = self.cart_products[cart_id][product].pop(0)
            self.products[product].append(removed_product_producer_id)
            self.products_nr[removed_product_producer_id] += 1
        return True

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.
        @warning NOT THREAD-SAFE. Accesses `self.cart_products` without a lock.
        """
        if cart_id not in self.cart_products:
            return []
        
        order_list = []
        for product, producers in self.cart_products[cart_id].items():
            product_list = [product] * len(producers)
            order_list.append(product_list)

        # Clear the cart after the order is "placed".
        self.cart_products[cart_id] = {}
        return order_list

# --- Unit Test Suite ---
class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.
    Note: These tests are sequential and will not expose the concurrency flaws.
    """
    max_queue = 3
    def setUp(self):
        self.marketplace = Marketplace(self.max_queue)

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)

    # ... other tests ...

# --- Producer Logic ---
class Producer(threading.Thread):
    """
    Represents a producer that supplies products to the marketplace.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.
        :param products: A list of product tuples (product_name, quantity, production_time).
        :param marketplace: The shared Marketplace instance.
        :param republish_wait_time: Time to wait before retrying a failed publish.
        """
        threading.Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Functional Utility: The producer registers itself once upon creation.
        self.producer_id = marketplace.register_producer()

    def run(self):
        """Main execution loop for the producer thread."""
        while True:
            for (product_type, remaining_quantity, production_time) in self.products:
                while remaining_quantity > 0:
                    # Implements a busy-wait retry loop for publishing.
                    if not self.marketplace.publish(self.producer_id, product_type):
                        time.sleep(self.republish_wait_time)
                    else:
                        time.sleep(production_time)
                        remaining_quantity -= 1
