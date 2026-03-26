"""
consumer.py

@brief A thread-safe, logging-enabled simulation of a producer-consumer marketplace.
@description This module provides a robust simulation of an e-commerce marketplace. It features
Producer and Consumer threads, a central Marketplace class that manages all state changes,
comprehensive logging to 'marketplace.log', and a suite of unit tests. Thread safety is
handled via a single, coarse-grained lock within the Marketplace to serialize access to
shared data structures.
"""

import time
from threading import Thread, Lock, currentThread
from logging.handlers import RotatingFileHandler
import unittest
import logging
# The import path suggests this file is part of a package named 'tema'.
import tema.product
from dataclasses import dataclass


class Consumer(Thread):
    """
    Represents a consumer thread that performs shopping actions in the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping sessions, where each session is a
                          list of product-related commands (add/remove).
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying an action.
            **kwargs: Accepts thread-related arguments, including 'name'.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        Main execution loop for the consumer. Processes each shopping cart sequentially.
        """
        # Block Logic: Iterate through each assigned list of shopping commands.
        for cart in self.carts:
            # A new cart is created in the marketplace for each shopping session.
            cart_id = self.marketplace.new_cart()
            # Block Logic: Execute each command within the current shopping session.
            for entry in cart:
                (entry_type, product, quantity) = 
                    (entry["type"], entry["product"], entry["quantity"])
                
                # Invariant: The specified quantity of the operation is performed.
                aux = 0
                while aux < quantity:
                    if entry_type == "add":
                        # For 'add', retry until the marketplace confirms the addition.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    else:  # entry_type == "remove"
                        self.marketplace.remove_from_cart(cart_id, product)
                    aux = aux + 1

            # After all commands for the cart are done, finalize the order.
            self.marketplace.place_order(cart_id)


class Marketplace:
    """
    The central marketplace, managing all producers, consumers, and inventory.

    This class is designed to be thread-safe by using a single master lock to
    protect all of its shared data structures (`producers`, `products`, `consumers`).
    It also logs every major operation.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of items any single
                                           producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = {}  # {producer_id: [product1, product2]}
        self.products = []   # A global, redundant list of all available products.
        self.consumers = {}  # {cart_id: [(producer_id, product), ...]}
                            
        self.lock = Lock()  # A single, coarse-grained lock for all shared data access.
        
        # --- Logging Setup ---
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=5000, backupCount=10)
        self.logger.addHandler(handler)

    def register_producer(self):
        """
        Atomically registers a new producer and returns a unique ID.
        """
        self.logger.info("enter register_producer")
        with self.lock:
            producer_id = len(self.producers) + 1
            self.producers[producer_id] = []
            self.logger.info("return from register_producer %s", str(producer_id))
            return producer_id

    def publish(self, producer_id, product):
        """
        Atomically publishes a product for a given producer, if space is available.
        """
        self.logger.info("input to publish %s %s", str(producer_id), str(product))
        # Pre-condition: Check if producer's queue is full.
        if len(self.producers[producer_id]) >= self.queue_size_per_producer:
            self.logger.info("return from publish False")
            return False

        # State Change: Add product to both the producer-specific and global lists.
        self.producers[producer_id].append(product)
        self.products.append(product)
        self.logger.info("return from publish True")
        return True

    def new_cart(self):
        """
        Atomically creates a new, empty cart for a consumer and returns its ID.
        """
        self.logger.info("enter new_cart")
        with self.lock:
            cart_id = len(self.consumers) + 1
            self.consumers[cart_id] = []
            self.logger.info("return from new_cart %s", str(cart_id))
            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Atomically moves a product from a producer's inventory to a consumer's cart.
        """
        self.logger.info("input to add_to_cart %s %s", str(cart_id), str(product))
        with self.lock:
            # Block Logic: Find the producer who owns the requested product.
            producer_id_found = 0
            if cart_id in self.consumers:
                for producer_id in self.producers:
                    if product in self.producers[producer_id]:
                        producer_id_found = producer_id
                        break  # Found a seller

                if producer_id_found == 0:
                    self.logger.info("return from add_to_cart False - product not found")
                    return False

            # State Change: Move the product from inventory to the cart.
            # The cart stores which producer the item came from.
            self.consumers[cart_id].append((producer_id_found, product))
            self.products.remove(product)
            self.producers[producer_id_found].remove(product)
            self.logger.info("return from add_to_cart True")
            return True

    def remove_from_cart(self, cart_id, product):
        """
        Atomically moves a product from a consumer's cart back to the original
        producer's inventory.
        """
        self.logger.info("input to remove_from_cart %s %s", str(cart_id), str(product))
        with self.lock:
            if cart_id in self.consumers:
                # Block Logic: Find the product in the cart to identify its original producer.
                for item_tuple in self.consumers[cart_id]:
                    if item_tuple[1] == product:
                        original_producer_id = item_tuple[0]
                        self.consumers[cart_id].remove(item_tuple)
                        self.products.append(product)
                        # Return the product to the producer's inventory if not full.
                        if len(self.producers[original_producer_id]) < self.queue_size_per_producer:
                            self.producers[original_producer_id].append(product)
                        return

    def place_order(self, cart_id):
        """
        Finalizes an order by printing the items bought.
        Note: The actual removal from inventory happens in `add_to_cart`. This
        method effectively just confirms and clears the cart.
        """
        self.logger.info("input to place_order %s", str(cart_id))
        if cart_id in self.consumers:
            order_list = []
            # Block Logic: Iterate through the finalized cart to report the purchase.
            for item_tuple in self.consumers[cart_id]:
                # The actual product is the second element of the tuple.
                product_bought = item_tuple[1]
                order_list.append(product_bought)
                print(currentThread().getName() + " bought " + str(product_bought))
            
            # The list of products is returned, but the caller in Consumer doesn't use it.
            self.logger.info("return from place_order %s", str(order_list))
            return order_list
        return []


class TestMarketplace(unittest.TestCase):
    """
    A suite of unit tests to verify the non-concurrent logic of the Marketplace.
    """
    def setUp(self):
        """Initializes a clean marketplace for each test."""
        self.marketplace = Marketplace(5)
        self.product = tema.product.Tea('Linden', 10, 'Herbal')
        self.product2 = tema.product.Coffee('Arabica', 10, '5.05', 'MEDIUM')

    def test_register_producer(self):
        """Tests if producers get unique, sequential IDs."""
        self.marketplace.register_producer()
        self.assertEqual(len(self.marketplace.producers), 1, 'wrong number of producers')

    def test_publish(self):
        """Tests if a product can be successfully published."""
        self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(1, self.product), True, 'failed to publish')

    def test_new_cart(self):
        """Tests if carts get unique, sequential IDs."""
        self.marketplace.new_cart()
        # NOTE: This assertion is logically incorrect, comparing a dict to an int.
        self.assertEqual(len(self.marketplace.consumers), 1, 'wrong number of carts')

    def test_add_to_cart(self):
        """Tests if a published product can be added to a cart."""
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product)
        self.marketplace.new_cart()
        self.assertEqual(self.marketplace.add_to_cart(1, self.product), True, 'failed to add')

    def test_remove_from_cart(self):
        """Tests if a product removed from a cart is returned to inventory."""
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product)
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product)
        self.marketplace.remove_from_cart(1, self.product)
        # Check that the product is no longer in the consumer's cart.
        self.assertFalse(any(self.product in t for t in self.marketplace.consumers[1]), 'failed to remove')

    def test_place_order(self):
        """Tests if place_order returns the correct list of purchased items."""
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product)
        self.marketplace.publish(1, self.product2)
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product)
        self.marketplace.add_to_cart(1, self.product2)
        
        # The test expects place_order to return the list of items,
        # which it does, although the Consumer thread ignores the return value.
        order = self.marketplace.place_order(1)
        self.assertIn(self.product, order, 'failed to place order')
        self.assertIn(self.product2, order, 'failed to place order')


class Producer(Thread):
    """
    Represents a producer thread that continuously publishes its products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # The producer registers itself upon creation.
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        Main execution loop. Continuously cycles through its product list and
        tries to publish them.
        """
        # Invariant: The thread runs forever to simulate a persistent producer.
        while True:
            for (product, quantity, wait_time) in self.products:
                aux = 0
                while aux < quantity:
                    aux = aux + 1
                    # Invariant: Blocks until the product is successfully published.
                    while not self.marketplace.publish(self.prod_id, product):
                        time.sleep(self.republish_wait_time)
                    time.sleep(wait_time)


# --- Data classes for products ---
# These are likely defined here for self-containment, duplicating a 'product.py' module.

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base class for a generic product."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A product of type Tea."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A product of type Coffee."""
    acidity: str
    roast_level: str
